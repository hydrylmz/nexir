// src/interop/external_texture.rs
// Binds a GPU texture's underlying memory to a CUDA array via external-memory
// import, so NVDEC/NVENC can read or write the same allocation wgpu renders into.
//
// The direction of the escape hatch matters and is the reason this file looks the
// way it does.  CUDA cannot import an arbitrary wgpu texture: the allocation has
// to be created up front with export-compatible flags
// (`D3D12_HEAP_FLAG_SHARED` on D3D12), and wgpu's own `create_texture` never sets
// them.  wgpu-hal also keeps `dx12::Texture`'s `ID3D12Resource` private, so there
// is no way to reach into a texture wgpu already made.
//
// So the allocation is created HERE, by us, as a shared committed resource, and
// then handed to wgpu with `Device::create_texture_from_hal` — see
// [`SharedTexture::new`].  Both sides end up pointing at one allocation:
//
//     ID3D12Resource (HEAP_FLAG_SHARED)
//        ├── CreateSharedHandle  → cuImportExternalMemory → CUarray  (CUDA / NVENC)
//        └── texture_from_raw    → create_texture_from_hal → wgpu::Texture
//
// [`ExternalTexture::import`] remains available for the case where the caller
// already holds an export-compatible resource handle.

use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::ffi::cuda_gl_vk_interop::*;
use crate::interop::ffi::cuda_driver::{CUarray, CUmipmappedArray, CUexternalMemory, CUDA_SUCCESS};
use crate::interop::capability::InteropTransport;
use crate::render::device::GpuDevice;
use std::sync::Arc;

/// How a wgpu texture format maps onto a CUDA array descriptor.
///
/// CUDA validates this against the imported resource's own footprint, so the
/// channel count and element type must describe the SAME bytes per pixel as the
/// D3D12/Vulkan format — not merely something of the right total size.
#[derive(Copy, Clone, Debug)]
pub struct CudaArrayFormat {
    /// `CU_AD_FORMAT_*`.
    pub format:       u32,
    /// Channels per element (1..=4).
    pub num_channels: u32,
    /// Bytes per pixel, used to size the import.
    pub bytes_per_px: u32,
}

impl CudaArrayFormat {
    /// The CUDA array descriptor for a wgpu texture format.
    ///
    /// Only the formats the interop paths actually use are listed; anything else
    /// is a programming error rather than a runtime condition, because the caller
    /// chose the format.
    pub fn for_wgpu(format: wgpu::TextureFormat) -> Option<Self> {
        use wgpu::TextureFormat as F;
        Some(match format {
            // ABGR10 packed by Abgr10RepackNode: one 32-bit word per pixel.
            F::R32Uint => Self {
                format: CU_AD_FORMAT_UNSIGNED_INT32, num_channels: 1, bytes_per_px: 4,
            },
            // NV12 luma plane.
            F::R8Unorm => Self {
                format: CU_AD_FORMAT_UNSIGNED_INT8, num_channels: 1, bytes_per_px: 1,
            },
            // NV12 interleaved chroma plane.
            F::Rg8Unorm => Self {
                format: CU_AD_FORMAT_UNSIGNED_INT8, num_channels: 2, bytes_per_px: 2,
            },
            F::Rgba8Unorm => Self {
                format: CU_AD_FORMAT_UNSIGNED_INT8, num_channels: 4, bytes_per_px: 4,
            },
            F::Rgba16Float => Self {
                format: CU_AD_FORMAT_HALF, num_channels: 4, bytes_per_px: 8,
            },
            _ => return None,
        })
    }
}

/// A GPU texture's memory additionally accessible to CUDA as a `CUarray`.
/// Owns the CUDA-side import; the texture itself is owned elsewhere.
pub struct ExternalTexture {
    cuda_array: CUarray,
    mipmap:     CUmipmappedArray,
    ext_mem:    CUexternalMemory,
    pub width:  u32,
    pub height: u32,
    /// A share of the CUDA primary context the import was made against.
    ///
    /// Load-bearing, not bookkeeping.  `cuMipmappedArrayDestroy` and
    /// `cuDestroyExternalMemory` in [`ExternalTexture::release`] are only valid
    /// while that context exists; if the last `Arc<CudaContext>` elsewhere drops
    /// first, `cuDevicePrimaryCtxRelease` tears the context down and this
    /// object's own teardown becomes an access violation.  Holding the `Arc`
    /// here makes that ordering impossible to get wrong from the outside —
    /// previously it depended on struct field declaration order in
    /// `EncodeInterop`, which is exactly the kind of accident that produced a
    /// STATUS_ACCESS_VIOLATION at end-of-export.
    cuda_ctx:   Arc<CudaContext>,
}

unsafe impl Send for ExternalTexture {}

/// A single allocation visible to both wgpu and CUDA.
///
/// Created by [`SharedTexture::new`], which allocates the export-compatible
/// resource itself rather than trying to export one of wgpu's.
pub struct SharedTexture {
    /// The wgpu view of the allocation — usable as a normal texture (storage
    /// binding, copy source, render target, whatever `usage` allowed).
    pub texture:  wgpu::Texture,
    /// The CUDA view of the same allocation.  Carries its own
    /// `Arc<CudaContext>`, so this struct needs no ordering discipline of its own.
    pub external: ExternalTexture,
    /// Kept so Drop can close the NT handle after CUDA has released its import.
    #[cfg(target_os = "windows")]
    shared_handle: *mut std::ffi::c_void,
}

unsafe impl Send for SharedTexture {}

impl SharedTexture {
    /// Allocate a texture whose memory both wgpu and CUDA can address.
    ///
    /// `usage` describes what wgpu is allowed to do with it and is translated to
    /// the matching D3D12 resource flags; `STORAGE_BINDING` becomes
    /// `ALLOW_UNORDERED_ACCESS`, which is what the ABGR10 repack compute pass
    /// needs.  The allocation always carries `D3D12_HEAP_FLAG_SHARED`.
    ///
    /// Takes the context by `Arc` and keeps a share of it (see
    /// [`ExternalTexture::cuda_ctx`]): the CUDA import cannot outlive the context
    /// it was made against, and a borrow would let the caller arrange exactly
    /// that.
    pub fn new(
        cuda_ctx:  Arc<CudaContext>,
        device:    &GpuDevice,
        transport: InteropTransport,
        label:     &'static str,
        width:     u32,
        height:    u32,
        format:    wgpu::TextureFormat,
        usage:     wgpu::TextureUsages,
    ) -> Result<Self, CudaError> {
        let array_fmt = CudaArrayFormat::for_wgpu(format).ok_or_else(|| {
            CudaError::Import(format!(
                "no CUDA array descriptor is defined for wgpu format {format:?} — \
                 add one to CudaArrayFormat::for_wgpu"
            ))
        })?;

        match transport {
            #[cfg(target_os = "windows")]
            InteropTransport::D3D12Win32Handle => {
                let (texture, shared_handle, alloc_size) =
                    d3d12_shared::create_shared_texture(device, label, width, height, format, usage)?;

                let external = match ExternalTexture::import_win32_d3d12(
                    cuda_ctx, shared_handle, alloc_size, width, height, array_fmt,
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        unsafe { d3d12_shared::close_handle(shared_handle) };
                        return Err(e);
                    }
                };

                Ok(Self { texture, external, shared_handle })
            }
            #[cfg(not(target_os = "windows"))]
            InteropTransport::D3D12Win32Handle => Err(CudaError::Import(
                "D3D12Win32Handle transport is Windows-only".into(),
            )),
            InteropTransport::VulkanOpaqueFd => {
                // Exporting a VkImage's memory needs VK_KHR_external_memory_fd
                // enabled at device-creation time, which GpuDevice does not
                // request.  Reported rather than half-attempted: a bogus fd makes
                // cuImportExternalMemory fail with a misleading "invalid argument".
                let _ = (cuda_ctx, device, label, width, height, array_fmt);
                Err(CudaError::Import(
                    "Vulkan external-memory export is not implemented: GpuDevice does not \
                     enable VK_KHR_external_memory_fd, so no exportable VkDeviceMemory exists"
                        .into(),
                ))
            }
            InteropTransport::None => {
                Err(CudaError::Import("No interop transport available".into()))
            }
        }
    }
}

impl Drop for SharedTexture {
    fn drop(&mut self) {
        // Order matters: CUDA's import must be destroyed (which happens when
        // `external` drops, after this body) before the NT handle is closed.
        // Rust drops fields after the body, so close the handle explicitly here
        // only on the Windows path — and do it AFTER manually releasing CUDA.
        #[cfg(target_os = "windows")]
        unsafe {
            self.external.release();
            d3d12_shared::close_handle(self.shared_handle);
            self.shared_handle = std::ptr::null_mut();
        }
    }
}

impl ExternalTexture {
    /// Import an already-exported resource handle into CUDA.
    ///
    /// Prefer [`SharedTexture::new`], which allocates an export-compatible
    /// resource in the first place.  This entry point exists for callers holding
    /// a handle from elsewhere.
    ///
    /// # Safety-relevant preconditions
    /// The handle must refer to memory allocated with export-compatible flags
    /// (`D3D12_HEAP_FLAG_SHARED`, or `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`
    /// on Vulkan) and must stay valid for the duration of this call.
    #[cfg(target_os = "windows")]
    pub fn import_win32_d3d12(
        cuda_ctx:   Arc<CudaContext>,
        handle:     *mut std::ffi::c_void,
        alloc_size: u64,
        width:      u32,
        height:     u32,
        array_fmt:  CudaArrayFormat,
    ) -> Result<Self, CudaError> {
        if handle.is_null() {
            return Err(CudaError::Import(
                "CreateSharedHandle produced a NULL handle".into(),
            ));
        }
        if alloc_size == 0 {
            return Err(CudaError::Import(
                "the shared resource reports a zero allocation size".into(),
            ));
        }

        Self::import_desc(
            cuda_ctx,
            CudaExternalMemoryHandleDesc {
                r#type: CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
                handle: HandleUnion {
                    win32: HandleWin32 { handle, name: std::ptr::null() },
                },
                size:  alloc_size,
                flags: CUDA_EXTERNAL_MEMORY_DEDICATED,
                reserved: [0; 16],
            },
            width,
            height,
            array_fmt,
        )
    }

    /// Shared tail of every import path: `cuImportExternalMemory` followed by
    /// mapping level 0 of a mipmapped array over the imported memory.
    fn import_desc(
        cuda_ctx:  Arc<CudaContext>,
        desc:      CudaExternalMemoryHandleDesc,
        width:     u32,
        height:    u32,
        array_fmt: CudaArrayFormat,
    ) -> Result<Self, CudaError> {
        let mut ext_mem: CUexternalMemory = std::ptr::null_mut();
        let ret = cuda_ctx.with_context(|_stream| unsafe {
            cuImportExternalMemory(&mut ext_mem, &desc)
        });
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Import(format!(
                "cuImportExternalMemory failed: {}",
                crate::interop::ffi::cuda_driver::cu_err_to_string(ret)
            )));
        }

        let array_desc = CudaArray3DDescriptor {
            width:        width as usize,
            height:       height as usize,
            depth:        0, // 2D
            format:       array_fmt.format,
            num_channels: array_fmt.num_channels,
            flags:        CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mipmap_desc = CudaExternalMemoryMipmappedArrayDesc {
            offset: 0,
            array_desc,
            num_levels: 1,
            reserved: [0; 16],
        };

        let mut mipmap: CUmipmappedArray = std::ptr::null_mut();
        let mut cuda_array: CUarray = std::ptr::null_mut();
        let ret = cuda_ctx.with_context(|_stream| unsafe {
            let r = cuExternalMemoryGetMappedMipmappedArray(&mut mipmap, ext_mem, &mipmap_desc);
            if r != CUDA_SUCCESS { return r; }
            cuMipmappedArrayGetLevel(&mut cuda_array, mipmap, 0)
        });
        if ret != CUDA_SUCCESS {
            unsafe { cuDestroyExternalMemory(ext_mem) };
            return Err(CudaError::MapArray(format!(
                "mapping the imported memory as a {}x{} CUDA array \
                 (format=0x{:x}, {} channel(s)) failed: {}",
                width, height, array_fmt.format, array_fmt.num_channels,
                crate::interop::ffi::cuda_driver::cu_err_to_string(ret)
            )));
        }

        Ok(Self { cuda_array, mipmap, ext_mem, width, height, cuda_ctx })
    }

    pub fn cuda_array(&self) -> CUarray {
        self.cuda_array
    }

    /// Release the CUDA import early, idempotently.
    ///
    /// [`SharedTexture`] uses this to guarantee CUDA is done with the memory
    /// before it closes the NT handle backing it.
    ///
    /// Runs the destroy calls with this object's own context current
    /// (`with_context`), which is safe precisely because `cuda_ctx` keeps that
    /// context alive for at least as long as `self`.
    ///
    /// # Safety
    /// No CUDA work referencing `cuda_array()` may still be in flight.
    unsafe fn release(&mut self) {
        let mipmap  = std::mem::replace(&mut self.mipmap, std::ptr::null_mut());
        let ext_mem = std::mem::replace(&mut self.ext_mem, std::ptr::null_mut());
        self.cuda_array = std::ptr::null_mut();
        if mipmap.is_null() && ext_mem.is_null() {
            return;
        }
        self.cuda_ctx.with_context(|_stream| {
            if !mipmap.is_null() {
                cuMipmappedArrayDestroy(mipmap);
            }
            if !ext_mem.is_null() {
                cuDestroyExternalMemory(ext_mem);
            }
        });
    }
}

impl Drop for ExternalTexture {
    fn drop(&mut self) {
        // SAFETY: nothing else holds the CUDA array once this owner is dropped.
        unsafe { self.release() };
        // NOTE: this releases ONLY the CUDA view of the memory.  The underlying
        // texture is freed by whoever owns it.
    }
}

/// D3D12 allocation of a shared, CUDA-importable texture.
///
/// Kept in its own module so the winapi/d3d12 imports do not leak into the
/// cross-platform part of this file.
#[cfg(target_os = "windows")]
mod d3d12_shared {
    use super::CudaError;
    use crate::render::device::GpuDevice;
    use winapi::shared::dxgiformat;
    use winapi::shared::dxgitype;
    use winapi::um::d3d12 as d3d12_ty;
    use winapi::Interface;

    /// `GENERIC_ALL` — the access mask `CreateSharedHandle` wants for a resource
    /// another API will both read and write.
    const GENERIC_ALL: u32 = 0x1000_0000;

    /// The DXGI format wgpu uses for a texture format, for the subset of formats
    /// the interop paths allocate.  Must agree with wgpu-hal's own mapping, or
    /// `create_texture_from_hal` hands wgpu a resource whose footprint does not
    /// match the descriptor it was told about.
    fn dxgi_format(format: wgpu::TextureFormat) -> Option<dxgiformat::DXGI_FORMAT> {
        use wgpu::TextureFormat as F;
        Some(match format {
            F::R32Uint     => dxgiformat::DXGI_FORMAT_R32_UINT,
            F::R8Unorm     => dxgiformat::DXGI_FORMAT_R8_UNORM,
            F::Rg8Unorm    => dxgiformat::DXGI_FORMAT_R8G8_UNORM,
            F::Rgba8Unorm  => dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
            F::Rgba16Float => dxgiformat::DXGI_FORMAT_R16G16B16A16_FLOAT,
            _ => return None,
        })
    }

    /// Resource flags implied by the wgpu usages, mirroring wgpu-hal's
    /// `map_texture_usage_to_resource_flags` (dx12/conv.rs).
    ///
    /// Keep this in step with that function.  In particular do NOT add
    /// `DENY_SHADER_RESOURCE` for colour formats that merely lack
    /// `TEXTURE_BINDING`: wgpu-hal sets it only alongside `ALLOW_DEPTH_STENCIL`,
    /// and combining it with `ALLOW_UNORDERED_ACCESS` is invalid — D3D12 then
    /// rejects the descriptor and `GetResourceAllocationInfo` returns
    /// `UINT64_MAX` rather than failing loudly.
    fn resource_flags(usage: wgpu::TextureUsages) -> d3d12_ty::D3D12_RESOURCE_FLAGS {
        let mut flags = 0;
        if usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
            flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }
        if usage.contains(wgpu::TextureUsages::STORAGE_BINDING) {
            flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        flags
    }

    pub unsafe fn close_handle(handle: *mut std::ffi::c_void) {
        if !handle.is_null() {
            winapi::um::handleapi::CloseHandle(handle.cast());
        }
    }

    /// Create a `D3D12_HEAP_FLAG_SHARED` committed texture, expose it to wgpu,
    /// and return an NT handle CUDA can import.
    ///
    /// Returns `(wgpu texture, shared NT handle, allocation size in bytes)`.  The
    /// caller owns the handle and must `close_handle` it once CUDA is finished.
    pub fn create_shared_texture(
        device: &GpuDevice,
        label:  &'static str,
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage:  wgpu::TextureUsages,
    ) -> Result<(wgpu::Texture, *mut std::ffi::c_void, u64), CudaError> {
        let dxgi = dxgi_format(format).ok_or_else(|| {
            CudaError::Import(format!(
                "no DXGI format is defined for wgpu format {format:?} — \
                 add one to external_texture::d3d12_shared::dxgi_format"
            ))
        })?;

        let raw_desc = d3d12_ty::D3D12_RESOURCE_DESC {
            Dimension: d3d12_ty::D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width:     width as u64,
            Height:    height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format:    dxgi,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout:    d3d12_ty::D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags:     resource_flags(usage),
        };

        let heap_props = d3d12_ty::D3D12_HEAP_PROPERTIES {
            Type:                 d3d12_ty::D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty:      d3d12_ty::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: d3d12_ty::D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask:     1,
            VisibleNodeMask:      1,
        };

        // Reach the raw ID3D12Device through wgpu's hal escape hatch, allocate the
        // shared resource on it, and export a handle — all inside one closure so
        // the borrow of the hal device is confined.
        type Created = Result<(d3d12::Resource, *mut std::ffi::c_void, u64), CudaError>;
        let created: Created = unsafe {
            device
                .device
                .as_hal::<wgpu_hal::api::Dx12, _, _>(|hal_device| {
                    let hal_device = hal_device.ok_or_else(|| {
                        CudaError::Import(
                            "the wgpu device is not a D3D12 device, so no shared \
                             resource can be created"
                                .into(),
                        )
                    })?;
                    let raw_device: &d3d12::Device = hal_device.raw_device();

                    let alloc_info = raw_device.GetResourceAllocationInfo(0, 1, &raw_desc);
                    if alloc_info.SizeInBytes == 0 || alloc_info.SizeInBytes == u64::MAX {
                        return Err(CudaError::Import(format!(
                            "GetResourceAllocationInfo rejected a {width}x{height} \
                             {format:?} resource (SizeInBytes={})",
                            alloc_info.SizeInBytes
                        )));
                    }

                    let mut resource = d3d12::Resource::null();
                    let hr = raw_device.CreateCommittedResource(
                        &heap_props,
                        d3d12_ty::D3D12_HEAP_FLAG_SHARED,
                        &raw_desc,
                        d3d12_ty::D3D12_RESOURCE_STATE_COMMON,
                        std::ptr::null(),
                        &d3d12_ty::ID3D12Resource::uuidof(),
                        resource.mut_void(),
                    );
                    if hr < 0 {
                        return Err(CudaError::Import(format!(
                            "CreateCommittedResource(HEAP_FLAG_SHARED) for a \
                             {width}x{height} {format:?} texture failed: HRESULT 0x{hr:08X}"
                        )));
                    }

                    let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
                    let hr = raw_device.CreateSharedHandle(
                        resource.as_mut_ptr().cast(),
                        std::ptr::null(),
                        GENERIC_ALL,
                        std::ptr::null(),
                        &mut handle as *mut _ as *mut _,
                    );
                    if hr < 0 || handle.is_null() {
                        return Err(CudaError::Import(format!(
                            "CreateSharedHandle failed: HRESULT 0x{hr:08X}"
                        )));
                    }

                    Ok((resource, handle, alloc_info.SizeInBytes))
                })
                .ok_or_else(|| {
                    CudaError::Import(
                        "wgpu::Device::as_hal returned None — this build is not using \
                         the wgpu-core backend, so the D3D12 escape hatch is unavailable"
                            .into(),
                    )
                })?
        };
        let (resource, handle, alloc_size) = created?;

        // Hand the resource to wgpu.  `texture_from_raw` builds the hal texture
        // around our ID3D12Resource with `allocation: None`, so wgpu will not try
        // to free it through its own allocator; the ComPtr refcount governs
        // lifetime.  The descriptor must describe the resource we just made.
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        };
        let texture = unsafe {
            let hal_texture = wgpu_hal::dx12::Device::texture_from_raw(
                resource,
                format,
                wgpu::TextureDimension::D2,
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                1, // mip_level_count
                1, // sample_count
            );
            device
                .device
                .create_texture_from_hal::<wgpu_hal::api::Dx12>(hal_texture, &desc)
        };

        Ok((texture, handle, alloc_size))
    }
}
