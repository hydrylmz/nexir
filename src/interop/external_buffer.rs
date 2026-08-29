// src/interop/external_buffer.rs
// The buffer sibling of `external_texture.rs`: one D3D12 allocation visible to
// both wgpu (as a `wgpu::Buffer` a compute shader can write) and CUDA (as a flat
// `CUdeviceptr` NVENC can be registered against).
//
// WHY A SECOND FILE RATHER THAN A FLAG ON SharedTexture.  The two differ on every
// axis that matters:
//
//                       SharedTexture                SharedBuffer
//   D3D12 Dimension     TEXTURE2D                    BUFFER
//   D3D12 Layout        UNKNOWN (driver-chosen)      ROW_MAJOR
//   CUDA mapping        GetMappedMipmappedArray      GetMappedBuffer
//   CUDA handle         CUarray (via mipmap level 0) CUdeviceptr
//   CUDA release        cuMipmappedArrayDestroy      cuMemFree
//   row stride          inferred by CUDA from the    OURS to state
//                       D3D12 footprint
//   NVENC resourceType  CUDAARRAY                    CUDADEVICEPTR
//
// Sharing an implementation would mean a struct where half the fields are null on
// each path and every method branches — which is how the `pitch = 0` mistake
// documented in `nvenc-nv12-zero-copy-allocation-shapes.md` travelled from the
// packed-single-plane case to the two-plane one.
//
// WHY IT EXISTS.  NV12 is two planes in one allocation and the chroma plane sits
// at `pitch * height`, so the pitch has to be a number we choose and state, not
// one CUDA derives from a texture footprint we cannot see.  `Nv12EncodeNode`
// (step 1) already writes exactly that layout into a `wgpu::Buffer`; this is the
// allocation that buffer has to be, for NVENC to read it without a copy.
//
// MEASURED FACTS THIS FILE DEPENDS ON.  All from `nvchk/extbuf_probe.c` on an RTX
// 3050, NVENC API 12.2, headers `n12.2.72.0`; the NVENC half from
// `nvchk/d3d12_buf_probe.c`:
//
//   * A `D3D12_RESOURCE_DIMENSION_BUFFER` committed resource with
//     `D3D12_HEAP_FLAG_SHARED` exports an NT handle CUDA imports, and
//     `cuExternalMemoryGetMappedBuffer` returns a usable `CUdeviceptr`.  61440
//     bytes written host→device and read back device→host compared equal byte for
//     byte.
//   * `GetResourceAllocationInfo` PADS the request (61440 → 65536, 64 KiB heap
//     alignment).  The import is given the PADDED size and the mapping the
//     LOGICAL one.  Note the scope: a rung importing the LOGICAL size instead
//     also succeeded and round-tripped, so this driver does not REQUIRE the
//     padded figure — the padded size is what CUDA's documentation describes
//     (the size of the memory object) and it is the one that stays correct if a
//     future driver starts validating, which is why it is what the code passes.
//     Do not "simplify" it to one number on the strength of the lenient rung.
//   * `CUDA_EXTERNAL_MEMORY_DEDICATED` is accepted for a buffer import.
//   * The mapped pointer is released with `cuMemFree`, then the import with
//     `cuDestroyExternalMemory`.  Both returned `CUDA_SUCCESS`; neither faulted.
//   * NVENC accepts the resulting pointer as
//     `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR` with `pitch = 320` on a
//     256-wide frame and decodes the right pixels, with chroma read at
//     `pitch * height`.
//
// SCOPE.  This file allocates, imports, and releases.  It does not register
// anything with NVENC and does not encode — `encode_interop.rs` is the next step.

use crate::interop::capability::InteropTransport;
use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::ffi::cuda_driver::{cu_err_to_string, CUexternalMemory, CUdeviceptr, CUDA_SUCCESS};
use crate::interop::ffi::cuda_gl_vk_interop::*;
use crate::render::device::GpuDevice;
use std::sync::Arc;

/// A GPU buffer's memory additionally accessible to CUDA as a `CUdeviceptr`.
/// Owns the CUDA-side import; the buffer itself is owned elsewhere.
pub struct ExternalBuffer {
    device_ptr: CUdeviceptr,
    ext_mem:    CUexternalMemory,
    /// Bytes mapped — the logical size, not the padded allocation.
    pub size:   u64,
    /// A share of the CUDA primary context the import was made against.
    ///
    /// Load-bearing, not bookkeeping, for exactly the reason spelled out on
    /// [`crate::interop::external_texture::ExternalTexture`]: `cuMemFree` and
    /// `cuDestroyExternalMemory` in [`ExternalBuffer::release`] are only valid
    /// while that context exists, and if the last `Arc<CudaContext>` elsewhere
    /// drops first, teardown here becomes an access violation rather than an
    /// error return.
    cuda_ctx:   Arc<CudaContext>,
}

unsafe impl Send for ExternalBuffer {}

/// A single allocation visible to both wgpu and CUDA, laid out as a flat buffer.
///
/// Created by [`SharedBuffer::new`], which allocates the export-compatible
/// resource itself: CUDA cannot import an allocation wgpu made, because
/// `Device::create_buffer` never sets `D3D12_HEAP_FLAG_SHARED` and wgpu-hal keeps
/// `dx12::Buffer`'s `ID3D12Resource` private.
pub struct SharedBuffer {
    /// The wgpu view of the allocation — a normal storage buffer as far as a
    /// compute pass is concerned.
    pub buffer:   wgpu::Buffer,
    /// The CUDA view of the same allocation.  Carries its own
    /// `Arc<CudaContext>`, so this struct needs no ordering discipline of its own.
    pub external: ExternalBuffer,
    /// Kept so Drop can close the NT handle after CUDA has released its import.
    #[cfg(target_os = "windows")]
    shared_handle: *mut std::ffi::c_void,
}

unsafe impl Send for SharedBuffer {}

impl SharedBuffer {
    /// Allocate a buffer whose memory both wgpu and CUDA can address.
    ///
    /// `size` is the LOGICAL size in bytes — what the shader writes and what the
    /// encoder reads.  D3D12 rounds the underlying allocation up to its heap
    /// alignment and the import is told the rounded size; the mapping and the
    /// `wgpu::Buffer` both stay at `size`, so nothing downstream has to know the
    /// padding exists.
    ///
    /// `usage` describes what wgpu may do with it.  `STORAGE` becomes
    /// `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`, which is what
    /// `Nv12EncodeNode`'s compute passes need.
    ///
    /// Takes the context by `Arc` and keeps a share of it, for the reason on
    /// [`ExternalBuffer::cuda_ctx`].
    pub fn new(
        cuda_ctx:  Arc<CudaContext>,
        device:    &GpuDevice,
        transport: InteropTransport,
        label:     &'static str,
        size:      u64,
        usage:     wgpu::BufferUsages,
    ) -> Result<Self, CudaError> {
        if size == 0 {
            return Err(CudaError::Import(
                "a zero-byte shared buffer was requested".into(),
            ));
        }

        match transport {
            #[cfg(target_os = "windows")]
            InteropTransport::D3D12Win32Handle => {
                let (buffer, shared_handle, alloc_size) =
                    d3d12_shared_buffer::create_shared_buffer(device, label, size, usage)?;

                let external = match ExternalBuffer::import_win32_d3d12(
                    cuda_ctx, shared_handle, alloc_size, size,
                ) {
                    Ok(e) => e,
                    Err(e) => {
                        // SAFETY: CUDA never took a reference to this handle on
                        // the failure path, so closing it now cannot orphan an
                        // import.
                        unsafe {
                            crate::interop::external_texture::d3d12_shared::close_handle(
                                shared_handle,
                            )
                        };
                        return Err(e);
                    }
                };

                Ok(Self { buffer, external, shared_handle })
            }
            #[cfg(not(target_os = "windows"))]
            InteropTransport::D3D12Win32Handle => Err(CudaError::Import(
                "D3D12Win32Handle transport is Windows-only".into(),
            )),
            InteropTransport::VulkanOpaqueFd => {
                // Same gap as on the texture path: exporting a VkBuffer's memory
                // needs VK_KHR_external_memory_fd enabled at device-creation
                // time, which GpuDevice does not request.  Reported rather than
                // half-attempted — a bogus fd makes cuImportExternalMemory fail
                // with a misleading "invalid argument".
                let _ = (cuda_ctx, device, label, size, usage);
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

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        // Order matters: CUDA's import must be released before the NT handle
        // backing it is closed.  Rust drops fields after this body, so release
        // CUDA explicitly here and close the handle after.
        #[cfg(target_os = "windows")]
        unsafe {
            self.external.release();
            crate::interop::external_texture::d3d12_shared::close_handle(self.shared_handle);
            self.shared_handle = std::ptr::null_mut();
        }
    }
}

impl ExternalBuffer {
    /// Import an already-exported D3D12 resource handle as a flat device pointer.
    ///
    /// Prefer [`SharedBuffer::new`], which allocates an export-compatible
    /// resource in the first place.  This entry point exists for callers holding
    /// a handle from elsewhere.
    ///
    /// `alloc_size` must be the size `GetResourceAllocationInfo` reported (the
    /// PADDED one); `map_size` is the logical size to map.  Getting these the
    /// wrong way round is the mistake the two parameters exist to make visible.
    ///
    /// # Safety-relevant preconditions
    /// The handle must refer to memory allocated with `D3D12_HEAP_FLAG_SHARED`
    /// and must stay valid for the duration of this call.
    #[cfg(target_os = "windows")]
    pub fn import_win32_d3d12(
        cuda_ctx:   Arc<CudaContext>,
        handle:     *mut std::ffi::c_void,
        alloc_size: u64,
        map_size:   u64,
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
        if map_size > alloc_size {
            return Err(CudaError::Import(format!(
                "cannot map {map_size} bytes of a {alloc_size}-byte allocation"
            )));
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
            map_size,
        )
    }

    /// Shared tail of every import path: `cuImportExternalMemory` followed by
    /// mapping `map_size` bytes of it as a device pointer.
    fn import_desc(
        cuda_ctx: Arc<CudaContext>,
        desc:     CudaExternalMemoryHandleDesc,
        map_size: u64,
    ) -> Result<Self, CudaError> {
        let alloc_size = desc.size;

        let mut ext_mem: CUexternalMemory = std::ptr::null_mut();
        let ret = cuda_ctx.with_context(|_stream| unsafe {
            cuImportExternalMemory(&mut ext_mem, &desc)
        });
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Import(format!(
                "cuImportExternalMemory failed for a {alloc_size}-byte buffer: {}",
                cu_err_to_string(ret)
            )));
        }

        let buffer_desc = CudaExternalMemoryBufferDesc {
            offset: 0,
            size:   map_size,
            flags:  0,
            reserved: [0; 16],
        };

        let mut device_ptr: CUdeviceptr = 0;
        let ret = cuda_ctx.with_context(|_stream| unsafe {
            cuExternalMemoryGetMappedBuffer(&mut device_ptr, ext_mem, &buffer_desc)
        });
        if ret != CUDA_SUCCESS {
            unsafe { cuDestroyExternalMemory(ext_mem) };
            return Err(CudaError::MapBuffer(format!(
                "mapping {map_size} bytes of a {alloc_size}-byte imported allocation \
                 as a CUdeviceptr failed: {}",
                cu_err_to_string(ret)
            )));
        }

        Ok(Self { device_ptr, ext_mem, size: map_size, cuda_ctx })
    }

    /// The mapped device pointer, for passing to NVENC as
    /// `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Copy the whole mapping out to host memory, through CUDA.
    ///
    /// This exists for VERIFICATION, not for the pipeline: an export never reads
    /// the buffer back, and doing so would defeat the point of zero-copy.  What it
    /// is for is proving that the CUDA pointer and the `wgpu::Buffer` are the same
    /// allocation — write with one API, read with the other, compare.  Nothing
    /// else can distinguish "one shared allocation" from "two allocations that
    /// both happen to hold plausible data".
    ///
    /// The caller is responsible for ordering: any wgpu work writing the buffer
    /// must have completed (`Device::poll(Maintain::Wait)`) before this runs.
    /// There is no cross-API fence here.
    pub fn read_to_host(&self) -> Result<Vec<u8>, CudaError> {
        let mut out = vec![0u8; self.size as usize];
        let ret = self.cuda_ctx.with_context(|stream| unsafe {
            let r = crate::interop::ffi::cuda_driver::cuMemcpyDtoH_v2(
                out.as_mut_ptr().cast(),
                self.device_ptr,
                out.len(),
            );
            if r != CUDA_SUCCESS {
                return r;
            }
            crate::interop::ffi::cuda_driver::cuStreamSynchronize(stream)
        });
        if ret != CUDA_SUCCESS {
            return Err(CudaError::MapBuffer(format!(
                "reading {} bytes back from the shared buffer failed: {}",
                self.size,
                cu_err_to_string(ret)
            )));
        }
        Ok(out)
    }

    /// Fill the whole mapping from host memory, through CUDA.
    ///
    /// The counterpart of [`ExternalBuffer::read_to_host`], and equally not part
    /// of the pipeline: it lets a test write through CUDA and read back through
    /// wgpu, which is the other direction of the same aliasing proof.
    pub fn write_from_host(&self, src: &[u8]) -> Result<(), CudaError> {
        if src.len() as u64 != self.size {
            return Err(CudaError::MapBuffer(format!(
                "write_from_host was given {} bytes for a {}-byte mapping",
                src.len(),
                self.size
            )));
        }
        let ret = self.cuda_ctx.with_context(|stream| unsafe {
            let r = crate::interop::ffi::cuda_driver::cuMemcpyHtoD_v2(
                self.device_ptr,
                src.as_ptr().cast(),
                src.len(),
            );
            if r != CUDA_SUCCESS {
                return r;
            }
            crate::interop::ffi::cuda_driver::cuStreamSynchronize(stream)
        });
        if ret != CUDA_SUCCESS {
            return Err(CudaError::MapBuffer(format!(
                "writing {} bytes into the shared buffer failed: {}",
                self.size,
                cu_err_to_string(ret)
            )));
        }
        Ok(())
    }

    /// Release the CUDA mapping and import early, idempotently.
    ///
    /// [`SharedBuffer`] uses this to guarantee CUDA is done with the memory
    /// before it closes the NT handle backing it.
    ///
    /// Runs with this object's own context current, which is safe precisely
    /// because `cuda_ctx` keeps that context alive for at least as long as
    /// `self`.
    ///
    /// # Safety
    /// No CUDA or NVENC work referencing `device_ptr()` may still be in flight.
    unsafe fn release(&mut self) {
        let device_ptr = std::mem::replace(&mut self.device_ptr, 0);
        let ext_mem    = std::mem::replace(&mut self.ext_mem, std::ptr::null_mut());
        if device_ptr == 0 && ext_mem.is_null() {
            return;
        }
        self.cuda_ctx.with_context(|_stream| {
            // `cuMemFree` on memory CUDA did not allocate is the documented
            // counterpart of cuExternalMemoryGetMappedBuffer, and returns
            // CUDA_SUCCESS here (measured, `nvchk/extbuf_probe.c buf`).
            if device_ptr != 0 {
                crate::interop::ffi::cuda_driver::cuMemFree_v2(device_ptr);
            }
            if !ext_mem.is_null() {
                cuDestroyExternalMemory(ext_mem);
            }
        });
    }
}

impl Drop for ExternalBuffer {
    fn drop(&mut self) {
        // SAFETY: nothing else holds the mapping once this owner is dropped.
        unsafe { self.release() };
        // NOTE: this releases ONLY the CUDA view of the memory.  The underlying
        // buffer is freed by whoever owns it.
    }
}

/// D3D12 allocation of a shared, CUDA-importable BUFFER.
///
/// The texture equivalent lives in
/// [`crate::interop::external_texture::d3d12_shared`]; the `as_hal` escape-hatch
/// dance is the same shape but the descriptor and the hal constructor differ, so
/// this is its own function rather than a parameter on that one.
#[cfg(target_os = "windows")]
mod d3d12_shared_buffer {
    use super::CudaError;
    use crate::render::device::GpuDevice;
    use winapi::shared::dxgiformat;
    use winapi::shared::dxgitype;
    use winapi::um::d3d12 as d3d12_ty;
    use winapi::Interface;

    /// `GENERIC_ALL` — the access mask `CreateSharedHandle` wants for a resource
    /// another API will both read and write.
    const GENERIC_ALL: u32 = 0x1000_0000;

    /// Resource flags implied by the wgpu buffer usages, mirroring wgpu-hal's
    /// `map_buffer_usage_to_resource_flags` (dx12/conv.rs), which sets exactly
    /// one flag and only for `STORAGE_READ_WRITE`.
    ///
    /// Keep this in step with that function: `create_buffer_from_hal` hands wgpu
    /// a resource it will treat as though its own `create_buffer` had made it,
    /// and a flag mismatch means wgpu records barriers for a resource state the
    /// resource cannot be in.
    fn resource_flags(usage: wgpu::BufferUsages) -> d3d12_ty::D3D12_RESOURCE_FLAGS {
        let mut flags = 0;
        if usage.contains(wgpu::BufferUsages::STORAGE) {
            flags |= d3d12_ty::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        }
        flags
    }

    /// Create a `D3D12_HEAP_FLAG_SHARED` committed buffer, expose it to wgpu, and
    /// return an NT handle CUDA can import.
    ///
    /// Returns `(wgpu buffer, shared NT handle, PADDED allocation size in bytes)`.
    /// The caller owns the handle and must close it once CUDA is finished.  The
    /// third value is what `cuImportExternalMemory` needs and is generally LARGER
    /// than `size` — 64 KiB-aligned on this hardware.
    pub fn create_shared_buffer(
        device: &GpuDevice,
        label:  &'static str,
        size:   u64,
        usage:  wgpu::BufferUsages,
    ) -> Result<(wgpu::Buffer, *mut std::ffi::c_void, u64), CudaError> {
        // A buffer resource descriptor: Width is the byte length, Height 1,
        // format UNKNOWN, layout ROW_MAJOR.  This is what wgpu-hal's own
        // `create_buffer` builds, and what the C probe measured.
        let raw_desc = d3d12_ty::D3D12_RESOURCE_DESC {
            Dimension: d3d12_ty::D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width:     size,
            Height:    1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format:    dxgiformat::DXGI_FORMAT_UNKNOWN,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout:    d3d12_ty::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags:     resource_flags(usage),
        };

        // HEAP_TYPE_DEFAULT: GPU-local, no CPU access.  A shared buffer cannot be
        // an upload/readback heap, and neither the compute shader nor NVENC needs
        // CPU visibility.
        let heap_props = d3d12_ty::D3D12_HEAP_PROPERTIES {
            Type:                 d3d12_ty::D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty:      d3d12_ty::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: d3d12_ty::D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask:     1,
            VisibleNodeMask:      1,
        };

        type Created = Result<(d3d12::Resource, *mut std::ffi::c_void, u64), CudaError>;
        let created: Created = unsafe {
            device
                .device
                .as_hal::<wgpu_hal::api::Dx12, _, _>(|hal_device| {
                    let hal_device = hal_device.ok_or_else(|| {
                        CudaError::Import(
                            "the wgpu device is not a D3D12 device, so no shared \
                             buffer can be created"
                                .into(),
                        )
                    })?;
                    let raw_device: &d3d12::Device = hal_device.raw_device();

                    let alloc_info = raw_device.GetResourceAllocationInfo(0, 1, &raw_desc);
                    if alloc_info.SizeInBytes == 0 || alloc_info.SizeInBytes == u64::MAX {
                        return Err(CudaError::Import(format!(
                            "GetResourceAllocationInfo rejected a {size}-byte buffer \
                             (SizeInBytes={})",
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
                            "CreateCommittedResource(HEAP_FLAG_SHARED, BUFFER {size} bytes) \
                             failed: HRESULT 0x{hr:08X}"
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

        // Hand the resource to wgpu.  `buffer_from_raw` builds the hal buffer
        // around our ID3D12Resource with `allocation: None`, so wgpu will not try
        // to free it through its own allocator; the ComPtr refcount governs
        // lifetime.  The descriptor's `size` is the LOGICAL size, deliberately
        // not the padded one: it bounds every wgpu-side binding and copy, and
        // letting wgpu believe the padding is addressable would let a shader
        // write into bytes CUDA never mapped.
        let desc = wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        };
        let buffer = unsafe {
            let hal_buffer = wgpu_hal::dx12::Device::buffer_from_raw(resource, size);
            device
                .device
                .create_buffer_from_hal::<wgpu_hal::api::Dx12>(hal_buffer, &desc)
        };

        Ok((buffer, handle, alloc_size))
    }
}
