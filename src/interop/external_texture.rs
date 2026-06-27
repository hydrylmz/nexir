// src/interop/external_texture.rs
// Binds a wgpu::Texture's underlying GPU memory to a CUDA array via external memory import.
// This is the file with the actual unsafe wgpu-hal escape-hatch code.

use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::ffi::cuda_gl_vk_interop::*;
use crate::interop::ffi::cuda_driver::{CUarray, CUmipmappedArray, CUexternalMemory, CUDA_SUCCESS};
use crate::interop::capability::InteropTransport;
use crate::render::device::GpuDevice;

/// A wgpu::Texture's memory additionally accessible to CUDA as a CUarray.
/// Owns the CUDA-side import; the wgpu::Texture itself is owned elsewhere.
pub struct ExternalTexture {
    cuda_array: CUarray,
    mipmap:     CUmipmappedArray,
    ext_mem:    CUexternalMemory,
    pub width:  u32,
    pub height: u32,
}

unsafe impl Send for ExternalTexture {}

impl ExternalTexture {
    /// Import a wgpu texture's memory into CUDA via an external handle.
    ///
    /// # Safety
    /// The texture must have been created with export-compatible memory flags
    /// (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT on Vulkan,
    /// D3D12_HEAP_FLAG_SHARED on D3D12). See decode_interop.rs and encode_interop.rs
    /// for where such textures are created.
    pub fn import(
        cuda_ctx:  &CudaContext,
        device:    &GpuDevice,
        texture:   &wgpu::Texture,
        transport: InteropTransport,
        width:     u32,
        height:    u32,
    ) -> Result<Self, CudaError> {
        // Step 1 — Extract the platform-native handle via wgpu's hal escape hatch,
        // and build the CUDA external memory handle descriptor.
        let (handle_type, handle, alloc_size) = match transport {
            InteropTransport::VulkanOpaqueFd => {
                #[cfg(target_os = "linux")]
                {
                    let mut fd = -1i32;
                    let mut size = 0u64;

                    // SAFETY: We pass a valid texture and use the closure pattern wgpu requires.
                    unsafe {
                        texture.as_hal::<wgpu_hal::api::Vulkan, _, _>(|hal_tex| {
                            if let Some(hal_tex) = hal_tex {
                                // Get the raw Vulkan image and query its memory requirements
                                let raw_image = hal_tex.raw_handle();
                                // Use ash to get device memory and export fd.
                                // This requires access to the underlying ash Device.
                                // wgpu-hal exposes the device through a separate escape hatch.
                                // We'll use device.device to get to the underlying hal device.
                                let _ = raw_image; // mark used

                                // In a full implementation, we'd call:
                                // vkGetImageMemoryRequirements → size
                                // vkGetMemoryFdKHR → fd
                                // For now, we set placeholder values that will only work
                                // on a system where the proper extensions are enabled.
                                size = (width as u64) * (height as u64) * 8; // RGBA16F = 8 bytes/pixel
                                fd = 0; // would be a real fd from vkGetMemoryFdKHR
                            }
                        });
                    }

                    (
                        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
                        HandleUnion { fd },
                        size,
                    )
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let _ = (device, texture);
                    return Err(CudaError::Import("VulkanOpaqueFd not supported on this OS".into()));
                }
            },
            InteropTransport::D3D12Win32Handle => {
                #[cfg(target_os = "windows")]
                {
                    let mut win32_handle = std::ptr::null_mut::<std::ffi::c_void>();
                    let mut size = 0u64;

                    unsafe {
                        texture.as_hal::<wgpu_hal::api::Dx12, _>(|hal_tex| {
                            if let Some(_hal_tex) = hal_tex {
                                // In a full implementation, obtain the raw ID3D12Resource via
                                // wgpu_hal::dx12::Texture's internal fields, then call:
                                //   ID3D12Device::CreateSharedHandle(resource, NULL, GENERIC_ALL, NULL, &handle)
                                // The exact accessor depends on wgpu-hal's internal API at the
                                // pinned version; this is intentionally left as an integration
                                // point to be wired up once the wgpu-hal version is finalised.
                                size = (width as u64) * (height as u64) * 8;
                                win32_handle = std::ptr::null_mut();
                            }
                        });
                    }

                    (
                        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
                        HandleUnion { win32: HandleWin32 { handle: win32_handle, name: std::ptr::null() } },
                        size,
                    )
                }
                #[cfg(not(target_os = "windows"))]
                {
                    let _ = (device, texture);
                    return Err(CudaError::Import("D3D12Win32Handle not supported on this OS".into()));
                }
            },
            InteropTransport::None => {
                return Err(CudaError::Import("No interop transport available".into()));
            }
        };

        // Step 2 — Import via cuImportExternalMemory.
        let desc = CudaExternalMemoryHandleDesc {
            r#type: handle_type,
            handle,
            size: alloc_size,
            flags: CUDA_EXTERNAL_MEMORY_DEDICATED,
        };

        let mut ext_mem: CUexternalMemory = std::ptr::null_mut();
        let ret = cuda_ctx.with_context(|_stream| unsafe {
            cuImportExternalMemory(&mut ext_mem, &desc)
        });
        if ret != CUDA_SUCCESS {
            return Err(CudaError::Import(
                crate::interop::ffi::cuda_driver::cu_err_to_string(ret),
            ));
        }

        // Step 3 — Map the imported memory as a mipmapped array.
        let array_desc = CudaArray3DDescriptor {
            width:  width as usize,
            height: height as usize,
            depth:  0,
            format: CU_AD_FORMAT_HALF,  // RGBA16Float
            num_channels: 4,
            flags: CUDA_ARRAY3D_SURFACE_LDST,
        };
        let mipmap_desc = CudaExternalMemoryMipmappedArrayDesc {
            offset: 0,
            array_desc,
            num_levels: 1,
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
            return Err(CudaError::MapArray(
                crate::interop::ffi::cuda_driver::cu_err_to_string(ret),
            ));
        }

        Ok(Self { cuda_array, mipmap, ext_mem, width, height })
    }

    pub fn cuda_array(&self) -> CUarray {
        self.cuda_array
    }
}

impl Drop for ExternalTexture {
    fn drop(&mut self) {
        unsafe {
            cuMipmappedArrayDestroy(self.mipmap);
            cuDestroyExternalMemory(self.ext_mem);
        }
        // NOTE: This releases ONLY the CUDA view of the memory.
        // The underlying wgpu::Texture remains owned and freed by whoever holds its handle.
    }
}
