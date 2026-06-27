use super::cuda_driver::{CUresult, CUarray, CUmipmappedArray, CUexternalMemory};

#[repr(C)]
pub struct CudaExternalMemoryHandleDesc {
    pub r#type: u32,           // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1, _WIN32 = 2
    pub handle: HandleUnion,
    pub size:   u64,
    pub flags:  u32,            // CUDA_EXTERNAL_MEMORY_DEDICATED = 1 — set for texture-backed allocations
}

#[repr(C)]
pub union HandleUnion {
    pub fd:    std::os::raw::c_int,            // Linux: dup'd file descriptor
    pub win32: HandleWin32,                     // Windows: NT handle
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HandleWin32 {
    pub handle: *mut std::ffi::c_void,
    pub name:   *const std::ffi::c_void,        // NULL when using a raw handle, not a named one
}

#[repr(C)]
pub struct CudaExternalMemoryMipmappedArrayDesc {
    pub offset:     u64,
    pub array_desc: CudaArray3DDescriptor,
    pub num_levels: u32,
}

#[repr(C)]
pub struct CudaArray3DDescriptor {
    pub width:        usize,
    pub height:       usize,
    pub depth:        usize,        // 0 for a 2D array
    pub format:       u32,          // CU_AD_FORMAT_HALF = 0x10 for our RGBA16Float textures
    pub num_channels: u32,          // 4 for RGBA
    pub flags:        u32,          // CUDA_ARRAY3D_SURFACE_LDST = 2 — required for read/write access
}

pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:        u32 = 1;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32:      u32 = 2;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE:    u32 = 4;
pub const CUDA_EXTERNAL_MEMORY_DEDICATED:                  u32 = 1;
pub const CU_AD_FORMAT_HALF:                                u32 = 0x10;
pub const CUDA_ARRAY3D_SURFACE_LDST:                        u32 = 2;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuImportExternalMemory(
        ext_mem: *mut CUexternalMemory,
        desc:    *const CudaExternalMemoryHandleDesc,
    ) -> CUresult;

    pub fn cuExternalMemoryGetMappedMipmappedArray(
        mipmap:  *mut CUmipmappedArray,
        ext_mem: CUexternalMemory,
        desc:    *const CudaExternalMemoryMipmappedArrayDesc,
    ) -> CUresult;

    pub fn cuMipmappedArrayGetLevel(
        level_array: *mut CUarray,
        mipmap:      CUmipmappedArray,
        level:       u32,
    ) -> CUresult;

    pub fn cuMipmappedArrayDestroy(mipmap: CUmipmappedArray) -> CUresult;
    pub fn cuDestroyExternalMemory(ext_mem: CUexternalMemory) -> CUresult;

    pub fn cuMemcpy2DAsync_v2(
        copy_params: *const CudaMemcpy2D,
        stream:      super::cuda_driver::CUstream,
    ) -> CUresult;
}

#[repr(C)]
pub struct CudaMemcpy2D {
    pub src_x_in_bytes: usize,
    pub src_y:          usize,
    pub src_member_type: u32,     // CU_MEMORYTYPE_DEVICE = 2, or _ARRAY = 3
    pub src_host:       *const std::ffi::c_void,
    pub src_device:     super::cuda_driver::CUdeviceptr,
    pub src_array:      CUarray,
    pub src_pitch:      usize,
    pub dst_x_in_bytes: usize,
    pub dst_y:          usize,
    pub dst_member_type: u32,
    pub dst_host:       *mut std::ffi::c_void,
    pub dst_device:     super::cuda_driver::CUdeviceptr,
    pub dst_array:      CUarray,
    pub dst_pitch:      usize,
    pub width_in_bytes: usize,
    pub height:         usize,
}
