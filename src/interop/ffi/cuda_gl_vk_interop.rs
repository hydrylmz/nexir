use super::cuda_driver::{CUresult, CUarray, CUmipmappedArray, CUexternalMemory};

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1`.
///
/// The trailing `reserved[16]` is NOT optional padding: the driver validates that
/// those words are zero and rejects the import with `CUDA_ERROR_INVALID_VALUE`
/// ("invalid argument") when they contain stack garbage.  Construct this only via
/// `..Default::default()` or an explicit zeroed `reserved`.
#[repr(C)]
pub struct CudaExternalMemoryHandleDesc {
    pub r#type: u32,           // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1, _WIN32 = 2
    pub handle: HandleUnion,
    pub size:   u64,
    pub flags:  u32,            // CUDA_EXTERNAL_MEMORY_DEDICATED = 1 — set for texture-backed allocations
    pub reserved: [u32; 16],
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

/// `CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1`.
///
/// `reserved[16]` must be zero for the same reason as in
/// [`CudaExternalMemoryHandleDesc`].
#[repr(C)]
pub struct CudaExternalMemoryMipmappedArrayDesc {
    pub offset:     u64,
    pub array_desc: CudaArray3DDescriptor,
    pub num_levels: u32,
    pub reserved:   [u32; 16],
}

/// `CUDA_ARRAY3D_DESCRIPTOR_v2` — note the field order: the three extents come
/// first, then `format`, `num_channels`, `flags`.  `width`/`height`/`depth` are
/// `size_t`.
#[repr(C)]
pub struct CudaArray3DDescriptor {
    pub width:        usize,
    pub height:       usize,
    pub depth:        usize,        // 0 for a 2D array
    pub format:       u32,          // CU_AD_FORMAT_* — must match the imported footprint
    pub num_channels: u32,          // 1..=4
    pub flags:        u32,          // CUDA_ARRAY3D_SURFACE_LDST = 2 — required for read/write access
}

pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:        u32 = 1;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32:      u32 = 2;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE:    u32 = 4;
pub const CUDA_EXTERNAL_MEMORY_DEDICATED:                  u32 = 1;
// CUarray_format values.  CUDA validates the (format, num_channels) pair against
// the imported allocation's footprint, so these must describe the same bytes per
// pixel as the D3D12/Vulkan format being imported — see
// `external_texture::CudaArrayFormat::for_wgpu`.
pub const CU_AD_FORMAT_UNSIGNED_INT8:                       u32 = 0x01;
pub const CU_AD_FORMAT_UNSIGNED_INT16:                      u32 = 0x02;
pub const CU_AD_FORMAT_UNSIGNED_INT32:                      u32 = 0x03;
pub const CU_AD_FORMAT_HALF:                                u32 = 0x10;
pub const CU_AD_FORMAT_FLOAT:                               u32 = 0x20;
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
