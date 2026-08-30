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

/// `CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1` — the buffer sibling of
/// [`CudaExternalMemoryMipmappedArrayDesc`], used to map imported memory as a
/// flat `CUdeviceptr` instead of a `CUarray`.
///
/// `reserved[16]` must be zero for the same reason as in
/// [`CudaExternalMemoryHandleDesc`].
///
/// Measured against `nv-codec-headers` `n12.2.72.0`'s `dynlink_cuda.h` with
/// `offsetof` (probe: `nvchk/extbuf_probe.c layout`):
///
/// ```text
///     size 88
///     offset    0
///     size      8
///     flags    16
///     reserved 20
/// ```
#[repr(C)]
pub struct CudaExternalMemoryBufferDesc {
    /// Byte offset into the imported allocation where the mapping starts.
    pub offset: u64,
    /// Bytes to map.  May be SMALLER than the size given to
    /// `cuImportExternalMemory` — D3D12 pads a committed resource up to its
    /// heap alignment (61440 → 65536 for the NV12 case), and the import wants
    /// the padded `GetResourceAllocationInfo` size while this wants the logical
    /// one.  Verified with exactly that pair — and separately verified that a
    /// LOGICAL-sized import also works on this driver, so the asymmetry is a
    /// correctness choice, not a hard requirement (see `external_buffer.rs`).
    pub size:   u64,
    pub flags:  u32,
    pub reserved: [u32; 16],
}

/// The three external-memory descriptors, checked against the `offsetof`/`sizeof`
/// the vendor header produces under MinGW gcc (`nvchk/extbuf_probe.c layout`).
/// A short struct here makes the driver write past the allocation.
const _: () = {
    assert!(std::mem::size_of::<CudaExternalMemoryHandleDesc>() == 104);
    assert!(std::mem::size_of::<CudaExternalMemoryBufferDesc>() == 88);
    assert!(std::mem::size_of::<CudaExternalMemoryMipmappedArrayDesc>() == 120);
};

pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:        u32 = 1;
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32:      u32 = 2;
/// A D3D12 *heap* handle.
///
/// This is the value the D3D12 import paths in this crate used to pass while
/// calling it `..._D3D12_RESOURCE`, and it works — see the note on
/// [`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`].  Kept named correctly so
/// the mistake cannot be made again from the constant list.
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP:        u32 = 4;
/// A D3D12 *resource* handle, i.e. what `ID3D12Device::CreateSharedHandle` on an
/// `ID3D12Resource` returns — which is what every import in this crate passes.
///
/// The header says 4 is `D3D12_HEAP` and 5 is `D3D12_RESOURCE`; this crate said
/// 4 was `D3D12_RESOURCE` and shipped it.  Measured (`nvchk/extbuf_probe.c`, RTX
/// 3050, driver API 12.2): the driver accepts a resource NT handle under BOTH
/// values, for both a committed TEXTURE2D imported as a mipmapped array and a
/// committed BUFFER imported as a device pointer, and in all four combinations
/// the full allocation round-trips byte-identically through
/// `cuMemcpy2D`/`cuMemcpyHtoD`+`DtoH`.  So the old value was not producing wrong
/// behaviour — it was a wrong NAME on a lenient driver, which is exactly the
/// kind of thing that stops being harmless on the next driver.  The imports now
/// pass 5, the value the header defines for the handle they actually hold.
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE:    u32 = 5;
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

    /// Map imported memory as a flat device pointer.
    ///
    /// The returned `CUdeviceptr` is released with `cuMemFree_v2` — NOT with a
    /// dedicated "unmap" call, and not by `cuDestroyExternalMemory` alone.
    /// Calling `cuMemFree` on memory CUDA did not allocate looks wrong; it is
    /// what the driver documents and what it accepts (measured:
    /// `nvchk/extbuf_probe.c buf`, both handle types — `CUDA_SUCCESS`, followed
    /// by a clean `cuDestroyExternalMemory`, no access violation).
    pub fn cuExternalMemoryGetMappedBuffer(
        dev_ptr: *mut super::cuda_driver::CUdeviceptr,
        ext_mem: CUexternalMemory,
        desc:    *const CudaExternalMemoryBufferDesc,
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
