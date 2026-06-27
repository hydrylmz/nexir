pub type CUdevice    = i32;
pub type CUresult    = i32;
pub type CUcontext   = *mut std::ffi::c_void;
pub type CUstream    = *mut std::ffi::c_void;
pub type CUdeviceptr = u64;          // CUDA device pointers are 64-bit handles, not real pointers
pub type CUarray     = *mut std::ffi::c_void;
pub type CUmipmappedArray = *mut std::ffi::c_void;
pub type CUexternalMemory = *mut std::ffi::c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:    i32 = 33;
pub const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: i32 = 34;
pub const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: i32 = 50;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: u32) -> CUresult;

    pub fn cuDriverGetVersion(version: *mut i32) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut i32) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceGetAttribute(value: *mut i32, attrib: i32, dev: CUdevice) -> CUresult;

    pub fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;

    pub fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPopCurrent(ctx: *mut CUcontext) -> CUresult;

    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    pub fn cuStreamDestroy(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    pub fn cuMemcpyDtoDAsync_v2(
        dst:    CUdeviceptr,
        src:    CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;

    pub fn cuGetErrorString(error: CUresult, str_ptr: *mut *const std::ffi::c_char) -> CUresult;
}

pub fn cu_err_to_string(result: CUresult) -> String {
    let mut ptr: *const std::ffi::c_char = std::ptr::null();
    unsafe {
        cuGetErrorString(result, &mut ptr);
        if ptr.is_null() {
            return format!("CUDA error {result} (no description)");
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
