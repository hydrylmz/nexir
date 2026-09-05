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

    /// Whether the device's primary context is active, and with which flags.
    ///
    /// Needed because FFmpeg's `AV_CUDA_USE_PRIMARY_CONTEXT` path *checks* this and
    /// refuses to proceed when the context is already active with flags other than
    /// the ones it wants (`hwcontext_cuda.c`: "Primary context already active with
    /// incompatible flags", `AVERROR(ENOTSUP)`). Since `CudaContext::new` may run
    /// before the first decoder is opened, this crate has to leave the primary
    /// context in the state FFmpeg will accept — see
    /// [`crate::interop::cuda_context::CudaContext::new`].
    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut u32,
        active: *mut i32,
    ) -> CUresult;

    /// Set the flags the device's primary context will be created with.
    ///
    /// Only legal while the context is INACTIVE — it returns
    /// `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE` otherwise, which is a condition to report
    /// rather than to ignore: it means something already retained the context and
    /// the flags cannot be reconciled.
    ///
    /// `_v2` is the name the driver exports; there is no unversioned symbol.
    pub fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: u32) -> CUresult;

    pub fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPopCurrent(ctx: *mut CUcontext) -> CUresult;

    /// The context current to the calling thread, or NULL when there is none.
    ///
    /// **The observation `CudaContext::with_context` is built on, and it exists
    /// because `cuCtxPushCurrent` cannot be used to find out.** The driver requires
    /// the pushed context to be *floating* — not current to any thread, including
    /// this one — so pushing a context FFmpeg has already made current returns
    /// `CUDA_ERROR_INVALID_CONTEXT` (201) rather than nesting.
    ///
    /// MEASURED, `examples/g2e_interop_target_probe.rs`: after
    /// `av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_CUDA, …, AV_CUDA_USE_PRIMARY_CONTEXT)`
    /// the primary context is current on the calling thread, so every subsequent
    /// `with_context` push failed and the pop that followed it took FFmpeg's context
    /// off the stack — one `cuImportExternalMemory` failing with "invalid device
    /// context", then a stack the next call corrupted further.
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;

    /// Bind a context to the calling thread, floating or not.
    ///
    /// **The API `with_context` uses, and the difference from `cuCtxPushCurrent` is
    /// the entire reason the interop path works at all.** `cuCtxPushCurrent` pushes
    /// onto the thread's context *stack* and requires the context to be floating —
    /// attached to no thread — so once FFmpeg's decoder threads hold the primary
    /// context, every push from the render thread returns
    /// `CUDA_ERROR_INVALID_CONTEXT` (201). `cuCtxSetCurrent` has no such
    /// requirement: since CUDA 4.0 one context may be current to many threads at
    /// once, which is precisely the sharing `AV_CUDA_USE_PRIMARY_CONTEXT` sets up.
    ///
    /// MEASURED (`examples/g2e_interop_target_probe.rs`): after `Decoder::open`
    /// attaches NVDEC, `cuCtxGetCurrent` reports NULL on the calling thread while
    /// `cuCtxPushCurrent(primary)` returns 201 and `cuCtxSetCurrent(primary)`
    /// returns 0 — same handle, same thread, same instant.
    ///
    /// Passing NULL unbinds, which is how [`super::super::cuda_context::CudaContext`]
    /// restores a thread that had no context before.
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;

    pub fn cuStreamCreate(stream: *mut CUstream, flags: u32) -> CUresult;
    pub fn cuStreamDestroy(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    /// Block until EVERY stream in the current context has finished.
    ///
    /// **The decode-side half of the interop path's ordering, and it cannot be
    /// replaced by `cuStreamSynchronize`.** NVDEC decodes into its output surface on
    /// *FFmpeg's* CUDA stream (`AVCUDADeviceContext::stream`), while
    /// `DecodeInteropTarget::copy_from_nvdec_frame` issues its `cuMemcpy2DAsync` on
    /// the stream `CudaContext` created. Two streams in one context are unordered
    /// with respect to each other, so synchronising ours proves only that OUR copy
    /// finished — it says nothing about whether the decoder had finished writing the
    /// surface the copy read.
    ///
    /// Measured symptom of the missing barrier, on `bench --interop`: 15-22% of the
    /// pixels that move between consecutive frames came back holding the PREVIOUS
    /// frame's content, the picture differed run to run (the CPU arm is
    /// bit-identical across runs), and the difference sat exactly on moving edges.
    /// No error, no counter, nothing in the pool's statistics.
    ///
    /// A context-wide sync rather than reading FFmpeg's stream handle out of
    /// `AVCUDADeviceContext` on purpose: that would be a hard-coded struct offset in
    /// an FFmpeg-version-dependent layout, which AGENTS.md requires a `nvchk/` probe
    /// to justify, and it would buy nothing measurable here — the decode is already
    /// complete by the time the copy is issued in the common case, so this returns
    /// at once (measured 0.02 ms/frame at 4K).
    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuMemcpyDtoDAsync_v2(
        dst:    CUdeviceptr,
        src:    CUdeviceptr,
        byte_count: usize,
        stream: CUstream,
    ) -> CUresult;

    /// Free a device allocation.
    ///
    /// Also the documented release for a pointer obtained from
    /// `cuExternalMemoryGetMappedBuffer`, where CUDA never allocated the memory
    /// in the first place — see the note on that function.
    pub fn cuMemFree_v2(dev_ptr: CUdeviceptr) -> CUresult;

    pub fn cuMemcpyHtoD_v2(
        dst:        CUdeviceptr,
        src:        *const std::ffi::c_void,
        byte_count: usize,
    ) -> CUresult;

    pub fn cuMemcpyDtoH_v2(
        dst:        *mut std::ffi::c_void,
        src:        CUdeviceptr,
        byte_count: usize,
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
