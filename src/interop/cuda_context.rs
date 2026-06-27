// src/interop/cuda_context.rs
// Safe wrapper around CUDA primary context and stream.
// Owns the primary context retain/release lifecycle.

use crate::interop::ffi::cuda_driver::*;
use crate::interop::capability::InteropCapability;

#[derive(Debug)]
pub enum CudaError {
    DeviceGet(String),
    CtxRetain(String),
    StreamCreate(String),
    Import(String),
    MapArray(String),
}

pub struct CudaContext {
    device:  CUdevice,
    ctx:     CUcontext,
    /// Non-blocking stream dedicated to interop copies, separate from FFmpeg's internal stream.
    stream:  CUstream,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    /// Retrieve the primary context for the device identified in `capability`.
    /// Shares the same context as FFmpeg's hwcontext_cuda.
    pub fn new(capability: &InteropCapability) -> Result<Self, CudaError> {
        // Step 1 — Resolve the CUdevice handle from the ordinal.
        let mut device: CUdevice = 0;
        let ret = unsafe { cuDeviceGet(&mut device, capability.cuda_device_ordinal) };
        if ret != CUDA_SUCCESS {
            return Err(CudaError::DeviceGet(cu_err_to_string(ret)));
        }

        // Step 2 — Retain the primary context (shared with FFmpeg's hwcontext_cuda).
        let mut ctx: CUcontext = std::ptr::null_mut();
        let ret = unsafe { cuDevicePrimaryCtxRetain(&mut ctx, device) };
        if ret != CUDA_SUCCESS {
            return Err(CudaError::CtxRetain(cu_err_to_string(ret)));
        }

        // Step 3 — Push context to create the stream, then pop it.
        unsafe { cuCtxPushCurrent(ctx) };
        let mut stream: CUstream = std::ptr::null_mut();
        let ret = unsafe { cuStreamCreate(&mut stream, 1 /* CU_STREAM_NON_BLOCKING */) };
        let mut popped: CUcontext = std::ptr::null_mut();
        unsafe { cuCtxPopCurrent(&mut popped) };

        if ret != CUDA_SUCCESS {
            unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(CudaError::StreamCreate(cu_err_to_string(ret)));
        }

        Ok(Self { device, ctx, stream })
    }

    /// Run `f` with this context current on the calling thread, then restore
    /// the previous context. Each OS thread has its own independent context stack.
    pub fn with_context<R>(&self, f: impl FnOnce(CUstream) -> R) -> R {
        unsafe { cuCtxPushCurrent(self.ctx) };
        let result = f(self.stream);
        let mut popped: CUcontext = std::ptr::null_mut();
        unsafe { cuCtxPopCurrent(&mut popped) };
        debug_assert_eq!(popped, self.ctx, "CUDA context stack corrupted by closure");
        result
    }

    pub fn raw_context(&self) -> CUcontext {
        self.ctx
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        self.with_context(|stream| unsafe {
            cuStreamSynchronize(stream);
            cuStreamDestroy(stream);
        });
        unsafe { cuDevicePrimaryCtxRelease(self.device) };
    }
}
