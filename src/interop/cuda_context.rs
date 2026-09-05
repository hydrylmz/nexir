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
    MapBuffer(String),
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
    ///
    /// **Shares the device's primary context with FFmpeg's `hwcontext_cuda`, and
    /// that sharing is load-bearing rather than incidental.** The interop decode
    /// path copies out of a surface NVDEC wrote on FFmpeg's stream, so ordering the
    /// copy after the decode requires both to live in ONE `CUcontext`: a
    /// `cuCtxSynchronize` in a context of our own waits for our own work and nothing
    /// else. `probe_hardware_device` passes `AV_CUDA_USE_PRIMARY_CONTEXT` for exactly
    /// this reason (see `HwDeviceType::create_flags`), which is what puts FFmpeg in
    /// the same context this retains.
    ///
    /// The flag dance below exists because FFmpeg's primary-context path *validates*
    /// the context's flags: `hwcontext_cuda.c` calls `cuDevicePrimaryCtxGetState` and
    /// fails with "Primary context already active with incompatible flags"
    /// (`AVERROR(ENOTSUP)`) unless the flags are `CU_CTX_SCHED_BLOCKING_SYNC`. This
    /// constructor usually runs FIRST — a `CudaContext` is built before any decoder
    /// is opened — so it is this call that decides what state FFmpeg finds. Setting
    /// the flags must therefore happen BEFORE the retain that activates the context;
    /// afterwards the driver refuses with `CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`.
    ///
    /// **The failure mode when this is wrong has no error at the call site.** With
    /// two contexts the decode still works and the copy still succeeds, but 15-22% of
    /// the pixels that move between consecutive frames come back holding the previous
    /// frame's content — measured on `bench --interop`, varying run to run, with the
    /// CPU arm bit-identical across runs. With incompatible flags the symptom moves
    /// to the other end: NVDEC is refused entirely and every source silently falls
    /// back to the CPU upload path, which the benchmark reports as `live targets 0`.
    pub fn new(capability: &InteropCapability) -> Result<Self, CudaError> {
        // Step 1 — Resolve the CUdevice handle from the ordinal.
        let mut device: CUdevice = 0;
        let ret = unsafe { cuDeviceGet(&mut device, capability.cuda_device_ordinal) };
        if ret != CUDA_SUCCESS {
            return Err(CudaError::DeviceGet(cu_err_to_string(ret)));
        }

        // Step 2 — Line the primary context's flags up with what FFmpeg demands,
        // while it is still inactive. `CU_CTX_SCHED_BLOCKING_SYNC` (0x04) is the
        // value `hwcontext_cuda.c` checks for.
        //
        // Only when the context is not already active: if something else retained it
        // first the flags cannot be changed, and the honest outcome is to proceed and
        // let FFmpeg decide (it accepts an active context whose flags already match).
        const CU_CTX_SCHED_BLOCKING_SYNC: u32 = 0x04;
        let mut flags: u32 = 0;
        let mut active: i32 = 0;
        if unsafe { cuDevicePrimaryCtxGetState(device, &mut flags, &mut active) } == CUDA_SUCCESS
            && active == 0
            && flags != CU_CTX_SCHED_BLOCKING_SYNC
        {
            let set = unsafe { cuDevicePrimaryCtxSetFlags_v2(device, CU_CTX_SCHED_BLOCKING_SYNC) };
            if set != CUDA_SUCCESS {
                // Not fatal: FFmpeg may still accept the context, and if it does not
                // the source falls back to the CPU path with a printed reason. Worth
                // a warning because it is the difference between the interop path
                // running and silently not running.
                log::warn!(
                    "[interop] could not set the primary context's flags to \
                     BLOCKING_SYNC ({}) — FFmpeg's AV_CUDA_USE_PRIMARY_CONTEXT may \
                     refuse it, and every source would then fall back to the CPU \
                     upload path",
                    cu_err_to_string(set)
                );
            }
        }

        // Step 3 — Retain the primary context (shared with FFmpeg's hwcontext_cuda).
        let mut ctx: CUcontext = std::ptr::null_mut();
        let ret = unsafe { cuDevicePrimaryCtxRetain(&mut ctx, device) };
        if ret != CUDA_SUCCESS {
            return Err(CudaError::CtxRetain(cu_err_to_string(ret)));
        }

        // Step 4 — Bind the context to create the stream, then restore.
        //
        // `cuCtxSetCurrent` rather than push/pop for the reason [`Self::with_context`]
        // documents at length: if a decoder was opened first (which happens whenever
        // an `IoLayer` outlives one project, or a second `CudaContext` is built after
        // playback started) the primary context is no longer floating and
        // `cuCtxPushCurrent` returns `CUDA_ERROR_INVALID_CONTEXT`. The old code
        // ignored that return, so `cuStreamCreate` ran against whatever the thread
        // happened to have — and the pop that followed removed a context this
        // function never put there.
        let mut previous: CUcontext = std::ptr::null_mut();
        let had_previous = unsafe { cuCtxGetCurrent(&mut previous) } == CUDA_SUCCESS;
        let bind = unsafe { cuCtxSetCurrent(ctx) };
        if bind != CUDA_SUCCESS {
            unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(CudaError::CtxRetain(format!(
                "the primary context was retained but could not be made current: {}",
                cu_err_to_string(bind)
            )));
        }
        let mut stream: CUstream = std::ptr::null_mut();
        let ret = unsafe { cuStreamCreate(&mut stream, 1 /* CU_STREAM_NON_BLOCKING */) };
        unsafe {
            cuCtxSetCurrent(if had_previous { previous } else { std::ptr::null_mut() })
        };

        if ret != CUDA_SUCCESS {
            unsafe { cuDevicePrimaryCtxRelease(device) };
            return Err(CudaError::StreamCreate(cu_err_to_string(ret)));
        }

        Ok(Self { device, ctx, stream })
    }

    /// Run `f` with this context current on the calling thread, then restore
    /// whatever was current before.
    ///
    /// **`cuCtxSetCurrent`, NOT `cuCtxPushCurrent`, and the difference is the whole
    /// reason the interop decode path works.** Push manipulates the thread's context
    /// *stack* and requires the context to be **floating** — attached to no thread at
    /// all. `AV_CUDA_USE_PRIMARY_CONTEXT` puts FFmpeg's decoder in this very context,
    /// so from the first `Decoder::open` onwards it is no longer floating and every
    /// push returns `CUDA_ERROR_INVALID_CONTEXT` (201). `cuCtxSetCurrent` has no such
    /// requirement: since CUDA 4.0 one context may be current to many threads at
    /// once, which is exactly the sharing this type exists to arrange.
    ///
    /// MEASURED, `examples/g2e_interop_target_probe.rs`, one process, one thread,
    /// consecutive lines of output after `Decoder::open` attached NVDEC:
    ///
    /// ```text
    ///   cuCtxGetCurrent            -> 0, NULL          (nothing current here)
    ///   cuDevicePrimaryCtxGetState -> 0, flags=0x4, active=1
    ///   cuCtxPushCurrent(ours)     -> 201 (invalid device context)
    ///   cuDevicePrimaryCtxRetain   -> 0, SAME handle   (our handle is not stale)
    ///   cuCtxPushCurrent(retained) -> 201              (nor is a fresh one)
    ///   cuCtxSetCurrent(ours)      -> 0                (and set just works)
    /// ```
    ///
    /// **What the push version cost.** Every `with_context` after the first decoder
    /// open silently ran `f` against the wrong context *and* then popped FFmpeg's
    /// context off the thread stack. The first visible symptom was
    /// `cuImportExternalMemory failed: invalid device context` inside
    /// `SharedTexture::new`, which `InteropDecodeTargets::ensure_target` flattens to
    /// "allocating the shared Y/UV texture pair failed" and caches as a per-source
    /// rejection — so every source fell back to the CPU upload path and
    /// `bench --interop` reported `live targets 0`, comparing the CPU path against
    /// itself while labelling one arm `interop`.
    ///
    /// The previous context is restored rather than left set, including the NULL case
    /// (`cuCtxSetCurrent(NULL)` unbinds): a thread that had no context before must not
    /// acquire one as a side effect, or an unrelated later call on that thread starts
    /// succeeding-by-accident against a context it never asked for.
    ///
    /// **Concurrency is now safe for this use.** Two threads may each set the same
    /// context current, which is the case that used to fail; what remains unsafe is
    /// the CUDA work inside `f`, which callers still serialise where it matters
    /// (`src/tests/export_validation.rs`, `src/tests/shared_buffer.rs`).
    pub fn with_context<R>(&self, f: impl FnOnce(CUstream) -> R) -> R {
        let mut previous: CUcontext = std::ptr::null_mut();
        let observed = unsafe { cuCtxGetCurrent(&mut previous) };
        if observed == CUDA_SUCCESS && previous == self.ctx {
            // Already ours — nothing to bind and, crucially, nothing to restore.
            return f(self.stream);
        }

        let set = unsafe { cuCtxSetCurrent(self.ctx) };
        if set != CUDA_SUCCESS {
            log::error!(
                "[interop] cuCtxSetCurrent failed: {} ({set}) — every CUDA call in \
                 this closure will run against whatever context this thread already \
                 had. This is not the floating-context condition cuCtxPushCurrent \
                 used to hit; a failing set means the context handle itself is \
                 invalid.",
                cu_err_to_string(set),
            );
            return f(self.stream);
        }

        let result = f(self.stream);

        // Restore, including to NULL. `observed != CUDA_SUCCESS` means the query
        // itself failed, so there is no known previous state to restore and
        // unbinding is the honest choice.
        let restore = if observed == CUDA_SUCCESS { previous } else { std::ptr::null_mut() };
        let back = unsafe { cuCtxSetCurrent(restore) };
        debug_assert_eq!(
            back, CUDA_SUCCESS,
            "restoring the previous CUDA context failed ({}); this thread is left \
             with {:?} current",
            cu_err_to_string(back),
            self.ctx
        );
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
