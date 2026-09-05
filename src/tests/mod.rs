// src/tests/mod.rs
//
// Each file here wraps its tests in a `mod <same name as the file>` block, so
// clippy's `module_inception` fires seven times (`tests::av_sync::av_sync`, and
// so on).  That nesting is deliberate: it is what lets a single test be named
// unambiguously on the command line, e.g.
//
//     cargo test -p nexir --lib tests::export_validation::export_validation::nvenc_export_matches_pattern
//
// Renaming the inner modules to satisfy the lint would change every test path
// for no benefit, so it is allowed for this module tree only.
#![allow(clippy::module_inception)]

pub mod render_integration;
// pub mod delta_e_tests;
// pub mod lut_tests;
pub mod playback_smoke;
pub mod yuv_upload;
pub mod av_sync;
pub mod colour_plumbing;
pub mod media_compat;
pub mod project_persistence;
pub mod interop_correctness;
pub mod abgr10_repack;
pub mod nv12_encode;
pub mod shared_buffer;
pub mod export_validation;
/// P2.3 — the fused colour-correction/LUT/chroma-key pass against the three nodes it
/// replaces. Needs a GPU and nothing else: no CUDA, no NVENC, no files.
pub mod fused_grade;

/// Serialises every test that puts the CUDA primary context current.
///
/// **Not because a second bind fails — that part is fixed.**
/// `CudaContext::with_context` is `cuCtxSetCurrent`, which has no floating
/// requirement and lets one context be current to many threads at once (see that
/// function's doc comment and `examples/g2e_interop_target_probe.rs`). The old
/// `cuCtxPushCurrent` implementation *did* fail that way: push needs the context
/// FLOATING, `AV_CUDA_USE_PRIMARY_CONTEXT` puts FFmpeg's decoder in it, and every
/// push after the first `Decoder::open` returned `CUDA_ERROR_INVALID_CONTEXT`
/// while the unconditional pop stripped a context it had never placed.
///
/// What the lock still guards is the CUDA WORK inside the closures. Every
/// `CudaContext` in this process wraps one `CUcontext` —
/// `CudaContext::new` calls `cuDevicePrimaryCtxRetain` for the device, so two test
/// modules each holding their "own" `Arc<CudaContext>` hold the same one — and one
/// stream, so two tests issuing copies or opening NVENC sessions concurrently
/// interleave on shared state. `cuCtxSynchronize` is context-wide: a test that
/// calls it waits for the other test's work too, and one that destroys a resource
/// does so while another is reading it.
///
/// So the lock lives HERE rather than per-module: `export_validation` had its own
/// `export_lock` and `shared_buffer` would have had another, and two locks guarding
/// one resource is not guarding it.
///
/// Held across the whole body of any test that touches CUDA, not just the CUDA
/// call: a test that maps a buffer, runs a wgpu dispatch and then reads back must
/// not have another test's session open and close in between.
pub fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panicking test poisons the mutex; ignore that rather than cascading one
    // failure into "every other CUDA test also failed".
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The ONE CUDA primary context for the whole test binary, retained once and
/// never released.
///
/// **One owner, for the same reason there is one lock**, and this was learned the
/// expensive way. `CudaContext::new` calls `cuDevicePrimaryCtxRetain` and `Drop`
/// calls the matching release, and every `CudaContext` in the process wraps the
/// same `CUcontext`. A second module holding its own `OnceLock<Arc<CudaContext>>`
/// is a second retain/release pair on one resource — and the symptom is not a
/// failure in the module that added it: `tests::export_validation`'s three NVENC
/// tests began reporting *"the export engine selected the FFmpeg backend"*,
/// because `EncodeInterop::open` was refused against a context a second owner had
/// disturbed. All six passed when that module ran alone, which is exactly the
/// shape gotcha 4 describes.
///
/// Never released on purpose: with tests in parallel threads, a context that dies
/// when one test finishes makes an unrelated test fail for reasons that have
/// nothing to do with the code under test.
///
/// Returns `None` when the host has no interop capability, which callers must turn
/// into a printed skip rather than a failure.
pub fn shared_cuda_ctx(
    capability: &crate::interop::capability::InteropCapability,
) -> Option<std::sync::Arc<crate::interop::cuda_context::CudaContext>> {
    static CUDA: std::sync::OnceLock<
        Option<std::sync::Arc<crate::interop::cuda_context::CudaContext>>,
    > = std::sync::OnceLock::new();
    CUDA.get_or_init(|| {
        if !capability.is_available() {
            return None;
        }
        match crate::interop::cuda_context::CudaContext::new(capability) {
            Ok(c) => Some(std::sync::Arc::new(c)),
            Err(e) => {
                eprintln!("[tests] CudaContext::new failed: {e:?}");
                None
            }
        }
    })
    .clone()
}
