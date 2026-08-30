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
pub mod av_sync;
pub mod colour_plumbing;
pub mod media_compat;
pub mod project_persistence;
pub mod interop_correctness;
pub mod abgr10_repack;
pub mod nv12_encode;
pub mod shared_buffer;
pub mod export_validation;

/// Serialises every test that puts the CUDA primary context current.
///
/// `CudaContext::with_context` is `cuCtxPushCurrent` / pop, and the driver
/// requires the context to be FLOATING to be pushed — a context already current
/// on another thread cannot be pushed on a second one.  Every `CudaContext` in
/// this process wraps the same allocation: `CudaContext::new` calls
/// `cuDevicePrimaryCtxRetain` for the device, so two test modules each holding
/// their "own" `Arc<CudaContext>` are holding one `CUcontext`.
///
/// So the lock has to live HERE rather than per-module: `export_validation` had
/// its own `export_lock` and `shared_buffer` would have had another, and two
/// locks guarding one resource is not guarding it.  Symptom when this is missing:
/// `cuCtxPushCurrent` returns `CUDA_ERROR_INVALID_CONTEXT`, the pop takes the
/// wrong context off the stack, and the debug assertion in `with_context` fires
/// with `left: 0x0` — from a test whose own code is correct and which passes when
/// run alone.
///
/// Held across the whole body of any test that touches CUDA, not just the CUDA
/// call: a test that maps a buffer, runs a wgpu dispatch and then reads back must
/// not have another test's push land between its own.
pub fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A panicking test poisons the mutex; ignore that rather than cascading one
    // failure into "every other CUDA test also failed".
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
