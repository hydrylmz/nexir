#![allow(
    clippy::large_enum_variant,
    clippy::missing_transmute_annotations,
    clippy::should_implement_trait,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod timeline;
pub mod project;
pub mod project_file;
pub mod autosave;
pub mod render;
pub mod colour;
pub mod io;
pub mod scheduler;
pub mod engine;
pub mod audio;
pub mod sync;
pub mod export;
pub mod interop;
pub mod profiling;
/// Locally generated real-media fixtures, shared by `src/bin/bench.rs` and
/// `src/tests/media_compat.rs`. Not `#[cfg(test)]`: a `[[bin]]` target cannot see
/// a test-only module, and the alternative was a second copy of the ffmpeg
/// locator and its skip contract.
pub mod bench_media;

#[cfg(test)]
mod tests;
