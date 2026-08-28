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

#[cfg(test)]
mod tests;
