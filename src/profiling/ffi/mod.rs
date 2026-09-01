// src/profiling/ffi/mod.rs
//
// Native FFI used by the profiler. Currently one surface: NVML, for the GPU/NVENC
// utilisation and driver-reported VRAM rows.
//
// Every layout constant in here is established by a standalone probe in `nvchk/`
// (see `nvchk/nvml_probe.c` and `nvchk/README.md`) rather than by a comment citing
// documentation — the rule AGENTS.md states for `src/interop/ffi/` and which applies
// for the same reason wherever hand-written struct layouts meet a vendor DLL.

/// NVML — `nvml.dll`, loaded dynamically. Windows-only; the module is `#![cfg(windows)]`
/// internally, so callers get a compile error rather than a link error elsewhere.
#[cfg(windows)]
pub mod nvml;
