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
pub mod interop_correctness;
pub mod abgr10_repack;
pub mod export_validation;
