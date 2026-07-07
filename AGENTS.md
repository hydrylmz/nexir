# Nexir — Agent Guide

GPU-accelerated non-linear video editor in Rust. Workspace with two packages: `nexir` (library) and `ui` (winit+egui desktop app).

## Build & Dev Commands

```bash
cargo build                   # builds both nexir and ui
cargo test                    # unit + integration tests
cargo test -p nexir --lib     # library unit tests only
cargo test -p nexir --test '*'  # integration tests
cargo run -p ui               # launch the desktop app
```

**Run tests with GPU** (integration tests need headless wgpu): `cargo test`

## Build Dependencies (Windows)

- **FFmpeg dev libs** at `C:/ffmpeg/` (set `VE_FFMPEG_LIB_DIR` to override)
- **CUDA 12.0 SDK** at `C:/Program Files/NVIDIA Corporation/CUDA/v12.0/lib/x64/`
- `.cargo/config.toml` sets `/DELAYLOAD:cuda.dll` and `/DELAYLOAD:nvEncodeAPI64.dll` — required for NVENC export

## Architecture

Labelled so the intent of each crate is clear:

**`nexir` (library crate)** — engine core:
| Module | Responsibility |
|---|---|
| `timeline/` | SoA clip store, tracks, query/mutation, effects |
| `render/` | wgpu render graph, WGSL shaders compute/effect chain |
| `io/` | FFmpeg demuxer/decoder, frame cache, prefetch worker |
| `audio/` | FFmpeg audio decoder, CPAL ring-buffer output |
| `export/` | Export engine, NVENC H.264/AAC, muxer |
| `interop/` | CUDA interop for GPU decode/encode texture sharing |
| `sync/` | Master clock, A/V drift correction, sync probing |
| `scheduler/` | Frame scheduler with render islands |
| `colour/` | Color space math, LUT parsing, DeltaE |

**`ui` (binary crate)** — desktop shell:
- `winit` event loop + `egui` panels (inspector, timeline, media pool, top bar, viewport)
- Preview rendered via blit shader onto egui texture
- Project save/open via `rfd` native dialogs (`.nexp` JSON format)

## Key Patterns & Conventions

### Timeline Store (SoA)
`TimelineStore` uses structure-of-arrays (parallel Vecs). Every clip property is a separate vec indexed by `ClipId`. **Never add a new property without extending all mutation paths.** See `mutation.rs` for insert/remove/trim patterns.

### Clip IDs & Track IDs
- `ClipId(u32)` — index into store arrays; generated sequentially
- `TrackId(u8)` — max 256 tracks
- `SourceId(u32)` — index into `SourceRegistry` arrays
- All use `.index()` for usize conversion

### FFmpeg FFI Pattern
FFmpeg struct fields accessed via C wrapper functions in `src/shim.c`. The Rust FFI modules (`src/io/ffi/`, `src/audio/ffi/`, `src/export/ffi/`, `src/interop/ffi/`) call these wrappers rather than accessing fields directly (FFmpeg exposes struct members as opaque in newer versions).

`build.rs` links: `avformat`, `avcodec`, `avutil`, `swscale`, `swresample`, `cuda`, `nvidia-encode`.

### WGSL Shaders
Compiled at startup via `ShaderRegistry::compile_all()`. Shaders live in `src/render/shader/*.wgsl` and are included at compile time (`include_str!`). Hot-reload supported per-shader.

Required wgpu features: `TEXTURE_BINDING_ARRAY`, `PUSH_CONSTANTS`, `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING`.

### Project File (.nexp)
JSON with `format_version: 1`. Saved/loaded via `ProjectFile::save()` / `ProjectFile::load()`.

### Timebase
All timeline PTS values use `Rational { num: 1, den: 90_000 }` (90 kHz MPEG timebase). Default frame rate is `30/1`. Conversions via `frame_to_pts()` / `pts_to_frame()`.

### UI Theme
Hardcoded dark theme in `NexirApp::new()`. No theme switching. Window size: 1280x720.

## Critical Gotchas

1. **`src/tests/mod.rs` has dead module references**: `delta_e_tests` and `lut_tests` are declared but their files don't exist. `cargo test` will fail if `#[cfg(test)]` activates these. Currently they compile because the outer `tests` module is `#[cfg(test)]` gated and the inner re-exports somehow don't trigger — but adding a new `#[cfg(test)]` block in `mod.rs` exposing them will break the build.

2. **`Cargo.lock` is committed** despite `.gitignore` listing it — the `.gitignore` entry is effectively overridden (checked in already). Do not touch.

3. **`GpuDevice` is `!Send` on some backends** — always wrap in `Arc` for cross-thread sharing. The `surface_format` field uses `Mutex` for interior mutability.

4. **Test GPU requirement**: Integration tests (`src/tests/`) create a headless wgpu device. They will fail in environments without a GPU/driver (most CI runners).

5. **Export pipeline**: The render graph is compiled per-frame from the current scheduler output. The `ExportEngine` reuses `FrameScheduler` and the same `RenderGraphCompiler` path as the preview pipeline.

## Style Notes

- `.gitignore` lists `Cargo.lock` but the file is committed — don't "fix" this
- `#![allow(dead_code)]` at top of `store.rs`: many accessor methods are intentionally unused while the API stabilizes
- Serde derives on all data structures — serde is not optional
- `pollster::block_on` used for async wgpu calls in synchronous contexts (app init, shader compilation)
- `log` crate for instrumentation, `env_logger` for UI binary
- UI uses `rfd` (native file dialogs) — test environments without a display will panic
