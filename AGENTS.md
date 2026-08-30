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
- **CUDA driver** — `cuda.dll` on the DLL search path (`target/debug/` holds a copy;
  the consumer NVIDIA driver installs it as `nvcuda.dll`, which is NOT the name the
  delay-load resolver looks for). The CUDA **Toolkit** is not required: `build.rs`
  generates the import library from `build/cuda.def` with `lib.exe`, so **adding a
  CUDA function to `src/interop/ffi/cuda_*.rs` means adding a line to
  `build/cuda.def`** or the link fails with `unresolved external symbol __imp_<name>`.
  Use the `_v2` name the driver actually exports (`cuMemFree_v2`, not `cuMemFree`).
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

### Native FFI provenance — `nvchk/`
Every "measured"/"verified" claim in `src/interop/` cites a standalone C probe in `nvchk/`, which is outside the Cargo build and not run by `cargo test`. Build them with `cd nvchk && ./fetch_headers.sh && ./build_probes.sh` (plain gcc, dynamic `LoadLibraryA` of `nvcuda.dll`/`nvEncodeAPI64.dll`, deliberately independent of `build.rs` and the delay-load config). Only `.c`/`.py`/`.sh` are committed; the vendor headers are pinned by tag *and* commit hash by `fetch_headers.sh` rather than vendored. **Adding a hard-coded struct offset, size, `_VER` word or enumerant to `src/interop/ffi/` means adding it to a probe** — an unfalsifiable layout comment is what turned one out-of-scope pitch measurement into a process kill. See `nvchk/README.md`.

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

4. **Test GPU requirement**: Integration tests (`src/tests/`) create a headless wgpu device. They will fail in environments without a GPU/driver (most CI runners). Tests needing **CUDA** as well (`tests::shared_buffer`, the NVENC half of `tests::export_validation`) must hold `tests::cuda_lock()` for their whole body: `CudaContext` wraps the device's *primary* context, so every instance in the process is one `CUcontext`, and `cuCtxPushCurrent` requires it to be floating. Without the lock they pass alone and fail in the suite, with a `left: 0x0` assertion or silently-zero readbacks.

5. **Export pipeline**: The render graph is compiled per-frame from the current scheduler output. The `ExportEngine` reuses `FrameScheduler` and the same `RenderGraphCompiler` path as the preview pipeline.

6. **Zero-copy NVENC input is NV12 in a shared buffer, and `pitch` is load-bearing.** `ExportBackend::GpuNvenc` runs `Nv12EncodeNode` (`src/interop/nv12_encode.rs`) to convert RGBA16Float → NV12 straight into the `SharedBuffer` NVENC reads, registered as `CUDADEVICEPTR` + `NV_ENC_BUFFER_FORMAT_NV12`. Three coupled invariants:
   - **One pitch, one source.** `EncodeInterop::pitch()` (= `Nv12EncodeNode::aligned_pitch(width)`, 256-byte rounding) feeds `NV_ENC_REGISTER_RESOURCE::pitch`, every `NV_ENC_PIC_PARAMS::inputPitch`, *and* the shader's push constants. Never recompute it at a call site — the chroma plane sits at `pitch * height`, so a stride disagreement is a hue shift or a sheared frame, never an error return. `pitch = 0` on a two-plane NV12 registration **kills the process** inside `nvEncEncodePicture` (the old "the driver ignores pitch" note was measured on a packed single-plane ABGR10 array and does not generalise).
   - **The shader owns the matrix, so the gate is about bit depth and range.** `ExportJob::nvenc_zero_copy_is_colour_safe` is `bit_depth <= 8 && effective_range() == Limited`. Matrix is irrelevant now that we convert; 10-bit (needs P010 + `profileGUID`/`pixelBitDepthMinus8`) and full range (needs `video_full_range_flag`) both live in NV_ENC_CONFIG's per-codec VUI union, which the FFI refuses to write, so they route to FFmpeg.
   - **`Nv12EncodeNode`'s WGSL push-constant struct nests `RgbToYuv` and must include its trailing padding.** Dropping `_colour_pad` shifts `pitch`/`chroma_plane_offset` by 8 bytes; the shader then reads pitch 0 and every row overwrites row 0.

7. **`nvenc_export_matches_pattern` is tagged BT.709 on purpose.** BT.601 is what the NVENC driver applies unprompted, so a BT.601 job cannot distinguish "our shader ran" from "the driver guessed right". Verified to have teeth: forcing `ColorInfo::bt601()` into `Nv12EncodeNode::new` fails the test with red decoding `[255, 24, 0]`. Run it with `NEXIR_REQUIRE_NVENC=1` so a missing-hardware skip becomes a failure.

8. **`Abgr10RepackNode` is off the export path.** It survives only as the tree's one 10-bit packing (a future P010/HDR reuse) with its own passing test. If it stops earning that, delete it *and* `src/tests/abgr10_repack.rs` together rather than leaving a second unexercised encode path.

9. **`src/bin/bench.rs` may only print what it measured.** It previously reported `gpu_utilization: 88.5`, `nvenc_utilization: 94.0` and `ram_used_bytes: 420 MB` as hardcoded literals, and its "Zero-Copy NVENC" path was a `std::thread::yield_now()` — so the NVENC column timed a thread yield next to a fabricated utilisation figure. Now: every `SystemMetrics` field is `Option`, unmeasured ones print `n/a` (pinned by `profiling::tests::unmeasured_system_metrics_print_as_not_available`), and the NVENC path opens a real `EncodeInterop` session and reports its packet count and bitstream size. **Missing hardware must stay a printed skip, never a zero row** — filling a metric from arithmetic instead of a driver query (NVML, `cuMemGetInfo`) is the regression. `allocated_gpu_bytes` is the one exception and is labelled a lower bound, because it counts only what the bench itself allocated, not the graph's texture pool. The bench needs `cuda.dll` beside the binary (`cp target/debug/*.dll target/release/`) or interop probes as `transport=None` and three of five benchmarks skip.

10. **`LutNode::set_size` is mandatory and `new` cannot do it for you.** `LutNode::new` only sees the LUT cube, so it leaves `params.width/height` at zero; `declare_resources` then asks wgpu for a zero-sized texture and the failure is `Dimension X is zero` from inside `Device::create_texture`, naming neither the node nor the missing call. There is now an assertion in `declare_resources` that names both. `EffectChainBuilder` constructs its own params with the canvas size and is unaffected — the bench was the only direct caller, which is why this stayed latent.

11. **Colour-metadata tests must use BT.601 sources, not BT.709.** BT.709 limited is simultaneously the commonest real input *and* every fallback in the chain — `ColorInfo::default()`, `from_ffmpeg`'s heuristic for an HD frame, and `luma_coefficients`' `Unknown` arm — so a test whose fixture is BT.709 passes even when the frame's metadata is dropped entirely and never reaches `YuvToRgbNode`. `tests::colour_plumbing` (P1.6) therefore tags its fixtures BT.601, where a mix-up is a ~24-level error on red. Two supporting rules:
    - **The fixtures are written by this crate's own `VideoEncoder` + `Muxer`,** not an external ffmpeg binary, because `VideoEncoder::open` pins swscale to `job.sws_colorspace()` — so the samples genuinely carry the matrix the VUI is tagged with. `the_two_matrices_are_not_interchangeable` asserts that the two fixtures actually differ (28 luma levels on saturated colour, 0 on grey); if `sws_setColorspaceDetails` ever silently fails, that test is the one that catches it and every other assertion in the file becomes vacuous.
    - **Each pixel assertion needs a mis-tagged control.** `node_decodes_the_pattern_using_the_signalled_matrix` re-renders the same decoded bytes with `matrix` forced to BT.709 and requires the result to move by MORE than the tolerance. Without it, a shader ignoring its push constants and hardcoding one matrix would pass, because one of the two would be right by accident.

## Style Notes

- `.gitignore` lists `Cargo.lock` but the file is committed — don't "fix" this
- `#![allow(dead_code)]` at top of `store.rs`: many accessor methods are intentionally unused while the API stabilizes
- Serde derives on all data structures — serde is not optional
- `pollster::block_on` used for async wgpu calls in synchronous contexts (app init, shader compilation)
- `log` crate for instrumentation, `env_logger` for UI binary
- UI uses `rfd` (native file dialogs) — test environments without a display will panic
