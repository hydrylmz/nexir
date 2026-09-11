# Nexir

A GPU-accelerated non-linear video editor written in Rust. Timeline editing,
a wgpu compute/render effect chain, FFmpeg decode, and H.264/HEVC export with a
zero-copy CUDA→NVENC path on NVIDIA hardware.

Nexir is a working editor rather than a finished product: the engine core
(timeline, render graph, decode, audio, export, colour) is covered by tests, and
the desktop shell is usable but rougher. See [Status](#status) for what is
verified and what is not.

## Workspace layout

Two crates:

| Crate | Kind | What it is |
|---|---|---|
| `nexir` | library | the engine: timeline, rendering, I/O, audio, sync, export |
| `ui` | binary | the desktop app: `winit` event loop + `egui` panels |

### `nexir` — engine modules

| Module | Responsibility |
|---|---|
| `timeline/` | structure-of-arrays clip store, tracks, query/mutation, effects, markers, timecode |
| `render/` | wgpu render graph, WGSL shaders, effect chain, still-image and text rasterisation |
| `io/` | FFmpeg demuxer/decoder, frame cache, slot pool, prefetch worker |
| `audio/` | FFmpeg audio decode, mixer (pan/fade/limiter), CPAL ring-buffer output |
| `sync/` | master clock, A/V drift corrector, presentation decisions, sync probing |
| `scheduler/` | frame scheduler with render islands |
| `export/` | export engine, partitioner, renderer, encoders, muxer, progress |
| `interop/` | CUDA/D3D12 interop: shared buffers, NVDEC decode, NVENC encode, RGB→NV12 |
| `colour/` | YUV matrices, HDR transfer functions, `.cube` LUT parsing, ΔE |
| `profiling.rs` | per-stage frame timing and report tables |
| `project.rs`, `project_file.rs`, `autosave.rs` | project model, `.nexp` save/load, autosave + recovery |

### `ui` — desktop shell

`winit` + `egui`, with panels for the viewport, timeline, inspector, media pool,
top bar, export settings, and the crash-recovery and media-relink dialogs.
Preview frames are blitted from the render graph's output texture onto an egui
texture. Projects save to `.nexp` (JSON, `format_version: 1`) through native
`rfd` file dialogs.

## Pipeline

Decode and render:

```
FFmpeg demux → decode (NVDEC when available) → YUV upload
  → YuvToRgbNode → per-clip effects → CompositeNode → tone map
  → RGBA16Float FINAL_COLOR
```

The render graph is compiled from the scheduler's per-frame clip set, cached, and
recompiled only when the active clips, their dimensions, or their effects change.
Everything downstream of `YuvToRgbNode` works in linear RGBA16Float.

Export takes one of two paths off `FINAL_COLOR`:

- **`CudaNvenc`** — zero-copy. `Nv12EncodeNode` converts RGBA16Float → NV12
  directly into a D3D12/CUDA shared buffer that NVENC reads as
  `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR`. The frame never touches system
  memory. Four pipeline slots keep the GPU ahead of the encoder.
- **`FfmpegEncoder`** — readback + libavcodec. Still probes `h264_nvenc` /
  `hevc_nvenc` (then AMF, then QSV) before libx264/libx265, so it is usually
  also a GPU encode; what it gives up is the zero-copy upload. Chosen when CUDA
  interop is unavailable, for codecs outside NVENC's H.264/HEVC scope (ProRes,
  VP9), or when the output's bit depth or range cannot be signalled by the direct
  interop session — 10-bit HDR10 exports go here for exactly that reason.

Effects (all WGSL compute): colour correction, gaussian blur, sharpen, vignette,
chroma key, 3D LUT, tone map, 2D transform with corner pin and mattes, plus
blend modes in the compositor.

## Colour

- Timeline PTS are in a 90 kHz timebase (`Rational { num: 1, den: 90_000 }`);
  the default frame rate is 30/1.
- BT.601 / BT.709 / BT.2020 matrices, limited and full range, 8/10/12-bit, in
  both directions, unit-tested against a round-trip.
- Colour metadata comes from the decoded frame rather than the container, and is
  part of the render graph's cache key.
- SDR export tone-maps HDR sources down to Rec.709. HDR10 export (BT.2020 + PQ,
  10-bit) skips the tone map and attaches SMPTE ST 2086 mastering-display and
  CTA-861.3 content-light-level metadata to both the encoder and the container.

## Building

### Requirements

- Rust (edition 2021 for `nexir`, 2024 for `ui`)
- **FFmpeg development libraries** at `C:/ffmpeg/` on Windows. Override the
  location with `VE_FFMPEG_LIB_DIR`. `build.rs` links `avformat`, `avcodec`,
  `avutil`, `swscale`, `swresample`.
- **A GPU** with `PUSH_CONSTANTS`. `TEXTURE_BINDING_ARRAY` +
  `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING` are used when
  present and fall back cleanly when not. DX12 is forced on Windows and Vulkan
  on Linux, because CUDA interop shares memory through
  `D3D12_HANDLE_TYPE_WIN32` / `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`.
- **For NVENC export:** an NVIDIA driver providing `cuda.dll` on the DLL search
  path. The consumer driver installs it as `nvcuda.dll`, which is *not* the name
  the delay-load resolver looks for — `target/debug/` holds a copy under the
  right name. The CUDA **Toolkit** is not required: `build.rs` generates the
  import library from `build/cuda.def` with `lib.exe`.
- `.cargo/config.toml` sets `/DELAYLOAD:cuda.dll` and
  `/DELAYLOAD:nvEncodeAPI64.dll`, so a machine without NVIDIA hardware still
  runs — CUDA absence becomes a runtime capability probe instead of a load-time
  process kill.

### Commands

```bash
cargo build                      # both crates
cargo run -p ui                  # launch the editor
cargo test                       # unit + integration tests
cargo test -p nexir --lib        # engine tests only
cargo run -p nexir --bin bench --release   # frame-timing benchmarks
```

The benchmark suite needs the FFmpeg and `cuda.dll` DLLs beside the binary it
builds; if `target/release/` lacks them, copy them from `target/debug/`.
Anything it cannot measure prints `n/a`, and a benchmark whose hardware is
missing prints a skip with the reason rather than a row of zeros.

### Windows releases and updates

Build a redistributable Windows archive from a Developer PowerShell prompt:

```powershell
.\package_windows.ps1
```

This creates `nexir-x86_64-pc-windows-msvc.zip`, containing `nexir.exe` and the
required FFmpeg runtime DLLs. Extract the complete archive into a directory the
current user can write to (for example, under `%LOCALAPPDATA%`); installing under
`Program Files` can prevent the updater from replacing the executable without
UAC elevation.

Releases use the workspace version in `Cargo.toml`. Push a matching
`v<version>` tag (for example, `v0.1.0`) to run the Windows release workflow; a
mismatched tag fails before publishing. Nexir checks `hydrylmz/nexir` releases
in the background and only accepts the exact asset name above. The initial ZIP
provides the runtime DLLs, while subsequent in-app updates replace only
`nexir.exe`, so do not delete the DLLs from the installation directory.

## Testing

Tests live beside the code they cover, plus `src/tests/` for the integration
suite: render graph hazards and pixel output, playback and seek, A/V sync,
CUDA shared buffers, the RGB→NV12 shader, and end-to-end export validation that
decodes the exported file and compares pixels against the source pattern.

Two environment properties gate parts of the suite:

- **A GPU is required.** `src/tests/` creates a headless wgpu device; without a
  driver these fail rather than skip. Most CI runners cannot run them.
- **CUDA/NVENC tests skip when the hardware is absent.** Set
  `NEXIR_REQUIRE_NVENC=1` to turn those skips into failures on a machine that is
  supposed to have it. `NEXIR_KEEP_EXPORT=1` keeps exported files for inspection.

`nvchk/` holds standalone C probes that back every "measured" or "verified"
claim in `src/interop/`. They are deliberately outside the Cargo build and are
not run by `cargo test`; see `nvchk/README.md`.

## Status

Verified by tests:

- timeline store mutation, query, effects, markers, timecode
- render graph scheduling, hazard ordering, effect pipeline, composite output
- YUV↔RGB matrix math across every matrix × range × bit depth
- RGB→NV12 encode shader: colour, chroma siting, two-plane layout, pitch
- CUDA/D3D12 shared buffers in both directions
- export decodes back to the source pattern on both backends, across resolutions
  and codecs, with sane PTS/DTS and correct colour tags in the container
- HDR10 export carries real 10-bit PQ pixels, not SDR pixels with HDR tags
- A/V drift stays bounded over a simulated 3-hour playback

Known gaps:

- audio effects (EQ, compression, noise reduction) are not implemented
- no media-compatibility matrix across codecs × containers × operations
- crash recovery, missing-media relinking, and corrupted-project handling have
  UI and autosave plumbing but no end-to-end tests
- editor UX is untested automatically — `rfd` needs a display
- HDR *input* (PQ/HLG or 10-bit source decode) is untested; only HDR output is
- A/V sync is tested in one configuration, not across track counts, sample
  rates, and frame rates

Contributor notes, conventions, and the gotchas worth knowing before touching
the FFI or the export path are in [`AGENTS.md`](AGENTS.md).
