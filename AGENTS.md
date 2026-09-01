# Nexir — Agent Guide

GPU-accelerated non-linear video editor in Rust. Workspace with two packages: `nexir` (library) and `ui` (winit+egui desktop app).

## Build & Dev Commands

```bash
cargo build                   # builds both nexir and ui
cargo test                    # unit + integration tests
cargo test -p nexir --lib     # ALL of this crate's tests, including src/tests/
cargo run -p ui               # launch the desktop app
```

**There is no `tests/` directory.** Integration tests live in `src/tests/` behind a
`#[cfg(test)] mod tests` in `lib.rs`, so they are `--lib` tests as far as Cargo is
concerned: `cargo test -p nexir --test '*'` fails with *"no test target matches
pattern `*`"* rather than running them. Use `cargo test -p nexir --lib
tests::colour_plumbing` to name one file's tests.

**Run tests with GPU** (integration tests need headless wgpu): `cargo test`

**Benchmarks** (`src/bin/bench.rs`) — see gotchas 9, 12 and 13 for what its numbers do and do not claim:

```bash
cargo build -p nexir --bin bench --release
cp target/debug/*.dll target/release/   # cuda.dll must sit beside the binary
./target/release/bench.exe              # all 6, median of 3 repeats each
./target/release/bench.exe 5 6          # only benchmarks 5 and 6
NEXIR_BENCH_REPEATS=1 ./target/release/bench.exe    # smoke run; prints that it is a sample
NEXIR_UPLOAD_PATH=write_texture ./target/release/bench.exe   # force one upload mechanism
NEXIR_FRAME_DUMP=1 ./target/release/bench.exe 5      # per-frame outliers + the latency/stage pairing
NEXIR_FRAME_CSV=target/run_%d.csv ./target/release/bench.exe 5   # per-frame series in ARRIVAL order
NEXIR_GPU_LOOKAHEAD=1 ./target/release/bench.exe 5   # collapse the pipeline (lowers only)
NEXIR_NODE_TIMINGS=1 ./target/release/bench.exe 5    # per-node GPU timings, in an EXTRA pass
./target/release/bench.exe --media    # real coded frames: decode/render/encode/E2E per class
./target/release/bench.exe --export   # end-to-end ExportEngine per class, every output verified
```

A single run of the 4K row spans ~40% between repeats, so **quote the median with its spread, never one run.**

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

12. **`YuvUploadNode` has two upload mechanisms, and `record` must branch on the DATA, not on the configured path.** `UploadPath::Auto` (the default) resolves per node: `WriteTexture` when the staging path would repack rows (1080p 8-bit luma pads 1920 → 2048), `StagingBuffer` when it would not (3840 is already aligned). Both earn their place — measured 3× each, `write_texture` wins every 1080p row by 25-40% and loses 4K by ~7%, so deleting either costs real performance. Force one with `NEXIR_UPLOAD_PATH=staging|write_texture`; that is how the comparison is reproduced from one binary. Two coupled invariants:
    - **`record` branches on whether a deferred frame is pending, never on `self.upload_path`.** `write_texture` needs the destination texture, which only exists during `record`, so `upload_frame_shared` parks an `Arc` and `record` completes it. But a `WriteTexture`-configured node still serves callers that lend a borrowed slice (`ui/src/app.rs`, `src/export/renderer.rs`) or hand over planar chroma — both of which route to the staging buffers. Branching on the configured path made `record` return early and skip the `copy_buffer_to_texture` for data already sitting in staging: **a black frame, with no error, no warning and no failing test**, on every 1080p preview frame. `upload_frame` therefore clears `pending_frame`, and `record` peeks rather than takes it so a re-record without a new upload still draws. Pinned by `a_write_texture_node_still_serves_the_borrowed_slice_api` and `a_borrowed_upload_supersedes_a_pending_deferred_one`.
    - **Adding a layout to the `write_texture` path means adding it to `upload_frame_shared`'s guard.** It accepts semi-planar only and falls back to staging otherwise, because planar U/V must be interleaved first and `upload_frame` already does that correctly — a second interleaver is a second thing to keep in step. `bytes_per_row` there is the UNPADDED source stride; passing the 256-aligned one shears the picture exactly like a wrong NVENC pitch (gotcha 6).

13. **A percentile over `CPU per frame` is not a frame time.** Since frames went in flight, the sum of one frame's CPU stages is the CPU's *share* of a frame (6.7 ms at 4K) while frames arrive 20.0 ms apart — so the report carries a separate measured `Frame latency` row, and that is the only row a "P95 ≤ 16.67 ms" target can be read off. `average_fps` is likewise frames ÷ wall time, never `1000 / CPU sum`: the old formula reported 237 FPS on a run delivering 58. Three rules follow, each with a test: unrecorded latency prints nothing rather than `0.00 ms`; the latency series must cover the WHOLE run (an unseeded `last_retire` silently dropped the pipeline-fill interval and printed 18.25 ms mean on a 21.1 ms/frame run); and because mean latency and throughput measure the same seconds two ways, `format_table` prints a `WARNING` when they disagree by >10%.

14. **`POOL_BUCKET_CAPACITY` is coupled to the graph's SIMULTANEOUS peak, and getting it wrong has no error message.** `CompiledGraph::execute` acquires every declared resource before the first node records and releases them all after the last, so a graph with N same-key intermediates holds N at once and hands back N together. `TransientTexturePool::release` drops whatever exceeds the per-key cap, so a cap below N discards the surplus every frame and re-creates it on the next frame's acquire — a permanent steady-state miss. **This was the P2.1 tail, and it was found by measurement, not by reading the code.** Benchmark 5's Heavy 4K graph peaks at 16 canvas-sized RGBA16Float textures in one bucket; the old cap of `8` therefore evicted ~1150 of them per 90-frame run at 66 MB each. Measured either way, three repeats (`target/bench_P21_cap8.txt` vs `target/bench_P21_final.txt`):

    | cap | `peak bucket` | miss rate | evicted | FPS | steady P95 | frame interval |
    |---|---|---|---:|---:|---:|---|
    | 8 | 8/8 (clamped) | 31.3% | ~1150 | 43.6 | 27.48 ms | **17.9 / 25.7 ms alternating** |
    | 32 | 16/32 | 0.6% | 0 | **57.3** | **18.24 ms** | 16.8 ms even |

    **The only symptom was the alternation** — `GPU transfer` split the same way (9.9 / 17.8 ms) while graph execution stayed flat at ~8.5 ms either way. No error, no warning, no failing test, and the pooled percentiles looked like an ordinary tail. Three rules:
    - **Read the cap off a measurement, not a guess.** `PoolStats::peak_bucket` is the observed high-water mark of any single bucket and the bench prints it as `peak bucket N/CAP`. `N == CAP` means the true peak is unknown and at least the cap — which is exactly when `evicted > 0` and the bench prints its `WARNING`. `render::resource::tests::the_bucket_cap_covers_a_whole_frames_peak` pins the constant against that measured 16, and `releasing_more_than_the_cap_evicts_the_surplus` pins the mechanism (surplus dropped → re-allocated next frame) so raising the constant does not leave the coupling untested.
    - **A cap costs nothing when it is not reached.** Buckets are created on demand and only hold what a frame actually returned, so `32` leaves the 1080p single-layer graph at `peak bucket 1/32` and 4 pooled textures.
    - **`miss_rate()` is `Option`.** Zero acquisitions is not a 0% miss rate; the pool follows gotcha 9's rule and prints `n/a`.

15. **The 4K "latency P99 = 54.6 ms" was the pipeline fill, not a stall — the latency series needs a split, never a trim.** A 90-frame run stamps ~90 intervals, so the P99 index resolves to the second-largest sample, and at 4K the largest samples are the first interval (measured 47-71 ms: the first `gpu_lookahead` submits, the first NVENC picture, first-frame allocation). Reporting that as a frame arriving late sent P2.1 hunting a tail stall that did not exist. But **dropping the fill is the bug gotcha 13 already records** (mean 18.25 ms on a 21.1 ms/frame run), so `ProfileReport` carries both: `frame_latency_stats` is whole-run and its mean must stay comparable with `average_fps`; `steady_latency_stats` excludes only the first interval and is the row a P95/P99 target is read off; `pipeline_fill_ms` prints that interval separately rather than hiding it. `format_table` says which row is which, and the bench summary has a `Fill` column. Two supporting rules:
    - **A distribution cannot see a cycle, so measure the parity split.** `latency_alternation` reports the mean of the even- and odd-indexed steady intervals, and `format_table` prints `ALTERNATING` above a 15% split. A pooled percentile is identical whether the slow frames alternate or cluster, and the two diagnoses lead opposite ways — a stall invites more pipeline depth, a cycle is made *worse* by it. `NEXIR_GPU_LOOKAHEAD` collapses the pipeline to tell those apart from one binary, and `NEXIR_FRAME_CSV` keeps arrival order, which is the only view a periodic pattern is visible in.
    - **Attribute a tail by the PAIRING, not by two percentiles.** `Upload`'s P99 being large next to a large latency P99 is equally consistent with "the late frame is the slow upload" and with "two unrelated frames were slow". `profiling::format_frame_dump` (`NEXIR_FRAME_DUMP=1`) prints the intersection of the two outlier sets and Pearson coefficients; on benchmark 5 it returned `0 in BOTH` with `r = +0.04`, which is what **refuted** the `write_buffer` lead and sent the investigation to the texture pool. Outliers are `median + 3 MAD`, not `mean + 3σ` (one 45 ms sample among 5 ms ones widens σ until it is no longer an outlier by its own measure) and a flat series must select **nothing** — `↳ submit` on the `write_texture` path is sub-microsecond every frame, so a "slowest N" ranking over it would name arbitrary frames and then report them as coinciding with the tail.

16. **The transient texture pool is FIFO, and a stack there costs nothing measurable while invalidating every bind-group cache in the graph.** `CompiledGraph::execute` acquires resources in ascending `ResourceId` order and releases them in ascending order too (`into_resources()` is indexed by id), so a LIFO bucket pops them back out **reversed**: resource 1 receives resource N's texture, resource 2 receives N-1's, and the whole assignment flips — then flips back next frame, alternating with period 2. Eleven nodes key their bind-group cache on the `ViewId`s they were handed (`lut.rs:241`, `composite.rs:398`, `tonemap.rs:350`, …), so all of them rebuilt every frame. **The pool's own counters cannot see this**: hits, misses (0.6%), `evicted` (0) and `peak bucket` (16/32) are identical either way, because every acquisition is still a hit — it is simply the *wrong* hit. `TransientTexturePool::buckets` is therefore a `VecDeque` with `push_back`/`pop_front`, pinned by `render::resource::tests::a_repeated_frame_shape_reuses_each_resources_own_texture`, which compares the ViewId *assignment* across three frames rather than the miss rate (two frames would pass a period-2 alternation). Measured on benchmark 5, three repeats each: 52.9 → 53.9 FPS median, whole-run mean latency 18.15 → 17.85 ms — i.e. **~2%, at the edge of the row's own 2-3% spread.** Keep it because it is free and the graph's caches are meaningless without it; do not quote it as a throughput win. The 4K frame is still ~10 ms `GPU transfer` + ~7.8 ms graph, and neither moved.

17. **`bench --export` measures the whole `ExportEngine`, and the harness's prefetch worker must be stopped by hand.** `PrefetchWorker::run` loops on a 500 µs sleep until its `shutdown` flag is set or its channel disconnects, and the export harness keeps the `SyncSender` alive — so one worker per class per repeat survives the run that spawned it. With 5 classes × 3 repeats that is fourteen live decode threads by the last row, making later rows slower for a reason that is the benchmark's own bookkeeping. `ExportHarness` implements `Drop` to set the flag; a new harness field is not enough. The profile also honours `NEXIR_BENCH_REPEATS` with the same default and the same wording as the synthetic sweep, and a **failed repeat fails the whole class** rather than being dropped from the median — the failures `verify_exported_file` returns are "the output does not decode" and "frames went missing", and a median over the repeats that happened to succeed would report a rate for a pipeline that is not reliably producing the file. Spread prints `n/a` on a single repeat, never `0%`.

## Style Notes

- `.gitignore` lists `Cargo.lock` but the file is committed — don't "fix" this
- `#![allow(dead_code)]` at top of `store.rs`: many accessor methods are intentionally unused while the API stabilizes
- Serde derives on all data structures — serde is not optional
- `pollster::block_on` used for async wgpu calls in synchronous contexts (app init, shader compilation)
- `log` crate for instrumentation, `env_logger` for UI binary
- UI uses `rfd` (native file dialogs) — test environments without a display will panic
