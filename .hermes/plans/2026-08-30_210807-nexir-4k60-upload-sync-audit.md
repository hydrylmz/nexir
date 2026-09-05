# Nexir 4K60 Performance Audit — Remaining Work

> Trimmed 2026-09-04 (seventh pass). **Everything in this file is unfinished.** Phases A/B/C, I0–I3, P2.1, G1, D2, D3, E1's API, F1b, E1b, F2, all of G2 (a–e), G3's engine fix, G2f.1–G2f.4, the fifth pass's Task 0 and Task 1, and the sixth pass's Tasks A, B, C, D, E and F's re-measurement are done and have been removed. Their outcomes survive as AGENTS.md gotchas 9/12/13/14/15/16/17/18/19/20/**21/22/23/24/25**, in the tests those gotchas name, and in `target/bench_P21_*.txt`, `target/bench_F1b.txt`, `target/bench_E1b.txt`, `target/bench_F2.txt`, `target/g2a_probe_3x.txt`, `target/bench_G2e_interop_3x.txt`, `target/bench_G2f_interop_3x.txt`, `target/bench_G2f_pipelined_3x.txt`, `target/bench_taskB_5_7_8_3x.txt`, `target/taskB_7_1.csv`, `target/taskB_bf0_7_1.csv`, `target/taskB_decode_split_queued.txt`, `target/taskC_single_source.txt`, `target/taskF_b7_nodes.txt`. G2's written re-costing is `.hermes/plans/2026-08-31_g2-recosting.md`.

## What landed on 2026-09-04 (sixth pass), for provenance only — do not re-do

| What | Where |
|---|---|
| **Task A** — five gotchas, not three. Gotcha 20's false first bullet rewritten; the G3 EOF design, the pts-transfer rule, the `seek_to` rule, and the four-source attribution rule added as **21, 22, 23, 25**. Line references (`io_layer.rs:330`) dropped rather than re-derived — the drain moved them and they will move again. | AGENTS.md gotchas 20–25 |
| **Task B** — the alternation is the **decoder's reorder buffer**, and it is not this crate's pipeline. | `decode_ms` column in `write_frame_csv` (+ `frame_csv_columns_agree` and two tests in `bench`); `examples/b_decode_split_probe.rs`; `Decoder::send_packet_only`/`receive_into`. Gotcha 24. |
| **Task B's controls** — four leads refuted by measurement, one confirmed. `NEXIR_GPU_LOOKAHEAD=1` keeps the cycle (16.81/52.35 ms); `NEXIR_FRAME_DUMP` gives `0 in BOTH`, `r = -0.095`; the probe reproduces it with no graph/NVENC/pipeline/lock (6.47/54.90 ms); a 4-deep decoder input queue does nothing (6.85/54.81). Same fixtures re-encoded `-bf 0`: **cycle gone**, `decode` 37.74/37.90 ms, no `ALTERNATING`. | `target/taskB_*.txt`, `target/taskB_*.csv` |
| **Task C** — the grain inversion is REAL and has two measured causes, neither a defect: the per-source block excludes the demux (1.5 ms/frame vs 0.04), and four sources share one NVDEC engine so the split reports who waited (single-source sum 31.0 ms vs four-source total 30.9, split moving wholesale). Gotcha 20's claim narrowed to one source at 1080p. | Gotcha 25; `MULTI_4K60_SOURCES`' doc comment; `target/taskC_single_source.txt` |
| **Task D** — G5 restated as a frame-time budget with a percentile, gate on benchmark 7. | AGENTS.md "Build & Dev Commands", and the `BENCHMARKS` comment above rows 7/8 |
| **Task E** — `bench --export` re-run on a binary carrying the pool, FIFO, G2c, G2e, G3, pts and seek fixes. All 5 classes NVENC, every output decoded, frame counts exact, spreads 0-17%. 4K30 60.4 → 66.9 FPS, 4K60 74.8 → 83.6. | `target/bench_F2.txt` (overwritten) |
| **Task F's re-measurement** — benchmark 7's own per-node figures, which E1b could not supply (it measured benchmark 5). `NEXIR_NODE_TIMINGS` now works on the real-media rows. | `target/taskF_b7_nodes.txt`; the new pass in `run_real_media_benchmark` |
| **P2.6** — closed as a decision with the note the plan asked for. | `src/export/readback.rs`'s header |

`cargo test -p nexir --lib`: **267 passed, 0 failed.** `cargo test --bin bench`: 2 passed.

## Where things stand, because the tasks below are judged against it

**The gate, restated (Task D, now in AGENTS.md):** steady-state frame interval ≤ 16.67 ms at P95 and ≤ 20 ms at P99 **on benchmark 7**, median of ≥3 repeats, read off `↳ steady`. Average FPS is throughput, not the gate.

| row | FPS | steady P95 | steady P99 | verdict |
|---|---:|---:|---:|---|
| 5 — synthetic, CPU upload | 55.6 | 19.07 ms | 20.57 ms | not the gate, and must never be quoted as one |
| **7 — real media, interop** | **26.5** | **72.87 ms** | **92.22 ms** | **short by ~4.4×** |
| 8 — real media, CPU upload | 18.6 | 68.51 ms | 73.58 ms | short by ~4.1× |

Median of 3, `target/bench_taskB_5_7_8_3x.txt`. Verified on both real-media rows: `live targets 4`, `4/4 clips imported` on every measured frame, `Upload 0.00 MB/frame` counted, `Δpx 0/255` (mean luma 140.3 both arms), `peak bucket 16/32` with 0 evicted. 7-vs-8 is 1.43×.

**Where benchmark 7's ~36 ms frame goes, all measured:**

| part | cost | source |
|---|---:|---|
| decode, 4 sources | ~30.9 ms | `Interop:` block; of 7.18 ms/decode, 6.66 ms is inside `send_packet`/`receive_frame` |
| graph (GPU) | 7.38 ms | `target/taskF_b7_nodes.txt` — Lut3D 1.550, ColorCorrection 1.492, ChromaKey 1.455, Composite 1.337, YuvToRgb 1.222, ToneMap 0.323 |
| composite recording (CPU) | 0.69–0.85 ms | `composite_ms`, flat across the alternation |
| NVENC | 0.23–0.34 ms | `nvenc_ms` |

CPU and GPU overlap once frames are in flight, so those do not sum to the frame. **The decode is the frame; a free graph still leaves ~29 ms.**

**Benchmark 7 still prints `ALTERNATING` (16.7 / 49.1 ms), and per gotcha 15 that fails the target on its own — but the cause is now known and is outside this crate's render path.** Gotcha 24. The only fix that does not change the workload is decoding a frame *before* the frame that needs it, which is Task G below.

---

## Non-negotiable rules that constrain every task here (from `AGENTS.md`)

- Every printed metric is **measured, counted, or `n/a`** — never estimated. `profiling::tests::unmeasured_system_metrics_print_as_not_available` pins this; do not weaken it.
- A missing GPU/CUDA/NVENC/NVML capability is a **printed skip with a reason**, never a zero row.
- Adding a CUDA function to `src/interop/ffi/cuda_*.rs` means adding a line to `build/cuda.def`; adding a hard-coded offset/size/enumerant to any FFI module means adding it to a probe in `nvchk/`.
- **An optimisation only counts if the improvement comes from the pipeline, not from the benchmark doing less work.** `-bf 0` fixtures are the standing example of a diagnostic that is not a fix (gotcha 24).
- **Quote the median of ≥3 repeats with its spread, never one run.**
- **Read P95/P99 off the `↳ steady` row and the mean off the whole-run row** (gotcha 15). On a 90-frame run the whole-run P99 *is* the pipeline fill.
- **`cargo test -p nexir --test '*'` does not work** — there is no `tests/` directory. Use `cargo test` or `cargo test -p nexir --lib`.
- **A benchmark pass must not run inside the session's timed span.** `generate_report` divides frames by `ProfilingSession`'s own elapsed time — which is why the new `NEXIR_NODE_TIMINGS` pass on rows 7/8 sits before `ProfilingSession::new`.
- **Every commit must build on its own.** Check with `git checkout <sha> && cargo check --all-targets`.
- **`cuda.dll` must sit beside the binary that loads it**, including test binaries (`cp target/debug/*.dll target/debug/deps/`) and example binaries (`target/release/examples/`) — without it every interop test *skips with a printed reason*, which looks like a green suite that exercised nothing.
- **Do not compare a serial `--interop` row with a pipelined synthetic row**, and do not compare benchmark 5 with 7 as if the difference were the decode path. 7-vs-8 is the decode-path comparison.
- **A test that allocates a `FrameSlotPool` per probe OOMs the test binary** (~386 MB each). Gotcha 23's last bullet has the workaround.
- **A claim about one source's decode cost must come from a single-source pass** (gotcha 25).

---

## Order of work

```text
G     the reorder-buffer cycle: a second interop target per source   ← the only
                                                                      unrefuted
                                                                      lead on the
                                                                      gate, and it
                                                                      re-opens G4
H     Phase H — pooling / lifetimes / fusion (P2.2–P2.5), re-scoped   ← now known
                                                                      to be ~20%
                                                                      of the frame
```

---

# Task G — the reorder-buffer cycle, i.e. G4 re-opened on new evidence

**This is the only remaining lead on the gate, and it is a fix rather than an investigation.** Gotcha 24 attributes benchmark 7's 16.7 / 49.1 ms cycle to the decoder's reorder buffer: a request for display-ordered frame N either finds it already in the buffer (cheap — a receive with no decode) or must decode the next anchor plus a B-frame (dear). Work per frame *pair* is unchanged, which is why the mean does not move and only the distribution does.

**Why this is G4 and why G4's old verdict does not apply.** G4 (double-buffering the interop target) was ruled out because the read-after-write guard is free — 376 of 384 decodes poll at 0.005–0.008 ms, and the barrier does not grow with N. **That is still true and it is a different question.** The guard being free says a second target buys nothing *for the guard*; it says nothing about whether a second target lets a decode run while the graph reads the previous frame, which is what smoothing a reorder cycle requires.

What to build, and the order matters because step 1 can kill the task:

1. **Measure the ceiling before writing the ownership change.** `examples/b_decode_split_probe.rs` already drives four sources through `DecodeInteropTarget` with no graph; add an arm that keeps **two** targets per source and decodes frame N+1 into the spare while frame N is still held. If the parity split does not flatten there — with no graph and no submission in the way — it will not flatten in the pipeline either, and the task stops.
2. **Then the ownership change**, and gotcha 18's five rules are the ones it must not break: one funnel (`GraphResource::into_transient`), a `ViewId` minted once per target rather than per frame, no descriptor for an import, an unbound import panicking by name, and `FrameState::imported` empty still meaning the CPU path. Two targets per source means `SourceTarget` holds a pair and `held_pts`/`last_read` become per-target — **the read-after-write guard has to follow the target that is about to be written, not the source.**
3. **VRAM is the cost and it is counted, not assumed.** 11.9 MB per 4K source, so four sources go 47.5 → 95.0 MB. `target_bytes()` already counts it and the bench prints it as a lower bound; the NVML row was 4.3–5.6 GB of 8 GB during rows 7 and 8, so this is affordable but must be stated in the row rather than discovered.

Acceptance, all four or it is a different workload under the same heading:

- `ALTERNATING` gone from benchmark 7, or its split under 15%.
- Steady P95/P99 improved, median of ≥3, quoted with spread.
- `4/4 clips imported` on every measured frame, `Upload 0.00 MB/frame`, `Δpx 0/255` against benchmark 8, and the `f` column unchanged.
- `peak bucket` re-read (gotcha 14): two targets per source changes what is live at once, and the cap must be read off the measurement rather than assumed to still be 16/32.

**What this task cannot do:** close the gate on its own. The cycle's two halves average ~24.6 ms; smoothing it perfectly leaves a ~24 ms frame against a 16.67 ms budget. The decode is 6.66 ms of NVDEC time per source per frame and four of them are serialised inside one `schedule_frame` — **the frame budget at four 4K60 sources may simply not exist on one NVDEC engine**, and the honest outcome of this task may be a measurement that says so. If it does, say it in the audit rather than moving the gate.

---

# Task H — Phase H: pooling, lifetimes, fusion (P2.2–P2.5)

**Re-measured 2026-09-05, and the per-node table it was scoped from was misleading in one specific way.** `print_node_timings`' `per pass` column was a median POOLED over a node's four instances, and on benchmark 7 those instances do not cost the same — three cheap passes and one dear one, so `×/fr × per pass` came out 22-30% below the measured `per frame`. Fixed: `aggregate_node_timings` keeps each instance's median, prints `UNEVEN` above a 10% spread, and lists the instances in execution order. Pinned by `bench`'s `a_node_whose_instances_differ_is_reported_per_instance` (+ `instances_that_cost_the_same_are_not_flagged` as the control); `cargo test --bin bench`: 4 passed.

**Per-instance medians, benchmark 7, median of 3 (`target/taskH_uneven_3x.txt`).** Execution order is `MULTI_4K60_SOURCES` order:

| Node | cam_4k60 | cam_4k60_rot | **cam_4k60_grain** | cam_4k60_flip | pooled `per pass` | ×4 | measured `per frame` |
|---|---:|---:|---:|---:|---:|---:|---:|
| ColorCorrection | 0.285 | 0.284 | **0.627** | 0.286 | 0.285 | 1.140 | **1.481** |
| Lut3D | 0.313 | 0.311 | **0.614** | 0.312 | 0.313 | 1.252 | **1.553** |
| ChromaKey | 0.307 | 0.307 | **0.529** | 0.310 | 0.307 | 1.228 | **1.455** |
| YuvToRgb | 0.274 | 0.271 | **0.400** | 0.273 | 0.274 | 1.096 | **1.220** |
| Composite (×1) | — | — | — | — | 1.335 | — | 1.335 |
| ToneMap (×1) | — | — | — | — | 0.322 | — | 0.322 |
| **GRAPH TOTAL** | | | | | | | **7.366 ms** |

The graph total is unchanged — it was always read off the per-frame column — so **Phase H is still ~20% of a ~36 ms frame and still cannot close the gate.** What changed is which item is worth opening.

## P2.4 is CLOSED, and it was not about imported views

The prediction was that `Composite` at 1.337 ms on benchmark 7 against 1.078 on benchmark 5 was the interop path re-creating something per frame. It is not:

- **Benchmark 8 reports 1.330 ms** — pooled textures, CPU upload, same 4-input 4K shape — i.e. within the row's own 0.8% spread of benchmark 7's 1.335. The source of the input views does not move it.
- **Re-run rows 7/8 against four low-entropy fixtures and `Composite` is 1.082 ms**, which is benchmark 5's 1.074 to **0.7%** (`target/taskH_cheapdecode_3x.txt`, median of 3). The whole rise is content.

So there is nothing to fix in `CompositeNode`, and its bind-group caching is not to be probed on this evidence.

## What the graph's cost actually tracks: source entropy

Same shaders, same 3840×2160, same clocks (SM 1980-1995 MHz on both extremes, `target/clk_bars.csv` / `target/clk_noise.csv`), four content sets through benchmark 7:

| content set | graph TOTAL | FPS | benchmark 5 control |
|---|---:|---:|---:|
| clean bars (`scroll` over `smptehdbars`) | **5.881 ms** | 45.5 | 6.802 ms |
| the repo's own fixtures | 7.366 | 26.5 | 6.802 |
| 3 clean + 1 noisy | 7.039 | 30.0 | 6.789 |
| all four `noise=alls=30` | **11.366** | 14.8 | 6.789 |

Benchmark 5 sat at 6.787-6.803 ms across all seven passes of every one of those runs (0.2% spread), so the machine is not drifting. Two things follow:

- **Per-pass GB/s implied by the reads+writes each node declares exceeds the RTX 3050's 224 GB/s on the cheap content (277-454) and falls below it on the noisy (128-231).** What varies is GPU-side lossless framebuffer compression, which is a property of the pixels. **A per-node figure is only comparable across rows carrying the same content.**
- **With content held equal, the interop graph is exactly benchmark 5's graph minus the node it deletes**: 6.802 − 0.935 (`YuvUpload`, absent on an interop clip per gotcha 18) = 5.867 predicted against 5.881 measured, **+0.2%**. That is the cleanest available confirmation that G2d's import path adds no shader cost, and it is only visible once content is equal.
- **`cam_4k60_grain` is dear in the GRAPH as well as the decoder, for a different reason than gotcha 25's.** Its ~35× bitrate makes the *decoded picture* incompressible, so every pass over it costs ~2×. Mixing one noisy source into three clean ones reproduces it: implied 4th instance 0.578/0.533/0.543/0.387 ms against the all-noisy per-pass 0.591/0.562/0.569/0.393, within 1.5-5.2% (`target/taskH_mixed_3x.txt`).

**Trap worth recording: `NEXIR_MEDIA_DIR` does not isolate a fixture set unless the files clear `ensure_fixture`'s 64 KB plausibility floor.** A 4K60 clip of near-static content encodes *below* it, so a directory of 48 KB fixtures was silently re-encoded with the default recipes and the run measured the repo's own fixtures under a "flat content" heading — md5-identical, which is why `target/taskH_flat.txt` matches the fixtures column to 0.1%. `scroll=horizontal=0.004` over `smptehdbars` clears it (0.2 MB); static `color=` does not.

## What is left, re-scoped

| Item | What is known now | First thing to look at |
|---|---|---|
| P2.2 Texture lifetimes | 4K VRAM pressure is real (4.3–5.6 GB of 8 GB on the NVML row); each interop source adds 11.9 MB, 47.5 MB at four. Task G is closed, so it will not double | The graph holds every declared resource for the whole frame (`resolve_resources` + the three `execute*` bodies); non-overlapping lifetimes could share. **`peak bucket 16/32` is the number this reduces**, and gotcha 14 means the cap must be re-read afterwards |
| P2.3 Effect fusion | **Precondition still met, and the per-instance split strengthens it.** On the CHEAP layers CC/LUT/ChromaKey are 0.285/0.313/0.307 ms — within 10% of each other, none an outlier. On the grain layer they are 0.627/0.614/0.529, i.e. all three roughly double together, which is what a bandwidth-bound chain does. Upper bound ~4.5 ms/frame, ~12% of a 36 ms frame | Four passes each read and write a full 4K RGBA16F texture (63.3 MB). Fusion saves bandwidth, not maths — and the entropy finding above is direct evidence that bandwidth is what these passes are spending. **Measure any fusion against BOTH content sets**, or a win on compressible content will not survive real footage |
| P2.4 Composite | **CLOSED — see above.** Not the imported views, not a per-frame rebuild; content, and `Composite` matches benchmark 5's to 0.7% on equal content | Nothing. Do not re-open |
| P2.5 NVENC tuning | **Explicitly last.** 0.23–0.34 ms measured, under 1% of the frame | Do not start here |

## Pending: AGENTS.md gotcha 27

The finding above belongs in AGENTS.md as gotcha 27 (the pooled-median rule, and the four consequences). **The write was blocked — AGENTS.md is a protected agent-instruction file and the approval prompt went unanswered — so it is recorded here only.** Add it before the next per-node measurement, or the next reader will multiply `per pass` by `×/fr` again.

---

# Settled — do not re-litigate

| Question | Answer |
|---|---|
| Does the RTX 3050 report `TIMESTAMP_QUERY`? | **Yes**, `available` in the bench banner. |
| Is a multi-slot staging ring needed? | **No.** `write_buffer` stages into wgpu's own buffer and the copy is GPU-ordered after the previous submission, so the hazard does not exist. |
| Is `write_texture` faster than `write_buffer` + copy? | **Depends on row alignment; both are kept.** Wins 1080p by 25–40% (1920 pads to 2048), loses 4K by ~7% (3840 is aligned). `UploadPath::Auto` picks per node; force either with `NEXIR_UPLOAD_PATH`. |
| Was the 4K P99 spike a stall inside `write_buffer`? | **No, three times over.** `NEXIR_FRAME_DUMP` refuted it on benchmark 5 (`0 in BOTH`, `r = +0.04`) and again on benchmark 7 (`0 in BOTH`, `r = -0.095`); the "54.6 ms P99" was the pipeline fill landing on the P99 index. The real defect was the pool's bucket cap — gotcha 14. |
| Can the upload copy be bracketed directly? | **No.** `write_buffer`'s copies live in wgpu's own `pending_writes` command buffer. Measured as the gap between consecutive frames' GPU timestamps instead — which is also why `gpu_transfer` is nonzero on a row that uploads nothing. |
| Is `nvml.dll` present, and is D3 done? | **Yes and yes.** |
| Is NVDEC selected on this machine? | **Yes** — `[decoder] Attached hardware decoder: Cuda`. |
| Is CUDA interop available on this machine? | **Yes** — `transport=D3D12Win32Handle, ordinal 0, driver 12.6`. |
| Is G2 worth the ownership refactor? | **Yes, measured rather than estimated.** Serial 4K60 single-source 9.10 → 5.38 ms; pipelined four-source real media 18.6 → 26.5 FPS (1.43×), counted upload at 0. |
| Does the per-frame `cuStreamSynchronize` cost more than the CPU round-trip it replaces? | **No.** G2a: interop 13.93–14.12 ms vs CPU 21.39–22.45 ms, and it is the *stable* pass (1.8% spread vs 17%). |
| Is the read-after-write wait the bottleneck? | **No, and this is settled at FOUR sources as well as one.** 376 of 384 decodes poll at 0.005–0.008 ms. |
| Should G4 (a second interop target per source) be started? | **RE-OPENED — Task G.** The old "no" was correct for the reason it was asked (the guard is free, and it still is). Gotcha 24's reorder cycle is a different reason, and the ceiling measurement in Task G step 1 is the new gate on it. |
| Are per-source CUDA streams the fix? | **No, not on this reading.** The barrier averages 0.305–0.313 ms per copy at four sources and does not grow across them. |
| Why is `grain_1080p60` slower on the interop path? | **NVDEC, not our copy — at ONE source and 1080p, which is the whole scope.** Gotcha 20's last bullet, narrowed by gotcha 25. The 4K grain source's apparent inversion is explained and is not a contradiction. |
| Is benchmark 7's alternation the pipeline, the upload, the graph, or the target-map lock? | **None of them.** It is the decoder's reorder buffer — gotcha 24, four controls. Do not re-open the upload lead. |
| Is `Composite` dearer on benchmark 7 because its inputs are IMPORTED? | **No, and P2.4 is closed.** Benchmark 8 (pooled, CPU upload) reports 1.330 ms against 7's 1.335, and on equal content both fall to 1.082 — benchmark 5's figure to 0.7%. It is source entropy, not the view's provenance. Task H. |
| Can a per-node `per pass` figure be multiplied by `×/fr`? | **Only when the row prints no `UNEVEN`.** The column is a median pooled over that node's instances, and benchmark 7's four layers differ ~2× — the product came out 22-30% under the measured per-frame total. Gotcha 27 / Task H. |
| Is the graph's cost a property of the code alone? | **No — it tracks source entropy.** Same shaders and clocks, graph TOTAL 5.881 → 7.366 → 11.366 ms across three content sets while benchmark 5 held at 6.79-6.80. Per-pass implied bandwidth crosses the 224 GB/s bus rate, so it is framebuffer compression. Compare per-node figures only across equal content. |
| Does `POOL_BUCKET_CAPACITY` move on the interop path? | **No, and it has been read from every shape.** `1/32` single-source, `4/32` serial four-source, `16/32` pipelined four-source with 0 evictions. Task G must re-read it. |
| Must FFmpeg and this crate share one `CUcontext`? | **Yes, and the barrier is meaningless without it.** Gotcha 19. |
| Is `cuCtxPushCurrent` usable for this? | **No.** Push needs the context floating, and FFmpeg holds it from the first `Decoder::open`. Gotcha 4. |
| Can the transient pool hold a texture the decoder owns? | **No, and the filter is in one place.** `GraphResource::into_transient` — gotcha 18. |
| Can a 10-bit source use the interop path? | **No.** The planes are `R8Unorm`/`Rg8Unorm`; a 10-bit source decodes to P010 at two bytes per sample. Refused before allocating. |
| Does colour metadata survive the upload bypass? | **Yes, verified where gotcha 11 says to verify it.** BT.601 fixtures: worst delta 1/255 signalled, 39/255 mis-tagged. |
| Does export use the interop decode path? | **No, deliberately** — throughput-bound with a 16-frame host lookahead vs latency-bound with per-frame GPU residency. `bench --export`: 5/5 classes NVENC, 4K60 at 83.6 FPS, every output verified. |
| Is P2.6 (CPU readback on the export path) open? | **No.** Closed as a decision, with the note in `src/export/readback.rs`. |
| Should `-bf 0` fixtures replace the real-media set? | **No.** They are the diagnostic that identified gotcha 24 and nothing more: the re-encode turned `cam_4k60_grain` into 82 I-frames of 120 and the row into 19.9 FPS. A fix has to come from the pipeline. |
| Are the four `MULTI_4K60_SOURCES` comparable decodes? | **No, and the heading now says so.** `cam_4k60_grain` carries ~35× its neighbours' bitrate. Gotcha 25. |
| Real footage for Phase F? | **No, and the generated classes have earned it** — `grain_1080p60` contradicted the interop win and was explained from counters; the 4K grain source did it again and was explained again. If a real-footage regression appears, add a class rather than replacing the set. |
| Is benchmark 1's spread worth chasing? | **No.** ~886 FPS median at 1080p render-only; it clears its target by 14×. |
| Can benchmark 5 be the 4K60 gate? | **No — and it now PASSES the restated budget, which is exactly why not.** It uploads `make_nv12` bars from host memory with no decoder in the process. Benchmark 7 is the gate. |
| Is the pipelined four-source real-media frame graph-bound? | **No, and this is now measured three ways.** 7.378 ms of graph in a ~36 ms frame (benchmark 7's own per-node pass), the graph is FLAT across the alternation while `decode` swings 8.3 → 40.5 ms, and 6.66 of 7.18 ms per decode is inside `send_packet`/`receive_frame`. |
| Was the last stale frame the drain? | **No.** It was `Decoder::seek_to` discarding the frame it lands on — 59 of 60 frames were the wrong picture. Gotcha 23. |
