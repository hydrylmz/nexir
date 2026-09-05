# Nexir 4K60 Performance Audit — Remaining Work

> Trimmed 2026-09-05 (eighth pass). **Everything in this file is unfinished.** Phases A/B/C, I0–I3, P2.1, G1, D2, D3, E1's API, F1b, E1b, F2, all of G2 (a–e), G3's engine fix, G2f.1–G2f.4, the fifth pass's Task 0 and Task 1, the sixth pass's Tasks A–F, **Task G, and Phase H's P2.4** are done and have been removed. Their outcomes survive as AGENTS.md gotchas 9/12/13/14/15/16/17/18/19/20/21/22/23/24/25/**26/27**, in the tests those gotchas name, and in `target/bench_P21_*.txt`, `target/bench_F1b.txt`, `target/bench_E1b.txt`, `target/bench_F2.txt`, `target/g2a_probe_3x.txt`, `target/bench_G2e_interop_3x.txt`, `target/bench_G2f_interop_3x.txt`, `target/bench_G2f_pipelined_3x.txt`, `target/bench_taskB_5_7_8_3x.txt`, `target/taskB_7_1.csv`, `target/taskB_bf0_7_1.csv`, `target/taskB_decode_split_queued.txt`, `target/taskC_single_source.txt`, `target/taskF_b7_nodes.txt`, `target/taskG_pingpong_3x.txt`, `target/taskH_uneven_3x.txt`, `target/taskH_cheapdecode_3x.txt`, `target/taskH_mixed_3x.txt`, `target/taskH_noise_3x.txt`, `target/clk_bars.csv`, `target/clk_noise.csv`. G2's written re-costing is `.hermes/plans/2026-08-31_g2-recosting.md`.

## Where things stand, because the tasks below are judged against it

**The gate:** steady-state frame interval ≤ 16.67 ms at P95 and ≤ 20 ms at P99 **on benchmark 7**, median of ≥3 repeats, read off `↳ steady`. Average FPS is throughput, not the gate.

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
| graph (GPU) | 7.37 ms | `target/taskH_uneven_3x.txt` — read off the per-FRAME column; the per-pass column is pooled and `UNEVEN` on four of six nodes (gotcha 27) |
| composite recording (CPU) | 0.69–0.85 ms | `composite_ms`, flat across the alternation |
| NVENC | 0.23–0.34 ms | `nvenc_ms` |

CPU and GPU overlap once frames are in flight, so those do not sum to the frame. **The decode is the frame; a free graph still leaves ~29 ms.**

**Benchmark 7 still prints `ALTERNATING` (16.1 / 52.0 ms), and per gotcha 15 that fails the target on its own.** The cause is the decoder's reorder buffer (gotcha 24), it is outside this crate's render path, and the one fix that does not change the workload has been measured and **refuted** (gotcha 26 — two targets per source keeps the cycle, does not improve the mean, and doubles the VRAM).

---

## Non-negotiable rules that constrain every task here (from `AGENTS.md`)

- Every printed metric is **measured, counted, or `n/a`** — never estimated. `profiling::tests::unmeasured_system_metrics_print_as_not_available` pins this; do not weaken it.
- A missing GPU/CUDA/NVENC/NVML capability is a **printed skip with a reason**, never a zero row.
- Adding a CUDA function to `src/interop/ffi/cuda_*.rs` means adding a line to `build/cuda.def`; adding a hard-coded offset/size/enumerant to any FFI module means adding it to a probe in `nvchk/`.
- **An optimisation only counts if the improvement comes from the pipeline, not from the benchmark doing less work.** `-bf 0` fixtures are the standing example of a diagnostic that is not a fix (gotcha 24); **low-entropy fixtures are the second** (gotcha 27 — they make the graph 20% cheaper without touching a line of code).
- **A per-node figure is only comparable across rows carrying the same content**, and `×/fr × per pass` is only valid on a row printing no `UNEVEN` (gotcha 27).
- **Quote the median of ≥3 repeats with its spread, never one run.**
- **Read P95/P99 off the `↳ steady` row and the mean off the whole-run row** (gotcha 15). On a 90-frame run the whole-run P99 *is* the pipeline fill.
- **`cargo test -p nexir --test '*'` does not work** — there is no `tests/` directory. Use `cargo test` or `cargo test -p nexir --lib`.
- **A benchmark pass must not run inside the session's timed span.** `generate_report` divides frames by `ProfilingSession`'s own elapsed time.
- **Every commit must build on its own.** Check with `git checkout <sha> && cargo check --all-targets`.
- **`cuda.dll` must sit beside the binary that loads it**, including test binaries (`cp target/debug/*.dll target/debug/deps/`) and example binaries (`target/release/examples/`) — without it every interop test *skips with a printed reason*, which looks like a green suite that exercised nothing.
- **`NEXIR_MEDIA_DIR` only isolates a fixture set whose files clear `ensure_fixture`'s 64 KB floor** (gotcha 27).
- **Do not compare a serial `--interop` row with a pipelined synthetic row**, and do not compare benchmark 5 with 7 as if the difference were the decode path. 7-vs-8 is the decode-path comparison.
- **A test that allocates a `FrameSlotPool` per probe OOMs the test binary** (~386 MB each). Gotcha 23's last bullet has the workaround.
- **A claim about one source's decode cost must come from a single-source pass** (gotcha 25).

---

## Order of work

```text
I   can four 4K60 sources share one NVDEC engine inside 16.67 ms at all?   ← the only
                                                                            remaining
                                                                            question
                                                                            about the
                                                                            GATE
H   Phase H — P2.2 lifetimes, P2.3 fusion                                 ← ~20% of
                                                                            the frame,
                                                                            cannot
                                                                            close it
```

---

# Task I — the gate's remaining question, and it may be answerable "no"

**Every lead inside this crate's render path is now closed, and the decode is 30.9 ms of a 36 ms frame.** Gotcha 24 attributes the alternation to the decoder's reorder buffer; gotcha 26 refutes the only fix that does not change the workload; gotcha 25 establishes that four sources share one NVDEC engine and the per-source split reports who waited rather than who costs. What has never been measured is the ceiling itself.

**The question, stated so it can be answered:** what is the minimum wall time in which this host's single NVDEC engine can deliver four decoded 4K60 frames, with no graph, no NVENC, no pipeline and no upload — and is it under 16.67 ms?

1. **Measure the engine, not the pipeline.** `examples/b_decode_split_probe.rs` already opens four sources and decodes into `DecodeInteropTarget`s with no graph; its concurrent arm already exists (gotcha 26). Add a pass that reports the frame-set wall interval as a **floor** — the sum of what NVDEC spent, and the wall time it took — over ≥3 repeats. Report it against 16.67 ms directly.
2. **Then vary only the source count**, 1 → 2 → 3 → 4, on the same four fixtures. If the floor scales linearly with sources, the engine is saturated and no scheduling change in this crate can help. If it scales sub-linearly, there is headroom and the next question is where the pipeline is failing to claim it.
3. **The honest outcome may be that the budget does not exist.** If four 4K60 sources cannot be decoded in 16.67 ms on one NVDEC engine, say so in the audit and state what the gate *can* be met at — two sources, or 4K30, or 1080p60 — rather than moving the gate silently. That is a result, not a failure.

Acceptance:

- A floor figure with its spread, median of ≥3, quoted against 16.67 ms.
- The source-count sweep, so "saturated" is a reading rather than an assertion.
- If the answer is no: the audit says which workload the gate IS met at, measured the same way.

**What this task must not do:** re-open the upload lead (gotcha 24, refuted twice), re-open double-buffered targets (gotcha 26), or quote a figure from `-bf 0` or low-entropy fixtures as an improvement.

---

# Task H — Phase H: pooling and fusion (P2.2, P2.3)

**P2.4 is closed** (gotcha 27: `Composite`'s rise is source entropy, not imported views — benchmark 8 reports 1.330 ms against 7's 1.335, and both fall to 1.082 on equal content, which is benchmark 5's 1.074 to 0.7%). **P2.5 is explicitly last** at 0.23–0.34 ms, under 1% of the frame; do not start there.

Per-instance medians, benchmark 7, median of 3 (`target/taskH_uneven_3x.txt`), execution order = `MULTI_4K60_SOURCES` order:

| Node | cam_4k60 | cam_4k60_rot | **cam_4k60_grain** | cam_4k60_flip | measured `per frame` |
|---|---:|---:|---:|---:|---:|
| ColorCorrection | 0.285 | 0.284 | **0.627** | 0.286 | 1.481 |
| Lut3D | 0.313 | 0.311 | **0.614** | 0.312 | 1.553 |
| ChromaKey | 0.307 | 0.307 | **0.529** | 0.310 | 1.455 |
| YuvToRgb | 0.274 | 0.271 | **0.400** | 0.273 | 1.220 |
| Composite (×1) | — | — | — | — | 1.335 |
| ToneMap (×1) | — | — | — | — | 0.322 |
| **GRAPH TOTAL** | | | | | **7.366 ms** |

**Phase H is ~20% of a ~36 ms frame and cannot close the gate.** Even a free graph leaves ~29 ms. Do it for the reasons below or not at all; the plan's constraint stands — *"do not sacrifice maintainability for theoretical savings."*

| Item | What is known | First thing to look at |
|---|---|---|
| **P2.3 Effect fusion** — the stronger of the two | On the CHEAP layers CC/LUT/ChromaKey are 0.285/0.313/0.307 ms, within 10% of each other, none an outlier. On the grain layer they are 0.627/0.614/0.529 — **all three roughly double together**, which is what a bandwidth-bound chain does. Upper bound ~4.5 ms/frame, ~12% of a 36 ms frame | Four passes each read and write a full 4K RGBA16F texture (63.3 MB). Fusion saves bandwidth, not maths, and gotcha 27's entropy finding is direct evidence that bandwidth is what these passes spend. **Measure any fusion against BOTH content sets** (`target/…/nexir_media_bars` is the low-entropy control, kept for this) — a win that only appears on compressible content will not survive real footage |
| P2.2 Texture lifetimes | 4K VRAM pressure is real (4.3–5.6 GB of 8 GB on the NVML row); each interop source adds 11.9 MB, 47.5 MB at four. Task G is closed, so it will not double. Buys VRAM, not frame time | The graph holds every declared resource for the whole frame (`resolve_resources` + the three `execute*` bodies); non-overlapping lifetimes could share. **`peak bucket 16/32` is the number this reduces**, and gotcha 14 means the cap must be re-read from a measurement afterwards |

---

# Settled — do not re-litigate

| Question | Answer |
|---|---|
| Does the RTX 3050 report `TIMESTAMP_QUERY`? | **Yes**, `available` in the bench banner. |
| Is a multi-slot staging ring needed? | **No.** `write_buffer` stages into wgpu's own buffer and the copy is GPU-ordered after the previous submission, so the hazard does not exist. |
| Is `write_texture` faster than `write_buffer` + copy? | **Depends on row alignment; both are kept.** Wins 1080p by 25–40% (1920 pads to 2048), loses 4K by ~7% (3840 is aligned). `UploadPath::Auto` picks per node; force either with `NEXIR_UPLOAD_PATH`. |
| Was the 4K P99 spike a stall inside `write_buffer`? | **No, three times over.** `NEXIR_FRAME_DUMP` refuted it on benchmark 5 (`0 in BOTH`, `r = +0.04`) and again on benchmark 7 (`0 in BOTH`, `r = -0.095`); the "54.6 ms P99" was the pipeline fill landing on the P99 index. The real defect was the pool's bucket cap — gotcha 14. |
| Can the upload copy be bracketed directly? | **No.** `write_buffer`'s copies live in wgpu's own `pending_writes` command buffer. Measured as the gap between consecutive frames' GPU timestamps instead. |
| Is `nvml.dll` present, and is D3 done? | **Yes and yes.** |
| Is NVDEC selected on this machine? | **Yes** — `[decoder] Attached hardware decoder: Cuda`. |
| Is CUDA interop available on this machine? | **Yes** — `transport=D3D12Win32Handle, ordinal 0, driver 12.6`. |
| Is G2 worth the ownership refactor? | **Yes, measured rather than estimated.** Serial 4K60 single-source 9.10 → 5.38 ms; pipelined four-source real media 18.6 → 26.5 FPS (1.43×), counted upload at 0. |
| Does the per-frame `cuStreamSynchronize` cost more than the CPU round-trip it replaces? | **No.** G2a: interop 13.93–14.12 ms vs CPU 21.39–22.45 ms, and it is the *stable* pass (1.8% spread vs 17%). |
| Is the read-after-write wait the bottleneck? | **No, and this is settled at FOUR sources as well as one.** 376 of 384 decodes poll at 0.005–0.008 ms. |
| Should G4 / a second interop target per source be started? | **No — measured and refuted.** Gotcha 26: the cycle survives (352% → 415% parity), the mean does not improve (32.11 → 32.40 ms), the VRAM doubles (47.5 → 94.9 MB). The ownership change is not started. |
| Are per-source CUDA streams the fix? | **No, not on this reading.** The barrier averages 0.305–0.313 ms per copy at four sources and does not grow across them. |
| Why is `grain_1080p60` slower on the interop path? | **NVDEC, not our copy — at ONE source and 1080p, which is the whole scope.** Gotcha 20's last bullet, narrowed by gotcha 25. |
| Is benchmark 7's alternation the pipeline, the upload, the graph, or the target-map lock? | **None of them.** It is the decoder's reorder buffer — gotcha 24, four controls. Do not re-open the upload lead. |
| Is `Composite` dearer on benchmark 7 because its inputs are IMPORTED? | **No, and P2.4 is closed.** Benchmark 8 (pooled, CPU upload) reports 1.330 ms against 7's 1.335, and on equal content both fall to 1.082 — benchmark 5's figure to 0.7%. Gotcha 27. |
| Can a per-node `per pass` figure be multiplied by `×/fr`? | **Only when the row prints no `UNEVEN`.** The column is a median pooled over that node's instances, and benchmark 7's four layers differ ~2× — the product came out 22-30% under the measured per-frame total. Gotcha 27. |
| Is the graph's cost a property of the code alone? | **No — it tracks source entropy.** Same shaders and clocks, graph TOTAL 5.881 → 7.366 → 11.366 ms across three content sets while benchmark 5 held at 6.79-6.80. Implied per-pass bandwidth crosses the 224 GB/s bus rate, so it is framebuffer compression. Gotcha 27. |
| Does the interop import path make the shaders dearer? | **No, and this is now exact.** With content held equal the interop graph is benchmark 5's graph minus `YuvUpload`: 6.802 − 0.935 = 5.867 predicted, 5.881 measured, +0.2%. Gotcha 27. |
| Does `POOL_BUCKET_CAPACITY` move on the interop path? | **No, and it has been read from every shape.** `1/32` single-source, `4/32` serial four-source, `16/32` pipelined four-source with 0 evictions, and `16/32` again on every content set. |
| Must FFmpeg and this crate share one `CUcontext`? | **Yes, and the barrier is meaningless without it.** Gotcha 19. |
| Is `cuCtxPushCurrent` usable for this? | **No.** Push needs the context floating, and FFmpeg holds it from the first `Decoder::open`. Gotcha 4. |
| Can the transient pool hold a texture the decoder owns? | **No, and the filter is in one place.** `GraphResource::into_transient` — gotcha 18. |
| Can a 10-bit source use the interop path? | **No.** The planes are `R8Unorm`/`Rg8Unorm`; a 10-bit source decodes to P010 at two bytes per sample. Refused before allocating. |
| Does colour metadata survive the upload bypass? | **Yes, verified where gotcha 11 says to verify it.** BT.601 fixtures: worst delta 1/255 signalled, 39/255 mis-tagged. |
| Does export use the interop decode path? | **No, deliberately** — throughput-bound with a 16-frame host lookahead vs latency-bound with per-frame GPU residency. `bench --export`: 5/5 classes NVENC, 4K60 at 83.6 FPS, every output verified. |
| Is P2.6 (CPU readback on the export path) open? | **No.** Closed as a decision, with the note in `src/export/readback.rs`. |
| Should `-bf 0` fixtures replace the real-media set? | **No.** They are the diagnostic that identified gotcha 24 and nothing more. A fix has to come from the pipeline. |
| Should low-entropy fixtures replace the real-media set? | **No, and for the same reason.** They make the graph 20% cheaper without touching a line of code (gotcha 27). They are the CONTROL that closed P2.4, and `nexir_media_bars` is kept for P2.3's two-content-set rule. |
| Are the four `MULTI_4K60_SOURCES` comparable decodes? | **No, and the heading now says so.** `cam_4k60_grain` carries ~35× its neighbours' bitrate, and gotcha 27 adds that this makes it ~2× dearer in the GRAPH as well. |
| Real footage for Phase F? | **No, and the generated classes have earned it** — `grain_1080p60` contradicted the interop win and was explained from counters; the 4K grain source did it twice. If a real-footage regression appears, add a class rather than replacing the set. |
| Is benchmark 1's spread worth chasing? | **No.** ~886 FPS median at 1080p render-only; it clears its target by 14×. |
| Can benchmark 5 be the 4K60 gate? | **No — and it now PASSES the restated budget, which is exactly why not.** It uploads `make_nv12` bars from host memory with no decoder in the process. Benchmark 7 is the gate. |
| Is the pipelined four-source real-media frame graph-bound? | **No, and this is now measured four ways.** 7.37 ms of graph in a ~36 ms frame, the graph is FLAT across the alternation while `decode` swings 8.3 → 40.5 ms, 6.66 of 7.18 ms per decode is inside `send_packet`/`receive_frame`, and making the graph 20% cheaper with low-entropy content leaves the row at 45 FPS — still short of 60. |
| Was the last stale frame the drain? | **No.** It was `Decoder::seek_to` discarding the frame it lands on — 59 of 60 frames were the wrong picture. Gotcha 23. |
