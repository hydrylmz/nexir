# Nexir 4K60 Performance Audit — Result

> Ninth pass, 2026-09-05. **Task I and Phase H are DONE.** This file is now a
> RESULT rather than a work list: the audit's remaining question has been answered,
> and the answer is that the 4K60 gate as stated is unreachable on this host for the
> workload it was stated over. What follows is that reading, what the gate IS met at,
> and the two Phase H items with their measured outcomes.
>
> Everything earlier — Phases A/B/C, I0–I3, P2.1, G1, D2, D3, E1's API, F1b, E1b, F2,
> all of G2 (a–e), G3's engine fix, G2f.1–G2f.4, the fifth pass's Task 0 and Task 1,
> the sixth pass's Tasks A–F, Task G, and Phase H's P2.4 — is done and removed. Their
> outcomes survive as AGENTS.md gotchas 9/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27
> plus **28/29/30 (pending: see "AGENTS.md is not yet updated" below)**, in the tests
> those gotchas name, and in the `target/*.txt` files each one cites.

---

## THE RESULT, in one table

**The gate, restated:** steady-state frame interval ≤ 16.67 ms at P95 and ≤ 20 ms at
P99 **on benchmark 7** (four distinct 4K60 files, Heavy graph, interop decode,
zero-copy NVENC), median of ≥3 repeats, read off `↳ steady`.

| what | measured | source |
|---|---:|---|
| **NVDEC floor, 4 × 4K60, pictures DISCARDED** | **32.03 ms** | `target/taskI_floor_4k_3x.txt` |
| the 60 Hz frame budget | 16.67 ms | definition |
| benchmark 7's whole frame, after P2.3 | 33.54 ms mean, P95 82.68 ms | `target/p23_fused_3x.txt` |

**The decode alone needs 1.92× the entire frame budget with nothing downstream of it
running.** No pooling, fusion, scheduling, upload or ownership change inside this crate
can close that, because none of them is in that measurement. The gate is not missed by
a margin this crate can work on; it is missed by the hardware's own throughput at that
source count.

### What the gate IS met at, measured the same way

| workload | floor | vs 16.67 ms | verdict |
|---|---:|---:|---|
| **2 × 4K60** | **9.78 ms** | 0.59× | **60 Hz, with 41% of the frame left** |
| 3 × 4K60 | 26.98 ms | 1.62× | 30 Hz only |
| 4 × 4K60 | 32.03 ms | 1.92× | 30 Hz only (33.33 ms, 4% spare) |
| **3 × 1080p60** | **7.53 ms** | 0.45× | **60 Hz, with 55% left** (`target/taskI_floor_1080p_3x.txt`) |

Recommended restatement of the target, if it is restated: **4K60 preview at two
concurrent sources**, or **four sources at 4K30**, or **1080p60 at three or more** —
each of which is a workload whose decode fits the budget with room for the graph.
Nothing above is extrapolated: the 1080p row was run on 1080p fixtures, and the 30 Hz
column is arithmetic on the same measurement (a 30 Hz frame *is* 33.33 ms).

---

## Task I — how the floor was measured, and why each choice was forced

`examples/b_decode_split_probe.rs --floor`. Eight arms per repeat per source count,
three repeats, `NEXIR_FLOOR_REPEATS` to override.

1. **The picture is DISCARDED.** `Decoder::receive_and_discard` takes the frame out of
   the decoder, reads its pts and unrefs it where NVDEC left it. No
   `av_hwframe_transfer_data`, no `cuCtxSynchronize`, no `cuMemcpy2DAsync` pair, no
   stream sync, no `dst`. Every other arm in that file ends in a copy, so its figure
   answers "what does a frame cost through this pipeline" — a different and always
   larger number. **This arm is not a decode path and cannot be quoted as one**; a
   pipeline cannot reach it by construction. It bounds what any change could ever buy.
2. **No engine mutex, and this is the one place that is right.** `run_concurrent_arm`
   (Task G) holds one lock across every `decode_into` because the copies share one
   `CUcontext` and one stream and the barrier is context-wide (gotchas 4, 19). With the
   copies gone there is no barrier to serialise, so whatever serialisation remains is
   the driver's own — which is exactly what belongs in a floor.
3. **The floor is the better of a CONCURRENT and a SERIAL arm, both printed.** "One
   thread per source" is a scheduling choice; if its contention made it slower than
   decoding in turn, calling it the floor would attribute this probe's own threading to
   the hardware. Measured they agree to 0.1–0.8%; the 4-source floor came off the
   serial arm (32.03 vs 32.07 ms).
4. **The saturation reading is `observed / additive`, never `n × floor(1 source)`.**
   `MULTI_4K60_SOURCES` is one very heavy source plus three ordinary ones by
   construction (gotcha 25 — `cam_4k60_grain` carries ~35× its neighbours' bitrate), so
   a ratio against the first fixture would charge *adding a source* for that one file
   being dear. Each source is therefore measured ALONE and the prediction is the sum of
   the subset's own solo floors:

   | source | solo floor |
   |---|---:|
   | cam_4k60 | 5.16 ms |
   | cam_4k60_rot | 4.90 ms |
   | **cam_4k60_grain** | **17.08 ms** |
   | cam_4k60_flip | 5.13 ms |

   Observed / additive is **0.97–0.99× at every source count**. The engine served them
   essentially in turn: **it is saturated**, which is Task I's step 2 answered as a
   reading rather than an assertion.
5. **Geometry, frame rate and decoder are read off the first fixture and printed.**
   A 1080p run cannot print a "4K60" verdict, and a host that attached software decode
   skips with a reason rather than reporting libavcodec's CPU throughput as an NVDEC
   ceiling.
6. **Repeats are the OUTER loop, source count the inner one.** Drift over a
   multi-minute sweep would otherwise land entirely on whichever count ran last, and
   the whole conclusion is a comparison *between* counts.

---

## Phase H — both items done

### P2.3 Effect fusion — DONE, −29 to −37% of the graph

`FusedGradeNode` (`src/render/nodes/fused_grade.rs`) + `fused_grade.wgsl` run colour
correction, the 3D LUT and the chroma key in one compute pass. The three passes each
read and write a full canvas-sized `Rgba16Float` texture, so the chain crosses the
canvas 6 times where the fused pass crosses it 2. `NEXIR_FUSE_GRADE=0` takes the
unfused arm from the same binary — one argument different, the discipline
`NEXIR_UPLOAD_PATH` already follows.

| row | graph TOTAL | FPS | peak bucket |
|---|---:|---:|---:|
| 5 — synthetic 4K60 | 6.783 → **4.814 ms** (−29%) | 57.5 → **67.1** | 16/32 → **8/32** |
| 7 — real media, interop | 7.387 → **4.682 ms** (−37%) | 27.8 → **28.6** | 16/32 → **8/32** |
| 8 — real media, CPU upload | (same graph) | 18.9 → **21.9** | 16/32 → **8/32** |
| 7 — `nexir_media_bars` control | 5.881 → **3.867 ms** (−34%) | 45.0 → **47.6** | — |

Medians of 3: `target/p23_unfused_3x.txt`, `target/p23_fused_3x.txt`,
`target/p23_bars_unfused_3x.txt`, `target/p23_bars_fused_3x.txt`.

- **Measured on BOTH content sets**, because a bandwidth win is exactly what
  low-entropy fixtures would flatter (gotcha 27). 34% on the compressible bars, 37% on
  the repo fixtures — it survives real content, which was the test that mattered.
- **It does not close the gate and is not claimed to.** Benchmark 7 still prints
  `ALTERNATING` (12.93 / 52.36 ms), which fails the target on its own (gotcha 15), and
  the decode floor is 32 ms. The graph was ~20% of the frame; it is now ~13%.
- **`peak bucket` re-read as gotcha 14 requires**: 16/32 → 8/32, 0 evicted.
  `POOL_BUCKET_CAPACITY` stays 32 — 16 is still a shape the tree builds
  (`NEXIR_FUSE_GRADE=0`, and `EffectChainBuilder` for a clip whose effect list is not
  the fusable triple), and an unreached bucket costs nothing.
- **Verified equivalent, not assumed.** `src/tests/fused_grade.rs` renders the same
  input through both paths in one process: non-identity grade in every field, a
  channel-rotating LUT, both sides of the chroma key's early-exit branch, compared
  within f16 quantisation (the chain rounds twice; the fused pass keeps f32 in
  registers, so equality would fail for the right reason and the wrong cause).
  `a_dropped_stage_would_exceed_the_tolerance` removes each stage in turn and measures
  the margin — 6–50× the tolerance — so the equality check can actually fail.

### P2.2 Texture lifetimes — MEASURED, and NOT worth writing

`CompiledGraph::lifetime_bounds()` computes the prize off the compiled graph's own
declarations instead of estimating it: `(held, ideal)` where `held` is what the frame
holds today and `ideal` is the maximum simultaneously *live*, i.e. the floor a perfect
aliasing scheme could reach. The bench prints both beside the pool stats.

**Benchmark 5 after P2.3: holds 18, ideal 15 — 3 fewer, ~190 MB of 4K residency, and
no frame time.** Three constraints keep it small, each in the analysis rather than
assumed:

- **A node binds its input and its output in one pass**, so a live range is
  `[first write … last read]` INCLUSIVE and consecutive stages can never share. A
  scheme treating a resource as dead at its last reader would build bind groups wgpu
  rejects outright. The floor for a chain of any length is 2, not 1.
- **The pool buckets by (format, USAGE, size)**, so resources with different usage
  flags cannot alias however their lifetimes fall.
- **Imports are excluded** (gotcha 18) — the decoder owns that texture beyond the
  frame, so counting it would inflate the prize with memory nothing can reclaim.

**Verdict: do not write it.** The pool already reuses textures across frames at a
0.5–0.8% miss rate with 0 evictions, so aliasing *within* a frame removes residency and
shortens nothing in the recording. Three textures of eighteen, against per-resource
lifetime tracking through `resolve_resources` and all three `execute*` bodies, is not a
trade the plan's own constraint ("do not sacrifice maintainability for theoretical
savings") permits. The measurement stays — `render::graph::tests::lifetime_bounds_*`
pins both shapes with their arithmetic, so the number can be re-read if the graph
changes shape again.

**P2.5 (NVENC, 0.23–0.34 ms, under 1% of the frame) remains untouched and should stay
that way.**

---

## AGENTS.md is not yet updated

Gotchas **28** (the NVDEC floor and what the gate is met at), **29** (the fusion and its
four rules) and **30** (P2.2's ceiling and why it was declined) are written but the
write to `AGENTS.md` was blocked pending approval. Everything they would say is above,
and the code carries it in doc comments:

- `examples/b_decode_split_probe.rs` — `run_floor_arm`, `run_serial_floor_arm`,
  `print_task_i_verdict`, `Source::one_frame_engine_floor`
- `src/io/decoder.rs` — `Decoder::receive_and_discard`
- `src/render/nodes/fused_grade.rs` — module comment and `FusedGradeParams`
- `src/render/shader/fused_grade.wgsl` — what it saves and the one behavioural difference
- `src/render/graph.rs` — `CompiledGraph::lifetime_bounds`
- `src/render/resource.rs` — `POOL_BUCKET_CAPACITY`'s re-read after P2.3
- `src/bin/bench.rs` — `fuse_grade`, `add_grade_chain`, `print_lifetime_bounds`

---

## Non-negotiable rules that still constrain any further work

- Every printed metric is **measured, counted, or `n/a`** — never estimated.
  `profiling::tests::unmeasured_system_metrics_print_as_not_available` pins this.
- A missing GPU/CUDA/NVENC/NVML capability is a **printed skip with a reason**, never a
  zero row.
- Adding a CUDA function to `src/interop/ffi/cuda_*.rs` means adding a line to
  `build/cuda.def`; a hard-coded offset/size/enumerant in any FFI module means a probe
  in `nvchk/`.
- **An optimisation only counts if the improvement comes from the pipeline.** `-bf 0`
  fixtures and low-entropy fixtures are both diagnostics, not fixes (gotchas 24, 27).
- **A per-node figure is only comparable across rows carrying the same content**, and
  `×/fr × per pass` is only valid on a row printing no `UNEVEN` (gotcha 27).
- **Quote the median of ≥3 repeats with its spread, never one run.**
- **Read P95/P99 off `↳ steady` and the mean off the whole-run row** (gotcha 15).
- **`cargo test -p nexir --test '*'` does not work** — use `cargo test` or
  `cargo test -p nexir --lib`.
- **A benchmark pass must not run inside the session's timed span.**
- **Every commit must build on its own** (`git checkout <sha> && cargo check --all-targets`).
- **`cuda.dll` must sit beside the binary that loads it**, including
  `target/debug/deps/` and `target/release/examples/`.
- **`NEXIR_MEDIA_DIR` only isolates a fixture set whose files clear the 64 KB floor.**
- **Do not compare a serial `--interop` row with a pipelined synthetic row**, and 7-vs-8
  is the decode-path comparison, not 5-vs-7.
- **A `FrameSlotPool` per probe OOMs the test binary** (~386 MB each).
- **A claim about one source's decode cost must come from a single-source pass**
  (gotcha 25).
- **On this shell a fixture path passed to a native binary must be `C:/...`** — an MSYS
  `/c/...` argument arrives as a file that does not exist. The floor probe now prints
  `n/a` and says so instead of `0.0 MB`.

---

## Settled — do not re-litigate

| Question | Answer |
|---|---|
| **Can four 4K60 sources be decoded inside 16.67 ms on this host?** | **No — 32.03 ms with the pictures discarded, 1.92× over.** Gotcha 28 / `target/taskI_floor_4k_3x.txt`. |
| **Is the NVDEC engine saturated at four sources?** | **Yes, measured: 0.99× of the additive prediction** built from each source's own solo floor. No scheduling change in this crate can overlap them. |
| **What IS the gate met at?** | **Two 4K60 sources (9.78 ms), or four at 30 Hz (32.03 of 33.33 ms), or three 1080p60 (7.53 ms).** Measured the same way, on fixtures at that geometry. |
| Is the decode floor a figure a pipeline can reach? | **No, by construction** — the picture is discarded. It bounds what a change could buy. |
| Should the concurrent floor arm hold the engine mutex the Task G arms hold? | **No, and only here.** With no copy there is no context-wide barrier to serialise; a lock would be timing a lock. |
| Was fusing colour correction + LUT + chroma key worth it? | **Yes: −29 to −37% of the graph on both content sets, `peak bucket` 16/32 → 8/32.** Gotcha 29. |
| Does the fused pass match the three-node chain? | **Yes, within f16 quantisation** — and the fused path is the more precise of the two. `tests::fused_grade`, with a per-stage control. |
| Did fusion close the 4K60 gate? | **No, and it was never going to.** The graph was ~20% of the frame and is now ~13%; benchmark 7 still prints `ALTERNATING`. |
| Should `POOL_BUCKET_CAPACITY` drop to fit the fused graph's 8? | **No.** 16 is still a shape the tree builds, and an unreached bucket costs nothing (`1/32` at 1080p). |
| Is P2.2 (texture lifetimes) worth writing? | **No — measured first.** 3 textures of 18 on benchmark 5, ~190 MB of residency and zero frame time, against lifetime tracking in `resolve_resources` and three `execute*` bodies. Gotcha 30. |
| Would aliasing collapse a chain to one texture? | **No, to two** — a node binds its input and output together. A scheme that thought otherwise would fail wgpu validation on its first frame. |
| Is P2.5 (NVENC, <1% of the frame) worth opening? | **No.** |
| Does the RTX 3050 report `TIMESTAMP_QUERY`? | **Yes.** |
| Is a multi-slot staging ring needed? | **No.** `write_buffer` stages into wgpu's own buffer and the copy is GPU-ordered. |
| Is `write_texture` faster than `write_buffer` + copy? | **Depends on row alignment; both are kept.** `UploadPath::Auto` picks per node. |
| Was the 4K P99 spike a stall inside `write_buffer`? | **No, three times over.** Refuted on 5 and on 7; the real defect was the pool's bucket cap (gotcha 14). |
| Is the read-after-write wait the bottleneck? | **No**, at one source and at four. 376 of 384 decodes poll at 0.005–0.008 ms. |
| Should a second interop target per source be started? | **No — measured and refuted.** Gotcha 26. |
| Are per-source CUDA streams the fix? | **No, not on this reading.** The barrier is 0.305–0.313 ms per copy and does not grow across sources. |
| Is benchmark 7's alternation the pipeline, the upload, the graph, or the target-map lock? | **None of them** — the decoder's reorder buffer, four controls. Gotcha 24. |
| Is `Composite` dearer on benchmark 7 because its inputs are IMPORTED? | **No, and P2.4 is closed.** It is content. Gotcha 27. |
| Is the graph's cost a property of the code alone? | **No — it tracks source entropy** through GPU framebuffer compression. Gotcha 27. |
| Does the interop import path make the shaders dearer? | **No: 6.802 − 0.935 = 5.867 predicted, 5.881 measured, +0.2%.** |
| Must FFmpeg and this crate share one `CUcontext`? | **Yes**, and the barrier is meaningless without it. Gotcha 19. |
| Is `cuCtxPushCurrent` usable for this? | **No.** Gotcha 4. |
| Can the transient pool hold a texture the decoder owns? | **No**, and the filter is in one place. Gotcha 18. |
| Can a 10-bit source use the interop path? | **No** — the planes are 8-bit; refused before allocating. |
| Does colour metadata survive the upload bypass? | **Yes, verified where gotcha 11 says to verify it.** |
| Does export use the interop decode path? | **No, deliberately** — throughput-bound vs latency-bound. |
| Is P2.6 (CPU readback on the export path) open? | **No.** Closed as a decision. |
| Should `-bf 0` or low-entropy fixtures replace the real-media set? | **No** to both. They are diagnostics and controls. |
| Are the four `MULTI_4K60_SOURCES` comparable decodes? | **No** — and gotcha 28's solo column now measures exactly how far apart they are (4.90 vs 17.08 ms). |
| Real footage for Phase F? | **No** — the generated classes have earned their place. |
| Can benchmark 5 be the 4K60 gate? | **No.** No decoder in the process. |
| Is the pipelined four-source real-media frame graph-bound? | **No, now five ways** — and gotcha 28 is the fifth: the decode alone exceeds the whole budget. |
| Was the last stale frame the drain? | **No** — `Decoder::seek_to` discarding the frame it lands on. Gotcha 23. |
