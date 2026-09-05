# Task G2 — the written re-costing, before any code

The plan requires this estimate to exist *before* the ownership refactor starts, and
to come from the **`GPU transfer` row**, not from G1's decode-call delta. Every
figure below is quoted from a run in `target/`, with its file named.

## What G1's number is, and why it is the wrong number

`target/g2_recost_probe.txt` (re-run today, `examples/g1_decode_probe.rs`, median of
60 frames per file):

| fixture | CPU round-trip | GPU interop | delta |
|---|---:|---:|---:|
| cam_1080p60 | 1.43 ms | 1.44 ms | **−0.01** |
| motion_1080p60 | 1.56 | 1.35 | 0.21 |
| grain_1080p60 | 5.21 | 4.88 | 0.33 |
| cam_4k30 | 5.20 | 5.05 | 0.15 |
| cam_4k60 | 5.06 | 4.95 | 0.11 |

Today's 4K delta is **0.11 ms**, at the bottom of G1's original 0.4–1.1 ms range and
smaller than the run-to-run spread of the decode figure itself. Taken alone this says
G2 is worthless. That reading is what the plan warns against: the probe swaps the
`None` argument but the frame still ends up in a CPU buffer being uploaded afterwards,
so it measures only the `av_hwframe_transfer_data` half. **The estimate has to come
from the transfer the ownership change deletes.**

## The number that decides it: the `GPU transfer` row

Benchmark 5, 4K60, 4 layers over 4 distinct sources, median of 3
(`target/bench_P24_fifo.txt`):

| | measured |
|---|---:|
| Frame interval (whole-run mean) | **17.85 ms** |
| `GPU transfer` (GPU avg) | **10.00 ms** |
| Graph execution (Composite bracket, GPU avg) | 7.82 ms |
| Bytes handed to the queue | 47.5 MB/frame |
| Implied transfer rate | 4.9 GB/s |
| CPU share of a frame | 5.82 ms (idle against the 17.85) |

10.00 of 17.85 ms is **56% of the frame**, and it is the only row large enough to
matter: 7.82 ms of graph and 5.82 ms of CPU both fit inside a 16.67 ms budget on
their own. Benchmark 6 confirms the row scales with bytes and not with layers —
11.9 MB/frame (4 layers sharing 1 source) transfers in 3.20 ms at the same ~3.9 GB/s
(`target/bench_P24_all.txt`), so this is a bus-rate row, not a fixed overhead.

### Does the interop path delete it, or move it?

Delete it. `DecodeInteropTarget::copy_from_nvdec_frame` issues
`cuMemcpy2DAsync` device→array: NVDEC's output surface and the destination texture
are both in VRAM on the same adapter. 47.5 MB at the RTX 3050's ~224 GB/s of VRAM
bandwidth is **~0.2 ms**, against 10.00 ms measured over PCIe at 4.9 GB/s. That is
the whole basis of the estimate and it is a ~50× ratio, not a marginal one.

### The estimate

| | now | with G2 | source |
|---|---:|---:|---|
| Bench-5-shaped frame (4 × 4K sources) | 17.85 ms | **~8.1 ms** | 17.85 − 10.00 + 0.2 |
| implied throughput | 53.9 FPS | ~120 FPS, GPU-graph-bound at 7.8 ms | |
| Real-media 4K60, 1 layer, serial E2E | 9.58 ms | **~6.3 ms** | 9.58 − 3.20 + 0.1, less the 0.11 decode delta |
| implied throughput | 104.4 FPS | ~160 FPS | `target/bench_G2_media.txt` |

**Verdict: G2 clears the refactor's cost, and nothing else does.** Phase H's own
data says so from the other side — the entire graph is 6.80 ms
(`target/bench_P24_nodes.txt`), so a graph reduced to *zero* still leaves a 10 ms
transfer row and a 17.9 ms frame. G2 is the only remaining change that can reach
60 FPS at 4K, exactly as the plan states.

## What it costs, stated honestly

Not "wire up one argument". The `None` at `src/io/io_layer.rs:175` is the last line
of the change, not the first.

1. **`FrameCache` caches CPU slots, and would have to cache textures.**
   `CachedFrame = (FrameSlotId, DecodedFrameMeta)` where `FrameSlotId` indexes
   `FrameSlotPool`'s `Vec<Mutex<Vec<u8>>>`. The interop path's product is a Y/UV
   texture pair. A cache of 32 entries × 12.4 MB of 4K NV12 textures is ~400 MB of
   VRAM held by the cache alone, so the cache depth becomes a VRAM budget decision
   rather than a RAM one.
2. **`RenderContext` owns its resources; interop textures are owned elsewhere.**
   `resources: Vec<Option<(Texture, TextureView, ViewId)>>`, acquired from and
   returned to `TransientTexturePool` inside `execute`. Binding a
   `DecodeInteropTarget`'s texture means the graph must accept an *imported*,
   externally-owned resource it must not release — a new resource kind in
   `render/resource.rs` and a branch in all three `execute*` bodies.
3. **`YuvUploadNode` must be absent, not fed.** Graph construction lives in three
   places that would each need the branch: `src/export/renderer.rs:270`,
   `ui/src/app.rs`, and the bench's own builders.
4. **`copy_from_nvdec_frame` ends in `cuStreamSynchronize`** (`decode_interop.rs:153`)
   — a blocking full-stream sync, per frame, *per layer*. Benchmark 5's shape would
   add four of them per frame to a pipeline whose current CPU cost is 5.82 ms and
   whose whole advantage is that frames overlap. **This is the one part of the
   estimate that could have gone the wrong way, and G2a measured it: it does not.**
   Four interop copies per frame are 6.0–9.7 ms *cheaper* than four CPU round-trips
   (`target/g2a_probe_3x.txt`, table at the end of this file). The sync stays for now;
   replacing it with an exported CUDA semaphore is a later optimisation, not a
   precondition.
5. **Colour metadata must survive the bypass** (gotcha 11). `emit_frame`'s interop
   arm already reads `read_frame_color` and pins the layout to NV12/P010, so the
   data is there; `tests::colour_plumbing`'s **BT.601** fixtures must be made to run
   through the interop path, because BT.709 ones pass even when metadata is dropped.
6. **The pool cap needs re-reading** (gotcha 14), though the prediction is that it
   does not move: `peak bucket 16/32` counts *canvas-sized `Rgba16Float`*, and the
   Y/UV planes are `R8Unorm`/`Rg8Unorm` at fixed sizes, i.e. different buckets.
   Confirm from a run rather than from this paragraph.

## Recommendation

Proceed with G2, staged so each step is independently verifiable:

- **G2a — DONE, and it clears.** See below.
- **G2b — DONE** (`eb80f53`). The imported-resource kind exists in
  `render/resource.rs` + `RenderContext`, and the pool cannot reclaim one. Two
  properties G2c and G2d both lean on:
  - **`GraphResource::into_transient` is the only release path.** `Transient`
    yields to the pool, `Imported` yields `None` and is dropped. A released
    import would be parked in a bucket keyed by (format, usage, size) and handed
    to an unrelated resource a frame or more later, while the decoder still
    writes into it — a torn frame, no error, and the pool reporting an ordinary
    hit. `render::resource::tests::an_imported_resource_is_never_released_to_the_pool`
    gives the import the SAME bucket key as the transient beside it, so a leak
    shows as `pooled == 2` rather than passing for the wrong reason.
  - **An import's `ViewId` is minted once and cloned per frame.** Eleven nodes
    key their bind-group cache on it (gotcha 16), so a fresh id per frame for one
    texture rebuilds all of them — the cost `e5623de` removed, through the other
    door.
  - `ResourceBuilder::import` registers the id as its own producer (so
    `MissingProducer` keeps its meaning) and creates NO descriptor, which is what
    stops the acquire loop allocating for it. `create` + `import` on one id is a
    compile-time `IncompatibleAccess`. An unbound import panics naming itself,
    rather than surfacing as `ctx.get` failing inside whichever node sampled it
    first.
  - `FrameState::imported` is the per-frame binding. Empty is the CPU path, so
    every existing caller is unchanged — which is why G2c can land behind a
    capability check without touching the fallback.
  - 253 tests pass (247 before). Each load-bearing assertion was verified by
    mutation: yielding the import from `into_transient`, a hand-written `Clone`
    minting a new `ViewId`, resolve skipping an unbound binding, and resolve also
    acquiring a pooled texture per import — each fails the test that claims to
    catch it.
- **G2c — DONE.** `src/io/interop_decode.rs`: `InteropDecodeTargets` owns one
  `DecodeInteropTarget` per source, allocated once and reused, behind a capability
  gate that rejects (with a remembered reason) a host without CUDA, a decoder that
  is not NVDEC (`Decoder::hw_type()` is new — `open` probes and falls back
  *silently*, `open_sw` never probes), a >8-bit source, and odd/zero dimensions.
  `IoLayer::decode_interop` is the entry point and `None` is the CPU fallback.
  Four decisions worth recording:
  - **No texture cache, one frame in flight per source.** That is the VRAM answer
    to cost #1 above: 32 cached 4K NV12 pairs is ~400 MB against ~3.7 GB of 8 GB
    already used, each needing its own D3D12 allocation and its own `ViewId`.
    `SourceTarget::held_pts` makes a repeat request for the resident frame free; a
    different pts re-decodes into the same textures.
  - **A read-after-write hazard the estimate above did not list.** One frame per
    source means the next decode overwrites textures the last frame's submitted
    graph may still be sampling. `cuStreamSynchronize` proves CUDA finished
    *writing* and says nothing about wgpu finishing *reading* — two APIs, no shared
    timeline. `mark_submitted` stamps the reading submission, `decode_into_target`
    waits on it before copying. **A caller that renders from imported planes and
    skips `mark_submitted` gets a torn frame with no error and no counter.**
  - **The 8-bit restriction is checked twice.** The planes are `R8Unorm`/`Rg8Unorm`;
    a 10-bit source decodes to P010 at two bytes per sample, so the same copy puts
    twice the bytes into half the array with `cuMemcpy2DAsync` returning success.
    Refused from the container's `PixelFormat` before allocating, and re-checked
    against the frame's own dimensions inside `copy_from_nvdec_frame` because a
    container may disagree with its frames.
  - **Export stays on the CPU path**, stated in code rather than left to whether the
    host has CUDA: `ExportDecodeWorker` decodes 16 frames ahead into the CPU
    `FrameCache` and a one-frame target cannot serve a lookahead.
- **G2d — construction DONE, measurement OPEN.** `YuvToRgbNode::with_imported_planes`
  switches `declare_resources` from `read` to `import`; both production builders
  (`src/export/renderer.rs`, `ui/src/app.rs`) **omit** the `YuvUploadNode` rather
  than leaving it unfed, because `declare_resources` *creates* both planes — an
  unfed node allocates two pooled textures nobody writes and renders a black clip
  with no error and the pool reporting two ordinary hits. Plane ids come from
  `FrameScheduler::interop_y_id`/`interop_uv_id`, the same functions that bind them.
  `ClipSignature::is_interop` forces a recompile when a source falls back
  mid-timeline.
  - **Colour verified where gotcha 11 says to verify it.**
    `tests::colour_plumbing::interop_decoded_frames_carry_their_matrix_to_the_shader`
    drives BT.601 fixtures through the real `InteropDecodeTargets`: worst delta
    **1/255** with the signalled matrix, **39/255** mis-tagged as BT.709. It also
    asserts the graph shape (`YuvUpload` absent, `pooled == 1`), because pixels alone
    cannot tell "imported correctly" from "someone fed an upload node".
  - `cargo test`: 259 passed, 0 failed (253 before). Interop confirmed available on
    this machine: `transport=D3D12Win32Handle`, driver 12.6. **`cuda.dll` must be
    copied into `target/debug/deps/` or every interop test skips with a printed
    reason** — which looks like a green suite that exercised nothing.
- **G2e — OPEN, and it is the whole remaining task.** `bench --media` still measures
  the path G2 replaced: `media_graph` builds a `YuvUploadNode` unconditionally from a
  host-memory `DecodedCache` and constructs no `IoLayer`, so no run of it can touch
  the interop path. Until a pass goes through `FrameScheduler`, the ~8.1 ms/frame
  figure below remains a prediction. See
  `.hermes/plans/2026-08-30_210807-nexir-4k60-upload-sync-audit.md` § G2e.

## G2a result — the stream sync does not sink it

`examples/g2a_interop_cost_probe.rs`, `target/g2a_probe_3x.txt`. Four independent
decoders on one file (benchmark 5's shape), median of 50 frames per pass, three
repeats. A "frame" is all four decodes; the first frame per source is primed
untimed, so the thread-delay fill is not in the median.

| fixture | pass | r1 | r2 | r3 | median |
|---|---|---:|---:|---:|---:|
| cam_4k30 | CPU round-trip | 20.13 | 21.39 | 22.40 | **21.39 ms** |
| | GPU interop | 14.12 | 14.12 | 14.18 | **14.12 ms** |
| cam_4k60 | CPU round-trip | 23.64 | 22.45 | 22.31 | **22.45 ms** |
| | GPU interop | 13.93 | 13.94 | 13.93 | **13.93 ms** |

**Interop is 6.0–9.7 ms/frame cheaper at four sources, in every one of six pass
pairs.** So four `cuStreamSynchronize` calls per frame do not cost more than the
`av_hwframe_transfer_data` they replace — the risk that would have sunk G2 is
measured and absent. Two things worth noting beyond the headline:

- **The interop pass is the stable one.** 13.93–14.18 ms across six runs (1.8%
  spread) against 20.13–23.64 ms for the CPU path (17%). The sync makes the cost
  *predictable*, which for a frame-paced pipeline is worth something on its own.
- **It scales better than per-source arithmetic suggests.** The single-source probe
  (`target/g2_recost_probe.txt`) measured 4.95 ms/frame interop; four sources cost
  13.93, i.e. 3.48 each rather than 4.95 — the four decoders overlap, and the syncs
  do not serialise them into 4 × 4.95 = 19.8 ms.

This does **not** re-measure the 10.00 ms `GPU transfer` row: neither pass uploads to
the graph, so the delta above is decode-side only. The transfer row is still the
reason to do G2 and is still measured by `bench.exe 5`. What G2a establishes is that
the decode side gets *better*, not worse, so the two effects add rather than cancel.

