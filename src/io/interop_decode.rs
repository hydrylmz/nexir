// src/io/interop_decode.rs
//
// G2c — the per-source ownership of NVDEC's destination surfaces.
//
// WHAT THIS IS FOR. `IoLayer::decode_blocking` used to call
// `dec.decode_into(&pkt, mapped, None)`: NVDEC decoded into VRAM, FFmpeg copied
// the frame back to host memory (`av_hwframe_transfer_data`), the frame landed in
// a `FrameSlotPool` buffer, and `YuvUploadNode` then pushed it *back* across the
// bus into a texture. That round trip is the `GPU transfer` row — 10.00 ms of a
// 17.85 ms 4K frame, 56% of it, and the only row large enough to be the reason
// benchmark 5 sits at 56.8 FPS instead of 60 (`.hermes/plans/2026-08-31_g2-
// recosting.md`). The interop path deletes it: `cuMemcpy2DAsync` device→array
// stays in VRAM at ~224 GB/s rather than crossing PCIe at 4.9.
//
// WHY THIS IS A SEPARATE FILE AND NOT A FIELD ON `IoLayer`. The thing being
// owned is not a buffer, it is a *decode target whose textures a render graph
// will bind*, and three properties follow from that which the slot pool does not
// have and must not grow:
//
//  1. **One target per SOURCE, allocated once.** The `ViewId`s the graph caches
//     bind groups on are minted inside `DecodeInteropTarget::new` (gotcha 18), so
//     a per-frame target rebuilds eleven bind-group caches every frame — the exact
//     cost `e5623de` removed, through the other door. It is also a
//     `CreateCommittedResource` + `CreateSharedHandle` + `cuImportExternalMemory`
//     per plane, which is not per-frame work.
//
//  2. **Therefore ONE FRAME IN FLIGHT per source, and therefore no cache.**
//     `FrameCache` holds 32 CPU slots; the interop path cannot, because 32 × 12.4
//     MB of 4K NV12 textures is ~400 MB of VRAM on top of the ~3.7 GB of 8 GB
//     already in use (the NVML row), and each would still need its own `ViewId`
//     and its own D3D12 allocation. So a target records which pts it currently
//     holds ([`SourceTarget::held_pts`]) and a request for a different pts
//     re-decodes into the same textures. Playback is sequential, so the common
//     case is a request for the next pts, which is a decode either way. A
//     *repeated* request for the pts already resident costs nothing —
//     [`InteropDecodeTargets::decode_into_target`] returns `Cached`.
//
//  3. **It is a capability, so every failure has to fall back rather than fail.**
//     No CUDA, no NVDEC on this source, a 10-bit source, a frame bigger than the
//     allocation, a `cuMemcpy2DAsync` that returns an error: each returns
//     `Unavailable`/`Err` and the caller uses the CPU path, which is untouched.
//     `FrameState::imported` being empty already means "CPU upload path", so the
//     fallback needs no branch of its own (gotcha 18).
//
// THE HAZARD THIS FILE EXISTS TO CONTAIN, and it has no error message. One frame
// per source means the NEXT decode overwrites the textures the LAST frame's graph
// is still reading. `copy_from_nvdec_frame` ends in `cuStreamSynchronize`, which
// proves CUDA finished writing — it says nothing about wgpu having finished
// reading, and the two APIs share no timeline. Overwriting a texture mid-read is a
// torn frame with the pool reporting ordinary hits, exactly like gotchas 14, 16 and
// 18. So [`InteropDecodeTargets::mark_submitted`] stamps each target with the
// submission that last read it, and [`InteropDecodeTargets::decode_into_target`]
// waits for that submission before issuing the copy — the same
// `WaitForSubmissionIndex` guard `ExportRenderer` uses before handing a slot to
// NVENC, and for the same reason. **A caller that renders from these planes and
// never calls `mark_submitted` gets the torn frame**, which is why the two live
// callers do it one line after their `submit`.
//
// THE 8-BIT RESTRICTION IS DELIBERATE AND CHECKED. `DecodeInteropTarget`'s planes
// are `R8Unorm`/`Rg8Unorm`. A 10-bit source decodes to P010 — two bytes per
// sample, `FrameLayout::P010` — so the same copy would put twice the bytes into
// half the array. [`Self::target_for`] refuses those *before* allocating, from the
// container's `PixelFormat`, and `copy_from_nvdec_frame` refuses an oversized
// frame as a second line of defence, because the container is allowed to disagree
// with the frames it contains.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::interop::capability::InteropCapability;
use crate::interop::cuda_context::CudaContext;
use crate::interop::decode_interop::DecodeInteropTarget;
use crate::io::decoder::Decoder;
use crate::io::demuxer::Packet;
use crate::io::ffi::hw_accel::HwDeviceType;
use crate::render::device::GpuDevice;
use crate::render::resource::InteropPlanes;
use crate::timeline::ids::SourceId;
use crate::timeline::source::{DecodedFrameMeta, VideoStreamInfo};

/// What one interop decode attempt produced.
///
/// Four outcomes rather than a `Result<Option<_>>`, because the caller does
/// something different with each and collapsing any two of them loses a decision:
/// `Pending` must feed another packet, `Unavailable` must fall back to the CPU
/// path *for this source*, an `Err` must fall back for this frame while leaving the
/// source's target in place, and `Decoded` may still not be the frame the caller
/// asked for — see [`InteropFrame::pts`].
#[derive(Debug)]
pub enum InteropDecode {
    /// A frame was decoded into the target's textures. **Not necessarily the one
    /// asked for**: a sequential read-forward passes through the frames before it,
    /// and each lands in these same textures. The caller compares
    /// [`InteropFrame::pts`] against its target and calls
    /// [`InteropDecodeTargets::mark_held`] for the one it keeps.
    Decoded(InteropFrame),
    /// The decoder swallowed the packet without emitting a frame. Feed another.
    Pending,
    /// This source cannot use the interop path at all (no CUDA, no NVDEC, wrong
    /// depth, no registered geometry). The caller must use the CPU path, and
    /// asking again for the same source will get the same answer.
    Unavailable(&'static str),
}

/// A decoded frame living in a source's interop textures.
#[derive(Clone, Debug)]
pub struct InteropFrame {
    /// The Y/UV textures, ready to bind via
    /// [`crate::render::resource::ImportedResources`].
    pub planes: InteropPlanes,
    /// Layout and colour as the DECODER reported them for these exact pixels —
    /// gotcha 11's rule, and the reason the interop path can be colour-correct at
    /// all: `emit_frame`'s interop arm reads `read_frame_color` from the AVFrame.
    pub meta: DecodedFrameMeta,
    /// The pts the DECODER reported for this frame, in the SOURCE stream's
    /// timebase.
    ///
    /// Carried because the read-forward loop needs it: a sequential request
    /// translates a project pts to a stream pts and then decodes until a frame at
    /// or past it arrives, and the frames before that one land in these same
    /// textures and are discarded. Without the emitted pts the loop cannot tell
    /// which frame it is looking at.
    pub pts: i64,
    /// Frame dimensions, which may be smaller than the target's allocation.
    pub width: u32,
    pub height: u32,
}

/// One source's decode target plus what it currently holds.
struct SourceTarget {
    target: DecodeInteropTarget,
    /// The project-timebase pts resident in the textures, or `None` when what they
    /// hold is unknown — before the first decode, after a seek, or after a frame
    /// the caller decided to discard.
    ///
    /// **This is what makes a one-frame target usable.** Without it a caller
    /// re-asking for the frame it just rendered (a paused playhead, a graph
    /// recompile, two layers over one source) would trigger a decode of a frame
    /// that is already there — and on a sequential decoder that means reading
    /// forward past it.
    held_pts: Option<i64>,
    /// Cached alongside `held_pts` so a `Cached` hit answers with the same
    /// metadata the decode reported rather than re-deriving it.
    held: Option<InteropFrame>,
    /// The last wgpu submission that READ these textures, if any.
    ///
    /// **The read-after-write guard, and the one hazard a one-frame target
    /// creates.** `cuStreamSynchronize` at the end of `copy_from_nvdec_frame`
    /// proves CUDA finished writing; nothing in it proves wgpu finished reading the
    /// PREVIOUS frame, and the two APIs share no timeline. Without this the next
    /// decode overwrites a texture a submitted graph is still sampling — a torn
    /// frame, no error, and the pool reporting ordinary hits.
    ///
    /// Cleared once waited on, so a repeated decode with no render in between does
    /// not wait twice.
    last_read: Option<wgpu::SubmissionIndex>,
}

/// Per-source NVDEC decode targets, and the capability gate in front of them.
///
/// Cheap to construct and inert when the host has no CUDA: [`Self::new`] takes the
/// probe result, and every method returns `Unavailable` without allocating
/// anything when `capability.is_available()` is false. That is what lets `IoLayer`
/// hold one unconditionally.
pub struct InteropDecodeTargets {
    device: Arc<GpuDevice>,
    capability: InteropCapability,
    /// `None` when interop is unavailable, or when `CudaContext::new` failed —
    /// which is a host condition, reported once and then remembered.
    cuda: Option<Arc<CudaContext>>,
    /// One entry per source that has successfully allocated a target.
    targets: Mutex<HashMap<SourceId, SourceTarget>>,
    /// Sources known NOT to be interop-capable, with the reason.
    ///
    /// Remembered so the reason is logged once rather than per frame, and so a
    /// per-frame call does not retry a `SharedTexture` allocation that already
    /// failed.
    rejected: Mutex<HashMap<SourceId, &'static str>>,
}

impl InteropDecodeTargets {
    /// Build the registry for a device and its probed capability, retaining the
    /// CUDA primary context itself.
    ///
    /// `GpuDevice` by `Arc` because a target's textures outlive any one frame and
    /// the registry may be used from the prefetch worker as well as the render
    /// thread (gotcha 3: `GpuDevice` is `!Send` on some backends, so it is always
    /// shared behind `Arc`).
    ///
    /// **Prefer [`Self::with_context`] when the caller already holds a
    /// `CudaContext`.** Every `CudaContext` in the process wraps the SAME primary
    /// context (`cuDevicePrimaryCtxRetain`), so a second one is a second
    /// retain/release pair on one resource — and one owner's `Drop` releasing while
    /// another is still retaining is a race that surfaces as an unrelated failure
    /// elsewhere (gotcha 4). `ui/src/app.rs` already holds one for export.
    pub fn new(device: Arc<GpuDevice>, capability: InteropCapability) -> Self {
        let cuda = if capability.is_available() {
            match CudaContext::new(&capability) {
                Ok(c) => Some(Arc::new(c)),
                Err(e) => {
                    log::warn!(
                        "[interop] CudaContext::new failed ({e:?}) — every source \
                         falls back to the CPU upload path"
                    );
                    None
                }
            }
        } else {
            None
        };
        Self::build(device, capability, cuda)
    }

    /// Build the registry against a context the caller already owns.
    ///
    /// The right constructor whenever one exists, for the reason [`Self::new`]
    /// documents: one primary context, one owner. `cuda = None` gives a registry
    /// that declines every source, which is what an unavailable capability means.
    pub fn with_context(
        device: Arc<GpuDevice>,
        capability: InteropCapability,
        cuda: Option<Arc<CudaContext>>,
    ) -> Self {
        let cuda = capability.is_available().then_some(cuda).flatten();
        Self::build(device, capability, cuda)
    }

    fn build(
        device: Arc<GpuDevice>,
        capability: InteropCapability,
        cuda: Option<Arc<CudaContext>>,
    ) -> Self {
        if cuda.is_some() {
            log::info!(
                "[interop] GPU decode targets enabled (transport={:?}, driver {})",
                capability.transport,
                capability.driver_version
            );
        }
        Self {
            device,
            capability,
            cuda,
            targets: Mutex::new(HashMap::new()),
            rejected: Mutex::new(HashMap::new()),
        }
    }

    /// A registry that can never hand out a target.
    ///
    /// For callers with no GPU decode ambitions (the export path's second
    /// `IoLayer`, tests) and for a host without CUDA. Distinct from `new` with an
    /// unavailable capability only in that it does not probe.
    pub fn disabled(device: Arc<GpuDevice>) -> Self {
        Self {
            device,
            capability: InteropCapability::none(),
            cuda: None,
            targets: Mutex::new(HashMap::new()),
            rejected: Mutex::new(HashMap::new()),
        }
    }

    /// Whether any source could use the interop path on this host.
    pub fn is_available(&self) -> bool {
        self.cuda.is_some()
    }

    /// The probed capability, for reporting.
    pub fn capability(&self) -> &InteropCapability {
        &self.capability
    }

    /// How many sources currently hold a target — i.e. how many Y/UV texture
    /// pairs this registry has allocated.
    ///
    /// Counted rather than estimated (gotcha 9): at 4K each pair is ~12.4 MB of
    /// VRAM, and this is the only place that number is knowable.
    pub fn live_targets(&self) -> usize {
        self.targets.lock().unwrap().len()
    }

    /// Why a source is on the CPU path, if it is.
    pub fn rejection_reason(&self, source_id: SourceId) -> Option<&'static str> {
        self.rejected.lock().unwrap().get(&source_id).copied()
    }

    /// Forget every target and every rejection.
    ///
    /// Called from `IoLayer::reset_for_new_project`: the old project's targets are
    /// sized for the old project's sources, and a `SourceId` is an index into a
    /// registry that has just been replaced — so keeping them would hand a new
    /// source the previous occupant's textures.
    pub fn clear(&self) {
        self.targets.lock().unwrap().clear();
        self.rejected.lock().unwrap().clear();
    }

    /// Drop one source's target, releasing its VRAM.
    pub fn evict(&self, source_id: SourceId) {
        self.targets.lock().unwrap().remove(&source_id);
        self.rejected.lock().unwrap().remove(&source_id);
    }

    /// Decide whether `source_id` may use the interop path, allocating its target
    /// on first use.
    ///
    /// Returns the rejection reason as `Err`, and caches it: the checks are
    /// properties of the source and the host, neither of which changes between
    /// frames.
    fn ensure_target(
        &self,
        source_id: SourceId,
        info: &VideoStreamInfo,
        decoder_hw: HwDeviceType,
    ) -> Result<(), &'static str> {
        if let Some(reason) = self.rejected.lock().unwrap().get(&source_id) {
            return Err(reason);
        }
        if self.targets.lock().unwrap().contains_key(&source_id) {
            return Ok(());
        }

        let reject = |reason: &'static str| -> Result<(), &'static str> {
            self.rejected.lock().unwrap().insert(source_id, reason);
            log::info!(
                "[interop] source {} stays on the CPU upload path: {reason}",
                source_id.index()
            );
            Err(reason)
        };

        let Some(cuda) = self.cuda.as_ref() else {
            return reject("CUDA interop is unavailable on this host");
        };

        // NVDEC specifically. `Decoder::open` probes and falls back silently, and
        // `open_sw` (still images) never probes — a target allocated for either
        // would be 12.4 MB of VRAM nothing ever writes into, with the graph bound
        // to an empty texture and no error anywhere.
        if decoder_hw != HwDeviceType::Cuda {
            return reject("this source's decoder is not NVDEC");
        }

        // 8-bit, even dimensions. Through the free function below rather than
        // inline, so the rule that is unit-tested without a GPU is the same rule
        // production applies — two copies of it is how one of them drifts.
        if let Err(reason) = source_geometry_is_interop_capable(info) {
            return reject(reason);
        }

        match DecodeInteropTarget::new(
            Arc::clone(cuda),
            &self.device,
            self.capability.transport,
            info.width,
            info.height,
        ) {
            Ok(target) => {
                log::info!(
                    "[interop] source {} decodes straight into a {}x{} NV12 texture pair \
                     (~{:.1} MB VRAM); its upload node is now absent from the graph",
                    source_id.index(),
                    info.width,
                    info.height,
                    (info.width as f64 * info.height as f64 * 1.5) / (1024.0 * 1024.0),
                );
                self.targets.lock().unwrap().insert(
                    source_id,
                    SourceTarget { target, held_pts: None, held: None, last_read: None },
                );
                Ok(())
            }
            Err(e) => {
                log::warn!(
                    "[interop] DecodeInteropTarget::new failed for source {}: {e:?}",
                    source_id.index()
                );
                reject("allocating the shared Y/UV texture pair failed")
            }
        }
    }

    /// What this source currently holds, if it is the pts being asked for.
    ///
    /// Separate from [`Self::decode_into_target`] so a caller can answer a
    /// cache-style query without holding a decoder lock — which is what
    /// `IoLayer::decode_blocking` needs before it decides to seek.
    pub fn held_frame(&self, source_id: SourceId, pts: i64) -> Option<InteropFrame> {
        let targets = self.targets.lock().unwrap();
        let entry = targets.get(&source_id)?;
        if entry.held_pts == Some(pts) {
            entry.held.clone()
        } else {
            None
        }
    }

    /// Feed one packet through the interop path, into `source_id`'s own textures.
    ///
    /// Returns whatever the decoder emitted, carrying the frame's own pts — **the
    /// caller decides whether that is the frame it wanted.** A sequential read
    /// forward passes through every frame between the decoder's position and the
    /// target, and each one lands in these same textures; the caller keeps the
    /// first at or past its target and calls [`Self::mark_held`] for it.
    ///
    /// This does NOT answer from what is already resident: [`Self::held_frame`] is
    /// that check and it is cheaper (no decoder lock), so doing it here as well
    /// would put a second copy of the same decision in the hot path.
    ///
    /// **Concurrency.** The copy is issued while the target map is locked, which
    /// serialises distinct sources against each other. That is deliberate rather
    /// than incidental: `CudaContext::with_context` is `cuCtxPushCurrent`, the
    /// driver requires the context to be floating, and every `CudaContext` in this
    /// process wraps the same primary context (gotcha 4). Two sources copying
    /// concurrently is precisely the case where the second push fails, the pop
    /// takes the wrong context, and every driver call runs against it — silently,
    /// in release. The serialised part is a ~0.2 ms VRAM copy; the decode itself is
    /// already serialised per source by the decoder's own mutex.
    pub fn decode_into_target(
        &self,
        source_id: SourceId,
        info: &VideoStreamInfo,
        decoder: &mut Decoder,
        packet: &Packet,
    ) -> InteropDecode {
        if let Err(reason) = self.ensure_target(source_id, info, decoder.hw_type()) {
            return InteropDecode::Unavailable(reason);
        }
        let Some(cuda) = self.cuda.as_ref() else {
            return InteropDecode::Unavailable("CUDA interop is unavailable on this host");
        };

        let mut targets = self.targets.lock().unwrap();
        let Some(entry) = targets.get_mut(&source_id) else {
            // `ensure_target` returned Ok, so the entry exists unless another
            // thread evicted it between the two locks. Falling back is correct and
            // costs one CPU frame.
            return InteropDecode::Unavailable("the source's target was evicted mid-decode");
        };

        // READ-AFTER-WRITE. The textures about to be overwritten may still be being
        // sampled by a submitted graph: `cuStreamSynchronize` inside the copy proves
        // CUDA finished WRITING, and says nothing about wgpu having finished
        // READING. Two APIs, no shared timeline — the same ordering requirement
        // `ExportRenderer` satisfies before handing a slot to NVENC, and the same
        // fix. Usually already complete, so this returns at once; when it is not,
        // waiting is the only alternative to a torn frame.
        if let Some(sid) = entry.last_read.take() {
            self.device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        }

        // Whatever the textures held is about to be overwritten. Cleared BEFORE the
        // copy, not after: a copy that fails half-way leaves them holding a torn
        // frame, and a `held_pts` surviving that would serve it as a hit.
        entry.held_pts = None;
        entry.held = None;

        // `decode_into` still takes a `&mut [u8]`, and the interop arm returns
        // before touching it. One byte, deliberately: if the branch is ever
        // bypassed this panics on the slice bounds instead of quietly reporting a
        // CPU round-trip as a GPU one — the same guard `examples/
        // g2a_interop_cost_probe.rs` uses.
        let mut unused = [0u8; 1];
        let decoded = decoder.decode_into(
            packet,
            &mut unused,
            Some((cuda.as_ref(), &entry.target, &self.capability)),
        );

        match decoded {
            Ok(Some(frame)) => InteropDecode::Decoded(InteropFrame {
                planes: InteropPlanes {
                    y: entry.target.y_import().clone(),
                    uv: entry.target.uv_import().clone(),
                },
                meta: frame.meta,
                pts: frame.pts,
                width: frame.width,
                height: frame.height,
            }),
            Ok(None) => InteropDecode::Pending,
            Err(e) => {
                log::warn!(
                    "[interop] decode into source {}'s target failed ({e:?}) — this \
                     frame falls back to the CPU path",
                    source_id.index()
                );
                InteropDecode::Unavailable("the interop decode returned an error")
            }
        }
    }

    /// Record that `source_id`'s textures now hold `pts`, so a repeat request for
    /// it is a [`Self::held_frame`] hit rather than a decode.
    ///
    /// Called by the caller rather than by [`Self::decode_into_target`] because
    /// only the caller knows which of the frames it read forward through is the one
    /// it kept — and marking a discarded frame as resident would serve the wrong
    /// picture on the next request, with no error.
    pub fn mark_held(&self, source_id: SourceId, pts: i64, frame: &InteropFrame) {
        if let Some(entry) = self.targets.lock().unwrap().get_mut(&source_id) {
            entry.held_pts = Some(pts);
            entry.held = Some(frame.clone());
        }
    }

    /// Record that `submission` reads every interop texture this frame bound, so the
    /// next decode into any of them waits for it first.
    ///
    /// **Call this immediately after `submit`, for every frame that rendered from
    /// imported planes.** It is the other half of the read-after-write guard in
    /// [`Self::decode_into_target`]: without it the next decode overwrites textures
    /// a submitted graph is still sampling, which is a torn frame with no error,
    /// nothing in the pool's counters, and no failing test — the same shape as
    /// gotchas 14, 16 and 18.
    ///
    /// Takes the clips rather than a source list so the caller cannot pass the
    /// wrong set: the sources to stamp are exactly the ones whose planes were bound.
    pub fn mark_submitted(
        &self,
        clips: &[crate::render::frame_state::ClipRenderEntry],
        submission: &wgpu::SubmissionIndex,
    ) {
        let mut targets = self.targets.lock().unwrap();
        for clip in clips {
            if clip.interop_planes.is_none() {
                continue;
            }
            if let Some(entry) = targets.get_mut(&clip.source_id) {
                entry.last_read = Some(submission.clone());
            }
        }
    }

    /// Note that `source_id`'s textures no longer hold a known frame.
    ///
    /// Called on seek and on decoder flush: the frame in the textures is from
    /// before the seek, so serving it as a `Cached` hit for the new position would
    /// show the wrong picture with no error.
    pub fn invalidate(&self, source_id: SourceId) {
        if let Some(entry) = self.targets.lock().unwrap().get_mut(&source_id) {
            entry.held_pts = None;
            entry.held = None;
        }
    }
}

/// Whether a source's container metadata permits the interop path at all.
///
/// Split out as a free function so it can be unit-tested without a GPU, a CUDA
/// driver or a decoder — the three things that make [`InteropDecodeTargets`]
/// untestable on a CI runner. `ensure_target` applies exactly these rules plus the
/// host and decoder checks.
pub fn source_geometry_is_interop_capable(info: &VideoStreamInfo) -> Result<(), &'static str> {
    if info.pixel_fmt.is_10bit() {
        return Err("the source is 10-bit or deeper, and the interop planes are 8-bit");
    }
    if info.width == 0
        || info.height == 0
        || !info.width.is_multiple_of(2)
        || !info.height.is_multiple_of(2)
    {
        return Err("the source's dimensions are zero or odd");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::rational::Rational;
    use crate::timeline::source::{ColorInfo, PixelFormat};

    /// A registry on the process-wide shared context.
    ///
    /// [`crate::tests::shared_cuda_ctx`], never `CudaContext::new`: a second
    /// retain/release pair on the one primary context makes an unrelated module's
    /// NVENC tests fail. See that function's doc comment for what it cost.
    fn live_registry(device: &Arc<GpuDevice>) -> InteropDecodeTargets {
        let capability = InteropCapability::probe(device);
        let cuda = crate::tests::shared_cuda_ctx(&capability);
        InteropDecodeTargets::with_context(Arc::clone(device), capability, cuda)
    }

    fn info(width: u32, height: u32, pixel_fmt: PixelFormat) -> VideoStreamInfo {
        VideoStreamInfo {
            width,
            height,
            frame_rate: Rational { num: 30, den: 1 },
            pixel_fmt,
            color_info: ColorInfo::bt709(),
            duration_pts: 0,
            is_vfr: false,
            time_base: Rational { num: 1, den: 90_000 },
            rotation: Default::default(),
        }
    }

    /// The depth gate, stated where it can be checked without hardware.
    ///
    /// THE BUG THIS PINS. `DecodeInteropTarget`'s planes are `R8Unorm` and
    /// `Rg8Unorm`. A 10-bit source decodes to P010 — `FrameLayout::P010`, two bytes
    /// per sample — and `copy_from_nvdec_frame` would then copy `width` bytes per
    /// row into an array whose rows are `width` *samples* of one byte. The result
    /// is half a picture and half of nothing, with `cuMemcpy2DAsync` returning
    /// success: the copy is well-formed, it just describes the wrong data.
    #[test]
    fn a_ten_bit_source_is_refused_before_anything_is_allocated() {
        for fmt in [
            PixelFormat::P010,
            PixelFormat::Yuv420p10,
            PixelFormat::Yuv422p10,
            PixelFormat::Yuv444p10,
            PixelFormat::Yuv420p12,
        ] {
            let err = source_geometry_is_interop_capable(&info(3840, 2160, fmt))
                .expect_err("a >8-bit source must be refused: the planes are 8-bit");
            assert!(
                err.contains("8-bit"),
                "the reason must name the depth mismatch, got {err:?}"
            );
        }
        // ...and the formats that ARE 8-bit 4:2:0 must pass, or the gate would put
        // every source on the CPU path and G2 would measure nothing.
        for fmt in [PixelFormat::Nv12, PixelFormat::Yuv420p] {
            source_geometry_is_interop_capable(&info(3840, 2160, fmt))
                .expect("an 8-bit source must be allowed");
        }
    }

    /// Odd or zero dimensions cannot describe a half-size chroma plane.
    ///
    /// `DecodeInteropTarget::new` allocates `width / 2` by `height / 2`, so a
    /// 1921-wide source would get a 960-wide chroma plane covering 1920 luma
    /// columns — the last column reading whatever integer division left behind.
    /// Zero is worse: `Dimension X is zero` from inside `Device::create_texture`,
    /// naming neither the source nor the cause (the failure mode gotcha 10
    /// records for `LutNode`).
    #[test]
    fn odd_or_zero_dimensions_are_refused() {
        for (w, h) in [(1921, 1080), (1920, 1081), (0, 1080), (1920, 0)] {
            source_geometry_is_interop_capable(&info(w, h, PixelFormat::Nv12))
                .expect_err(&format!("{w}x{h} must be refused"));
        }
        source_geometry_is_interop_capable(&info(1920, 1080, PixelFormat::Nv12))
            .expect("an even 8-bit source must be allowed");
    }

    /// A disabled registry must answer `Unavailable` without touching CUDA.
    ///
    /// This is the fallback's entry condition and the reason `IoLayer` can hold one
    /// unconditionally: on a host with no CUDA, or for an `IoLayer` that does not
    /// want GPU decode, nothing here may allocate, probe, or log a warning per
    /// frame.
    #[test]
    fn a_disabled_registry_allocates_nothing() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let targets = InteropDecodeTargets::disabled(Arc::new(device));
        assert!(!targets.is_available());
        assert_eq!(targets.live_targets(), 0);
        assert!(targets.held_frame(SourceId::new(0), 0).is_none());
        // `ensure_target` must reject on the host check, before the depth check,
        // so a disabled registry never even looks at the source.
        let err = targets
            .ensure_target(SourceId::new(0), &info(1920, 1080, PixelFormat::Nv12), HwDeviceType::Cuda)
            .expect_err("a disabled registry must reject every source");
        assert!(err.contains("CUDA"), "got {err:?}");
        assert_eq!(targets.live_targets(), 0, "nothing may be allocated");
        assert_eq!(targets.rejection_reason(SourceId::new(0)), Some(err));
    }

    /// A software decoder must be refused even when CUDA itself is available.
    ///
    /// `Decoder::open` probes for hardware and falls back to software silently, and
    /// `open_sw` (still images, per `IoLayer::get_or_open_decoder`) never probes.
    /// Allocating a target for either would hold 12.4 MB of VRAM per source that
    /// nothing writes into, while `emit_frame`'s interop arm — which is gated on
    /// `hw_type == Cuda` — sends every frame down the CPU path. The graph would
    /// then sample an empty texture: a black or garbage clip, no error.
    #[test]
    fn a_software_decoder_is_refused_even_with_cuda_present() {
        let _cuda = crate::tests::cuda_lock();
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let device = Arc::new(device);
        let targets = live_registry(&device);
        if !targets.is_available() {
            eprintln!("SKIP: no CUDA interop on this machine — the host check answers first");
            return;
        }

        let err = targets
            .ensure_target(
                SourceId::new(3),
                &info(1920, 1080, PixelFormat::Nv12),
                HwDeviceType::None,
            )
            .expect_err("a software decoder must not get an interop target");
        assert!(err.contains("NVDEC"), "the reason must name the decoder, got {err:?}");
        assert_eq!(targets.live_targets(), 0, "nothing may be allocated for it");

        // The rejection is remembered, so the reason is logged once and the failed
        // check is not retried per frame.
        assert_eq!(targets.rejection_reason(SourceId::new(3)), Some(err));
        assert!(targets
            .ensure_target(SourceId::new(3), &info(1920, 1080, PixelFormat::Nv12), HwDeviceType::Cuda)
            .is_err());
    }

    /// One target per source, reused — and its `ViewId`s stable across frames.
    ///
    /// The property gotcha 18 exists for, checked end to end on real hardware: a
    /// second `ensure_target` for the same source must NOT allocate a second
    /// texture pair, and the planes handed out must carry the same `ViewId`s, or
    /// eleven bind-group caches rebuild every frame.
    #[test]
    fn a_sources_target_is_allocated_once_and_keeps_its_view_ids() {
        let _cuda = crate::tests::cuda_lock();
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let device = Arc::new(device);
        let targets = live_registry(&device);
        if !targets.is_available() {
            eprintln!("SKIP: no CUDA interop on this machine");
            return;
        }

        let sid = SourceId::new(0);
        let source = info(1920, 1080, PixelFormat::Nv12);
        if targets.ensure_target(sid, &source, HwDeviceType::Cuda).is_err() {
            eprintln!("SKIP: the shared texture pair could not be allocated on this host");
            return;
        }
        assert_eq!(targets.live_targets(), 1);

        let first = {
            let map = targets.targets.lock().unwrap();
            let t = &map[&sid].target;
            (t.y_import().view_id(), t.uv_import().view_id())
        };
        assert_ne!(first.0, first.1, "the two planes are distinct textures");

        // Three "frames" of re-binding, which is what the failure mode needs: an
        // id minted per frame alternates or drifts rather than failing at once.
        for frame in 0..3 {
            targets
                .ensure_target(sid, &source, HwDeviceType::Cuda)
                .expect("an allocated source stays allocated");
            assert_eq!(
                targets.live_targets(),
                1,
                "frame {frame}: a second target was allocated for one source"
            );
            let map = targets.targets.lock().unwrap();
            let t = &map[&sid].target;
            assert_eq!(
                (t.y_import().view_id(), t.uv_import().view_id()),
                first,
                "frame {frame}: the planes' ViewIds moved, so every bind-group \
                 cache keyed on them is rebuilt"
            );
        }

        // Eviction is what releases the VRAM, and it must actually do so.
        targets.evict(sid);
        assert_eq!(targets.live_targets(), 0);
        targets.clear();
        assert_eq!(targets.live_targets(), 0);
    }
}
