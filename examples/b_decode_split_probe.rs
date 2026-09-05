// examples/b_decode_split_probe.rs
//
// Task B — WHICH CPU stage inside benchmark 7's `Decode` column alternates.
//
// WHAT IS ALREADY MEASURED, so this probe does not re-measure it. Benchmark 7's
// frames arrive 16.66 / 49.06 ms in a two-frame cycle (`target/taskB_7_1.csv`,
// parity split of the steady series), and with `decode_ms` now in that CSV the
// whole 32 ms swing sits in the `Decode` stage: 40.46 ms on the even frames
// against 8.33 ms on the odd ones, while `gpu_graph_ms` is FLAT (8.35 vs 8.66 —
// the fast half is if anything dearer). `NEXIR_GPU_LOOKAHEAD=1` does not remove
// the cycle (16.81 / 52.35 ms), which refutes the standing hypothesis that it is
// the pipeline having `gpu_lookahead` frames outstanding.
//
// WHAT IS STILL UNATTRIBUTED, and it is the reason for this file. On the
// real-media rows `PipelineStage::Decode` is `submitted.schedule`, i.e. the whole
// of `FrameScheduler::schedule_frame`, and inside that a source's frame costs
// THREE things which no counter separates:
//
//   1. `Demuxer::next_video_packet` — reading the coded bitstream. **Not covered
//      by `InteropDecodeStats`**: `decode_into_target`'s timer starts after the
//      packet is already in hand, so every figure in the bench's `Interop:` block
//      excludes it. One of the four fixtures is 243.6 MB (`cam_4k60_grain`, ~35x
//      its neighbours' bitrate), so this is not a rounding term.
//   2. `avcodec_send_packet` / `avcodec_receive_frame` — NVDEC itself.
//   3. our device->array copy, already split three ways by `CopyTiming`.
//
// So this probe replicates one frame of benchmark 7's decode work OUTSIDE the
// scheduler — four independent sources, one frame each, in the same serial order —
// and times (1) and (2+3) separately, per source, per frame, keeping ARRIVAL
// ORDER. That is the only view a period-2 cycle is visible in (gotcha 15), and
// splitting demux from decode is what decides whether the cycle is I/O or codec.
//
// It also settles Task C from the same series: the bench's per-source block reports
// `cam_4k60_grain` as the CHEAPEST of the four (2.96 ms/frame against 6.3-8.0),
// which contradicts gotcha 20's `grain_1080p60` finding. If that inversion is the
// demux exclusion above, this probe shows it as a large `demux` column on src 2 and
// gotcha 20's claim needs narrowing rather than a new mechanism.
//
// WHAT IT DELIBERATELY DOES NOT CONTAIN: no render graph, no NVENC session, no
// wgpu submission, no pipeline. If the cycle reproduces here it belongs to the
// demux/decode pair alone; if it does not, the cause is an interaction with the
// graph or the encoder and the next split is which of those.
//
// ── TASK G — the ceiling on a SECOND interop target per source ────────────────
//
// The fifth arm was added later, and it is the gate on Task G rather than another
// reading of Task B. Gotcha 24 attributes the cycle to the decoder's reorder
// buffer and closes with the one fix that does not change the workload: decode a
// frame BEFORE the frame that needs it, which needs a second `DecodeInteropTarget`
// per source (11.9 MB of VRAM each) so a decode can run while the graph still
// reads the previous frame.
//
// **That is an ownership refactor across `SourceTarget`, `held_pts`, `last_read`
// and gotcha 18's five rules, so it is measured here first.** The `PING-PONG` arm
// keeps TWO targets per source and drives them exactly as the refactor would:
// every step releases the target the previous frame was served from, decodes the
// NEXT frame into it while the frame being served stays resident in the other, and
// asserts that no decode ever writes the target the consumer still holds. Same
// four sources, same order, same serial loop — one argument different, the same
// discipline `--interop`'s two arms follow.
//
// What the arm can and cannot settle. It measures whether the DECODE SERIES
// itself changes shape when a decode no longer has to wait for its own target to
// be free. It cannot measure the overlap with graph work, because there is no
// graph here — which is what makes it the cheap gate: work per frame PAIR is
// fixed (gotcha 24), so if the parity split survives with nothing in the way, the
// ownership change cannot flatten it either and Task G stops at this reading.
//
// Every figure is timed with `Instant::now()` around the call it is attributed to.
// A missing capability is a printed skip with a reason, never a zero row (gotcha 9).
//
// Run:
//     cargo build -p nexir --example b_decode_split_probe --release
//     cp target/debug/*.dll target/release/examples/   # cuda.dll beside the binary
//     ./target/release/examples/b_decode_split_probe.exe
//
// With no arguments it uses the four `MULTI_4K60_SOURCES` fixtures, which are
// benchmark 7's own inputs. Pass paths to override.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use nexir::interop::capability::InteropCapability;
use nexir::interop::cuda_context::CudaContext;
use nexir::interop::decode_interop::DecodeInteropTarget;
use nexir::io::decoder::Decoder;
use nexir::io::demuxer::Demuxer;
use nexir::render::device::GpuDevice;

/// Frames per arm. Benchmark 7's own count, so the series is directly comparable
/// with `target/taskB_7_1.csv` rather than being a differently-shaped sample.
const FRAMES: usize = 90;

/// Untimed frames before the measured loop, matching the bench's warm-up in
/// purpose: the first frames out of a decoder carry the open, the keyframe and the
/// reorder-delay fill, and charging those to frame 0 would put a one-off cost in a
/// per-frame series that is being read for a *periodic* pattern.
const WARMUP: usize = 8;

/// Packets kept outstanding in the decoder's input queue by the third and fourth
/// arms.
///
/// Four rather than one because that is what the pipeline can absorb without
/// changing which frames are produced: a source's decoder emits pictures in
/// presentation order regardless of how many packets are queued behind the one
/// being waited for, and the fixtures' GOP is `I B B P B B P …` (probed with
/// `ffprobe -show_entries frame=pict_type`), so four covers a full B-frame group.
const QUEUE_DEPTH: usize = 4;

/// How one arm drives a source.
///
/// An enum rather than the old `Option<usize>` queue depth because the fifth arm
/// differs in what it OWNS rather than in how deep it queues, and the two are
/// independent: `PingPong` allocates a second `DecodeInteropTarget` per source and
/// alternates between them, which is the ownership change Task G would land in
/// `SourceTarget`.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ArmMode {
    /// One packet in, one frame out, into the source's single target — what
    /// `IoLayer` does today.
    Single,
    /// The same, with `n` packets outstanding in the decoder's input queue.
    Queued(usize),
    /// TWO targets per source: each step decodes the NEXT frame into the target
    /// the previous frame is NOT resident in. Task G's step 1.
    PingPong,
}

impl ArmMode {
    /// Targets to allocate per source on the interop arm.
    ///
    /// Counted rather than assumed, because it is also the VRAM figure: 11.9 MB per
    /// 4K pair, so four sources go 47.5 → 95.0 MB on the `PingPong` arm and the arm
    /// has to say so rather than have it discovered.
    fn targets_per_source(self) -> usize {
        match self {
            Self::PingPong => 2,
            _ => 1,
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration series"));
    Some(s[s.len() / 2])
}

/// One source: its own demuxer, decoder and (on the interop arm) its own target,
/// so N of these are N independent decode streams exactly as N timeline sources
/// are.
struct Source {
    label: String,
    demuxer: Demuxer,
    decoder: Decoder,
    /// The source's decode target(s). Empty on the CPU arm, one entry on the
    /// `Single`/`Queued` arms, TWO on `PingPong` — which is the whole difference
    /// Task G's step 1 measures.
    targets: Vec<DecodeInteropTarget>,
    /// Which target the NEXT decode writes into. Only moves on `PingPong`.
    write_slot: usize,
    /// Which target the frame the consumer is currently "holding" lives in, if any
    /// — i.e. the one a decode must NOT overwrite.
    ///
    /// **This is the invariant the arm exists to exercise**, and it is checked
    /// rather than assumed: the whole point of a second target is that a decode can
    /// proceed while the previous frame is still being read, so an arm that
    /// accidentally decoded into the held target would report a flattened series
    /// for the wrong reason (it would be the one-target arm with extra VRAM).
    held_slot: Option<usize>,
    /// Packets offered to the decoder that have not yet come back as pictures —
    /// the input queue depth, carried across frames by
    /// [`Self::one_frame_queued`]. Always 0 on the `one_frame` arms.
    sent: usize,
    /// Per-frame demux time, in arrival order.
    demux_ms: Vec<f64>,
    /// Per-frame decode time (send/receive plus, on the interop arm, the copy).
    decode_ms: Vec<f64>,
    /// Packets consumed per frame — a source needing two packets for one frame
    /// pays two demux reads, and a mean over frames would hide that.
    packets: Vec<usize>,
}

/// What one frame's worth of work on one source produced.
enum Step {
    /// A picture came out, with the two halves timed separately.
    Frame {
        demux: Duration,
        decode: Duration,
        packets: usize,
    },
    /// The file ran out of packets before a picture arrived.
    Eof,
}

impl Source {
    /// Feed packets until this source emits one picture, timing demux and decode
    /// separately.
    ///
    /// A loop rather than one packet per call for the reason
    /// `g2a_interop_cost_probe` documents: frame-level threading and B-frame
    /// reordering mean one packet in is not one frame out, so a fixed one-packet
    /// step would drift further behind every frame and never see a picture during
    /// the reorder fill.
    ///
    /// `engine` serialises the decode itself against other sources when the arm runs
    /// them on their own threads — see [`run_concurrent_arm`]. `None` on every
    /// single-threaded arm, where there is nothing to serialise against.
    fn one_frame(
        &mut self,
        cuda: Option<&CudaContext>,
        capability: &InteropCapability,
        dst: &mut [u8],
        engine: Option<&std::sync::Mutex<()>>,
    ) -> Result<Step, String> {
        let mut demux = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut packets = 0usize;
        loop {
            let t = Instant::now();
            let pkt = self
                .demuxer
                .next_video_packet()
                .map_err(|e| format!("{}: next_video_packet failed: {e:?}", self.label))?;
            demux += t.elapsed();
            let Some(pkt) = pkt else {
                return Ok(Step::Eof);
            };
            packets += 1;

            // The interop triple is assembled per call so this is the SAME call
            // `InteropDecodeTargets::decode_into_target` makes — including the
            // `cuCtxSynchronize` barrier and the two `cuMemcpy2DAsync` calls, which
            // are inside `emit_frame` rather than out here.
            //
            // `self.targets.get(self.write_slot)` rather than an accessor method:
            // `decode_into` needs `&mut self.decoder` at the same time, and only a
            // direct field borrow splits. On the ping-pong arm `write_slot` is the
            // target the previous frame is NOT resident in.
            let write_slot = self.write_slot;
            let interop = match (cuda, self.targets.get(write_slot)) {
                (Some(ctx), Some(target)) => Some((ctx, target, capability)),
                _ => None,
            };
            // Held across send/receive/copy, exactly as `decode_into_target` holds
            // the target map across the same call: the copy's pre-barrier is
            // `cuCtxSynchronize` and every source shares one context and one stream
            // (gotcha 4), so two sources copying concurrently is the thing that
            // design already refuses. Timed INSIDE the guard, so a producer's
            // `decode_ms` includes what it waited for — a figure that excluded the
            // queue would report a saturated engine as four cheap decodes.
            let _guard = engine.map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
            let t = Instant::now();
            let got = self
                .decoder
                .decode_into(&pkt, dst, interop)
                .map_err(|e| format!("{}: decode failed: {e:?}", self.label))?;
            decode += t.elapsed();
            drop(_guard);
            if got.is_some() {
                return Ok(Step::Frame {
                    demux,
                    decode,
                    packets,
                });
            }
        }
    }

    /// One frame, decoded into the target the previous frame is NOT resident in —
    /// **Task G's step 1.**
    ///
    /// This is [`Self::one_frame`] with the one thing the ownership change would
    /// add: a second `DecodeInteropTarget`, so the decode of frame N+1 does not
    /// have to wait for frame N to stop being read. Everything else — the same
    /// demuxer, the same decoder, the same `decode_into` with the same barrier and
    /// copies — is held fixed, which is what makes the two series comparable.
    ///
    /// **What this arm can and cannot show, because it is a weak test on purpose.**
    /// It is the plan's own step 1 taken literally, and single-threaded: nothing here
    /// *triggers* the decode of N+1 early, so the spare target is storage the loop
    /// never uses ahead of time. A flat result here would be strong evidence; an
    /// unchanged result is only evidence that STORAGE ALONE changes nothing, which is
    /// why [`run_concurrent_arm`] exists and why the verdict is read off that.
    ///
    /// **The invariant is checked, not assumed.** `held_slot` is the target holding
    /// the frame the consumer has in hand; writing into it would make this arm the
    /// one-target arm with extra VRAM, and the timings would not say so. A
    /// violation is an `Err` naming itself rather than an assertion, so a run that
    /// hits it reports the reason instead of dying with a slot index.
    fn one_frame_ping_pong(
        &mut self,
        cuda: Option<&CudaContext>,
        capability: &InteropCapability,
        dst: &mut [u8],
    ) -> Result<Step, String> {
        if self.targets.len() < 2 {
            return Err(format!(
                "{}: the ping-pong arm needs two targets and has {}",
                self.label,
                self.targets.len()
            ));
        }
        if self.held_slot == Some(self.write_slot) {
            return Err(format!(
                "{}: about to decode into target {} while the consumer still holds \
                 it — the arm would be measuring one target with twice the VRAM",
                self.label, self.write_slot
            ));
        }
        let step = self.one_frame(cuda, capability, dst, None)?;
        if let Step::Frame { .. } = step {
            // The frame just decoded is now the one being read, and the target it
            // did NOT land in is free for the next decode. That is the whole
            // mechanism: one frame of decode always has somewhere to go.
            self.held_slot = Some(self.write_slot);
            self.advance_slot();
        }
        Ok(step)
    }

    /// Move to the next target in round-robin order.
    ///
    /// Separate from [`Self::one_frame_ping_pong`] because the concurrent arm
    /// advances it too, but takes its "the previous frame has been released" signal
    /// from the consumer's credit rather than from single-threaded bookkeeping.
    fn advance_slot(&mut self) {
        if !self.targets.is_empty() {
            self.write_slot = (self.write_slot + 1) % self.targets.len();
        }
    }

    /// The same frame, with `depth` packets offered to the decoder BEFORE the first
    /// receive — the third arm's mechanism.
    ///
    /// **This is the candidate fix, and it is why `Decoder` grew
    /// `send_packet_only`/`receive_into`.** `one_frame` above gives the hardware an
    /// input queue of exactly one: it sends a packet, then immediately blocks in
    /// `avcodec_receive_frame` waiting for that packet's picture, and while it is
    /// copying the result nothing is decoding. This sends `depth` packets first and
    /// then receives, so the decoder has work queued behind the picture being
    /// waited for.
    ///
    /// The `sent` field carries the queue across frames: the pre-feed happens once,
    /// on the first frame, and after that one send per receive maintains the depth.
    /// Re-filling it every frame would be a growing queue rather than a fixed one.
    fn one_frame_queued(
        &mut self,
        cuda: Option<&CudaContext>,
        capability: &InteropCapability,
        dst: &mut [u8],
        depth: usize,
    ) -> Result<Step, String> {
        let mut demux = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut packets = 0usize;
        // The queue may have to go DEEPER than `depth` while the decoder's reorder
        // delay fills — it holds roughly one frame per thread before emitting
        // anything, so the first receives return EAGAIN however many packets are in.
        // Raising the ceiling by one per empty receive is what makes the fill
        // terminate; from then on the steady state is `depth`.
        let mut ceiling = depth.max(1);
        let mut eof = false;
        loop {
            while self.sent < ceiling && !eof {
                let t = Instant::now();
                let pkt = self
                    .demuxer
                    .next_video_packet()
                    .map_err(|e| format!("{}: next_video_packet failed: {e:?}", self.label))?;
                demux += t.elapsed();
                let Some(pkt) = pkt else {
                    // No more input. Fall through to the receive: the decoder still
                    // holds whatever was already offered.
                    eof = true;
                    break;
                };
                packets += 1;
                let t = Instant::now();
                self.decoder
                    .send_packet_only(&pkt)
                    .map_err(|e| format!("{}: send failed: {e:?}", self.label))?;
                decode += t.elapsed();
                self.sent += 1;
            }
            if self.sent == 0 {
                // Nothing offered and nothing left to offer. `drain_into` would find
                // the reorder tail; this probe measures the steady state, so running
                // dry is a clean stop (the same rule `--interop` follows).
                return Ok(Step::Eof);
            }

            // Same disjoint-field borrow as `one_frame`: `receive_into` takes
            // `&mut self.decoder`, so the target reference must come from the field
            // rather than from a method borrowing all of `self`.
            let write_slot = self.write_slot;
            let interop = match (cuda, self.targets.get(write_slot)) {
                (Some(ctx), Some(target)) => Some((ctx, target, capability)),
                _ => None,
            };
            let t = Instant::now();
            let got = self
                .decoder
                .receive_into(dst, interop)
                .map_err(|e| format!("{}: receive failed: {e:?}", self.label))?;
            decode += t.elapsed();
            if got.is_some() {
                // One picture out, so one packet's worth of queue is free again.
                self.sent -= 1;
                return Ok(Step::Frame {
                    demux,
                    decode,
                    packets,
                });
            }
            if eof {
                // Out of input and the decoder wants more: the rest of this file is
                // only reachable through the drain.
                return Ok(Step::Eof);
            }
            ceiling += 1;
        }
    }
}

/// One arm's readings: N sources driven together, one frame at a time.
struct ArmReading {
    label: &'static str,
    /// Whole-frame demux + decode, in arrival order — the quantity benchmark 7
    /// records as `PipelineStage::Decode`.
    ///
    /// **On the CONCURRENT arm this is the WALL interval of the frame set**, not a
    /// sum: see [`Self::concurrent`].
    frame_ms: Vec<f64>,
    /// The same series split into its two halves.
    frame_demux_ms: Vec<f64>,
    frame_decode_ms: Vec<f64>,
    sources: Vec<Source>,
    /// VRAM this arm's targets occupy — see [`arm_target_bytes`]. A lower bound,
    /// and printed as one.
    target_bytes: u64,
    /// True when the sources ran on their own threads.
    ///
    /// **Load-bearing for how the rows are read.** On a serial arm `frame_ms` IS
    /// `demux + decode` summed over the sources, because they happened one after
    /// another. On the concurrent arm `frame_ms` is the wall time of the whole frame
    /// set while the two halves still sum each source's own timings — so the halves
    /// EXCEED the frame whenever the sources overlapped, and that excess is the
    /// reading. Printing them the same way without saying so is how an overlap
    /// factor would be read as a bookkeeping error.
    concurrent: bool,
}

impl ArmReading {
    /// Mean of the even- and odd-indexed frames, separately — gotcha 15's rule.
    ///
    /// A distribution cannot see a cycle, and this is the split that showed the
    /// cycle is entirely in `Decode`. `None` below four samples, matching
    /// `ProfileReport::latency_alternation`: two means each drawn from one or two
    /// samples say nothing.
    fn parity(v: &[f64]) -> Option<(f64, f64)> {
        if v.len() < 4 {
            return None;
        }
        let even: Vec<f64> = v.iter().step_by(2).copied().collect();
        let odd: Vec<f64> = v.iter().skip(1).step_by(2).copied().collect();
        Some((mean(&even)?, mean(&odd)?))
    }

    /// How far apart the two parities are, as a percentage of the cheaper one.
    ///
    /// The verdict is computed from this rather than read off the printed table,
    /// because Task G's acceptance is a threshold (`ALTERNATING` is 15% in
    /// `format_table`) and an arm that "looks flatter" is not a reading.
    fn parity_gap_pct(v: &[f64]) -> Option<f64> {
        let (e, o) = Self::parity(v)?;
        let lo = e.min(o);
        let hi = e.max(o);
        (lo > 0.0).then(|| (hi - lo) / lo * 100.0)
    }

    fn print(&self) {
        println!("\n  ── {} ──", self.label);
        if self.frame_ms.is_empty() {
            println!("    no complete frames measured, so there is nothing to report");
            return;
        }
        println!(
            "    {} frame(s) measured, {:.2} ms/frame median ({:.2} ms mean){}",
            self.frame_ms.len(),
            median(&self.frame_ms).unwrap_or(f64::NAN),
            mean(&self.frame_ms).unwrap_or(f64::NAN),
            if self.concurrent {
                " — WALL interval of the frame set"
            } else {
                ""
            },
        );
        println!(
            "      of which demux {:.2} ms and decode {:.2} ms (means)",
            mean(&self.frame_demux_ms).unwrap_or(f64::NAN),
            mean(&self.frame_decode_ms).unwrap_or(f64::NAN),
        );
        if self.concurrent {
            // Said explicitly, because the two rows above are not additive here and a
            // reader who assumed they were would report the overlap as a bookkeeping
            // error. The ratio IS the overlap, so it is printed rather than left to
            // arithmetic.
            //
            // **AND ITS SCOPE LIMIT, which matters more than the number.** This arm
            // holds ONE engine mutex around every `decode_into` — deliberately, so the
            // two concurrent arms differ only in depth — so an overlap near 1.0x is
            // what this arm was BUILT to produce and is NOT evidence about whether
            // NVDEC parallelises across sources. Answering that needs an arm without
            // the lock, which cannot be written honestly here: without it every
            // source's context-wide `cuCtxSynchronize` waits for the others' decodes
            // too (gotcha 19), which inflates the per-source figures rather than
            // revealing concurrency. Quote this ratio only as "the lock did what it
            // says".
            let wall = mean(&self.frame_ms).unwrap_or(f64::NAN);
            let work = mean(&self.frame_demux_ms).unwrap_or(0.0)
                + mean(&self.frame_decode_ms).unwrap_or(0.0);
            println!(
                "      the halves are each source's OWN time summed ({work:.2} ms) against a \
                 {wall:.2} ms wall\n\
                 \x20     frame — {:.2}x overlap. NOT additive on this arm, and NOT a reading \
                 about\n\
                 \x20     NVDEC concurrency: one engine lock is held fixed across both \
                 concurrent arms.",
                if wall > 0.0 { work / wall } else { f64::NAN },
            );
        }
        // Counted, and labelled a lower bound — Task G's step 3. On the CPU arm
        // there are no targets at all, so this is 0 rather than absent.
        let targets: usize = self.sources.iter().map(|s| s.targets.len()).sum();
        println!(
            "      {} interop target(s), {:.1} MB VRAM (lower bound, counted from each \
             target's own geometry)",
            targets,
            self.target_bytes as f64 / (1024.0 * 1024.0),
        );

        // THE PARITY SPLIT — the whole point of the probe. Printed for the total
        // and for each half, because the question is not "does it alternate" (the
        // bench already answered that) but WHICH HALF does.
        println!("    parity split (even / odd, and the gap):");
        for (name, series) in [
            ("frame total", &self.frame_ms),
            ("  ↳ demux", &self.frame_demux_ms),
            ("  ↳ decode", &self.frame_decode_ms),
        ] {
            match Self::parity(series) {
                Some((e, o)) => {
                    let lo = e.min(o);
                    let hi = e.max(o);
                    let gap = if lo > 0.0 {
                        format!("{:.0}% apart", (hi - lo) / lo * 100.0)
                    } else {
                        "n/a".to_string()
                    };
                    println!("      {name:<14} {e:>8.2} ms / {o:>8.2} ms   {gap}");
                }
                None => println!("      {name:<14} n/a (fewer than 4 frames)"),
            }
        }

        println!("    per source (mean per frame, and its own parity split):");
        for s in &self.sources {
            let pk = mean(&s.packets.iter().map(|p| *p as f64).collect::<Vec<_>>());
            println!(
                "      {:<16} demux {:>7.2} ms   decode {:>7.2} ms   {:>4.2} packet(s)/frame",
                s.label,
                mean(&s.demux_ms).unwrap_or(f64::NAN),
                mean(&s.decode_ms).unwrap_or(f64::NAN),
                pk.unwrap_or(f64::NAN),
            );
            // PER SOURCE, because the aggregate cannot tell "every source
            // alternates" from "one source alternates and dominates the sum", and
            // the two lead opposite ways: the first is a property of a single
            // decoder's send/receive pattern, the second is one dear file.
            match Self::parity(&s.decode_ms) {
                Some((e, o)) => {
                    let lo = e.min(o);
                    let hi = e.max(o);
                    let gap = if lo > 0.0 {
                        format!("{:.0}% apart", (hi - lo) / lo * 100.0)
                    } else {
                        "n/a".to_string()
                    };
                    println!(
                        "                       ↳ decode parity {e:>7.2} ms / {o:>7.2} ms   {gap}"
                    );
                }
                None => println!("                       ↳ decode parity n/a (fewer than 4 frames)"),
            }
        }

        // Arrival order, because a periodic pattern is invisible in a summary.
        let show = self.frame_ms.len().min(24);
        println!("    first {show} frame(s), total (demux+decode) in ms:");
        print!("     ");
        for v in self.frame_ms.iter().take(show) {
            print!(" {v:>5.1}");
        }
        println!();
        print!("      demux ");
        for v in self.frame_demux_ms.iter().take(show) {
            print!(" {v:>5.1}");
        }
        println!();
        print!("      decode");
        for v in self.frame_decode_ms.iter().take(show) {
            print!(" {v:>5.1}");
        }
        println!();
    }
}

fn open_sources(
    paths: &[PathBuf],
    device: &GpuDevice,
    cuda: Option<&std::sync::Arc<CudaContext>>,
    capability: &InteropCapability,
    mode: ArmMode,
) -> Result<Vec<Source>, String> {
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let demuxer = Demuxer::open(path).map_err(|e| format!("Demuxer::open failed: {e:?}"))?;
        let stream = demuxer
            .video_stream
            .clone()
            .ok_or_else(|| format!("{} has no video stream", path.display()))?;
        let w = stream.width.ok_or("the stream has no width")?;
        let h = stream.height.ok_or("the stream has no height")?;
        // Hardware decode ENABLED, because benchmark 7's decoders have it and the
        // whole question is what an NVDEC surface costs to reach.
        let decoder = Decoder::open(&stream, stream.codecpar, true)
            .map_err(|e| format!("Decoder::open failed: {e:?}"))?;
        // One target per source normally, TWO on the ping-pong arm. The CPU arm gets
        // none, which is what keeps its `dst` slice load-bearing.
        let mut targets = Vec::new();
        if let Some(ctx) = cuda {
            for _ in 0..mode.targets_per_source() {
                targets.push(
                    DecodeInteropTarget::new(
                        std::sync::Arc::clone(ctx),
                        device,
                        capability.transport,
                        w,
                        h,
                    )
                    .map_err(|e| format!("DecodeInteropTarget::new failed: {e:?}"))?,
                );
            }
        }
        out.push(Source {
            label: path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string()),
            demuxer,
            decoder,
            targets,
            write_slot: 0,
            held_slot: None,
            sent: 0,
            demux_ms: Vec::with_capacity(FRAMES),
            decode_ms: Vec::with_capacity(FRAMES),
            packets: Vec::with_capacity(FRAMES),
        });
    }
    Ok(out)
}

/// Bytes of VRAM one arm's targets occupy, counted from the geometry each was
/// allocated at.
///
/// The same NV12 arithmetic `InteropDecodeTargets::target_bytes` uses and a LOWER
/// BOUND for the same reason (gotcha 9 / G2f.2): it counts what this process asked
/// for and knows nothing about alignment padding. Printed because the ping-pong arm
/// doubles it and Task G's step 3 says that must be stated in the row rather than
/// discovered.
fn arm_target_bytes(sources: &[Source]) -> u64 {
    sources
        .iter()
        .flat_map(|s| s.targets.iter())
        .map(|t| {
            let (w, h) = t.dimensions();
            let luma = w as u64 * h as u64;
            luma + (w as u64 / 2) * (h as u64 / 2) * 2
        })
        .sum()
}

/// One arm: N sources driven together, one frame at a time.
///
/// `mode` is [`ArmMode::Single`] for one packet in / one frame out into one target
/// — what `IoLayer` does today — [`ArmMode::Queued`] to keep packets outstanding in
/// the decoder's input queue, and [`ArmMode::PingPong`] for two targets per source.
fn run_arm(
    label: &'static str,
    paths: &[PathBuf],
    device: &GpuDevice,
    cuda: Option<&std::sync::Arc<CudaContext>>,
    capability: &InteropCapability,
    mode: ArmMode,
) -> Result<ArmReading, String> {
    let mut sources = open_sources(paths, device, cuda, capability, mode)?;
    let target_bytes = arm_target_bytes(&sources);

    // The CPU arm needs a real destination; the interop arm gets one byte, which is
    // the same evidence `g2a_interop_cost_probe` relies on — if the interop branch
    // is ever bypassed this panics on the slice bounds instead of quietly reporting
    // a CPU round-trip as a GPU one.
    let mut dst: Vec<u8> = if cuda.is_some() {
        vec![0u8; 1]
    } else {
        vec![0u8; 3840 * 2160 * 6]
    };

    // One closure, so the warm-up and the measured loop run the SAME code — the
    // reason `run_interop_pass` in the bench is written the same way.
    fn step(
        s: &mut Source,
        cuda: Option<&std::sync::Arc<CudaContext>>,
        capability: &InteropCapability,
        dst: &mut [u8],
        mode: ArmMode,
    ) -> Result<Step, String> {
        let ctx = cuda.map(|c| c.as_ref());
        match mode {
            ArmMode::Queued(d) => s.one_frame_queued(ctx, capability, dst, d),
            // Only meaningful with targets; on a CPU arm it degrades to `Single`
            // rather than failing, so the arm can be run for a control.
            ArmMode::PingPong if s.targets.len() >= 2 => {
                s.one_frame_ping_pong(ctx, capability, dst)
            }
            // No engine lock on a serial arm: there is nothing to serialise against,
            // and taking one would time a lock the production path does not have here.
            _ => s.one_frame(ctx, capability, dst, None),
        }
    }

    for _ in 0..WARMUP {
        for s in sources.iter_mut() {
            if let Step::Eof = step(s, cuda, capability, &mut dst, mode)? {
                return Err(format!("{} ran out of packets during the warm-up", s.label));
            }
        }
    }

    let mut frame_ms = Vec::with_capacity(FRAMES);
    let mut frame_demux_ms = Vec::with_capacity(FRAMES);
    let mut frame_decode_ms = Vec::with_capacity(FRAMES);

    'outer: for _ in 0..FRAMES {
        let mut demux_total = Duration::ZERO;
        let mut decode_total = Duration::ZERO;
        for s in sources.iter_mut() {
            match step(s, cuda, capability, &mut dst, mode)? {
                Step::Frame {
                    demux,
                    decode,
                    packets,
                } => {
                    s.demux_ms.push(ms(demux));
                    s.decode_ms.push(ms(decode));
                    s.packets.push(packets);
                    demux_total += demux;
                    decode_total += decode;
                }
                // A frame is all N pictures: one source running dry makes the frame
                // incomplete, and recording it would put a partial frame in a
                // per-frame series.
                Step::Eof => break 'outer,
            }
        }
        frame_demux_ms.push(ms(demux_total));
        frame_decode_ms.push(ms(decode_total));
        frame_ms.push(ms(demux_total + decode_total));
    }

    Ok(ArmReading {
        label,
        frame_ms,
        frame_demux_ms,
        frame_decode_ms,
        sources,
        target_bytes,
        concurrent: false,
    })
}

/// Benchmark 7's own four inputs.
fn default_fixtures() -> Vec<PathBuf> {
    let dir = nexir::bench_media::fixture_dir();
    nexir::bench_media::MULTI_4K60_SOURCES
        .iter()
        .map(|c| dir.join(c.file_name()))
        .filter(|p| p.exists())
        .collect()
}

/// One frame handed from a producer thread to the consumer.
///
/// Carries the slot so the consumer can return the credit that frees it — the
/// bookkeeping that makes "a decode may run while the previous frame is still held"
/// checkable rather than assumed.
struct Produced {
    slot: usize,
}

/// N sources on their own threads, with a credit depth equal to the TARGET count —
/// the arm Task G's claim actually needs.
///
/// **Why the single-threaded ping-pong arm is not enough.** That arm gives a source
/// a spare target and then never uses it early: the loop still decodes frame N+1
/// only after frame N has been consumed, so the second target is storage nothing
/// reaches for. The plan's mechanism is *"a decode can run while the graph still
/// reads the previous frame"*, and nothing in a serial loop can do that. Here each
/// source runs on its own thread and may be **`targets_per_source` frames ahead of
/// what the consumer has released** — with one target that is zero frames ahead
/// (starting the next decode would overwrite the texture the consumer holds), and
/// with two it is one frame. The two arms therefore differ by exactly the thing the
/// ownership change would add.
///
/// **What is held fixed, so the comparison is one change.** The engine mutex: every
/// `decode_into` runs under one lock, because `InteropDecodeTargets::decode_into_target`
/// serialises the copy behind the target map for a documented reason (one
/// `CUcontext`, one stream, a context-wide `cuCtxSynchronize` — gotcha 4), and the
/// four sources share one NVDEC engine anyway (gotcha 25). So this arm does NOT
/// claim concurrency makes the decode cheaper; total engine work per frame is
/// unchanged by construction. What it can move is *when* that work happens, which is
/// the only thing a second target could ever buy.
///
/// **The invariant is checked.** A producer refuses to decode into a slot the
/// consumer has not returned, and says so by name — otherwise a credit bug would
/// silently make the deep arm the shallow arm with more VRAM, and the timings would
/// look like a null result.
fn run_concurrent_arm(
    label: &'static str,
    paths: &[PathBuf],
    device: &GpuDevice,
    cuda: &std::sync::Arc<CudaContext>,
    capability: &InteropCapability,
    targets_per_source: usize,
) -> Result<ArmReading, String> {
    use std::sync::mpsc;

    let mode = if targets_per_source >= 2 {
        ArmMode::PingPong
    } else {
        ArmMode::Single
    };
    let sources = open_sources(paths, device, Some(cuda), capability, mode)?;
    let target_bytes = arm_target_bytes(&sources);
    for s in &sources {
        if s.targets.len() != targets_per_source {
            return Err(format!(
                "{}: asked for {targets_per_source} target(s) per source and got {}",
                s.label,
                s.targets.len()
            ));
        }
    }

    // One NVDEC engine, one CUDA context, one stream — so one lock, exactly as
    // production has. See this function's doc comment.
    let engine = std::sync::Mutex::new(());
    let n = sources.len();

    // Per source: frames out, credits back. A credit is permission to overwrite one
    // target, so `targets_per_source` of them are outstanding at the start and the
    // consumer returns one per frame it has finished with.
    let mut frame_rx = Vec::with_capacity(n);
    let mut credit_tx = Vec::with_capacity(n);

    let mut frame_ms: Vec<f64> = Vec::with_capacity(FRAMES);
    let mut short: Option<String> = None;

    let sources = std::thread::scope(|scope| -> Result<Vec<Source>, String> {
        let mut handles = Vec::with_capacity(n);
        for mut src in sources {
            let (f_tx, f_rx) = mpsc::channel::<Produced>();
            let (c_tx, c_rx) = mpsc::channel::<usize>();
            frame_rx.push(f_rx);
            credit_tx.push(c_tx);
            let engine = &engine;
            handles.push(scope.spawn(move || -> (Source, Option<String>) {
                // The interop arm never touches `dst`; one byte, so a bypassed
                // interop branch panics on the slice bounds instead of quietly
                // measuring a CPU round-trip (the same guard the serial arms use).
                let mut dst = [0u8; 1];
                // Slots the consumer has not released yet — the depth, and the whole
                // difference between the two concurrent arms. `targets_per_source`
                // outstanding is permitted, so one target means the next decode
                // cannot start until the consumer releases the frame it is holding
                // (0 frames ahead) and two means it can (1 frame ahead).
                let mut in_flight: std::collections::VecDeque<usize> =
                    std::collections::VecDeque::with_capacity(targets_per_source);
                let mut err = None;
                for i in 0..(WARMUP + FRAMES) {
                    // Block only at the depth limit. This is the ONE place a credit is
                    // consumed: taking one per iteration as well would consume two per
                    // frame against the consumer's one, and the arm would deadlock
                    // after the primed credits ran out rather than reporting anything.
                    let mut gone = false;
                    while in_flight.len() >= targets_per_source {
                        match c_rx.recv() {
                            Ok(released) => in_flight.retain(|s| *s != released),
                            // Consumer gone: a clean stop.
                            Err(_) => {
                                gone = true;
                                break;
                            }
                        }
                    }
                    if gone {
                        break;
                    }
                    // Anything already released, taken without blocking, so a producer
                    // that got ahead does not carry stale entries into the check below.
                    while let Ok(released) = c_rx.try_recv() {
                        in_flight.retain(|s| *s != released);
                    }
                    if in_flight.contains(&src.write_slot) {
                        err = Some(format!(
                            "{}: about to decode into target {} while the consumer \
                             still holds it — the arm would be measuring one target \
                             with {targets_per_source}x the VRAM",
                            src.label, src.write_slot
                        ));
                        break;
                    }
                    let step = match src.one_frame(
                        Some(cuda.as_ref()),
                        capability,
                        &mut dst,
                        Some(engine),
                    ) {
                        Ok(s) => s,
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    };
                    match step {
                        Step::Frame { demux, decode, packets } => {
                            if i >= WARMUP {
                                src.demux_ms.push(ms(demux));
                                src.decode_ms.push(ms(decode));
                                src.packets.push(packets);
                            }
                            let slot = src.write_slot;
                            in_flight.push_back(slot);
                            src.advance_slot();
                            if f_tx.send(Produced { slot }).is_err() {
                                break;
                            }
                        }
                        // Out of packets. The whole set is incomplete from here on,
                        // so stop and let the consumer see the closed channel — the
                        // same clean stop `--interop` takes.
                        Step::Eof => break,
                    }
                }
                (src, err)
            }));
        }

        // ── The consumer: collect one frame from every source, per frame ──────
        //
        // Takes them as fast as they arrive and does no work of its own, so the
        // interval it stamps is the interval at which four sources can DELIVER a
        // complete frame — which is the quantity benchmark 7's `Decode` column is,
        // and the one whose parity split is the cycle.
        //
        // No priming: a producer starts with nothing outstanding, so its depth is
        // exactly `targets_per_source` frames from the first one.
        let mut last = Instant::now();
        'sets: for i in 0..(WARMUP + FRAMES) {
            let mut got = Vec::with_capacity(n);
            for (s, rx) in frame_rx.iter().enumerate() {
                match rx.recv() {
                    Ok(p) => got.push((s, p.slot)),
                    Err(_) => {
                        if i < WARMUP {
                            short = Some(format!(
                                "source {s} ran out of packets during the warm-up"
                            ));
                        }
                        break 'sets;
                    }
                }
            }
            let now = Instant::now();
            if i >= WARMUP {
                frame_ms.push(ms(now.duration_since(last)));
            }
            last = now;
            // Release every slot this set occupied — the consumer is done reading.
            for (s, slot) in got {
                if credit_tx[s].send(slot).is_err() {
                    break 'sets;
                }
            }
        }
        // Dropping the credit senders is what lets a blocked producer exit.
        drop(credit_tx);
        drop(frame_rx);

        let mut out = Vec::with_capacity(n);
        for h in handles {
            let (src, err) = h.join().map_err(|_| "a producer thread panicked".to_string())?;
            if let Some(e) = err {
                return Err(e);
            }
            out.push(src);
        }
        Ok(out)
    })?;

    if let Some(why) = short {
        return Err(why);
    }

    // The two halves are each source's OWN timings summed, so on this arm they
    // exceed the wall interval whenever the sources overlapped — see
    // `ArmReading::concurrent`. Recorded rather than recomputed, so the printed rows
    // are the same quantities the serial arms print.
    let per_frame = |pick: fn(&Source) -> &Vec<f64>| -> Vec<f64> {
        let len = sources.iter().map(|s| pick(s).len()).min().unwrap_or(0);
        (0..len)
            .map(|i| sources.iter().map(|s| pick(s)[i]).sum())
            .collect()
    };
    let frame_demux_ms = per_frame(|s| &s.demux_ms);
    let frame_decode_ms = per_frame(|s| &s.decode_ms);

    Ok(ArmReading {
        label,
        frame_ms,
        frame_demux_ms,
        frame_decode_ms,
        sources,
        target_bytes,
        concurrent: true,
    })
}

/// The `ALTERNATING` threshold `ProfileReport::latency_alternation` uses, so the
/// verdict below is read against the same line the bench draws rather than a second
/// one that could drift from it.
const ALTERNATION_THRESHOLD_PCT: f64 = 15.0;

/// Task G's step 1, stated as a decision rather than left to the reader.
///
/// **The plan's own words: "If the parity split does not flatten there — with no
/// graph and no submission in the way — it will not flatten in the pipeline either,
/// and the task stops."** So this says which of those two outcomes the numbers are,
/// because the whole value of the arm is that it is cheap enough to run before an
/// ownership refactor and only useful if it is allowed to say no.
///
/// **Read off the CONCURRENT pair, not the serial one, and that distinction is the
/// point.** A serial loop cannot decode ahead however many targets it owns, so the
/// serial ping-pong arm can only ever show that storage alone changes nothing. The
/// concurrent pair differs by the depth a producer may run ahead of the consumer —
/// 1 target = 0 frames ahead, 2 = 1 frame ahead — which is exactly the mechanism the
/// ownership change would add. The serial rows are printed alongside as the control.
fn print_task_g_verdict(
    serial_single: Option<&ArmReading>,
    serial_ping: Option<&ArmReading>,
    conc_single: Option<&ArmReading>,
    conc_ping: Option<&ArmReading>,
) {
    println!("\n  ══ Task G step 1 — the ceiling on a second target per source ══");

    // The control first, and labelled as one: it is the plan's step 1 read
    // literally, and its null result is expected rather than informative.
    match (serial_single, serial_ping) {
        (Some(a), Some(b)) => {
            match (
                ArmReading::parity_gap_pct(&a.frame_decode_ms),
                ArmReading::parity_gap_pct(&b.frame_decode_ms),
            ) {
                (Some(x), Some(y)) => println!(
                    "    SERIAL control  : decode parity {x:.0}% → {y:.0}%   \
                     (a serial loop cannot decode ahead, so this arm tests STORAGE only)"
                ),
                _ => println!("    SERIAL control  : n/a (fewer than 4 frames on one arm)"),
            }
        }
        _ => println!("    SERIAL control  : n/a (an arm did not run)"),
    }

    let (Some(single), Some(ping)) = (conc_single, conc_ping) else {
        println!(
            "    NOT DECIDABLE: the concurrent arms are the ones that test the mechanism \
             and at\n\
             \x20   least one did not run. A verdict from the serial pair alone would be a \
             verdict\n\
             \x20   about storage, which is not what Task G claims."
        );
        return;
    };
    let (Some(a), Some(b)) = (
        ArmReading::parity_gap_pct(&single.frame_ms),
        ArmReading::parity_gap_pct(&ping.frame_ms),
    ) else {
        println!("    NOT DECIDABLE: fewer than 4 measured frames on one concurrent arm.");
        return;
    };

    // Read off the WALL interval, because that is the quantity the gate is stated
    // over (gotcha 15: a frame-time target is about when frames arrive) and the one
    // whose cycle a viewer sees. The per-source sums are printed too, since a change
    // in one without the other is itself a reading.
    println!(
        "    CONCURRENT      : frame-arrival parity {a:.0}% with ONE target/source  →  \
         {b:.0}% with TWO\n\
         \x20                    (threshold {ALTERNATION_THRESHOLD_PCT:.0}%; \
         1 target = 0 frames ahead, 2 = 1 frame ahead)"
    );
    println!(
        "    frame interval  : {:.2} ms → {:.2} ms mean, {:.2} → {:.2} ms median \
         ({} vs {} frames)",
        mean(&single.frame_ms).unwrap_or(f64::NAN),
        mean(&ping.frame_ms).unwrap_or(f64::NAN),
        median(&single.frame_ms).unwrap_or(f64::NAN),
        median(&ping.frame_ms).unwrap_or(f64::NAN),
        single.frame_ms.len(),
        ping.frame_ms.len(),
    );
    println!(
        "    decode per src  : {:.2} ms → {:.2} ms summed (engine work per frame, \
         unchanged by design)",
        mean(&single.frame_decode_ms).unwrap_or(f64::NAN),
        mean(&ping.frame_decode_ms).unwrap_or(f64::NAN),
    );
    println!(
        "    VRAM (lower bd) : {:.1} MB → {:.1} MB",
        single.target_bytes as f64 / (1024.0 * 1024.0),
        ping.target_bytes as f64 / (1024.0 * 1024.0),
    );
    // One invocation is one repeat. Said here rather than left to the reader,
    // because the plan's standing rule is a median of ≥3 with its spread and a
    // verdict quoted off a single run would breach it silently.
    println!(
        "    (one run per invocation — run this 3x and quote the median with its \
         spread before\n\
         \x20    acting on the verdict; the 4K rows span ~40% between repeats.)"
    );

    // The frame-time consequence, which is the only thing the gate is about. A
    // parity split that flattens while the mean rises is not a win, so both are
    // required before the verdict says "on evidence".
    let mean_single = mean(&single.frame_ms).unwrap_or(f64::NAN);
    let mean_ping = mean(&ping.frame_ms).unwrap_or(f64::NAN);
    let mean_ok = mean_ping <= mean_single * 1.05;

    if b <= ALTERNATION_THRESHOLD_PCT && mean_ok {
        println!(
            "    VERDICT: the arrival cycle FLATTENED below the threshold with the mean \
             held.\n\
             \x20            Task G's step 2 (the ownership change in `SourceTarget`, \
             `held_pts`\n\
             \x20            and `last_read` per target) is ON EVIDENCE. It must hold \
             gotcha 18's\n\
             \x20            five rules, and `peak bucket` must be re-read afterwards \
             (gotcha 14).\n\
             \x20            Note the ceiling: this arm has no graph, so it bounds what \
             step 2\n\
             \x20            can buy rather than predicting it."
        );
    } else if b < a * 0.5 && mean_ok {
        println!(
            "    VERDICT: the cycle MOVED but did not clear the threshold. A second \
             target is\n\
             \x20            part of the answer and not all of it — and this arm has no \
             graph in\n\
             \x20            the way, so the pipeline cannot improve on what is shown \
             here."
        );
    } else if b <= ALTERNATION_THRESHOLD_PCT {
        println!(
            "    VERDICT: the cycle flattened but the MEAN interval rose \
             ({mean_single:.2} → {mean_ping:.2} ms).\n\
             \x20            That is a different workload, not a smoother one: a frame-time \
             budget\n\
             \x20            is not met by spacing frames out. Step 2 is NOT on evidence."
        );
    } else {
        println!(
            "    VERDICT: the cycle SURVIVED two targets per source, with the producers \
             free to\n\
             \x20            run a frame ahead and no graph, no submission and no \
             encoder in the\n\
             \x20            way. Work per frame PAIR is fixed (gotcha 24), so a second \
             target can\n\
             \x20            only move WHEN that work happens — and given the chance to, \
             it did\n\
             \x20            not. **Task G stops at this reading.** The ownership change \
             cannot\n\
             \x20            flatten what this arm could not, and it would cost 11.9 MB \
             of VRAM\n\
             \x20            per 4K source for the refactor's risk. Say so in the audit \
             rather\n\
             \x20            than moving the gate."
        );
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!(
        "\n=== Task B — which half of benchmark 7's `Decode` column alternates? ===\n\n\
         Benchmark 7's frames arrive 16.66 / 49.06 ms in a two-frame cycle and the whole\n\
         swing is in `Decode` (40.46 / 8.33 ms) while the graph is flat. `Decode` there is\n\
         all of `schedule_frame`, which is demux + NVDEC + our copy; the bench's per-source\n\
         counters start AFTER the packet is read, so the demux is unmeasured. This probe\n\
         drives the same four sources with no graph, no NVENC and no pipeline, and splits\n\
         demux from decode per frame in arrival order.\n\n\
         The fifth arm is TASK G STEP 1: the same decode with TWO interop targets per\n\
         source, which is the ceiling on the ownership change before it is written.\n"
    );

    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<PathBuf> = if args.is_empty() {
        let found = default_fixtures();
        if found.len() != nexir::bench_media::MULTI_4K60_SOURCES.len() {
            println!(
                "SKIP: found {} of {} MULTI_4K60_SOURCES fixture(s) in {}. Generate them with\n\
                 \n    ./target/release/bench.exe --media\n\n\
                 or pass paths. A run on fewer sources is a different workload from\n\
                 benchmark 7 and its parity split could not be read against it.",
                found.len(),
                nexir::bench_media::MULTI_4K60_SOURCES.len(),
                nexir::bench_media::fixture_dir().display(),
            );
            return;
        }
        println!(
            "Using benchmark 7's own {} fixture(s) in {}\n",
            found.len(),
            nexir::bench_media::fixture_dir().display()
        );
        found
    } else {
        args.iter().map(PathBuf::from).collect()
    };
    for p in &paths {
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!(
            "  {:<20} {:>9.1} MB",
            p.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            size as f64 / (1024.0 * 1024.0)
        );
    }

    let device = match pollster::block_on(GpuDevice::new_headless()) {
        Ok(d) => d,
        Err(e) => {
            println!("\nSKIP: no GPU device ({e:?}) — nothing here can be measured.");
            return;
        }
    };
    let capability = InteropCapability::probe(&device);
    println!(
        "\nCUDA interop : {}",
        if capability.is_available() {
            format!("available (transport={:?})", capability.transport)
        } else {
            "UNAVAILABLE — only the CPU arm runs".to_string()
        }
    );
    let cuda = if capability.is_available() {
        match CudaContext::new(&capability) {
            Ok(c) => Some(std::sync::Arc::new(c)),
            Err(e) => {
                println!("CUDA context : failed ({e:?}) — only the CPU arm runs");
                None
            }
        }
    } else {
        None
    };

    // The interop arm FIRST, because it is benchmark 7's path and the one the cycle
    // was measured on. The CPU arm is benchmark 8's, which alternates mildly, so
    // running both says whether the cycle is a property of the interop copy.
    //
    // Kept for the Task G verdict below: the single-target interop arm is the
    // BASELINE the ping-pong arm is read against, and comparing against a remembered
    // figure from a previous run would be comparing two different machine states.
    let mut interop_single: Option<ArmReading> = None;
    if let Some(ctx) = &cuda {
        match run_arm(
            "INTEROP decode, one packet in / one frame out (benchmark 7's path)",
            &paths,
            &device,
            Some(ctx),
            &capability,
            ArmMode::Single,
        ) {
            Ok(r) => {
                r.print();
                interop_single = Some(r);
            }
            Err(e) => println!("\n  INTEROP arm FAILED: {e}"),
        }
    } else {
        println!("\n  INTEROP arm skipped: no CUDA context, so its cycle cannot be reproduced");
    }
    match run_arm(
        "CPU round-trip, one packet in / one frame out (benchmark 8's path)",
        &paths,
        &device,
        None,
        &capability,
        ArmMode::Single,
    ) {
        Ok(r) => r.print(),
        Err(e) => println!("\n  CPU arm FAILED: {e}"),
    }

    // ── The third arm: the same decode with a DEEPER input queue ──────────────
    //
    // The candidate fix, on the same sources in the same order, differing by one
    // argument — the same discipline `--interop`'s two arms follow. If the cycle is
    // the hardware being starved between a send and its own receive, this removes it
    // without changing the frames produced; if the cycle survives, the queue depth is
    // not the mechanism and the next split is inside NVDEC.
    if let Some(ctx) = &cuda {
        match run_arm(
            "INTEROP decode, QUEUED input (4 packets outstanding)",
            &paths,
            &device,
            Some(ctx),
            &capability,
            ArmMode::Queued(QUEUE_DEPTH),
        ) {
            Ok(r) => r.print(),
            Err(e) => println!("\n  QUEUED INTEROP arm FAILED: {e}"),
        }
    }
    match run_arm(
        "CPU round-trip, QUEUED input (4 packets outstanding)",
        &paths,
        &device,
        None,
        &capability,
        ArmMode::Queued(QUEUE_DEPTH),
    ) {
        Ok(r) => r.print(),
        Err(e) => println!("\n  QUEUED CPU arm FAILED: {e}"),
    }

    // ── The fifth arm: TWO interop targets per source — Task G's step 1 ───────
    //
    // The plan's step 1 read literally, and kept as the CONTROL rather than the
    // answer: nothing but the target count differs from the first arm, and a serial
    // loop never decodes ahead however many targets it owns. So a null result here
    // means "storage alone changes nothing", which is not the claim Task G makes.
    // The pair that tests the claim is below.
    let mut serial_ping: Option<ArmReading> = None;
    if let Some(ctx) = &cuda {
        match run_arm(
            "INTEROP decode, PING-PONG targets (two per source), SERIAL — Task G control",
            &paths,
            &device,
            Some(ctx),
            &capability,
            ArmMode::PingPong,
        ) {
            Ok(r) => {
                r.print();
                serial_ping = Some(r);
            }
            Err(e) => println!("\n  SERIAL PING-PONG arm FAILED: {e}"),
        }
    } else {
        println!(
            "\n  PING-PONG arms skipped: no CUDA context. Task G's step 1 needs real \
             `DecodeInteropTarget`s, so this is a printed skip rather than a zero row."
        );
    }

    // ── The sixth and seventh arms: the SAME depth question, CONCURRENTLY ─────
    //
    // **These are the arms Task G's claim needs**, and the difference between them is
    // the one thing the ownership change would add: how far a source's decoder may run
    // ahead of the frame the consumer still holds. One target per source is zero
    // frames ahead — starting the next decode would overwrite the texture being read,
    // which is precisely the hazard `SourceTarget::last_read` guards today. Two is one
    // frame ahead.
    //
    // Everything else is held: the same four fixtures in the same order, the same
    // `decode_into` with the same barrier and copies, and the same single engine lock
    // (one NVDEC engine, one CUDA context, one stream — gotchas 4 and 25), so the
    // total decode work per frame cannot change and only its TIMING can.
    let mut conc_single: Option<ArmReading> = None;
    let mut conc_ping: Option<ArmReading> = None;
    if let Some(ctx) = &cuda {
        match run_concurrent_arm(
            "INTEROP decode, CONCURRENT sources, ONE target each (0 frames ahead)",
            &paths,
            &device,
            ctx,
            &capability,
            1,
        ) {
            Ok(r) => {
                r.print();
                conc_single = Some(r);
            }
            Err(e) => println!("\n  CONCURRENT 1-target arm FAILED: {e}"),
        }
        match run_concurrent_arm(
            "INTEROP decode, CONCURRENT sources, TWO targets each (1 frame ahead) — Task G",
            &paths,
            &device,
            ctx,
            &capability,
            2,
        ) {
            Ok(r) => {
                r.print();
                conc_ping = Some(r);
            }
            Err(e) => println!("\n  CONCURRENT 2-target arm FAILED: {e}"),
        }
    }

    print_task_g_verdict(
        interop_single.as_ref(),
        serial_ping.as_ref(),
        conc_single.as_ref(),
        conc_ping.as_ref(),
    );

    println!(
        "\nHow to read this. `↳ demux` is stable on every arm at ~1.6 ms/frame, so the\n\
         cycle is NOT reading the bitstream. If the QUEUED arms flatten while producing\n\
         the same number of frames, the cycle is NVDEC being starved between a send and\n\
         its own receive, and the fix belongs in `IoLayer`'s read-forward loop rather\n\
         than anywhere in the graph. If they alternate too, the queue depth is not the\n\
         mechanism.\n\n\
         The PING-PONG and CONCURRENT arms are Task G's step 1, and they answer a\n\
         different question: not WHERE the cycle is (gotcha 24 settled that — the\n\
         decoder's reorder buffer) but whether a second target per source can smooth it.\n\
         The serial ping-pong arm is only a control — a serial loop never decodes ahead,\n\
         so it tests storage. The two CONCURRENT arms differ by how far a producer may\n\
         run ahead of the frame the consumer holds (0 frames with one target, 1 with\n\
         two), which is the mechanism itself, and the verdict block is read off them.\n\
         It is allowed to say no.\n\n\
         Task C, from the per-source rows: the bench's `Interop:` block times only what\n\
         is inside `decode_into_target`, which starts AFTER the packet is in hand — so\n\
         `cam_4k60_grain`'s demux (1.5 ms against 0.04 for the other three, its file\n\
         being ~35x their size) is invisible there. That is why the bench reports it as\n\
         the CHEAPEST of the four while single-source runs of this probe show it as by\n\
         far the DEAREST (17.0 ms/frame against 4.8-5.1).\n"
    );
}
