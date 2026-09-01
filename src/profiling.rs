// src/profiling/mod.rs

/// OS-level process metrics (RSS, CPU utilisation). Separate file because it is
/// the only `#[cfg(windows)]` FFI in the profiler and everything else here is
/// portable arithmetic.
pub mod sysinfo;

/// Native FFI for the driver-only metrics: NVML for GPU/NVENC utilisation and VRAM.
/// Layout constants are established by `nvchk/nvml_probe.c`.
pub mod ffi;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Discrete pipeline stages measured during frame processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineStage {
    Decode,
    /// Handing already-decoded pixel data to the GPU (`queue.write_texture`).
    ///
    /// Distinct from [`Self::Decode`] on purpose: a caller that feeds the graph
    /// pre-made sample data is measuring an upload, and recording that under
    /// `Decode` would report a decode cost for work no decoder did.
    Upload,
    /// The CPU-side half of an upload: building row-padded staging buffers.
    ///
    /// Split out from [`Self::Upload`] because the two have completely different
    /// cures. At 4K this was the majority of a 20 ms upload stage — heap
    /// allocation and memcpy, not PCIe transfer — and no amount of transfer
    /// tuning would have touched it.
    UploadPrepare,
    /// The transfer-submission half of an upload: `queue.write_buffer`.
    UploadSubmit,
    /// GPU-timeline time between one frame's graph finishing and the next
    /// frame's graph starting.
    ///
    /// **This stage has no CPU time and never will**, which is the point. wgpu's
    /// `queue.write_buffer` does not touch the destination buffer from the CPU: it
    /// memcpys into an internal staging buffer and records a
    /// `copy_buffer_to_buffer` into its own `pending_writes` encoder, which
    /// `pre_submit()` prepends to the *next* `queue.submit` (wgpu-core-0.19.4
    /// `device/queue.rs:231-242`, `:1443`). Those copies therefore execute in a
    /// command buffer this crate never encodes and cannot bracket — they land
    /// between the previous frame's closing timestamp and this frame's opening
    /// one.
    ///
    /// Before this stage existed, that time was known only by subtraction: 18.0 ms
    /// of wall interval minus 8.2 ms of bracketed graph minus 5.0 ms of CPU work
    /// left "about 9.8 ms, probably the upload". This measures it.
    GpuTransfer,
    Scheduler,
    YuvToRgb,
    Effects,
    Composite,
    GpuSubmit,
    GpuWait,
    Nvenc,
    Mux,
    AudioDecode,
    AudioMix,
}

impl PipelineStage {
    pub fn all() -> &'static [PipelineStage] {
        &[
            PipelineStage::Decode,
            PipelineStage::Upload,
            PipelineStage::UploadPrepare,
            PipelineStage::UploadSubmit,
            PipelineStage::GpuTransfer,
            PipelineStage::Scheduler,
            PipelineStage::YuvToRgb,
            PipelineStage::Effects,
            PipelineStage::Composite,
            PipelineStage::GpuSubmit,
            PipelineStage::GpuWait,
            PipelineStage::Nvenc,
            PipelineStage::Mux,
            PipelineStage::AudioDecode,
            PipelineStage::AudioMix,
        ]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            PipelineStage::Decode => "Decode",
            PipelineStage::Upload => "Upload",
            PipelineStage::UploadPrepare => "  ↳ prepare",
            PipelineStage::UploadSubmit => "  ↳ submit",
            PipelineStage::GpuTransfer => "GPU transfer",
            PipelineStage::Scheduler => "Scheduler",
            PipelineStage::YuvToRgb => "YUV → RGB",
            PipelineStage::Effects => "Effects",
            PipelineStage::Composite => "Composite",
            PipelineStage::GpuSubmit => "GPU submit",
            PipelineStage::GpuWait => "GPU wait",
            PipelineStage::Nvenc => "NVENC",
            PipelineStage::Mux => "Mux",
            PipelineStage::AudioDecode => "Audio Decode",
            PipelineStage::AudioMix => "Audio Mix",
        }
    }

    /// Whether this stage is a **breakdown** of another one rather than a
    /// distinct slice of the frame.
    ///
    /// `UploadPrepare` + `UploadSubmit` decompose `Upload`; all three are
    /// recorded, so summing every stage would count the upload twice and report
    /// a frame time longer than the frame. [`FrameProfile::total_time`] skips
    /// these, and the report indents them under their parent.
    pub fn is_breakdown(&self) -> bool {
        matches!(
            self,
            PipelineStage::UploadPrepare | PipelineStage::UploadSubmit
        )
    }
}

/// Per-frame timing profile capturing duration for each pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct FrameProfile {
    pub frame_index: usize,
    pub stage_durations: BTreeMap<PipelineStage, Duration>,
    /// GPU **execution** time per stage, from timestamp queries
    /// ([`crate::render::gpu_timer::GpuTimer`]).
    ///
    /// Deliberately a separate map, and deliberately NOT part of
    /// [`Self::total_time`]: CPU and GPU time overlap by construction once
    /// frames are in flight — the CPU records frame N+1 while the GPU executes
    /// frame N — so a sum of the two is a number no clock ever saw.
    ///
    /// Empty when the device lacks `TIMESTAMP_QUERY`, or for any stage nobody
    /// bracketed. An absent entry reports as `n/a`, never as `0.00 ms`.
    pub gpu_durations: BTreeMap<PipelineStage, Duration>,
    /// Wall-clock interval between this frame retiring and the one before it.
    ///
    /// **This is the quantity a "P95 ≤ 16.67 ms" target is about**, and it is not
    /// [`Self::total_time`]. Once frames are in flight the CPU's per-frame work
    /// stops being a frame duration: the 4K row reads 5.0 ms of CPU stages while
    /// frames actually arrive 18.0 ms apart, so a percentile over the CPU sum
    /// answers a question nobody asked. It also cuts the other way — a pipeline
    /// deep enough to absorb a 50 ms CPU hiccup still delivers every frame on
    /// time, and reporting that hiccup as the frame's P99 claims a stall the
    /// viewer never saw.
    ///
    /// `None` on the first frame (there is no interval before it) and whenever
    /// nobody stamped it; absent reports as `n/a`, never as `0.00 ms`.
    pub latency: Option<Duration>,
    pub frame_pts: i64,
}

impl FrameProfile {
    pub fn new(frame_index: usize, frame_pts: i64) -> Self {
        Self {
            frame_index,
            stage_durations: BTreeMap::new(),
            gpu_durations: BTreeMap::new(),
            latency: None,
            frame_pts,
        }
    }

    /// Record the elapsed time for a given stage.
    pub fn record_stage(&mut self, stage: PipelineStage, duration: Duration) {
        *self.stage_durations.entry(stage).or_default() += duration;
    }

    /// Record GPU execution time for a stage, as measured by timestamp queries.
    ///
    /// Accumulates like [`Self::record_stage`], so a stage spanning several
    /// nodes (four LUT passes, say) can be recorded per node and read as a
    /// total.
    pub fn record_gpu(&mut self, stage: PipelineStage, duration: Duration) {
        *self.gpu_durations.entry(stage).or_default() += duration;
    }

    /// Execute a closure and measure its duration under the specified stage.
    pub fn measure<F, R>(&mut self, stage: PipelineStage, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.record_stage(stage, start.elapsed());
        result
    }

    /// Total measured time for this frame across all recorded stages.
    ///
    /// Breakdown stages are excluded — see [`PipelineStage::is_breakdown`].
    /// Counting `UploadPrepare` alongside the `Upload` it decomposes would
    /// double-count and report a frame longer than the frame took.
    pub fn total_time(&self) -> Duration {
        self.stage_durations
            .iter()
            .filter(|(stage, _)| !stage.is_breakdown())
            .map(|(_, d)| *d)
            .sum()
    }

    /// Total measured time in milliseconds.
    pub fn total_time_ms(&self) -> f64 {
        self.total_time().as_secs_f64() * 1000.0
    }

    /// Duration of a specific stage in milliseconds.
    pub fn stage_time_ms(&self, stage: PipelineStage) -> f64 {
        self.stage_durations
            .get(&stage)
            .copied()
            .unwrap_or(Duration::ZERO)
            .as_secs_f64()
            * 1000.0
    }

    /// GPU execution time for a stage in milliseconds, or `None` when it was not
    /// measured.
    ///
    /// `Option` rather than a defaulted zero on purpose — see
    /// [`Self::gpu_durations`].
    pub fn gpu_time_ms(&self, stage: PipelineStage) -> Option<f64> {
        self.gpu_durations
            .get(&stage)
            .map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Whether this frame carries any GPU measurement at all.
    pub fn has_gpu_times(&self) -> bool {
        !self.gpu_durations.is_empty()
    }

    /// Record the wall-clock interval since the previous frame retired.
    ///
    /// See [`Self::latency`]: this is the frame *interval*, the quantity the
    /// P95/P99 targets are stated in, and it is deliberately not derived from the
    /// stage timings.
    pub fn record_latency(&mut self, interval: Duration) {
        self.latency = Some(interval);
    }

    /// Frame interval in milliseconds, or `None` when unmeasured.
    pub fn latency_ms(&self) -> Option<f64> {
        self.latency.map(|d| d.as_secs_f64() * 1000.0)
    }
}

/// Scoped RAII timer for a pipeline stage.
pub struct StageTimer {
    stage: PipelineStage,
    start: Instant,
}

impl StageTimer {
    pub fn start(stage: PipelineStage) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }

    pub fn stop_and_record(self, profile: &mut FrameProfile) {
        let elapsed = self.start.elapsed();
        profile.record_stage(self.stage, elapsed);
    }
}

/// Aggregate statistics for a single pipeline stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageStats {
    pub count: usize,
    pub min_ms: f64,
    pub max_ms: f64,
    pub avg_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub total_ms: f64,
    pub percentage_of_total: f64,
    /// GPU execution time for this stage, when timestamp queries measured it.
    ///
    /// `None` means unmeasured — no `TIMESTAMP_QUERY` on the device, or nothing
    /// bracketed this stage. It does not mean the GPU did no work.
    pub gpu_avg_ms: Option<f64>,
    pub gpu_p95_ms: Option<f64>,
    pub gpu_p99_ms: Option<f64>,
}

/// Resource counters that accompany a [`ProfileReport`].
///
/// Every field is an `Option` and every one of them is `None` until something
/// **measures** it. That is deliberate, and it is the whole design of this
/// struct: it used to be a bag of `f32`/`u64` that a caller filled with plausible
/// constants (`gpu_utilization: 88.5`, `ram_used_bytes: 420 MB`), which the
/// report then printed indistinguishably from a real reading. A number that was
/// never measured is not a small inaccuracy in a benchmark — it is the benchmark
/// reporting a result it does not have.
///
/// So: fill a field only from a real measurement, leave the rest `None`, and
/// [`ProfileReport::format_table`] prints `n/a` for them. Adding a driver query
/// (NVML for GPU/NVENC utilisation, `cuMemGetInfo` for VRAM, an OS call for RAM)
/// means setting the corresponding field; nothing else has to change.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemMetrics {
    /// GPU core utilization percentage (0.0 .. 100.0).
    ///
    /// Requires a driver query (NVML `nvmlDeviceGetUtilizationRates`); wgpu
    /// exposes nothing equivalent, so this stays `None` in-process.
    pub gpu_utilization: Option<f32>,
    /// CPU core utilization percentage (0.0 .. 100.0). Needs an OS-specific
    /// query (`GetSystemTimes` / `/proc/stat`).
    pub cpu_utilization: Option<f32>,
    /// NVENC encoder hardware utilization percentage (0.0 .. 100.0). NVML's
    /// `nvmlDeviceGetEncoderUtilization`.
    pub nvenc_utilization: Option<f32>,
    /// VRAM in use, as reported by the driver (`cuMemGetInfo`) — NOT a sum of
    /// what the caller believes it allocated. For that, see
    /// [`Self::allocated_gpu_bytes`].
    pub vram_used_bytes: Option<u64>,
    /// Bytes of GPU memory the caller itself allocated, computed from its own
    /// texture and buffer sizes.
    ///
    /// Honest but narrow: it counts what the caller asked for and knows nothing
    /// about the render graph's internal pool, alignment padding, or driver
    /// overhead, so it is a lower bound on real VRAM use rather than a
    /// measurement of it.
    pub allocated_gpu_bytes: Option<u64>,
    /// Process resident set size in bytes, from an OS query.
    pub ram_used_bytes: Option<u64>,
    /// Bytes uploaded CPU → GPU, counted by the caller as it uploads.
    pub gpu_upload_bytes: Option<u64>,
    /// Bytes read back GPU → CPU, counted by the caller as it reads back.
    pub gpu_download_bytes: Option<u64>,
    /// Frame queue depth, from the queue itself.
    pub frame_queue_depth: Option<usize>,
    /// Decode queue depth, from the queue itself.
    pub decode_queue_depth: Option<usize>,
    /// Encode queue depth, from the queue itself.
    pub encode_queue_depth: Option<usize>,
}

impl SystemMetrics {
    /// The `System Metrics` block of a report table.
    ///
    /// Unmeasured fields print `n/a` rather than a zero: `0.0%` GPU utilisation
    /// and "not measured" are different claims, and a reader has no way to tell
    /// them apart once both are printed as a number.
    pub fn format_block(&self) -> String {
        fn pct(v: Option<f32>) -> String {
            v.map(|x| format!("{x:>5.1}%")).unwrap_or_else(|| format!("{:>6}", "n/a"))
        }
        fn mb(v: Option<u64>) -> String {
            v.map(|b| format!("{:>7.1} MB", b as f64 / (1024.0 * 1024.0)))
                .unwrap_or_else(|| format!("{:>10}", "n/a"))
        }
        fn depth(v: Option<usize>) -> String {
            v.map(|d| d.to_string()).unwrap_or_else(|| "n/a".into())
        }

        let mut out = String::from("System Metrics (n/a = not measured, never estimated):\n");
        out.push_str(&format!(
            "  GPU Util:   {}  |  NVENC Util: {}  |  CPU Util:   {}\n",
            pct(self.gpu_utilization),
            pct(self.nvenc_utilization),
            pct(self.cpu_utilization),
        ));
        out.push_str(&format!(
            "  VRAM (driver): {} |  RAM (RSS): {}\n",
            mb(self.vram_used_bytes),
            mb(self.ram_used_bytes),
        ));
        if let Some(allocated) = self.allocated_gpu_bytes {
            out.push_str(&format!(
                "  GPU bytes this run allocated itself: {} (lower bound; excludes \
                 the graph's texture pool)\n",
                mb(Some(allocated)),
            ));
        }
        out.push_str(&format!(
            "  Transfers: up {} | down {}\n",
            mb(self.gpu_upload_bytes),
            mb(self.gpu_download_bytes),
        ));
        out.push_str(&format!(
            "  Queue Depths: Frame={}, Decode={}, Encode={}\n",
            depth(self.frame_queue_depth),
            depth(self.decode_queue_depth),
            depth(self.encode_queue_depth),
        ));
        out
    }
}

/// Summary report over a sequence of profiled frames.
#[derive(Debug, Clone, Default)]
pub struct ProfileReport {
    pub total_frames: usize,
    pub total_wall_time: Duration,
    /// Throughput: frames divided by elapsed wall time.
    ///
    /// **This is the honest answer to "how fast is it".** It counts real elapsed
    /// seconds, so it cannot be improved by moving work off the measured path.
    ///
    /// Use this, not [`Self::cpu_bound_fps`], for any claim about performance.
    pub average_fps: f64,
    /// What the frame rate would be if the CPU's own per-frame work were the only
    /// limit: `1000 / (sum of CPU stages)`.
    ///
    /// Equal to [`Self::average_fps`] in a serial pipeline, and **far higher than
    /// it** once frames overlap — because then the CPU finishes its part of frame
    /// N+1 while the GPU is still executing frame N, so the CPU sum stops being a
    /// frame duration.
    ///
    /// Kept because the gap between the two is the useful diagnostic: it says how
    /// much headroom the CPU has. But it is a ceiling, not a throughput, and
    /// reporting it as the latter is how a pipelined benchmark claims 237 FPS on a
    /// machine delivering 58.
    pub cpu_bound_fps: f64,
    pub p99_fps: f64,
    pub realtime_factor: f64,
    pub stage_stats: BTreeMap<PipelineStage, StageStats>,
    /// Distribution of the CPU's per-frame work: the sum of one frame's CPU
    /// stages.
    ///
    /// **Not a frame duration once frames overlap.** For the interval frames
    /// actually arrive at — and therefore for any P95/P99 target — use
    /// [`Self::frame_latency_stats`].
    pub total_frame_stats: StageStats,
    /// Distribution of the wall-clock interval between frames retiring.
    ///
    /// **Whole-run**, including the pipeline-fill interval — which is what keeps
    /// its mean comparable with [`Self::average_fps`], since both then cover the
    /// same seconds. For the percentiles a 60 FPS target is read off, use
    /// [`Self::steady_latency_stats`]: on a 90-frame run the P99 index resolves to
    /// the largest sample, and the largest sample is the fill.
    ///
    /// `None` when nothing recorded a latency, which is the honest answer for a
    /// caller that never stamped one: a P95 of `0.00 ms` and "we did not measure
    /// the frame interval" are different claims.
    pub frame_latency_stats: Option<StageStats>,
    /// The first stamped interval: the pipeline **fill**, from the loop starting to
    /// the first frame retiring.
    ///
    /// Reported on its own because it is a startup cost of a different kind from
    /// everything after it — it contains the submit work for the first
    /// `gpu_lookahead` frames, the first NVENC picture, and any first-frame
    /// allocation, none of which recurs. Measured at 4K: 67-91 ms against a 20 ms
    /// steady-state mean.
    ///
    /// It is **kept in the series** rather than dropped, because dropping it is a
    /// bug this project already had once — an unseeded `last_retire` silently
    /// omitted it and the mean then read 18.25 ms on a run delivering a frame every
    /// 21.1 ms. Separated and printed, not discarded.
    pub pipeline_fill_ms: Option<f64>,
    /// Distribution of the frame interval **after** the pipeline has filled, i.e.
    /// every stamped interval except the first.
    ///
    /// **This is the row a "P95 ≤ 16.67 ms" target is read off.** The whole-run
    /// series answers "did the instrument cover the run"; this one answers "how
    /// evenly did frames arrive once the pipeline was running", which is the
    /// question a viewer experiences and the only one a percentile over ~90 samples
    /// can answer at all — see [`Self::pipeline_fill_ms`].
    ///
    /// `None` when fewer than two intervals were stamped.
    pub steady_latency_stats: Option<StageStats>,
    /// Mean steady-state interval of the even- and odd-indexed frames, separately.
    ///
    /// **A distribution cannot see a cycle.** The 4K row's steady intervals split
    /// cleanly into 17.6 ms and 25.4 ms by frame parity, which is why its P95 sits
    /// 25% above its mean — not a stall on a few frames, but every other frame
    /// arriving late in a repeating two-state pattern. A percentile over the pooled
    /// series is identical whether the slow frames alternate or cluster, so the
    /// split has to be measured separately or the shape is invisible.
    ///
    /// Parity of the frame index within the steady series, so `(even, odd)`.
    /// `None` when there are fewer than four steady intervals — below that, two
    /// means each drawn from one or two samples say nothing.
    pub latency_alternation: Option<(f64, f64)>,
    pub system_metrics: SystemMetrics,
}

impl ProfileReport {
    /// The frame rate the measured mean latency implies, or `None` when latency
    /// was not measured.
    ///
    /// Exists to be compared against [`Self::average_fps`]: the two are computed
    /// from independent clocks over the same run, so a disagreement means the
    /// latency series has gaps and its percentiles cover less than the whole run.
    /// [`Self::format_table`] prints a warning when they diverge by more than 10%.
    pub fn latency_implied_fps(&self) -> Option<f64> {
        self.frame_latency_stats
            .filter(|l| l.avg_ms > 0.0)
            .map(|l| 1000.0 / l.avg_ms)
    }

    /// How closely the GPU-timeline spans add up to the measured frame interval,
    /// as a percentage — or `None` without both GPU times and latency.
    ///
    /// **This is a consistency check, NOT evidence that the frame is understood.**
    /// `GpuTransfer` is measured as the gap between one frame's closing timestamp
    /// and the next frame's opening one, so `Composite + GpuTransfer` tiles the
    /// GPU timeline by construction and ~100% is the expected reading, not an
    /// achievement. Reporting it as coverage would be a tautology dressed as a
    /// verification.
    ///
    /// What a *deviation* means is the useful part: the two figures come from
    /// different clocks (the queue's tick counter vs `Instant`), so a gap between
    /// them is real evidence that tick samples are missing — a dropped resolve, a
    /// driver reordering ticks across a submission, or frames retiring out of
    /// submission order. Any of those invalidate the `GpuTransfer` row.
    ///
    /// For the question this does *not* answer — whether that transfer time is
    /// really the upload — see [`Self::transfer_bandwidth_gbps`].
    pub fn gpu_timeline_consistency_pct(&self) -> Option<f64> {
        let latency_avg = self.frame_latency_stats.filter(|l| l.avg_ms > 0.0)?.avg_ms;
        let gpu_sum: f64 = self
            .stage_stats
            .values()
            .filter_map(|s| s.gpu_avg_ms)
            .sum();
        if gpu_sum <= 0.0 {
            return None;
        }
        Some(gpu_sum / latency_avg * 100.0)
    }

    /// Bytes-per-second implied by dividing the counted upload bytes by the
    /// measured `GpuTransfer` time, in GB/s — or `None` when either is unmeasured.
    ///
    /// **This is the falsifiable version of "the remaining 4K cost is upload
    /// bandwidth".** The `GpuTransfer` span on its own cannot distinguish transfer
    /// work from an idle queue: both look like time in which no graph is running.
    /// Dividing by bytes turns it into a claim that can be wrong — a figure near
    /// the host's PCIe rate supports the transfer explanation, and a figure far
    /// below it says the queue was idle and something else is pacing the frame.
    ///
    /// It is a lower bound on achieved bandwidth: the numerator counts only the
    /// bytes this process handed to `write_buffer`, while the span may also contain
    /// queue idle time and any other work wgpu prepended.
    pub fn transfer_bandwidth_gbps(&self) -> Option<f64> {
        let transfer_ms = self
            .stage_stats
            .get(&PipelineStage::GpuTransfer)?
            .gpu_avg_ms
            .filter(|ms| *ms > 0.0)?;
        let total_bytes = self.system_metrics.gpu_upload_bytes?;
        if self.total_frames == 0 {
            return None;
        }
        let bytes_per_frame = total_bytes as f64 / self.total_frames as f64;
        Some(bytes_per_frame / (transfer_ms / 1000.0) / 1e9)
    }

    /// Formats the profile report into a human-readable table matching the roadmap spec.
    pub fn format_table(&self) -> String {
        /// A GPU column, or `n/a` when nothing measured it.
        ///
        /// The same rule as [`SystemMetrics::format_block`]: an unmeasured stage
        /// prints `n/a`, never `0.00 ms`, because "the GPU was idle" and "we did
        /// not look" are different claims.
        fn gpu(v: Option<f64>) -> String {
            v.map(|x| format!("{x:>8.2} ms"))
                .unwrap_or_else(|| format!("{:>11}", "n/a"))
        }

        let any_gpu = self
            .stage_stats
            .values()
            .any(|s| s.gpu_avg_ms.is_some());

        let mut out = String::new();
        out.push_str("========================================================================\n");
        out.push_str("                        NEXIR PROFILING REPORT                          \n");
        out.push_str("========================================================================\n");
        out.push_str(&format!(
            "Frames: {:<6} | Wall Time: {:<7.2}s | Avg FPS: {:<7.1} | Realtime: {:<5.2}x\n",
            self.total_frames,
            self.total_wall_time.as_secs_f64(),
            self.average_fps,
            self.realtime_factor
        ));
        // The CPU ceiling, and how far throughput sits below it. Printed on its
        // own line and labelled, because a reader who sees only one FPS number
        // must see the one that counts elapsed seconds.
        if self.cpu_bound_fps > self.average_fps * 1.05 {
            out.push_str(&format!(
                "  Avg FPS above is THROUGHPUT (frames / wall time). CPU-bound ceiling: \
                 {:.1} FPS\n  — the CPU finishes its share of a frame in {:.2} ms; the \
                 gap is GPU/encoder-bound headroom.\n",
                self.cpu_bound_fps, self.total_frame_stats.avg_ms,
            ));
        }
        out.push_str("------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "{:<16} {:>9} {:>9} {:>9} {:>9} {:>8}",
            "Stage", "Avg (ms)", "Min (ms)", "P95 (ms)", "P99 (ms)", "Share %"
        ));
        if any_gpu {
            out.push_str(&format!(
                " {:>11} {:>11} {:>11}",
                "GPU Avg", "GPU P95", "GPU P99"
            ));
        }
        out.push('\n');
        out.push_str("------------------------------------------------------------------------\n");

        for (stage, stats) in &self.stage_stats {
            let has_cpu = stats.count > 0 && stats.total_ms > 0.0;
            let has_gpu = stats.gpu_avg_ms.is_some();
            if !has_cpu && !has_gpu {
                continue;
            }
            if has_cpu {
                out.push_str(&format!(
                    "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>7.1}%",
                    stage.display_name(),
                    stats.avg_ms,
                    stats.min_ms,
                    stats.p95_ms,
                    stats.p99_ms,
                    stats.percentage_of_total
                ));
            } else {
                // A GPU-only stage — `GpuTransfer` is one by construction, since
                // the work happens in a command buffer wgpu submits on this
                // crate's behalf. Its CPU columns are `n/a`, not `0.00 ms`: this
                // thread genuinely spent no time there, and printing zeros would
                // put it in the same visual class as a stage that was measured
                // and found free.
                out.push_str(&format!(
                    "{:<16} {:>11} {:>11} {:>11} {:>11} {:>8}",
                    stage.display_name(),
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a"
                ));
            }
            if any_gpu {
                out.push_str(&format!(
                    " {} {} {}",
                    gpu(stats.gpu_avg_ms),
                    gpu(stats.gpu_p95_ms),
                    gpu(stats.gpu_p99_ms),
                ));
            }
            out.push('\n');
        }

        out.push_str("------------------------------------------------------------------------\n");
        // "CPU per frame", not "Total Frame". It is the sum of one frame's CPU
        // stages, which stopped being a frame duration the moment frames began
        // overlapping — and a row labelled "Total Frame" invites exactly the
        // misreading the latency row below exists to prevent.
        out.push_str(&format!(
            "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>7.1}%\n",
            "CPU per frame",
            self.total_frame_stats.avg_ms,
            self.total_frame_stats.min_ms,
            self.total_frame_stats.p95_ms,
            self.total_frame_stats.p99_ms,
            100.0
        ));
        // The frame interval. Two rows on purpose: the whole-run series (whose mean
        // must agree with throughput) and the steady-state series after the pipeline
        // has filled (whose percentiles are what a 60 FPS target is about). Omitted
        // entirely when unmeasured rather than printed as zeros.
        if let Some(lat) = &self.frame_latency_stats {
            out.push_str(&format!(
                "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8}\n",
                "Frame latency", lat.avg_ms, lat.min_ms, lat.p95_ms, lat.p99_ms, "—"
            ));
            // The steady row, and the fill it was separated from. Both printed:
            // hiding the fill would be trimming the sample that made the P99 large,
            // and printing only the whole-run P99 attributes a startup cost to a
            // frame that arrived late.
            if let (Some(steady), Some(fill)) = (&self.steady_latency_stats, self.pipeline_fill_ms) {
                out.push_str(&format!(
                    "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8}\n",
                    "  ↳ steady",
                    steady.avg_ms,
                    steady.min_ms,
                    steady.p95_ms,
                    steady.p99_ms,
                    "—"
                ));
                out.push_str(&format!(
                    "  Pipeline fill (first interval, excluded from ↳ steady only): \
                     {fill:.2} ms\n"
                ));
                // The two-state pattern, when there is one. Printed only above a
                // threshold because a few percent apart is noise, and a line
                // announcing a "pattern" on every run would train the reader to skip
                // it. Above ~15% it is the dominant fact about the row: the P95 is
                // then not a tail at all but the slow half of a cycle.
                if let Some((even, odd)) = self.latency_alternation {
                    let (lo, hi) = if even <= odd { (even, odd) } else { (odd, even) };
                    if lo > 0.0 && (hi - lo) / lo > 0.15 {
                        out.push_str(&format!(
                            "  ALTERNATING: frames arrive {lo:.2} ms / {hi:.2} ms in a \
                             two-frame cycle\n   ({:.0}% apart). The P95 above is the \
                             slow half of that cycle, NOT a rare\n   stall — a fix must \
                             change the cycle, and pipeline depth will not.\n",
                            (hi - lo) / lo * 100.0,
                        ));
                    }
                }
                // Said explicitly, because the two P99s can differ by 2-3x and a
                // reader needs to know which one the target is about.
                out.push_str(
                    "  Read P95/P99 off ↳ steady: the whole-run row's tail IS the fill \
                     on a\n  short run (the P99 index lands on the largest sample). \
                     Read the MEAN off\n  the whole-run row, which covers the same \
                     seconds as Avg FPS.\n",
                );
            }
            out.push_str(
                "  (Frame latency is wall-clock spacing between frames retiring — the \
                 quantity\n   P95/P99 targets are about. CPU per frame is the CPU's \
                 share of one frame and\n   is smaller once frames overlap; it is not a \
                 frame duration.)\n",
            );
            // SELF-CHECK, printed rather than asserted. Mean latency and
            // throughput measure the same seconds two different ways, so they
            // must agree; when they do not, the latency series is missing samples
            // (an unstamped fill or drain) and its percentiles are being read off
            // a different run length than the FPS. Printing the disagreement is
            // what keeps a plausible-looking P95 from being trusted.
            if let Some(implied) = self.latency_implied_fps() {
                if self.average_fps > 0.0
                    && (implied - self.average_fps).abs() > self.average_fps * 0.1
                {
                    out.push_str(&format!(
                        "  WARNING: mean latency implies {implied:.1} FPS but throughput \
                         is {:.1} FPS.\n   The latency series does not cover the whole \
                         run — treat its percentiles with suspicion.\n",
                        self.average_fps,
                    ));
                }
            }
        }
        if any_gpu {
            // Stated explicitly because the columns invite the addition: the CPU
            // total and the GPU column measure overlapping wall time once frames
            // are in flight, so they do not sum to a frame duration.
            out.push_str(
                "  (CPU per frame sums the CPU stages; GPU columns are concurrent GPU \
                 execution, not additive)\n",
            );
            // Whether the tick series is self-consistent. ~100% is expected, not
            // an achievement — `Composite` and `GpuTransfer` tile the GPU timeline
            // by construction — so only a DEVIATION carries information, and it
            // means tick samples are missing.
            if let Some(pct) = self.gpu_timeline_consistency_pct() {
                if (pct - 100.0).abs() > 10.0 {
                    out.push_str(&format!(
                        "  WARNING: GPU spans sum to {pct:.0}% of the {:.2} ms frame \
                         interval, not ~100%.\n   The tick series is incomplete — the \
                         GPU transfer row is not trustworthy.\n",
                        self.frame_latency_stats.map_or(0.0, |l| l.avg_ms),
                    ));
                }
            }
            // What the transfer span means, stated as a falsifiable rate rather
            // than as an assumption. An idle queue and a saturated bus look
            // identical in the span alone; dividing by counted bytes separates
            // them.
            if let Some(gbps) = self.transfer_bandwidth_gbps() {
                out.push_str(&format!(
                    "  GPU transfer implies {gbps:.2} GB/s for the {:.1} MB/frame this run \
                     uploaded\n   (lower bound: the span may also hold queue idle time).\n",
                    self.system_metrics
                        .gpu_upload_bytes
                        .map_or(0.0, |b| b as f64 / self.total_frames.max(1) as f64
                            / (1024.0 * 1024.0)),
                ));
            }
        }
        out.push_str("------------------------------------------------------------------------\n");
        out.push_str(&self.system_metrics.format_block());
        out.push_str("========================================================================\n");
        out
    }
}

/// Thread-safe active profiling session.
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    inner: Arc<Mutex<ProfilingSessionInner>>,
}

#[derive(Debug)]
struct ProfilingSessionInner {
    start_time: Instant,
    target_fps: f64,
    frames: Vec<FrameProfile>,
    system_metrics: SystemMetrics,
    max_history: usize,
}

impl ProfilingSession {
    pub fn new(target_fps: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ProfilingSessionInner {
                start_time: Instant::now(),
                target_fps: if target_fps > 0.0 { target_fps } else { 30.0 },
                frames: Vec::new(),
                system_metrics: SystemMetrics::default(),
                max_history: 10_000,
            })),
        }
    }

    /// Push a completed frame profile into the session.
    pub fn push_frame(&self, profile: FrameProfile) {
        let mut inner = self.inner.lock().unwrap();
        if inner.frames.len() >= inner.max_history {
            inner.frames.remove(0);
        }
        inner.frames.push(profile);
    }

    /// Update the current system metrics snapshot.
    pub fn update_system_metrics(&self, metrics: SystemMetrics) {
        let mut inner = self.inner.lock().unwrap();
        inner.system_metrics = metrics;
    }

    /// A copy of every frame profile recorded so far, in arrival order.
    ///
    /// For per-frame analysis that a summary cannot answer: [`ProfileReport`]
    /// carries distributions, and a distribution cannot say whether the frame with
    /// the worst latency is the frame with the worst upload. Percentiles taken
    /// over two series independently are compatible with any pairing between them,
    /// so a tail explanation built on "the P99s are both large" is not evidence —
    /// see [`format_frame_dump`].
    pub fn frames_snapshot(&self) -> Vec<FrameProfile> {
        self.inner.lock().unwrap().frames.clone()
    }
}

/// Pearson correlation between two equal-length per-frame series.
///
/// `None` when there are fewer than three pairs, when the lengths differ, or when
/// either series has no variance — in all three cases a coefficient would be
/// arithmetic without meaning, and printing one anyway is how a tail gets
/// attributed to whatever was measured next to it.
pub fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 3 {
        return None;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let dx = x - mx;
        let dy = y - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return None;
    }
    Some(cov / (vx * vy).sqrt())
}

/// The value above which a sample counts as an outlier: median + 3 scaled MADs.
///
/// Robust rather than mean+3σ because the series this is applied to is exactly the
/// kind with a long tail: one 45 ms upload among 5 ms ones drags a standard
/// deviation far enough that the outlier stops being an outlier by its own measure
/// (σ ≈ 8.7 there, so mean+3σ ≈ 33 and the mean has already moved to 7).
///
/// **A flat series must select nothing, not everything.** When the MAD is zero —
/// which happens whenever more than half the samples are identical — `median + 3
/// MADs` collapses to the median itself, and "anything above the median" would flag
/// half the run. The zero-MAD arm therefore doubles the median instead, so a series
/// of equal values selects nothing while a single large value among them still
/// stands out. This case is live, not hypothetical: `↳ submit` on the
/// `write_texture` path is sub-microsecond on every frame, and a "top N slowest"
/// ranking over it would name whichever frames happened to sort first and then
/// report them as coinciding with the latency tail.
///
/// `None` for an empty series.
pub fn outlier_threshold(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = v[v.len() / 2];
    let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = dev[dev.len() / 2];
    if mad > 0.0 {
        // 1.4826 puts the MAD on the same scale as a standard deviation for
        // normal data, so "3 MADs" reads like the familiar 3σ.
        Some(med + 3.0 * 1.4826 * mad)
    } else {
        // No spread to scale by. Twice the median: a flat series selects nothing
        // (nothing exceeds 2× its own value when every value is the median), while
        // a lone spike among identical samples still does.
        Some(med * 2.0)
    }
}

/// Per-frame dump of the worst-latency frames, with the stages that could explain
/// them.
///
/// **This exists because a percentile cannot attribute a tail.** The 4K row shows
/// a 54.6 ms latency P99 against a 20.0 ms mean, and an `Upload` P99 of 43-46 ms
/// against a 5.3 ms mean; those two facts are equally consistent with "the late
/// frame is the slow upload" and with "two unrelated frames were slow". Only the
/// pairing decides, so this prints the pairing: the `worst` latest frames, each
/// with its own upload/prepare/submit and GPU spans, plus
///
///   * the Pearson coefficient between latency and each candidate stage, and
///   * the **intersection of the outlier sets** — which of the frames that arrived
///     anomalously late also uploaded anomalously slowly.
///
/// The intersection is stated over outlier *sets* rather than as "how many of the
/// top N overlap" because the latter cannot survive ties: on the `write_texture`
/// path every frame's `↳ submit` is a fraction of a microsecond, so a
/// slowest-N ranking over it names arbitrary frames and then reports them as
/// coinciding with the tail. See [`outlier_threshold`].
///
/// An empty intersection **refutes** the upload explanation, which is the outcome
/// that matters: it sends the investigation to the texture pool rather than to
/// `write_buffer`.
pub fn format_frame_dump(frames: &[FrameProfile], worst: usize) -> String {
    let mut out = String::new();
    out.push_str(
        "── Per-frame outliers (NEXIR_FRAME_DUMP) ────────────────────────────────\n",
    );

    // Only frames that carry a latency stamp: frame 0 has no interval before it,
    // and a frame without one cannot be ranked by lateness at all.
    let mut rows: Vec<(usize, f64, &FrameProfile)> = frames
        .iter()
        .filter_map(|f| f.latency_ms().map(|l| (f.frame_index, l, f)))
        .collect();
    if rows.len() < 3 {
        out.push_str(
            "  Fewer than 3 frames carry a latency stamp; nothing to attribute.\n",
        );
        return out;
    }

    let lat_series: Vec<f64> = rows.iter().map(|(_, l, _)| *l).collect();
    let submit_series: Vec<f64> = rows
        .iter()
        .map(|(_, _, f)| f.stage_time_ms(PipelineStage::UploadSubmit))
        .collect();
    let upload_series: Vec<f64> = rows
        .iter()
        .map(|(_, _, f)| f.stage_time_ms(PipelineStage::Upload))
        .collect();
    let cpu_series: Vec<f64> = rows.iter().map(|(_, _, f)| f.total_time_ms()).collect();
    // The GPU-timeline spans, over the frames that carry them. Included because
    // this is the candidate the upload lead competes with: if latency tracks the
    // transfer span and not `↳ submit`, the tail is queue time and not
    // `write_buffer`. Frames missing a reading (the first, which has no previous
    // frame's closing tick) are dropped from BOTH series so the pairs stay aligned
    // — substituting 0.0 would invent a fast frame and flatten the coefficient.
    let (gpu_lat_series, xfer_series): (Vec<f64>, Vec<f64>) = rows
        .iter()
        .filter_map(|(_, l, f)| f.gpu_time_ms(PipelineStage::GpuTransfer).map(|x| (*l, x)))
        .unzip();
    let (graph_lat_series, graph_series): (Vec<f64>, Vec<f64>) = rows
        .iter()
        .filter_map(|(_, l, f)| f.gpu_time_ms(PipelineStage::Composite).map(|x| (*l, x)))
        .unzip();

    // Ranked by lateness, worst first.
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let take = worst.min(rows.len());

    out.push_str(&format!(
        "{:>7} {:>10} {:>10} {:>10} {:>10} {:>11} {:>11}\n",
        "frame", "latency", "CPU sum", "Upload", "↳ submit", "GPU graph", "GPU xfer"
    ));
    fn opt(v: Option<f64>) -> String {
        v.map(|x| format!("{x:>8.2} ms")).unwrap_or_else(|| format!("{:>11}", "n/a"))
    }
    for (idx, lat, f) in rows.iter().take(take) {
        out.push_str(&format!(
            "{idx:>7} {lat:>7.2} ms {:>7.2} ms {:>7.2} ms {:>7.2} ms {} {}\n",
            f.total_time_ms(),
            f.stage_time_ms(PipelineStage::Upload),
            f.stage_time_ms(PipelineStage::UploadSubmit),
            opt(f.gpu_time_ms(PipelineStage::Composite)),
            opt(f.gpu_time_ms(PipelineStage::GpuTransfer)),
        ));
    }

    // ── The pairing, as an intersection of outlier sets ───────────────────────
    // Ranked in the same frame order as the series above, so index i of every
    // series is the same frame.
    let ordered: Vec<usize> = frames
        .iter()
        .filter(|f| f.latency.is_some())
        .map(|f| f.frame_index)
        .collect();
    let pick = |series: &[f64]| -> std::collections::BTreeSet<usize> {
        match outlier_threshold(series) {
            Some(t) => ordered
                .iter()
                .zip(series)
                .filter(|(_, v)| **v > t)
                .map(|(i, _)| *i)
                .collect(),
            None => Default::default(),
        }
    };
    let late = pick(&lat_series);
    let slow_submit = pick(&submit_series);
    let both: Vec<usize> = late.intersection(&slow_submit).copied().collect();

    out.push_str(&format!(
        "  Outlier sets (> median + 3 MAD): {} late frame(s), {} slow-submit \
         frame(s),\n  {} in BOTH{}\n",
        late.len(),
        slow_submit.len(),
        both.len(),
        if both.is_empty() {
            " — the late frames are NOT the slow uploads.".to_string()
        } else {
            format!(" — frames {both:?}.")
        },
    ));

    fn r(v: Option<f64>) -> String {
        v.map(|x| format!("{x:+.3}")).unwrap_or_else(|| "n/a (no variance)".into())
    }
    out.push_str(&format!(
        "  Pearson r vs latency over {} frames:  ↳submit {}   Upload {}   CPU sum {}\n",
        lat_series.len(),
        r(pearson(&lat_series, &submit_series)),
        r(pearson(&lat_series, &upload_series)),
        r(pearson(&lat_series, &cpu_series)),
    ));
    // The competing explanation, on the same footing. `GPU xfer` is the span
    // between one frame's graph closing and the next's opening — where wgpu's own
    // staging copies execute — so a latency series that tracks it rather than
    // `↳ submit` says the time is on the queue, not in this thread's `write_buffer`
    // call.
    out.push_str(&format!(
        "  {:<38}  GPU xfer {}   GPU graph {}\n",
        format!("(over {} frame(s) carrying GPU ticks)", xfer_series.len()),
        r(pearson(&gpu_lat_series, &xfer_series)),
        r(pearson(&graph_lat_series, &graph_series)),
    ));
    out.push_str(
        "  (A coefficient near 0 with an empty intersection means the late frame is not \
         the\n   slow upload, and the tail is somewhere this dump does not measure.)\n",
    );
    out
}

impl ProfilingSession {
    /// Compute summary report over all accumulated frame profiles.
    pub fn generate_report(&self) -> ProfileReport {
        let inner = self.inner.lock().unwrap();
        let frame_count = inner.frames.len();
        let wall_time = inner.start_time.elapsed();

        if frame_count == 0 {
            return ProfileReport {
                total_frames: 0,
                total_wall_time: wall_time,
                realtime_factor: 0.0,
                ..Default::default()
            };
        }

        // Collect total frame times
        let mut total_times_ms: Vec<f64> = inner.frames.iter().map(|f| f.total_time_ms()).collect();
        let total_frame_stats = compute_distribution_stats(&mut total_times_ms);

        let sum_total_ms: f64 = inner.frames.iter().map(|f| f.total_time_ms()).sum();

        // Collect per-stage statistics
        let mut stage_stats = BTreeMap::new();
        for &stage in PipelineStage::all() {
            let mut stage_durations: Vec<f64> = inner
                .frames
                .iter()
                .map(|f| f.stage_time_ms(stage))
                .collect();
            let mut stats = compute_distribution_stats(&mut stage_durations);
            if sum_total_ms > 0.0 {
                stats.percentage_of_total = (stats.total_ms / sum_total_ms) * 100.0;
            }

            // GPU times are gathered only from the frames that actually carry a
            // reading for this stage. Defaulting a missing frame to 0.0 the way
            // the CPU series does would drag the average toward zero and report
            // a fast GPU where there was simply no measurement.
            let mut gpu_samples: Vec<f64> = inner
                .frames
                .iter()
                .filter_map(|f| f.gpu_time_ms(stage))
                .collect();
            if !gpu_samples.is_empty() {
                let gpu = compute_distribution_stats(&mut gpu_samples);
                stats.gpu_avg_ms = Some(gpu.avg_ms);
                stats.gpu_p95_ms = Some(gpu.p95_ms);
                stats.gpu_p99_ms = Some(gpu.p99_ms);
            }

            stage_stats.insert(stage, stats);
        }

        // THROUGHPUT, from the clock on the wall.
        //
        // This used to be `1000 / mean(CPU stage sum)`, which is the same number
        // only while the pipeline is serial. Once frames overlap, the CPU's
        // per-frame work stops being a frame duration — it finishes frame N+1's
        // share while the GPU is still on frame N — and that formula reports a
        // rate the machine never achieved. Measured here: the 4K row's CPU sum
        // implied 237 FPS on a run that delivered 58.
        //
        // Frames divided by elapsed seconds cannot be gamed that way.
        let avg_fps = if wall_time.as_secs_f64() > 0.0 {
            frame_count as f64 / wall_time.as_secs_f64()
        } else {
            0.0
        };

        // The CPU-bound ceiling, kept as a diagnostic rather than as the headline:
        // the gap between it and `avg_fps` is how much CPU headroom exists.
        let cpu_bound_fps = if total_frame_stats.avg_ms > 0.0 {
            1000.0 / total_frame_stats.avg_ms
        } else {
            0.0
        };

        let p99_fps = if total_frame_stats.p99_ms > 0.0 {
            1000.0 / total_frame_stats.p99_ms
        } else {
            0.0
        };

        // Realtime multiplier against throughput, for the same reason: "we export
        // 4x faster than realtime" is a claim about elapsed time.
        let realtime_factor = if wall_time.as_secs_f64() > 0.0 {
            (frame_count as f64 / inner.target_fps) / wall_time.as_secs_f64()
        } else {
            0.0
        };

        // Frame latency: the interval frames actually arrived at, gathered only
        // from the frames that carry one. Defaulting a missing stamp to 0.0 would
        // drag the mean toward zero and report a pipeline faster than the clock.
        //
        // Collected in ARRIVAL order first, because the first stamped interval is
        // the pipeline fill and has to be separable from the rest — see
        // `ProfileReport::pipeline_fill_ms`. Sorting happens inside
        // `compute_distribution_stats`, on copies.
        let ordered_latencies: Vec<f64> =
            inner.frames.iter().filter_map(|f| f.latency_ms()).collect();
        let mut latency_samples = ordered_latencies.clone();
        let frame_latency_stats = if latency_samples.is_empty() {
            None
        } else {
            Some(compute_distribution_stats(&mut latency_samples))
        };

        // The fill, and the steady state after it.
        //
        // Split rather than trimmed: the whole-run series above still contains the
        // fill, so its mean stays comparable with `average_fps`, while the
        // percentiles a 60 FPS target is read off come from the steady series. On a
        // 90-frame run the P99 index lands on the largest sample, and at 4K the
        // largest sample is the 67-91 ms fill — so the old P99 of 54.6 ms was
        // reporting the pipeline starting up, not a frame arriving late.
        let pipeline_fill_ms = ordered_latencies.first().copied();
        let steady_latency_stats = if ordered_latencies.len() > 1 {
            let mut steady = ordered_latencies[1..].to_vec();
            Some(compute_distribution_stats(&mut steady))
        } else {
            None
        };

        // The two-state pattern, measured rather than eyeballed off a CSV. See
        // `ProfileReport::latency_alternation`: pooled percentiles cannot
        // distinguish "every other frame is late" from "a few frames stalled", and
        // the two call for completely different fixes.
        let latency_alternation = if ordered_latencies.len() >= 5 {
            let steady = &ordered_latencies[1..];
            let (mut even, mut odd) = (Vec::new(), Vec::new());
            for (i, v) in steady.iter().enumerate() {
                if i % 2 == 0 {
                    even.push(*v)
                } else {
                    odd.push(*v)
                }
            }
            if even.len() >= 2 && odd.len() >= 2 {
                Some((
                    even.iter().sum::<f64>() / even.len() as f64,
                    odd.iter().sum::<f64>() / odd.len() as f64,
                ))
            } else {
                None
            }
        } else {
            None
        };

        ProfileReport {
            total_frames: frame_count,
            total_wall_time: wall_time,
            average_fps: avg_fps,
            cpu_bound_fps,
            p99_fps,
            realtime_factor,
            stage_stats,
            total_frame_stats,
            frame_latency_stats,
            pipeline_fill_ms,
            steady_latency_stats,
            latency_alternation,
            system_metrics: inner.system_metrics,
        }
    }
}

/// Computes percentile, average, min, and max statistics over a mutable sample slice.
fn compute_distribution_stats(samples: &mut [f64]) -> StageStats {
    if samples.is_empty() {
        return StageStats::default();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = samples.len();
    let min_ms = samples[0];
    let max_ms = samples[count - 1];
    let sum: f64 = samples.iter().sum();
    let avg_ms = sum / count as f64;

    let p95_idx = ((count as f64 * 0.95).ceil() as usize).saturating_sub(1).min(count - 1);
    let p99_idx = ((count as f64 * 0.99).ceil() as usize).saturating_sub(1).min(count - 1);

    StageStats {
        count,
        min_ms,
        max_ms,
        avg_ms,
        p95_ms: samples[p95_idx],
        p99_ms: samples[p99_idx],
        total_ms: sum,
        percentage_of_total: 0.0,
        // Filled by `generate_report` from the frames' `gpu_durations`; this
        // function only sees one series at a time.
        gpu_avg_ms: None,
        gpu_p95_ms: None,
        gpu_p99_ms: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_profile_stage_recording() {
        let mut fp = FrameProfile::new(0, 0);
        fp.record_stage(PipelineStage::Decode, Duration::from_millis(5));
        fp.record_stage(PipelineStage::Effects, Duration::from_millis(15));
        fp.record_stage(PipelineStage::Composite, Duration::from_millis(10));

        assert_eq!(fp.stage_time_ms(PipelineStage::Decode), 5.0);
        assert_eq!(fp.stage_time_ms(PipelineStage::Effects), 15.0);
        assert_eq!(fp.stage_time_ms(PipelineStage::Composite), 10.0);
        assert_eq!(fp.total_time_ms(), 30.0);
    }

    #[test]
    fn test_profiling_session_report_generation() {
        let session = ProfilingSession::new(30.0);

        for i in 0..100 {
            let mut fp = FrameProfile::new(i, i as i64 * 3000);
            fp.record_stage(PipelineStage::Decode, Duration::from_millis(2));
            fp.record_stage(PipelineStage::Scheduler, Duration::from_micros(500));
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(3));
            fp.record_stage(PipelineStage::Nvenc, Duration::from_millis(4));
            session.push_frame(fp);
        }

        let report = session.generate_report();
        assert_eq!(report.total_frames, 100);
        assert!(report.total_frame_stats.avg_ms >= 9.4);
        assert!(report.average_fps > 0.0);
        assert!(report.realtime_factor > 1.0);

        let table = report.format_table();
        assert!(table.contains("NEXIR PROFILING REPORT"));
        assert!(table.contains("Decode"));
        assert!(table.contains("Composite"));
        assert!(table.contains("NVENC"));
    }

    #[test]
    fn unmeasured_system_metrics_print_as_not_available() {
        // The regression this pins: `SystemMetrics` used to be plain numbers, a
        // caller filled them with plausible constants, and the table printed
        // those as if they had been read from the driver.
        let metrics = SystemMetrics::default();
        let block = metrics.format_block();
        assert!(
            block.contains("n/a"),
            "an all-unmeasured SystemMetrics must print n/a, got:\n{block}"
        );
        for forbidden in ["0.0%", "0.0 MB"] {
            assert!(
                !block.contains(forbidden),
                "unmeasured metrics must not print {forbidden} — a zero reading and \
                 no reading are different claims:\n{block}"
            );
        }
        assert!(
            !block.contains("Frame=0"),
            "an unmeasured queue depth must not print as 0:\n{block}"
        );
    }

    #[test]
    fn measured_system_metrics_are_printed() {
        let metrics = SystemMetrics {
            gpu_utilization: Some(42.5),
            gpu_upload_bytes: Some(8 * 1024 * 1024),
            frame_queue_depth: Some(3),
            ..Default::default()
        };
        let block = metrics.format_block();
        assert!(block.contains("42.5%"), "measured GPU util missing:\n{block}");
        assert!(block.contains("8.0 MB"), "measured upload bytes missing:\n{block}");
        assert!(block.contains("Frame=3"), "measured queue depth missing:\n{block}");
        // The ones still unmeasured stay honest in the same block.
        assert!(block.contains("n/a"), "unset fields must still say n/a:\n{block}");
    }

    /// GPU time is stored separately and must NEVER be summed into the frame's
    /// CPU total.
    ///
    /// The two overlap by construction — the CPU records frame N+1 while the GPU
    /// executes frame N — so adding them reports a frame time no clock ever saw.
    #[test]
    fn gpu_time_does_not_inflate_the_cpu_total() {
        let mut fp = FrameProfile::new(0, 0);
        fp.record_stage(PipelineStage::Upload, Duration::from_millis(4));
        fp.record_gpu(PipelineStage::Composite, Duration::from_millis(9));

        assert_eq!(fp.total_time_ms(), 4.0, "GPU time must not enter the CPU total");
        assert_eq!(fp.gpu_time_ms(PipelineStage::Composite), Some(9.0));
        assert_eq!(
            fp.gpu_time_ms(PipelineStage::Upload),
            None,
            "a stage nobody bracketed must report None, not 0.0"
        );
        assert!(fp.has_gpu_times());
    }

    /// A run without timestamp support must print `n/a` for GPU columns — or omit
    /// them — and must never print a zero that reads as a measurement.
    #[test]
    fn a_run_without_gpu_timing_prints_no_gpu_zeros() {
        let session = ProfilingSession::new(60.0);
        for i in 0..10 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_millis(2));
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(3));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        assert!(
            report.stage_stats[&PipelineStage::Composite].gpu_avg_ms.is_none(),
            "no GPU samples were recorded, so there must be no GPU average"
        );
        let table = report.format_table();
        assert!(
            !table.contains("GPU Avg"),
            "with nothing measured the GPU columns should not appear at all:\n{table}"
        );
    }

    /// A stage measured on only some frames must average over those frames, not
    /// over all of them.
    ///
    /// The bug this pins: treating a missing GPU reading as 0.0 the way the CPU
    /// series does. With 2 of 10 frames measured at 8 ms each, that would report
    /// 1.6 ms — a fast GPU that was never observed — instead of 8 ms.
    #[test]
    fn partial_gpu_samples_average_over_measured_frames_only() {
        let session = ProfilingSession::new(60.0);
        for i in 0..10 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(3));
            if i < 2 {
                fp.record_gpu(PipelineStage::Composite, Duration::from_millis(8));
            }
            session.push_frame(fp);
        }
        let report = session.generate_report();
        let gpu_avg = report.stage_stats[&PipelineStage::Composite]
            .gpu_avg_ms
            .expect("two frames carried a GPU reading");
        assert!(
            (gpu_avg - 8.0).abs() < 1e-9,
            "expected 8.0 ms (the mean of the frames that were measured), got {gpu_avg} \
             — unmeasured frames are being counted as zero"
        );
        let table = report.format_table();
        assert!(table.contains("GPU Avg"), "measured GPU time must be shown:\n{table}");
        assert!(
            table.contains("not additive"),
            "the table must say GPU and CPU columns do not sum:\n{table}"
        );
    }

    /// A breakdown stage must not be counted twice.
    ///
    /// `UploadPrepare` + `UploadSubmit` decompose `Upload`; all three get
    /// recorded. Summing everything would report a 8 ms frame as 16 ms and every
    /// derived FPS would be halved.
    #[test]
    fn breakdown_stages_do_not_double_count_the_frame() {
        let mut fp = FrameProfile::new(0, 0);
        fp.record_stage(PipelineStage::Upload, Duration::from_millis(5));
        fp.record_stage(PipelineStage::UploadPrepare, Duration::from_millis(4));
        fp.record_stage(PipelineStage::UploadSubmit, Duration::from_millis(1));
        fp.record_stage(PipelineStage::Composite, Duration::from_millis(3));

        assert_eq!(
            fp.total_time_ms(),
            8.0,
            "the frame is Upload + Composite = 8 ms; the prepare/submit split is \
             inside the 5 ms upload, not additional to it"
        );
        // The breakdown is still readable on its own.
        assert_eq!(fp.stage_time_ms(PipelineStage::UploadPrepare), 4.0);
        assert_eq!(fp.stage_time_ms(PipelineStage::UploadSubmit), 1.0);
    }

    /// Reported FPS must be throughput, not the inverse of the CPU's own work.
    ///
    /// THE BUG THIS PINS. `average_fps` was `1000 / mean(sum of CPU stages)`,
    /// which is correct only while the pipeline is serial. Once frames overlap,
    /// the CPU records frame N+1 while the GPU executes frame N, so the CPU sum is
    /// no longer a frame duration — and the formula reports a rate the machine
    /// never reached. Observed for real: a 4K run whose CPU sum implied 237 FPS
    /// took 1.54 s for 90 frames, i.e. 58 FPS.
    ///
    /// A benchmark that reports the first number gets faster every time work moves
    /// off the CPU's measured path, whether or not any frame arrives sooner.
    #[test]
    fn reported_fps_is_throughput_not_the_inverse_of_cpu_work() {
        let session = ProfilingSession::new(60.0);
        // 60 frames, each costing the CPU 1 ms — but the loop below takes real
        // time, so wall-clock throughput is far below 1000 FPS.
        for i in 0..60 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_millis(1));
            session.push_frame(fp);
        }
        std::thread::sleep(Duration::from_millis(120));

        let report = session.generate_report();

        // 60 frames in >= 120 ms is <= 500 FPS. The CPU-sum formula would say 1000.
        assert!(
            report.average_fps < 600.0,
            "average_fps must be frames/wall-time; got {:.1}, which is the inverse of \
             the 1 ms CPU sum rather than a throughput",
            report.average_fps
        );
        assert!(
            report.average_fps > 100.0,
            "sanity: 60 frames in ~0.12 s is a few hundred FPS, got {:.1}",
            report.average_fps
        );

        // The CPU ceiling is still available, and is much higher.
        assert!(
            report.cpu_bound_fps > report.average_fps * 1.5,
            "the CPU ceiling ({:.1}) should far exceed throughput ({:.1}) here",
            report.cpu_bound_fps,
            report.average_fps
        );

        // And the table must say which is which, so no reader mistakes the ceiling
        // for the result.
        let table = report.format_table();
        assert!(
            table.contains("THROUGHPUT"),
            "the report must label its FPS as throughput when a ceiling is also \
             shown:\n{table}"
        );
        assert!(
            table.contains("CPU-bound ceiling"),
            "the CPU ceiling must be labelled as such:\n{table}"
        );
    }

    /// The realtime multiplier is a claim about elapsed time and must be computed
    /// from it.
    #[test]
    fn realtime_factor_is_measured_against_wall_time() {
        let session = ProfilingSession::new(60.0);
        for i in 0..60 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_micros(100));
            session.push_frame(fp);
        }
        std::thread::sleep(Duration::from_millis(500));

        let report = session.generate_report();
        // 60 frames at a 60 FPS target is 1.0 s of content, produced in ~0.5 s,
        // so roughly 2x realtime — NOT the 10x the 0.1 ms CPU sum would imply.
        assert!(
            report.realtime_factor > 1.0 && report.realtime_factor < 4.0,
            "expected ~2x realtime from 1 s of content in ~0.5 s, got {:.2}x",
            report.realtime_factor
        );
    }

    /// Frame latency is wall-clock spacing between retirements, and its mean must
    /// agree with throughput.
    ///
    /// This is the quantity the audit's "P95 <= 16.67 ms" criterion is about.
    /// After frames-in-flight, the CPU-stage sum is NOT that quantity — it reads
    /// 5 ms on a pipeline delivering a frame every 18 ms — so a percentile taken
    /// over it answers a question nobody asked.
    #[test]
    fn frame_latency_mean_agrees_with_throughput() {
        let session = ProfilingSession::new(60.0);
        for i in 0..30 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_micros(200));
            fp.record_latency(Duration::from_millis(10));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        let lat = report.frame_latency_stats.expect("latency was recorded");
        assert!((lat.avg_ms - 10.0).abs() < 0.01, "got {}", lat.avg_ms);
        // 10 ms/frame is 100 FPS, and that must be what the latency implies —
        // not the 5000 FPS the 0.2 ms CPU sum would.
        assert!((1000.0 / lat.avg_ms - 100.0).abs() < 1.0);

        // And the table must name the two quantities differently, so nobody reads
        // the CPU's share of a frame as the frame interval.
        let table = report.format_table();
        assert!(
            table.contains("CPU per frame"),
            "the CPU stage sum must not be labelled as a frame duration:\n{table}"
        );
        assert!(
            table.contains("Frame latency"),
            "measured latency must be reported:\n{table}"
        );
    }

    /// Unrecorded latency reports as absent, not as zero.
    #[test]
    fn latency_is_absent_when_not_recorded() {
        let session = ProfilingSession::new(60.0);
        session.push_frame(FrameProfile::new(0, 0));
        let report = session.generate_report();
        assert!(report.frame_latency_stats.is_none());
        assert!(
            !report.format_table().contains("Frame latency"),
            "an unmeasured latency must not print a row at all"
        );
    }

    /// Latency percentiles must come from the latency series, not the CPU one.
    ///
    /// The failure this pins is subtle and was the actual state of the code: a
    /// pipeline that absorbs a 50 ms CPU hiccup delivers frames on time, and a
    /// P99 taken over CPU sums would report a spike the viewer never saw. The
    /// reverse is worse — a smooth CPU hiding a stalled presentation.
    #[test]
    fn latency_percentiles_are_independent_of_cpu_stage_times() {
        let session = ProfilingSession::new(60.0);
        for i in 0..100 {
            let mut fp = FrameProfile::new(i, 0);
            // One frame costs the CPU 50 ms; every frame still retires 10 ms apart
            // because the pipeline had the depth to absorb it.
            fp.record_stage(
                PipelineStage::Upload,
                if i == 50 {
                    Duration::from_millis(50)
                } else {
                    Duration::from_millis(1)
                },
            );
            fp.record_latency(Duration::from_millis(10));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        let lat = report.frame_latency_stats.expect("latency was recorded");
        assert!(
            (lat.max_ms - 10.0).abs() < 0.01,
            "no frame arrived late — the worst latency must still be 10 ms, got {}",
            lat.max_ms
        );
        assert!(
            report.total_frame_stats.max_ms > 40.0,
            "the CPU spike must still be visible in the CPU row, got {}",
            report.total_frame_stats.max_ms
        );
    }

    /// A latency series that does not cover the whole run must announce itself.
    ///
    /// The bug this pins is one the bench actually had: `last_retire` started as
    /// `None`, so the pipeline-fill interval before the first retirement was never
    /// stamped, and the 4K row printed an 18.25 ms mean latency on a run
    /// delivering a frame every 21.1 ms. Nothing in the table contradicted it —
    /// the percentiles looked entirely plausible while being taken over less time
    /// than the run took. Mean latency and throughput measure the same seconds, so
    /// a disagreement is a defect in the instrument, and the report has to say so
    /// rather than leave the reader to divide two numbers themselves.
    #[test]
    fn a_latency_series_that_disagrees_with_throughput_is_flagged() {
        let session = ProfilingSession::new(60.0);
        for i in 0..40 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_micros(100));
            // 1 ms apart, i.e. 1000 FPS — nowhere near what the sleep below allows.
            fp.record_latency(Duration::from_millis(1));
            session.push_frame(fp);
        }
        std::thread::sleep(Duration::from_millis(200));

        let report = session.generate_report();
        let implied = report.latency_implied_fps().expect("latency was recorded");
        assert!(
            implied > report.average_fps * 1.5,
            "fixture is wrong: implied {implied:.1} should far exceed throughput {:.1}",
            report.average_fps
        );
        let table = report.format_table();
        assert!(
            table.contains("WARNING") && table.contains("latency series"),
            "a latency series inconsistent with throughput must be flagged:\n{table}"
        );
    }

    /// ...and a consistent one must NOT be flagged, or the warning is noise.
    #[test]
    fn a_consistent_latency_series_is_not_flagged() {
        let session = ProfilingSession::new(60.0);
        // 20 frames × 10 ms of real sleep = 200 ms of wall time, stamped as 10 ms
        // intervals. Throughput and implied FPS should agree within 10%.
        for i in 0..20 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_micros(100));
            fp.record_latency(Duration::from_millis(10));
            session.push_frame(fp);
            std::thread::sleep(Duration::from_millis(10));
        }
        let report = session.generate_report();
        let table = report.format_table();
        assert!(
            !table.contains("WARNING"),
            "throughput {:.1} FPS vs latency-implied {:.1} FPS agree; no warning \
             expected:\n{table}",
            report.average_fps,
            report.latency_implied_fps().unwrap_or(0.0),
        );
    }

    /// A GPU-only stage must appear in the table even with no CPU time, and its
    /// CPU columns must read `n/a` rather than `0.00 ms`.
    ///
    /// `GpuTransfer` is GPU-only by construction: wgpu's `write_buffer` staging
    /// copies execute in a command buffer this crate never encodes, so no CPU
    /// stage covers them. The old row filter was `count > 0 && total_ms > 0.0`,
    /// which dropped such a stage from the report entirely — the 10.24 ms that
    /// dominates the 4K frame would have been measured and then not printed.
    #[test]
    fn a_gpu_only_stage_is_printed_with_na_cpu_columns() {
        let session = ProfilingSession::new(60.0);
        for i in 0..10 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(1));
            // No CPU time for GpuTransfer, ever.
            fp.record_gpu(PipelineStage::GpuTransfer, Duration::from_millis(10));
            fp.record_latency(Duration::from_millis(19));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        assert_eq!(report.stage_stats[&PipelineStage::GpuTransfer].total_ms, 0.0);
        assert_eq!(
            report.stage_stats[&PipelineStage::GpuTransfer].gpu_avg_ms,
            Some(10.0)
        );

        let table = report.format_table();
        let row = table
            .lines()
            .find(|l| l.contains("GPU transfer"))
            .expect("a measured GPU-only stage must still appear in the table");
        assert!(
            row.contains("n/a"),
            "its CPU columns must be n/a, not zeros: {row}"
        );
        // Split off the GPU columns before checking for zeros: "10.00 ms" in the
        // GPU column contains "0.00 ms" as a substring.
        let cpu_part = &row[..row.find("10.00 ms").unwrap_or(row.len())];
        assert!(
            !cpu_part.contains("0.00 ms"),
            "a zero CPU reading would claim this thread was measured there: {row}"
        );
        assert!(row.contains("10.00 ms"), "the GPU reading must be shown: {row}");
    }

    /// An incomplete tick series must be flagged rather than presented as a
    /// finding.
    ///
    /// `Composite` and `GpuTransfer` tile the GPU timeline by construction, so
    /// they must sum to the frame interval. When they do not, tick samples are
    /// missing — a dropped resolve, a driver reordering ticks, frames retiring out
    /// of order — and the `GPU transfer` row is then an arbitrary number. The
    /// fixture here measures 9 ms of graph inside a 19 ms interval with no
    /// transfer span at all: 47%, i.e. half the timeline unaccounted.
    #[test]
    fn an_incomplete_gpu_tick_series_is_flagged() {
        let session = ProfilingSession::new(60.0);
        for i in 0..20 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(1));
            fp.record_gpu(PipelineStage::Composite, Duration::from_millis(9));
            fp.record_latency(Duration::from_millis(19));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        let pct = report
            .gpu_timeline_consistency_pct()
            .expect("GPU time and latency were both measured");
        assert!(
            (pct - 9.0 / 19.0 * 100.0).abs() < 0.1,
            "9 ms of 19 ms is ~47%, got {pct:.1}%"
        );
        let table = report.format_table();
        assert!(
            table.contains("tick series is incomplete"),
            "a GPU timeline that does not add up must be flagged:\n{table}"
        );
    }

    /// ...and a complete one must not carry that warning.
    #[test]
    fn a_complete_gpu_tick_series_is_not_flagged() {
        let session = ProfilingSession::new(60.0);
        for i in 0..20 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(1));
            fp.record_gpu(PipelineStage::Composite, Duration::from_millis(9));
            fp.record_gpu(PipelineStage::GpuTransfer, Duration::from_millis(10));
            fp.record_latency(Duration::from_millis(19));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        let pct = report.gpu_timeline_consistency_pct().expect("both measured");
        assert!((pct - 100.0).abs() < 0.1, "19 of 19 ms is 100%, got {pct:.1}%");
        assert!(
            !report.format_table().contains("tick series is incomplete"),
            "a complete timeline must not be flagged"
        );
    }

    /// The transfer span must be reported as a bandwidth, because that is the form
    /// in which the claim "the bottleneck is the upload" can be wrong.
    ///
    /// The span alone cannot tell a saturated bus from an idle queue — both are
    /// time with no graph running. 47.5 MB in 10 ms is ~5 GB/s, a plausible PCIe
    /// figure; the same span with 1 MB/frame would be 0.1 GB/s and would refute the
    /// transfer explanation outright. Without this division, Phase C's mistake is
    /// available again in a new place: a measured number attributed to a cause
    /// nobody checked.
    #[test]
    fn the_transfer_span_is_reported_as_a_falsifiable_bandwidth() {
        let session = ProfilingSession::new(60.0);
        const MB: u64 = 1024 * 1024;
        for i in 0..20 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Composite, Duration::from_millis(1));
            fp.record_gpu(PipelineStage::GpuTransfer, Duration::from_millis(10));
            fp.record_latency(Duration::from_millis(19));
            session.push_frame(fp);
        }
        // 47.5 MB per frame, counted as the bench counts it.
        session.update_system_metrics(SystemMetrics {
            gpu_upload_bytes: Some(20 * 475 * MB / 10),
            ..Default::default()
        });
        let report = session.generate_report();
        let gbps = report
            .transfer_bandwidth_gbps()
            .expect("bytes and transfer time were both measured");
        // 47.5 MiB / 10 ms = 4.98 GB/s.
        assert!(
            (gbps - 4.98).abs() < 0.05,
            "47.5 MB in 10 ms is ~4.98 GB/s, got {gbps:.2}"
        );
        assert!(
            report.format_table().contains("GB/s"),
            "the bandwidth must be printed, not just computable"
        );
    }

    /// No counted bytes means no bandwidth claim — not a zero.
    #[test]
    fn transfer_bandwidth_is_absent_without_counted_bytes() {
        let session = ProfilingSession::new(60.0);
        for i in 0..5 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_gpu(PipelineStage::GpuTransfer, Duration::from_millis(10));
            fp.record_latency(Duration::from_millis(19));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        assert!(report.transfer_bandwidth_gbps().is_none());
        assert!(!report.format_table().contains("GB/s"));
    }

    /// The pipeline-fill interval must be separated from the steady state, and
    /// kept.
    ///
    /// THE BUG THIS PINS. The 4K row's "latency P99 = 54.6 ms against a 20.0 ms
    /// mean" was read off the whole-run series. A 90-frame run stamps ~90
    /// intervals, so `ceil(90 × 0.99) - 1 = 88` — the second-largest sample — and
    /// the largest two are the pipeline fill (measured 67-91 ms at 4K) and whatever
    /// sat next to it. The "tail stall" was the pipeline starting up, reported as a
    /// frame arriving late in steady playback.
    ///
    /// The fix is a split, NOT a trim: dropping the fill is the other bug this
    /// project already had, where an unstamped fill made the mean disagree with
    /// throughput. So the whole-run row keeps it (its mean stays comparable with
    /// `average_fps`) and the steady row excludes it (its percentiles answer the
    /// 60 FPS question).
    #[test]
    fn the_pipeline_fill_is_separated_from_the_steady_state_and_not_discarded() {
        let session = ProfilingSession::new(60.0);
        for i in 0..90 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(PipelineStage::Upload, Duration::from_millis(4));
            // One 80 ms fill, then a steady 20 ms — exactly the 4K shape.
            fp.record_latency(if i == 0 {
                Duration::from_millis(80)
            } else {
                Duration::from_millis(20)
            });
            session.push_frame(fp);
        }
        let report = session.generate_report();

        let whole = report.frame_latency_stats.expect("latency recorded");
        let steady = report.steady_latency_stats.expect("89 steady intervals");
        assert_eq!(report.pipeline_fill_ms, Some(80.0), "the fill must be reported");

        // The whole-run tail is the fill.
        assert!(
            whole.p99_ms > 70.0,
            "the whole-run P99 is the fill by construction, got {:.2}",
            whole.p99_ms
        );
        // The steady tail is the steady state, and it passes a 16.67 ms-class
        // target's shape (here 20 ms, but with no 80 ms sample in it).
        assert!(
            (steady.p99_ms - 20.0).abs() < 0.01,
            "the steady P99 must exclude the fill, got {:.2}",
            steady.p99_ms
        );
        // And the fill is still inside the whole-run series, so the mean covers the
        // whole run.
        assert!(
            whole.avg_ms > steady.avg_ms,
            "the whole-run mean must include the fill: whole {:.2} vs steady {:.2}",
            whole.avg_ms,
            steady.avg_ms
        );

        let table = report.format_table();
        assert!(table.contains("↳ steady"), "the steady row must be printed:\n{table}");
        assert!(
            table.contains("Pipeline fill"),
            "the fill must be printed, not silently dropped:\n{table}"
        );
        assert!(
            table.contains("Read P95/P99 off ↳ steady"),
            "the table must say which row a target is read off:\n{table}"
        );
    }

    /// A run with a single interval has no steady state to report, and must say
    /// nothing rather than repeat the fill as if it were one.
    #[test]
    fn a_single_interval_yields_no_steady_series() {
        let session = ProfilingSession::new(60.0);
        let mut fp = FrameProfile::new(0, 0);
        fp.record_latency(Duration::from_millis(80));
        session.push_frame(fp);
        let report = session.generate_report();
        assert_eq!(report.pipeline_fill_ms, Some(80.0));
        assert!(
            report.steady_latency_stats.is_none(),
            "one interval is a fill and nothing else"
        );
        assert!(!report.format_table().contains("↳ steady"));
    }

    /// An alternating interval must be reported as a cycle, not as a tail.
    ///
    /// THE FINDING THIS PINS. Benchmark 5's steady intervals are 17.6 ms on
    /// even-indexed frames and 25.4 ms on odd ones — measured, `target/p21_run1.csv`
    /// — and its `GPU transfer` span splits the same way (9.3 / 17.6 ms) while its
    /// graph execution does not (8.5 ms either way). So the row's P95 sitting 25%
    /// above its mean is not a rare stall; it is every other frame. A pooled
    /// percentile is blind to the difference, and the two diagnoses lead opposite
    /// ways: a stall invites more pipeline depth, a cycle is made worse by it.
    #[test]
    fn an_alternating_latency_is_reported_as_a_cycle() {
        let session = ProfilingSession::new(60.0);
        for i in 0..90 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_latency(if i == 0 {
                Duration::from_millis(66)
            } else if i % 2 == 0 {
                Duration::from_micros(17_600)
            } else {
                Duration::from_micros(25_400)
            });
            session.push_frame(fp);
        }
        let report = session.generate_report();
        let (even, odd) = report.latency_alternation.expect("89 steady intervals");
        // Steady series starts at frame 1 (odd), so its index-0 bucket is the
        // odd-numbered frames.
        let (lo, hi) = if even <= odd { (even, odd) } else { (odd, even) };
        assert!((lo - 17.6).abs() < 0.2, "fast half should be ~17.6 ms, got {lo:.2}");
        assert!((hi - 25.4).abs() < 0.2, "slow half should be ~25.4 ms, got {hi:.2}");

        let table = report.format_table();
        assert!(
            table.contains("ALTERNATING"),
            "a 44% two-state split must be called out:\n{table}"
        );
        assert!(
            table.contains("NOT a rare"),
            "the table must say the P95 is a cycle rather than a tail:\n{table}"
        );
    }

    /// ...and an evenly paced run must NOT carry that line, or it is noise.
    #[test]
    fn a_steady_latency_is_not_reported_as_alternating() {
        let session = ProfilingSession::new(60.0);
        for i in 0..40 {
            let mut fp = FrameProfile::new(i, 0);
            // A few percent of jitter, alternating — below the threshold.
            fp.record_latency(Duration::from_micros(if i % 2 == 0 { 16_500 } else { 17_000 }));
            session.push_frame(fp);
        }
        let report = session.generate_report();
        assert!(report.latency_alternation.is_some(), "the split is still computed");
        assert!(
            !report.format_table().contains("ALTERNATING"),
            "3% apart is jitter, not a cycle"
        );
    }

    /// A per-frame dump must decide the pairing, not restate the percentiles.
    ///
    /// Fixture: latency and upload spike on DIFFERENT frames. Both series then
    /// have exactly the distributions of the "they line up" case below, so anything
    /// reading only percentiles cannot tell the two apart — and telling them apart
    /// is the whole purpose of P2.1's first step. Here the answer must be "no
    /// frame is in both sets".
    #[test]
    fn a_frame_dump_refutes_the_upload_lead_when_the_spikes_do_not_line_up() {
        let session = ProfilingSession::new(60.0);
        for i in 0..40 {
            let mut fp = FrameProfile::new(i, 0);
            fp.record_stage(
                PipelineStage::UploadSubmit,
                if i == 10 { Duration::from_millis(45) } else { Duration::from_millis(5) },
            );
            fp.record_stage(PipelineStage::Upload, Duration::from_millis(5));
            fp.record_latency(if i == 30 {
                Duration::from_millis(54)
            } else {
                Duration::from_millis(20)
            });
            session.push_frame(fp);
        }
        let frames = session.frames_snapshot();
        let dump = format_frame_dump(&frames, 3);
        assert!(
            dump.contains("0 in BOTH") && dump.contains("NOT the slow uploads"),
            "the spikes are on frames 30 and 10; the dump must say they do not \
             coincide:\n{dump}"
        );
        assert!(
            dump.contains("1 late frame(s), 1 slow-submit frame(s)"),
            "each series has exactly one outlier:\n{dump}"
        );
        assert!(
            dump.lines().any(|l| l.trim_start().starts_with("30 ")),
            "the late frame must be named:\n{dump}"
        );
    }

    /// ...and must confirm it when they do, or an empty intersection means nothing.
    #[test]
    fn a_frame_dump_confirms_the_upload_lead_when_the_spikes_coincide() {
        let session = ProfilingSession::new(60.0);
        for i in 0..40 {
            let mut fp = FrameProfile::new(i, 0);
            let slow = i == 30;
            fp.record_stage(
                PipelineStage::UploadSubmit,
                if slow { Duration::from_millis(45) } else { Duration::from_millis(5) },
            );
            fp.record_latency(if slow {
                Duration::from_millis(54)
            } else {
                Duration::from_millis(20)
            });
            session.push_frame(fp);
        }
        let frames = session.frames_snapshot();
        let dump = format_frame_dump(&frames, 3);
        assert!(
            dump.contains("1 in BOTH") && dump.contains("frames [30]"),
            "the coinciding spike must be named as the shared outlier:\n{dump}"
        );
        let r = pearson(
            &frames.iter().filter_map(|f| f.latency_ms()).collect::<Vec<_>>(),
            &frames
                .iter()
                .filter(|f| f.latency.is_some())
                .map(|f| f.stage_time_ms(PipelineStage::UploadSubmit))
                .collect::<Vec<_>>(),
        )
        .expect("both series vary");
        assert!(r > 0.9, "one shared spike must correlate strongly, got {r:.3}");
    }

    /// A stage that is flat across every frame has NO outliers, so it can never be
    /// the shared one.
    ///
    /// This is the case a rank-based overlap gets wrong, and it is the live one:
    /// `↳ submit` on the `write_texture` path is sub-microsecond every frame, so
    /// "the 3 slowest uploads" is 3 arbitrary frames. Ranking would then report
    /// them as coinciding with the latency tail and the dump would confirm the
    /// upload lead on a run where the upload does nothing.
    #[test]
    fn a_flat_stage_is_never_the_shared_outlier() {
        let session = ProfilingSession::new(60.0);
        for i in 0..40 {
            let mut fp = FrameProfile::new(i, 0);
            // Identical on every frame — the write_texture path's submit cost.
            fp.record_stage(PipelineStage::UploadSubmit, Duration::from_micros(1));
            fp.record_latency(if i == 30 {
                Duration::from_millis(54)
            } else {
                Duration::from_millis(20)
            });
            session.push_frame(fp);
        }
        let frames = session.frames_snapshot();
        let dump = format_frame_dump(&frames, 3);
        assert!(
            dump.contains("0 slow-submit frame(s)"),
            "a flat series has no outliers at all:\n{dump}"
        );
        assert!(
            dump.contains("0 in BOTH"),
            "a stage with no outliers cannot share one with latency:\n{dump}"
        );
    }

    /// The outlier rule must be robust to the tail it is looking for.
    #[test]
    fn outlier_threshold_is_robust_and_empty_on_a_flat_series() {
        // 19 fives and one 45: the 45 must be above the threshold and the fives
        // below it. A mean+3σ rule fails here — the mean is already 7 and σ ≈ 8.7,
        // so mean+3σ ≈ 33 and the spike that produced the spread is what widened
        // the gate.
        let mut s = vec![5.0; 19];
        s.push(45.0);
        let t = outlier_threshold(&s).expect("non-empty");
        assert!(t > 5.0 && t < 45.0, "threshold {t} must separate 5 from 45");

        // Flat: nothing may be selected, at any magnitude — including all-zero,
        // which is what an unmeasured stage's series looks like.
        for flat in [vec![2.0; 10], vec![0.0; 10], vec![0.0009; 40]] {
            let t = outlier_threshold(&flat).expect("non-empty");
            assert!(
                !flat.iter().any(|v| *v > t),
                "a flat series must select no outliers; {:?} against threshold {t}",
                &flat[..2]
            );
        }
        assert!(outlier_threshold(&[]).is_none());
    }

    /// A correlation over a flat series is not 0.0 — it does not exist.
    #[test]
    fn pearson_refuses_a_series_without_variance() {
        assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(pearson(&[1.0, 2.0], &[1.0, 2.0]).is_none(), "2 pairs is not a correlation");
        assert!(pearson(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_none(), "lengths must match");
        let r = pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).expect("both vary");
        assert!((r - 1.0).abs() < 1e-9, "got {r}");
    }

    #[test]
    fn test_percentile_computation() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = compute_distribution_stats(&mut samples);
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, 10.0);
        assert_eq!(stats.avg_ms, 5.5);
        assert_eq!(stats.p95_ms, 10.0);
        assert_eq!(stats.p99_ms, 10.0);
    }
}
