// src/profiling/mod.rs

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
    pub frame_pts: i64,
}

impl FrameProfile {
    pub fn new(frame_index: usize, frame_pts: i64) -> Self {
        Self {
            frame_index,
            stage_durations: BTreeMap::new(),
            gpu_durations: BTreeMap::new(),
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
    pub fn total_time(&self) -> Duration {
        self.stage_durations.values().copied().sum()
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
    pub average_fps: f64,
    pub p99_fps: f64,
    pub realtime_factor: f64,
    pub stage_stats: BTreeMap<PipelineStage, StageStats>,
    pub total_frame_stats: StageStats,
    pub system_metrics: SystemMetrics,
}

impl ProfileReport {
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
            if stats.count > 0 && stats.total_ms > 0.0 {
                out.push_str(&format!(
                    "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>7.1}%",
                    stage.display_name(),
                    stats.avg_ms,
                    stats.min_ms,
                    stats.p95_ms,
                    stats.p99_ms,
                    stats.percentage_of_total
                ));
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
        }

        out.push_str("------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>7.1}%\n",
            "Total Frame",
            self.total_frame_stats.avg_ms,
            self.total_frame_stats.min_ms,
            self.total_frame_stats.p95_ms,
            self.total_frame_stats.p99_ms,
            100.0
        ));
        if any_gpu {
            // Stated explicitly because the columns invite the addition: the CPU
            // total and the GPU column measure overlapping wall time once frames
            // are in flight, so they do not sum to a frame duration.
            out.push_str(
                "  (CPU stages sum to Total Frame; GPU columns are concurrent GPU \
                 execution, not additive)\n",
            );
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

        let avg_fps = if total_frame_stats.avg_ms > 0.0 {
            1000.0 / total_frame_stats.avg_ms
        } else {
            0.0
        };

        let p99_fps = if total_frame_stats.p99_ms > 0.0 {
            1000.0 / total_frame_stats.p99_ms
        } else {
            0.0
        };

        let target_frame_dur_ms = 1000.0 / inner.target_fps;
        let realtime_factor = if total_frame_stats.avg_ms > 0.0 {
            target_frame_dur_ms / total_frame_stats.avg_ms
        } else {
            0.0
        };

        ProfileReport {
            total_frames: frame_count,
            total_wall_time: wall_time,
            average_fps: avg_fps,
            p99_fps,
            realtime_factor,
            stage_stats,
            total_frame_stats,
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
