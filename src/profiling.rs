// src/profiling/mod.rs

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Discrete pipeline stages measured during frame processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineStage {
    Decode,
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
    pub frame_pts: i64,
}

impl FrameProfile {
    pub fn new(frame_index: usize, frame_pts: i64) -> Self {
        Self {
            frame_index,
            stage_durations: BTreeMap::new(),
            frame_pts,
        }
    }

    /// Record the elapsed time for a given stage.
    pub fn record_stage(&mut self, stage: PipelineStage, duration: Duration) {
        *self.stage_durations.entry(stage).or_default() += duration;
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
}

/// System and resource utilization metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemMetrics {
    /// GPU core utilization percentage (0.0 .. 100.0)
    pub gpu_utilization: f32,
    /// CPU core utilization percentage (0.0 .. 100.0)
    pub cpu_utilization: f32,
    /// NVENC encoder hardware utilization percentage (0.0 .. 100.0)
    pub nvenc_utilization: f32,
    /// VRAM memory in use (bytes)
    pub vram_used_bytes: u64,
    /// System RAM memory in use (bytes)
    pub ram_used_bytes: u64,
    /// GPU memory upload transfer bytes (CPU -> GPU)
    pub gpu_upload_bytes: u64,
    /// GPU memory download transfer bytes (GPU -> CPU)
    pub gpu_download_bytes: u64,
    /// Frame queue depth
    pub frame_queue_depth: usize,
    /// Decode queue depth
    pub decode_queue_depth: usize,
    /// Encode queue depth
    pub encode_queue_depth: usize,
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
            "{:<16} {:>9} {:>9} {:>9} {:>9} {:>8}\n",
            "Stage", "Avg (ms)", "Min (ms)", "P95 (ms)", "P99 (ms)", "Share %"
        ));
        out.push_str("------------------------------------------------------------------------\n");

        for (stage, stats) in &self.stage_stats {
            if stats.count > 0 && stats.total_ms > 0.0 {
                out.push_str(&format!(
                    "{:<16} {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>8.2} ms {:>7.1}%\n",
                    stage.display_name(),
                    stats.avg_ms,
                    stats.min_ms,
                    stats.p95_ms,
                    stats.p99_ms,
                    stats.percentage_of_total
                ));
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
        out.push_str("------------------------------------------------------------------------\n");
        out.push_str(&format!(
            "System Metrics:\n  GPU Util:   {:>5.1}%  |  NVENC Util: {:>5.1}%  |  CPU Util:   {:>5.1}%\n  VRAM Usage: {:>5.1} MB |  RAM Usage:  {:>5.1} MB\n  Queue Depths: Frame={}, Decode={}, Encode={}\n",
            self.system_metrics.gpu_utilization,
            self.system_metrics.nvenc_utilization,
            self.system_metrics.cpu_utilization,
            self.system_metrics.vram_used_bytes as f64 / (1024.0 * 1024.0),
            self.system_metrics.ram_used_bytes as f64 / (1024.0 * 1024.0),
            self.system_metrics.frame_queue_depth,
            self.system_metrics.decode_queue_depth,
            self.system_metrics.encode_queue_depth
        ));
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
