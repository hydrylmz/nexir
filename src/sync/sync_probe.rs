// src/sync/sync_probe.rs

#[cfg(any(test, debug_assertions))]
pub struct SyncProbe {
    /// Recorded (video_pts, audio_pts) pairs, one per presented frame.
    samples: Vec<(i64, i64)>,
    project_tb: crate::timeline::rational::Rational,
}

#[cfg(any(test, debug_assertions))]
impl SyncProbe {
    pub fn new(project_tb: crate::timeline::rational::Rational) -> Self {
        Self { samples: Vec::new(), project_tb }
    }

    /// Record one frame's A/V PTS pair at the moment of presentation.
    pub fn record(&mut self, video_pts: i64, audio_pts: i64) {
        self.samples.push((video_pts, audio_pts));
    }

    /// Compute the RMS A/V synchronisation error in nanoseconds.
    pub fn rms_error_ns(&self) -> f64 {
        if self.samples.is_empty() { return 0.0; }
        let mut errors = Vec::with_capacity(self.samples.len());
        for (video_pts, audio_pts) in &self.samples {
            let diff_pts = video_pts - audio_pts;
            let diff_ns  = self.project_tb.pts_to_ns(diff_pts);
            errors.push(diff_ns as f64);
        }
        let mean_sq = errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64;
        mean_sq.sqrt()
    }

    /// Maximum absolute error across all samples (nanoseconds).
    pub fn max_error_ns(&self) -> f64 {
        self.samples.iter()
            .map(|(v, a)| self.project_tb.pts_to_ns(v - a).abs() as f64)
            .fold(0.0f64, f64::max)
    }

    /// Number of frames recorded.
    pub fn frame_count(&self) -> usize {
        self.samples.len()
    }

    /// Histogram of errors bucketed into 1 ms bins (for diagnostics).
    pub fn histogram_ms(&self) -> std::collections::HashMap<i64, usize> {
        let mut histogram = std::collections::HashMap::new();
        for (v, a) in &self.samples {
            let bin = self.project_tb.pts_to_ns(v - a) / 1_000_000;
            *histogram.entry(bin).or_insert(0) += 1;
        }
        histogram
    }
}
