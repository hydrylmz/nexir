// src/sync/master_clock.rs

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use crate::timeline::rational::Rational;

/// Authoritative engine clock.
/// PTS is advanced by the audio callback thread and read by all other threads.
pub struct MasterClock {
    /// Total audio samples consumed since clock start (or last seek).
    /// Updated atomically by the audio callback. Never decreases.
    total_samples: AtomicU64,
    /// PTS of the clock at the moment total_samples was last reset to 0.
    /// Set on construction and on each seek.
    start_pts:     AtomicI64,
    /// Wall-clock nanoseconds at the moment start_pts was set.
    /// Used for drift estimation when audio is absent.
    wall_ref_ns:   AtomicI64,
    /// Project timebase.
    project_tb:    Rational,
    /// Audio output sample rate (Hz). Immutable after construction.
    sample_rate:   u32,
}

impl MasterClock {
    pub fn new(project_tb: Rational, sample_rate: u32) -> Arc<Self> {
        Arc::new(Self {
            total_samples: AtomicU64::new(0),
            start_pts:     AtomicI64::new(0),
            wall_ref_ns:   AtomicI64::new(now_ns()),
            project_tb,
            sample_rate,
        })
    }

    /// Read the current PTS.
    pub fn pts(&self) -> i64 {
        let s = self.total_samples.load(Ordering::Acquire);
        let pts_offset = (s as u128 * self.project_tb.den as u128
                          / self.sample_rate as u128) as i64;
        self.start_pts.load(Ordering::Acquire) + pts_offset
    }

    /// Advance the clock by `sample_count` audio samples.
    pub fn advance_samples(&self, sample_count: usize) {
        self.total_samples.fetch_add(sample_count as u64, Ordering::Release);
    }

    /// Reset the clock to a new start PTS (called on seek).
    pub fn seek(&self, new_pts: i64) {
        self.total_samples.store(0, Ordering::Release);
        self.start_pts.store(new_pts, Ordering::Release);
        self.wall_ref_ns.store(now_ns(), Ordering::Release);
    }

    /// Estimate drift between audio clock and wall clock (nanoseconds).
    pub fn wall_drift_ns(&self) -> i64 {
        let now = now_ns();
        let wall_ref = self.wall_ref_ns.load(Ordering::Acquire);
        let pts_elapsed = self.pts() - self.start_pts.load(Ordering::Acquire);
        let expected_elapsed_ns = self.project_tb.pts_to_ns(pts_elapsed);
        now - (wall_ref + expected_elapsed_ns)
    }
}

/// Current wall clock in nanoseconds since an arbitrary epoch.
fn now_ns() -> i64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_nanos() as i64
}
