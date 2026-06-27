// src/sync/drift_corrector.rs

use crate::sync::master_clock::MasterClock;
use std::sync::Arc;

pub const KP: f64 = 0.1;
pub const KI: f64 = 0.01;
pub const MAX_CORRECTION_NS: f64 = 5_000_000.0; // 5 ms/s

pub struct DriftCorrector {
    clock:    Arc<MasterClock>,
    integral: f64,       // accumulated integral term (nanoseconds)
    last_update_ns: i64, // wall clock of last update() call
}

impl DriftCorrector {
    pub fn new(clock: Arc<MasterClock>) -> Self {
        Self {
            clock,
            integral: 0.0,
            last_update_ns: now_ns(),
        }
    }

    /// Update the controller with a new measurement.
    pub fn update(&mut self) -> f64 {
        let now = now_ns();
        let mut dt = (now - self.last_update_ns) as f64 / 1_000_000_000.0;
        self.last_update_ns = now;
        
        dt = dt.clamp(0.001, 0.1);

        let error = self.clock.wall_drift_ns() as f64;

        self.integral += KI * error * dt;
        self.integral = self.integral.clamp(-MAX_CORRECTION_NS, MAX_CORRECTION_NS);

        let output = KP * error + self.integral;
        output.clamp(-MAX_CORRECTION_NS, MAX_CORRECTION_NS)
    }

    /// Reset the controller (call on seek to discard accumulated integral).
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_update_ns = now_ns();
    }
}

/// Current wall clock in nanoseconds since an arbitrary epoch.
fn now_ns() -> i64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_nanos() as i64
}
