// src/sync/presentation.rs

use crate::sync::master_clock::MasterClock;
use crate::timeline::rational::Rational;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentAction {
    /// Frame is on time or within acceptable window — render and display it.
    Present,
    /// Frame is too early — skip this vsync, try again next frame.
    Hold,
    /// Frame is late — decode is behind; discard this frame and show the next one.
    Drop,
}

pub struct PresentationDecider {
    clock:          Arc<MasterClock>,
    project_tb:     Rational,
    frame_rate:     Rational,   // e.g. 30/1 or 60/1
    vsync_period_ns: i64,       // nanoseconds between display refreshes
}

impl PresentationDecider {
    pub fn new(
        clock:           Arc<MasterClock>,
        project_tb:      Rational,
        frame_rate:      Rational,
        vsync_period_ns: i64,
    ) -> Self {
        Self { clock, project_tb, frame_rate, vsync_period_ns }
    }

    /// Decide whether to drop, hold, or present a frame with the given PTS.
    pub fn decide(&self, frame_pts: i64) -> PresentAction {
        let frame_dur_ns = (self.frame_rate.den * 1_000_000_000i64) / self.frame_rate.num;
        let master_pts = self.clock.pts();
        let pts_diff   = frame_pts - master_pts;
        let drift_ns   = self.project_tb.pts_to_ns(pts_diff);

        if drift_ns < -frame_dur_ns {
            PresentAction::Drop
        } else if drift_ns > (self.vsync_period_ns / 2) {
            PresentAction::Hold
        } else {
            PresentAction::Present
        }
    }

    /// Compute the expected PTS of the next frame that should be presented.
    pub fn next_frame_pts(&self) -> i64 {
        let frame_dur_pts = self.project_tb.den / self.frame_rate.num;
        self.clock.pts() + frame_dur_pts
    }
}
