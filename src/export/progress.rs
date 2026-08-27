use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub frames_done: usize,
    pub total_frames: usize,
    pub elapsed: Duration,
    pub eta: Option<Duration>,
    pub fps: f64,
    pub phase: ExportPhase,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportPhase {
    Rendering,
    Encoding,
    Muxing,
    Paused,
    Cancelled,
    Done,
    Failed(String),
}

impl ProgressUpdate {
    pub fn compute_eta(frames_done: usize, total_frames: usize, elapsed: Duration) -> Option<Duration> {
        if frames_done < 5 || frames_done >= total_frames {
            return None;
        }
        let fps = frames_done as f64 / elapsed.as_secs_f64();
        if fps < 0.001 {
            return None;
        }
        let remaining_frames = total_frames - frames_done;
        let remaining_secs = remaining_frames as f64 / fps;
        Some(Duration::from_secs_f64(remaining_secs))
    }
}

/// Thread-safe controller for pausing, resuming, and gracefully cancelling an active export.
#[derive(Debug, Clone, Default)]
pub struct ExportControl {
    cancel: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
}

impl ExportControl {
    pub fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            pause: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn pause(&self) {
        self.pause.store(true, Ordering::SeqCst);
    }

    pub fn resume(&self) {
        self.pause.store(false, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn is_paused(&self) -> bool {
        self.pause.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
pub struct ProgressSender {
    tx: mpsc::Sender<ProgressUpdate>,
    start: Instant,
    total: usize,
    control: ExportControl,
}

impl ProgressSender {
    pub fn report(&self, frames_done: usize, phase: ExportPhase) {
        let elapsed = self.start.elapsed();
        let eta = ProgressUpdate::compute_eta(frames_done, self.total, elapsed);
        let fps = if elapsed.as_secs_f64() > 0.0 {
            frames_done as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let _ = self.tx.send(ProgressUpdate {
            frames_done,
            total_frames: self.total,
            elapsed,
            eta,
            fps,
            phase,
        });
    }

    pub fn control(&self) -> &ExportControl {
        &self.control
    }
}

pub struct ProgressReceiver {
    rx: mpsc::Receiver<ProgressUpdate>,
    control: ExportControl,
}

impl ProgressReceiver {
    pub fn try_recv(&self) -> Option<ProgressUpdate> {
        self.rx.try_recv().ok()
    }

    pub fn recv(&self) -> Option<ProgressUpdate> {
        self.rx.recv().ok()
    }

    pub fn control(&self) -> &ExportControl {
        &self.control
    }

    pub fn cancel(&self) {
        self.control.cancel();
    }

    pub fn pause(&self) {
        self.control.pause();
    }

    pub fn resume(&self) {
        self.control.resume();
    }
}

pub fn progress_channel(total_frames: usize) -> (ProgressSender, ProgressReceiver) {
    let (tx, rx) = mpsc::channel();
    let control = ExportControl::new();
    (
        ProgressSender {
            tx,
            start: Instant::now(),
            total: total_frames,
            control: control.clone(),
        },
        ProgressReceiver { rx, control },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_control_flags() {
        let control = ExportControl::new();
        assert!(!control.is_cancelled());
        assert!(!control.is_paused());

        control.pause();
        assert!(control.is_paused());

        control.resume();
        assert!(!control.is_paused());

        control.cancel();
        assert!(control.is_cancelled());
    }

    #[test]
    fn test_eta_computation() {
        let elapsed = Duration::from_secs(10);
        let eta = ProgressUpdate::compute_eta(100, 200, elapsed);
        assert!(eta.is_some());
        let eta_dur = eta.unwrap();
        assert_eq!(eta_dur.as_secs(), 10);
    }
}

