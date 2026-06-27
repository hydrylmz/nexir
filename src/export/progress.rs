use std::time::{Duration, Instant};
use std::sync::mpsc;

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub frames_done:  usize,
    pub total_frames: usize,
    pub elapsed:      Duration,
    pub eta:          Option<Duration>,
    pub fps:          f64,
    pub phase:        ExportPhase,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExportPhase {
    Rendering,
    Encoding,
    Muxing,
    Done,
    Failed(String),
}

impl ProgressUpdate {
    pub fn compute_eta(frames_done: usize, total_frames: usize, elapsed: Duration) -> Option<Duration> {
        if frames_done < 10 {
            return None;
        }
        let fps = frames_done as f64 / elapsed.as_secs_f64();
        if fps < 0.001 {
            return None;
        }
        let remaining = (total_frames - frames_done) as f64 / fps;
        Some(Duration::from_secs_f64(remaining))
    }
}

#[derive(Clone)]
pub struct ProgressSender {
    tx:    mpsc::Sender<ProgressUpdate>,
    start: Instant,
    total: usize,
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
}

pub struct ProgressReceiver {
    rx: mpsc::Receiver<ProgressUpdate>,
}

impl ProgressReceiver {
    pub fn try_recv(&self) -> Option<ProgressUpdate> {
        self.rx.try_recv().ok()
    }

    pub fn recv(&self) -> Option<ProgressUpdate> {
        self.rx.recv().ok()
    }
}

pub fn progress_channel(total_frames: usize) -> (ProgressSender, ProgressReceiver) {
    let (tx, rx) = mpsc::channel();
    (
        ProgressSender { tx, start: Instant::now(), total: total_frames },
        ProgressReceiver { rx }
    )
}
