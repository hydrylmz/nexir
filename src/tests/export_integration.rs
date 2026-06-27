#[cfg(test)]
mod export_integration {
    use std::path::PathBuf;
    use crate::export::job::{ExportJob, VideoCodec, AudioCodec, Container, VideoQuality};
    use crate::export::partitioner::SegmentPartitioner;
    use crate::export::progress::ExportPhase;
    use crate::timeline::rational::Rational;

    const TB: Rational = Rational { num: 1, den: 90_000 };
    const FPS: Rational = Rational { num: 30, den: 1 };

    #[test]
    fn total_frames_10_seconds() {
        let job = ExportJob::preset_web_h264(
            PathBuf::from("/tmp/test.mp4"), 0, 900_000, 1920, 1080, FPS, TB
        );
        assert_eq!(job.total_frames(), 300);
    }

    #[test]
    fn frame_pts_increments_correctly() {
        let job = ExportJob::preset_web_h264(
            PathBuf::from("/tmp/test.mp4"), 0, 900_000, 1920, 1080, FPS, TB
        );
        assert_eq!(job.frame_pts(0), 0);
        assert_eq!(job.frame_pts(1), 3_000);
        assert_eq!(job.frame_pts(299), 299 * 3_000);
    }

    #[test]
    fn validate_rejects_zero_duration() {
        let mut job = ExportJob::preset_web_h264(
            PathBuf::from("/tmp/test.mp4"), 0, 900_000, 1920, 1080, FPS, TB
        );
        job.pts_in = 100;
        job.pts_out = 100;
        assert!(job.validate().is_err());
    }

    #[test]
    fn partitioner_even_division() {
        let mut job = ExportJob::preset_web_h264(
            PathBuf::from("/tmp/test.mp4"), 0, 900_000, 1920, 1080, FPS, TB
        );
        job.render_threads = 4;
        let segs = SegmentPartitioner::partition(&job);
        assert_eq!(segs.len(), 4);
        for seg in &segs {
            assert_eq!(seg.frame_count(), 75);
        }
        assert!(SegmentPartitioner::are_contiguous(&segs[0], &segs[1]));
        assert!(SegmentPartitioner::are_contiguous(&segs[2], &segs[3]));
        assert_eq!(segs.last().unwrap().frame_end, 300);
    }

    #[test]
    fn partitioner_uneven_division() {
        let mut job = ExportJob::preset_web_h264(
            PathBuf::from("/tmp/test.mp4"), 0, 903_000, 1920, 1080, FPS, TB
        );
        job.render_threads = 4;
        let segs = SegmentPartitioner::partition(&job);
        assert_eq!(segs.len(), 4);
        let total_frames: usize = segs.iter().map(|s| s.frame_count()).sum();
        assert_eq!(total_frames, 301);
    }

    #[test]
    fn partitioner_segments_are_contiguous() {
        let mut job = ExportJob::preset_web_h264(
            PathBuf::from("/tmp/test.mp4"), 0, 300_000, 1920, 1080, FPS, TB
        );
        job.render_threads = 7;
        let segs = SegmentPartitioner::partition(&job);
        for i in 0..segs.len()-1 {
            assert!(SegmentPartitioner::are_contiguous(&segs[i], &segs[i+1]));
        }
    }

    #[test]
    fn encoder_queue_backpressure() {
        use crate::export::queue::{EncoderQueue, QueueItem};
        use crate::export::renderer::RawFrame;
        use std::sync::Arc;
        use std::thread;
        use std::time::{Duration, Instant};

        let queue = Arc::new(EncoderQueue::new());
        let qc = Arc::clone(&queue);

        thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            while let Some(_) = qc.pop() {}
        });

        let start = Instant::now();
        for i in 0..=4 {
            queue.push(QueueItem::Frame(RawFrame {
                frame_index: i, pts: 0, data: vec![0u8; 64]
            }));
        }
        assert!(start.elapsed() >= Duration::from_millis(50));
    }

    #[test]
    fn progress_eta_none_below_10_frames() {
        use std::time::Duration;
        use crate::export::progress::ProgressUpdate;
        let eta = ProgressUpdate::compute_eta(9, 100, Duration::from_secs(1));
        assert!(eta.is_none());
    }

    #[test]
    fn progress_eta_correct() {
        use std::time::Duration;
        use crate::export::progress::ProgressUpdate;
        let eta = ProgressUpdate::compute_eta(50, 100, Duration::from_secs(5));
        let eta_secs = eta.unwrap().as_secs_f64();
        assert!((eta_secs - 5.0).abs() < 0.1, "eta={}", eta_secs);
    }
}
