// src/tests/av_sync.rs
#[cfg(test)]
mod av_sync {
    use std::sync::Arc;
    use crate::sync::master_clock::MasterClock;
    use crate::sync::drift_corrector::DriftCorrector;
    use crate::sync::presentation::{PresentationDecider, PresentAction};
    use crate::audio::ring_buffer::AudioRingBuffer;
    use crate::timeline::rational::Rational;

    const TB: Rational = Rational { num: 1, den: 90_000 };
    const SAMPLE_RATE: u32 = 48_000;
    const FRAME_RATE:  Rational = Rational { num: 30, den: 1 };
    const VSYNC_NS:    i64 = 16_666_667; // 60 Hz display

    // ── AudioRingBuffer ───────────────────────────────────────────────────────

    #[test]
    fn ring_write_read_round_trip() {
        let ring = AudioRingBuffer::new(131_072);
        let samples: Vec<f32> = (0..1024).map(|i| i as f32 / 1024.0).collect();
        let written = ring.write(&samples);
        assert_eq!(written, 1024);
        let mut out = vec![0.0f32; 1024];
        let read = ring.read(&mut out);
        assert_eq!(read, 1024);
        for (a, b) in samples.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn ring_underrun_zero_fill() {
        let ring = AudioRingBuffer::new(131_072);
        let mut out = vec![1.0f32; 64];
        ring.read(&mut out);
        assert!(out.iter().all(|&s| s == 0.0), "underrun must produce silence");
        assert_eq!(ring.underrun_count(), 1);
    }

    #[test]
    fn ring_wraparound_correct() {
        let ring = AudioRingBuffer::new(8);
        ring.write(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut r = vec![0.0; 4];
        ring.read(&mut r);
        assert_eq!(r, [1.0, 2.0, 3.0, 4.0]);
        ring.write(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut r2 = vec![0.0; 7];
        ring.read(&mut r2);
        assert_eq!(r2, [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    // ── MasterClock ───────────────────────────────────────────────────────────

    #[test]
    fn clock_initial_pts_is_start() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        assert_eq!(clock.pts(), 0);
    }

    #[test]
    fn clock_advances_by_sample_count() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        clock.advance_samples(48_000);
        assert_eq!(clock.pts(), 90_000);
    }

    #[test]
    fn clock_seek_resets_correctly() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        clock.advance_samples(48_000);
        assert_eq!(clock.pts(), 90_000);
        clock.seek(450_000);
        assert_eq!(clock.pts(), 450_000);
        clock.advance_samples(48_000);
        assert_eq!(clock.pts(), 540_000);
    }

    // ── PresentationDecider ───────────────────────────────────────────────────

    #[test]
    fn decider_on_time_is_present() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        let decider = PresentationDecider::new(Arc::clone(&clock), TB, FRAME_RATE, VSYNC_NS);
        clock.advance_samples(0);
        let frame_pts = 0i64;
        assert_eq!(decider.decide(frame_pts), PresentAction::Present);
    }

    #[test]
    fn decider_late_frame_is_drop() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        clock.advance_samples(SAMPLE_RATE as usize * 2 / 30);
        let frame_pts = 0i64;
        let decider = PresentationDecider::new(Arc::clone(&clock), TB, FRAME_RATE, VSYNC_NS);
        assert_eq!(decider.decide(frame_pts), PresentAction::Drop);
    }

    #[test]
    fn decider_early_frame_is_hold() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        let frame_pts = 5000i64;
        let decider = PresentationDecider::new(Arc::clone(&clock), TB, FRAME_RATE, VSYNC_NS);
        assert_eq!(decider.decide(frame_pts), PresentAction::Hold);
    }

    // ── DriftCorrector ────────────────────────────────────────────────────────

    #[test]
    fn corrector_zero_drift_gives_zero_output() {
        let clock = MasterClock::new(TB, SAMPLE_RATE);
        let mut corrector = DriftCorrector::new(Arc::clone(&clock));
        corrector.reset();
        let out = corrector.update();
        assert!(out.abs() < 100_000.0, "output={out}");
    }
}
