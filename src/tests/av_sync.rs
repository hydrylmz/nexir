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

    /// The correction can never exceed its clamp, however bad the drift gets.
    ///
    /// This is the safety property of the controller: `update()` feeds a rate
    /// adjustment, so an unclamped output — or an integral that winds up over a
    /// long session — would yank the clock and produce an audible pitch jump
    /// rather than a slow correction.
    ///
    /// The drift is made pathological on purpose: advancing the clock by an hour
    /// of samples in an instant makes `wall_drift_ns` about -3.6e12, seven orders
    /// of magnitude past the 5 ms/s clamp. `KP * error` alone is 3.6e11.
    #[test]
    fn corrector_output_stays_clamped_under_extreme_drift() {
        use crate::sync::drift_corrector::MAX_CORRECTION_NS;

        let clock = MasterClock::new(TB, SAMPLE_RATE);
        let mut corrector = DriftCorrector::new(Arc::clone(&clock));

        // One hour of audio consumed with no wall time passing.
        clock.advance_samples(SAMPLE_RATE as usize * 3600);

        let mut worst = 0.0f64;
        for _ in 0..1000 {
            let out = corrector.update();
            worst = worst.max(out.abs());
            assert!(
                out.abs() <= MAX_CORRECTION_NS,
                "correction {out} exceeded the {MAX_CORRECTION_NS} ns/s clamp — \
                 the integral term has wound up"
            );
        }
        // The clamp must actually be the thing limiting the output, otherwise this
        // test would pass on a controller that does nothing at all.
        assert!(
            (worst - MAX_CORRECTION_NS).abs() < 1.0,
            "expected the output to saturate at {MAX_CORRECTION_NS}, got {worst}"
        );

        // A seek re-references the clock and `reset` discards the integral, so the
        // controller must come back to rest rather than carrying the hour of error.
        clock.seek(0);
        corrector.reset();
        let after = corrector.update();
        assert!(
            after.abs() < 1_000_000.0,
            "after a seek + reset the correction is still {after} ns/s — the \
             integral survived the reset"
        );
    }

    // ── P1.8 — long-duration A/V sync ─────────────────────────────────────────

    /// Three hours of simulated playback, asserting the A/V error stays bounded.
    ///
    /// WHAT THIS COVERS, and why the short unit tests above do not: every
    /// assertion in this file until now looks at a single decision. Sync failures
    /// in a real editor are cumulative — a timebase conversion that truncates a
    /// fraction of a tick per frame, or a presentation rule that never recovers
    /// the frames it falls behind by, is invisible after ten frames and half a
    /// second off after two hours. This drives the real `MasterClock`,
    /// `PresentationDecider` and `SyncProbe` across three hours of playback and
    /// asserts the error does not grow.
    ///
    /// SIMULATED, deliberately: no audio device, no GPU, no sleeping. Wall time is
    /// a counter, so three hours of playback run in milliseconds. What is real is
    /// the arithmetic under test — sample counting, PTS conversion and the
    /// drop/hold/present decision.
    ///
    /// THE MODEL, and each piece is there to create a specific failure mode:
    ///
    /// * **The audio device runs 200 ppm fast** (`DEVICE_PPM`), within spec for
    ///   consumer hardware. `MasterClock` derives PTS from the sample count, so
    ///   the master clock gains on the wall clock — about 2.16 s over three hours.
    ///   The video timeline advances by exactly one frame duration per presented
    ///   frame, so it must shed roughly 65 frames across the run to keep up. This
    ///   is the drift the presenter has to correct.
    /// * **The decoder runs faster than playback** (`DECODE_NS` = 20 ms per frame
    ///   against a 33.3 ms frame interval), which is what the prefetch worker and
    ///   frame cache buy in the real engine. Without that headroom a dropped frame
    ///   could never be recovered — the presenter would fall behind permanently
    ///   and drop everything, which is a property of the model, not of the code.
    /// * **Every 997th frame stalls an extra 250 ms** — a decode hiccup, a cache
    ///   miss after a scrub, a shader recompile. 250 ms is chosen to be larger
    ///   than one frame duration plus the hold window, which is what makes the
    ///   frames behind it genuinely late and forces `PresentAction::Drop`. A
    ///   smaller stall is absorbed inside the present window and never exercises
    ///   the drop path at all (a 40 ms stall does not: it leaves the next frame
    ///   27 ms late, still inside the 33.3 ms window).
    ///
    /// `dropped > 0` below is what proves the recovery path actually engaged
    /// rather than the test being vacuous.
    ///
    /// WHAT IS ASSERTED. Bounded RMS error; a worst case within one frame
    /// duration plus the hold window; the second half's RMS no worse than the
    /// first half's (the assertion that catches accumulation — a truncating
    /// conversion sails past the first two and fails this); and `MasterClock::pts`
    /// still exact after 3 h of callbacks.
    #[test]
    fn long_duration_playback_drift_stays_bounded() {
        use crate::sync::sync_probe::SyncProbe;
        use crate::timeline::rational::frame_to_pts;

        /// Simulated playback length. Three hours is past the point where a
        /// per-frame truncation of even one 90 kHz tick (11 µs) would show up: it
        /// would reach 3.6 s.
        const HOURS: i64 = 3;
        const TOTAL_NS: i64 = HOURS * 3600 * 1_000_000_000;
        /// Audio callback size. 441 samples does NOT divide evenly into the 90 kHz
        /// timebase (441 * 90000 / 48000 = 826.875 ticks), so if `MasterClock`
        /// truncated per callback instead of deriving PTS from the cumulative
        /// sample count it would lose 0.875 ticks every 9 ms — 3.4 s over this run.
        const CALLBACK: usize = 441;
        /// Audio device rate error in parts per million. Positive = running fast.
        const DEVICE_PPM: f64 = 200.0;
        /// Wall time the decoder needs per frame. Below the 33.3 ms frame interval
        /// so the pipeline has the headroom to catch up after falling behind.
        const DECODE_NS: i64 = 20_000_000;
        /// Extra decode cost on every 997th frame — a hiccup, not a trend. Larger
        /// than one frame duration plus the hold window so the frames behind it
        /// are genuinely late and the presenter has to drop to recover.
        const STALL_NS: i64 = 250_000_000;
        /// How far wall time advances while the presenter waits for an early frame.
        const HOLD_QUANTUM_NS: i64 = 1_000_000;

        let clock = MasterClock::new(TB, SAMPLE_RATE);
        let decider =
            PresentationDecider::new(Arc::clone(&clock), TB, FRAME_RATE, VSYNC_NS);

        let mut probe = SyncProbe::new(TB);
        let mut first_half = SyncProbe::new(TB);
        let mut second_half = SyncProbe::new(TB);

        // Wall nanoseconds per audio callback, with the device's rate error folded
        // in: a fast device delivers the same samples in less wall time.
        let callback_wall_ns =
            (CALLBACK as f64 * 1_000_000_000.0 / (SAMPLE_RATE as f64 * (1.0 + DEVICE_PPM / 1e6)))
                as i64;

        let mut sim_wall_ns: i64 = 0;
        let mut audio_deadline_ns: i64 = callback_wall_ns;
        let mut decode_ready_ns: i64 = 0;
        let mut frame_idx: i64 = 0;
        let mut presented: usize = 0;
        let mut dropped: usize = 0;
        let mut held: usize = 0;
        let mut total_samples: u64 = 0;

        while sim_wall_ns < TOTAL_NS {
            // Deliver every audio callback whose deadline has passed. This is the
            // only thing that advances the master clock, exactly as in the real
            // engine where the CPAL callback owns it.
            while audio_deadline_ns <= sim_wall_ns {
                clock.advance_samples(CALLBACK);
                total_samples += CALLBACK as u64;
                audio_deadline_ns += callback_wall_ns;
            }

            // Nothing decoded yet: wall time moves to when the next frame lands.
            if decode_ready_ns > sim_wall_ns {
                sim_wall_ns = decode_ready_ns;
                continue;
            }

            let frame_pts = frame_to_pts(frame_idx, FRAME_RATE, TB);
            let stall = if frame_idx % 997 == 0 { STALL_NS } else { 0 };

            match decider.decide(frame_pts) {
                PresentAction::Present => {
                    let audio_pts = clock.pts();
                    probe.record(frame_pts, audio_pts);
                    if sim_wall_ns * 2 < TOTAL_NS {
                        first_half.record(frame_pts, audio_pts);
                    } else {
                        second_half.record(frame_pts, audio_pts);
                    }
                    presented += 1;
                    frame_idx += 1;
                    decode_ready_ns = sim_wall_ns + DECODE_NS + stall;
                    // A presented frame occupies the display for one refresh.
                    sim_wall_ns += VSYNC_NS;
                }
                PresentAction::Drop => {
                    // Discarding costs only the decode of the replacement, which
                    // is how the pipeline claws back the drift.
                    dropped += 1;
                    frame_idx += 1;
                    decode_ready_ns = sim_wall_ns + DECODE_NS + stall;
                }
                PresentAction::Hold => {
                    held += 1;
                    sim_wall_ns += HOLD_QUANTUM_NS;
                }
            }
        }

        let rms_ms = probe.rms_error_ns() / 1e6;
        let max_ms = probe.max_error_ns() / 1e6;
        eprintln!(
            "[av_sync] {HOURS}h simulated: {presented} presented, {dropped} dropped, \
             {held} holds, RMS {rms_ms:.3} ms, max {max_ms:.3} ms"
        );

        // The run has to have actually happened: ~30 fps over three hours.
        let expected_frames = (TOTAL_NS / 33_333_333) as usize;
        assert!(
            presented > expected_frames * 9 / 10,
            "only {presented} frames were presented in {HOURS}h, expected roughly \
             {expected_frames} — the simulation stalled instead of playing"
        );
        assert_eq!(
            probe.frame_count(), presented,
            "the probe did not record every presented frame"
        );

        // The drop path must have engaged: a 200 ppm fast audio clock puts the
        // video ~2.16 s behind over three hours, and shedding frames is the only
        // mechanism that recovers it.  Without this the test would assert nothing
        // about recovery.
        assert!(
            dropped > 0,
            "no frames were dropped over {HOURS}h with a {DEVICE_PPM} ppm audio \
             clock error — the drop-to-recover path is untested"
        );

        // One frame duration is 33.3 ms and the hold window is half a vsync
        // (8.3 ms), so a presenter that recovers keeps every sample inside
        // [-33.3, +8.3] ms.  An RMS beyond that window means the error is no
        // longer bounded by the decision rule.
        assert!(
            rms_ms < 25.0,
            "RMS A/V error over {HOURS}h is {rms_ms:.3} ms, expected under 25 ms \
             (one frame is 33.3 ms) — the presenter is not recovering the drift"
        );
        assert!(
            max_ms < 45.0,
            "worst-case A/V error over {HOURS}h is {max_ms:.3} ms; anything past \
             one frame plus the hold window (41.6 ms) means a frame was presented \
             that should have been dropped"
        );

        // The accumulation check.  A per-frame truncation passes both assertions
        // above for the first minutes and fails here.
        let first_ms  = first_half.rms_error_ns() / 1e6;
        let second_ms = second_half.rms_error_ns() / 1e6;
        eprintln!("[av_sync] RMS first half {first_ms:.3} ms, second half {second_ms:.3} ms");
        assert!(
            first_half.frame_count() > 0 && second_half.frame_count() > 0,
            "both halves of the run must contain presented frames"
        );
        assert!(
            second_ms <= first_ms + 3.0,
            "RMS error grew from {first_ms:.3} ms in the first half to \
             {second_ms:.3} ms in the second — the error is accumulating rather \
             than staying bounded"
        );

        // `MasterClock::pts` must still be exact after three hours of callbacks:
        // it derives PTS from the cumulative sample count in 128-bit arithmetic,
        // so the only error is the sub-tick truncation of a single read.
        let expected_pts =
            (total_samples as u128 * TB.den as u128 / SAMPLE_RATE as u128) as i64;
        assert_eq!(
            clock.pts(), expected_pts,
            "after {total_samples} samples the clock reads {} but should read \
             {expected_pts} — PTS is accumulating conversion error",
            clock.pts()
        );
    }
}
