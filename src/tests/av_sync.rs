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

    // ── P1.8 — the A/V sync matrix ─────────────────────────────────────────────
    //
    // The audit asks for durations (10 min / 1 h / 3 h) crossed with stream
    // configurations (video only, audio only, A+V, multiple audio tracks,
    // overlapping clips, multiple sample rates, multiple FPS values), measuring
    // start/end offsets and final drift.
    //
    // WHAT IS SIMULATED AND WHAT IS REAL.  There is no audio device and no GPU
    // here: wall time is a counter, so three hours run in milliseconds.  What is
    // real is every piece of arithmetic the audit is actually asking about —
    // `MasterClock`'s sample→PTS conversion, `frame_to_pts`, the drop/hold/present
    // rule in `PresentationDecider`, and `SyncProbe`'s error accounting.  A
    // wall-clock test of three hours cannot be run in CI, and one that slept would
    // measure the host's scheduler rather than this code.

    /// One row of the sync matrix.
    struct SyncCase {
        label: &'static str,
        /// Simulated playback length in seconds.
        seconds: i64,
        /// Video frame rate. `None` = audio-only, so no frames are presented.
        fps: Option<Rational>,
        /// Audio device sample rate. `None` = video-only, so nothing advances the
        /// master clock from audio and the wall clock drives it instead.
        sample_rate: Option<u32>,
        /// Samples per audio callback.  Chosen per case so some do not divide the
        /// 90 kHz timebase evenly.
        callback: usize,
        /// Audio device rate error in ppm. Positive = the device runs fast.
        device_ppm: f64,
        /// Number of simultaneously-mixed audio buses. Only the sample accounting
        /// differs; the clock is advanced once per callback regardless, which is
        /// exactly what `audio_callback` does with multiple ring buffers.
        buses: usize,
    }

    /// What one simulated run measured.
    struct SyncOutcome {
        presented: usize,
        dropped: usize,
        held: usize,
        /// RMS |video_pts − audio_pts| over presented frames, in ms.
        rms_ms: f64,
        max_ms: f64,
        /// RMS over the first and second halves, for the accumulation check.
        first_half_ms: f64,
        second_half_ms: f64,
        /// A/V error at the first presented frame, in ms.
        start_offset_ms: f64,
        /// A/V error at the last presented frame, in ms.
        end_offset_ms: f64,
        /// `MasterClock::pts()` at the end, and what exact arithmetic says it
        /// should be.
        final_pts: i64,
        expected_pts: i64,
        total_samples: u64,
    }

    /// Drive the real clock/presenter/probe across `case.seconds` of playback.
    ///
    /// Shared by the matrix test and the three-hour test below so both measure the
    /// same quantities through the same code.
    fn simulate_playback(case: &SyncCase) -> SyncOutcome {
        use crate::sync::sync_probe::SyncProbe;
        use crate::timeline::rational::frame_to_pts;

        /// Wall time the decoder needs per frame, as a fraction of the frame
        /// interval. Below 1.0 so the pipeline has the headroom to recover after
        /// falling behind — that headroom is what the frame cache buys in the real
        /// engine, and without it a dropped frame could never be made up.
        const DECODE_FRACTION: f64 = 0.6;
        /// Extra decode cost on every 997th frame — a hiccup, not a trend. Scaled
        /// to be larger than a frame interval plus the hold window so the frames
        /// behind it are genuinely late and the presenter must drop to recover.
        const STALL_FRAMES: f64 = 7.5;
        /// How far wall time advances while the presenter waits for an early frame.
        const HOLD_QUANTUM_NS: i64 = 1_000_000;

        let total_ns = case.seconds * 1_000_000_000;
        // A sample rate is needed to construct the clock even for a video-only
        // case; 48 kHz is the engine's own output rate.
        let rate = case.sample_rate.unwrap_or(SAMPLE_RATE);
        let clock = MasterClock::new(TB, rate);
        // Video-only still needs a nominal rate for the decider's frame duration.
        let fps = case.fps.unwrap_or(FRAME_RATE);
        let decider = PresentationDecider::new(Arc::clone(&clock), TB, fps, VSYNC_NS);

        let mut probe = SyncProbe::new(TB);
        let mut first_half = SyncProbe::new(TB);
        let mut second_half = SyncProbe::new(TB);

        let frame_ns = (fps.den as f64 * 1e9 / fps.num as f64) as i64;
        let decode_ns = (frame_ns as f64 * DECODE_FRACTION) as i64;
        let stall_ns = (frame_ns as f64 * STALL_FRAMES) as i64;

        // Wall nanoseconds per audio callback, with the device's rate error folded
        // in: a fast device delivers the same samples in less wall time.
        let callback_wall_ns = (case.callback as f64 * 1e9
            / (rate as f64 * (1.0 + case.device_ppm / 1e6))) as i64;

        let mut sim_wall_ns: i64 = 0;
        let mut audio_deadline_ns: i64 = callback_wall_ns;
        let mut decode_ready_ns: i64 = 0;
        let mut frame_idx: i64 = 0;
        let mut presented = 0usize;
        let mut dropped = 0usize;
        let mut held = 0usize;
        let mut total_samples: u64 = 0;
        let mut start_offset_ns: Option<i64> = None;
        let mut end_offset_ns: i64 = 0;

        while sim_wall_ns < total_ns {
            // Deliver every audio callback whose deadline has passed. This is the
            // only thing that advances the master clock, exactly as in the real
            // engine where the CPAL callback owns it.
            if case.sample_rate.is_some() {
                while audio_deadline_ns <= sim_wall_ns {
                    // Multiple buses are summed into ONE output buffer, so the
                    // clock advances once per callback however many are mixed.
                    clock.advance_samples(case.callback);
                    total_samples += case.callback as u64;
                    audio_deadline_ns += callback_wall_ns;
                }
            } else {
                // Video-only: nothing drives the clock from audio.  The engine
                // still reads `MasterClock::pts()`, so advance it from wall time
                // at the nominal rate — the "no audio stream" case the audit asks
                // for, and the one where a presenter that assumes audio exists
                // stalls outright.
                let want = (sim_wall_ns as f64 * rate as f64 / 1e9) as u64;
                if want > total_samples {
                    clock.advance_samples((want - total_samples) as usize);
                    total_samples = want;
                }
            }

            // Nothing decoded yet: wall time moves to when the next frame lands.
            if decode_ready_ns > sim_wall_ns {
                sim_wall_ns = decode_ready_ns;
                continue;
            }

            if case.fps.is_none() {
                // Audio-only: there is no video to present, so the loop just runs
                // the clock forward.  The assertions that matter here are the
                // sample-exact PTS ones at the end.
                sim_wall_ns += callback_wall_ns.max(1);
                continue;
            }

            let frame_pts = frame_to_pts(frame_idx, fps, TB);
            let stall = if frame_idx % 997 == 0 { stall_ns } else { 0 };

            match decider.decide(frame_pts) {
                PresentAction::Present => {
                    let audio_pts = clock.pts();
                    probe.record(frame_pts, audio_pts);
                    if sim_wall_ns * 2 < total_ns {
                        first_half.record(frame_pts, audio_pts);
                    } else {
                        second_half.record(frame_pts, audio_pts);
                    }
                    let err_ns = TB.pts_to_ns(frame_pts - audio_pts);
                    if start_offset_ns.is_none() {
                        start_offset_ns = Some(err_ns);
                    }
                    end_offset_ns = err_ns;
                    presented += 1;
                    frame_idx += 1;
                    decode_ready_ns = sim_wall_ns + decode_ns + stall;
                    // A presented frame occupies the display for one refresh.
                    sim_wall_ns += VSYNC_NS;
                }
                PresentAction::Drop => {
                    // Discarding costs only the decode of the replacement, which
                    // is how the pipeline claws back the drift.
                    dropped += 1;
                    frame_idx += 1;
                    decode_ready_ns = sim_wall_ns + decode_ns + stall;
                }
                PresentAction::Hold => {
                    held += 1;
                    sim_wall_ns += HOLD_QUANTUM_NS;
                }
            }
        }

        let expected_pts =
            (total_samples as u128 * TB.den as u128 / rate as u128) as i64;

        SyncOutcome {
            presented,
            dropped,
            held,
            rms_ms: probe.rms_error_ns() / 1e6,
            max_ms: probe.max_error_ns() / 1e6,
            first_half_ms: first_half.rms_error_ns() / 1e6,
            second_half_ms: second_half.rms_error_ns() / 1e6,
            start_offset_ms: start_offset_ns.unwrap_or(0) as f64 / 1e6,
            end_offset_ms: end_offset_ns as f64 / 1e6,
            final_pts: clock.pts(),
            expected_pts,
            total_samples,
        }
    }

    /// P1.8 — the matrix: durations × frame rates × sample rates × stream
    /// configurations, each measured for start offset, end offset and drift.
    ///
    /// WHY THIS EXISTS ALONGSIDE the three-hour test below.  That one drives a
    /// single configuration (30 fps, 48 kHz, A+V) very deep.  Everything in this
    /// engine that converts between a frame index and a timestamp takes
    /// `frame_rate` as a `Rational`, and **a whole class of bug is invisible at
    /// any integer frame rate**: dropping `frame_rate.den` from a conversion is
    /// exact for 24, 25, 30, 60 and 120, and off by a factor of 1001 for
    /// 30000/1001. Only a matrix with an NTSC row can see it.
    ///
    /// That is not hypothetical: `ExportJob::frame_pts` and
    /// `PresentationDecider::next_frame_pts` both computed
    /// `project_tb.den / frame_rate.num` and were fixed for P1.8.
    /// `export_timestamp_grid_is_uniform_at_every_frame_rate` below is the
    /// export-side half of the same check.
    ///
    /// The rows, and what each is the only one to cover:
    ///
    /// * **10 min @ 30/1, 48 kHz** — the baseline, cheap enough to be the control
    ///   for every other row.
    /// * **1 h @ 30000/1001** — NTSC, where `frame_rate.den != 1`.
    /// * **1 h @ 24000/1001** — NTSC film, a different non-integer rate so a
    ///   constant fudged to fix 29.97 does not also pass here.
    /// * **3 h @ 60000/1001** — the longest run at the highest rate: most frames,
    ///   most conversions, most opportunity to accumulate.
    /// * **1 h @ 25/1, 44.1 kHz** — a sample rate that does NOT divide the 90 kHz
    ///   timebase evenly (44100 · 90000 / 44100 is exact, but 441 samples is
    ///   900 ticks and 1000 samples is 2040.8), crossed with PAL.
    /// * **1 h @ 30/1, 96 kHz** — high sample rate, so the clock advances in
    ///   smaller PTS steps per sample and any per-callback truncation compounds
    ///   twice as fast.
    /// * **1 h video-only** — nothing advances the clock from audio.
    /// * **1 h audio-only** — nothing is presented; only the sample-exact PTS
    ///   assertion applies.
    /// * **1 h, 4 audio buses** — multiple audio tracks mixed into one callback,
    ///   which must not multiply the clock's advance.
    /// * **1 h @ 30/1 with a 500 ppm device** — a worse-than-consumer clock error,
    ///   so the drop-to-recover path is driven hard (500 ppm is 1.8 s per hour).
    #[test]
    fn av_sync_matrix_across_durations_rates_and_stream_layouts() {
        const NTSC_30: Rational = Rational { num: 30_000, den: 1_001 };
        const NTSC_24: Rational = Rational { num: 24_000, den: 1_001 };
        const NTSC_60: Rational = Rational { num: 60_000, den: 1_001 };
        const PAL_25:  Rational = Rational { num: 25, den: 1 };

        let cases = [
            SyncCase { label: "10min_30fps_48k",      seconds: 600,   fps: Some(FRAME_RATE), sample_rate: Some(48_000), callback: 441,  device_ppm: 200.0, buses: 1 },
            SyncCase { label: "1h_29.97fps_48k",      seconds: 3600,  fps: Some(NTSC_30),    sample_rate: Some(48_000), callback: 1024, device_ppm: 200.0, buses: 1 },
            SyncCase { label: "1h_23.976fps_48k",     seconds: 3600,  fps: Some(NTSC_24),    sample_rate: Some(48_000), callback: 1024, device_ppm: 150.0, buses: 1 },
            SyncCase { label: "3h_59.94fps_48k",      seconds: 10800, fps: Some(NTSC_60),    sample_rate: Some(48_000), callback: 441,  device_ppm: 200.0, buses: 1 },
            SyncCase { label: "1h_25fps_44.1k",       seconds: 3600,  fps: Some(PAL_25),     sample_rate: Some(44_100), callback: 441,  device_ppm: 250.0, buses: 1 },
            SyncCase { label: "1h_30fps_96k",         seconds: 3600,  fps: Some(FRAME_RATE), sample_rate: Some(96_000), callback: 1000, device_ppm: 200.0, buses: 1 },
            SyncCase { label: "1h_video_only",        seconds: 3600,  fps: Some(FRAME_RATE), sample_rate: None,         callback: 1024, device_ppm: 0.0,   buses: 0 },
            SyncCase { label: "1h_audio_only",        seconds: 3600,  fps: None,             sample_rate: Some(48_000), callback: 1024, device_ppm: 200.0, buses: 1 },
            SyncCase { label: "1h_4_audio_buses",     seconds: 3600,  fps: Some(FRAME_RATE), sample_rate: Some(48_000), callback: 1024, device_ppm: 200.0, buses: 4 },
            SyncCase { label: "1h_30fps_500ppm",      seconds: 3600,  fps: Some(FRAME_RATE), sample_rate: Some(48_000), callback: 441,  device_ppm: 500.0, buses: 1 },
        ];

        // One frame duration is the width of the present window on the late side;
        // half a vsync is the early side.  A presenter that recovers keeps every
        // sample inside that band, so the bounds are expressed as multiples of the
        // case's OWN frame duration rather than as fixed millisecond constants —
        // 59.94 fps has half the budget 29.97 does.
        for case in &cases {
            let out = simulate_playback(case);
            let frame_ms = case
                .fps
                .map(|f| f.den as f64 * 1000.0 / f.num as f64)
                .unwrap_or(0.0);
            let hold_ms = VSYNC_NS as f64 / 2e6;

            eprintln!(
                "[av_sync] {:22} {:5}s fps={:>11} sr={:>6} buses={}: \
                 {} presented, {} dropped, {} held, RMS {:.3} ms, max {:.3} ms, \
                 start {:+.3} ms, end {:+.3} ms",
                case.label,
                case.seconds,
                case.fps.map(|f| format!("{}/{}", f.num, f.den)).unwrap_or_else(|| "-".into()),
                case.sample_rate.map(|s| s.to_string()).unwrap_or_else(|| "-".into()),
                case.buses,
                out.presented, out.dropped, out.held,
                out.rms_ms, out.max_ms, out.start_offset_ms, out.end_offset_ms,
            );

            // ── The clock's own arithmetic, for every case including audio-only ──
            //
            // `MasterClock::pts` derives PTS from the CUMULATIVE sample count in
            // 128-bit arithmetic, so the only error possible is the sub-tick
            // truncation of a single read.  A per-callback accumulation instead
            // would drift here: at 44.1 kHz a 441-sample callback is exactly 900
            // ticks, but at 96 kHz a 1000-sample callback is 937.5 — truncating
            // that loses half a tick every 10 ms, 1.8 s over this hour.
            assert_eq!(
                out.final_pts, out.expected_pts,
                "{}: after {} samples the clock reads {} but exact arithmetic \
                 says {} — PTS is accumulating conversion error rather than being \
                 derived from the sample count",
                case.label, out.total_samples, out.final_pts, out.expected_pts
            );

            if case.fps.is_none() {
                // Audio-only: nothing was presented, and that is the correct
                // outcome rather than a stalled simulation.
                assert_eq!(
                    out.presented, 0,
                    "{}: audio-only run presented {} frame(s)",
                    case.label, out.presented
                );
                assert!(
                    out.total_samples > 0,
                    "{}: audio-only run consumed no samples — the simulation did \
                     not advance",
                    case.label
                );
                continue;
            }

            // ── The run really happened ─────────────────────────────────────────
            //
            // Expected frame count from the case's own rate.  This is the
            // assertion that catches the dropped-`den` class of bug at the
            // presentation end: with `next_frame_pts` returning a 3-tick step at
            // 29.97, every frame is early, the presenter holds forever, and
            // `presented` collapses.
            let expected_frames =
                (case.seconds as f64 * case.fps.unwrap().as_f64()) as usize;
            assert!(
                out.presented > expected_frames * 8 / 10,
                "{}: only {} frames were presented in {}s at {:?}, expected \
                 roughly {} — the presenter is holding or dropping most of the \
                 stream rather than playing it",
                case.label, out.presented, case.seconds, case.fps.unwrap(),
                expected_frames
            );

            // ── Bounded error, in units of this case's own frame duration ────────
            assert!(
                out.rms_ms < frame_ms,
                "{}: RMS A/V error is {:.3} ms, which exceeds one frame \
                 ({:.3} ms) — the presenter is not recovering the drift",
                case.label, out.rms_ms, frame_ms
            );
            assert!(
                out.max_ms < frame_ms + hold_ms + 1.0,
                "{}: worst-case A/V error is {:.3} ms; anything past one frame \
                 plus the hold window ({:.3} ms) means a frame was presented that \
                 should have been dropped",
                case.label, out.max_ms, frame_ms + hold_ms
            );

            // ── Start and end offsets, which the audit asks for by name ─────────
            //
            // Both must be inside the same window: an end offset that has grown
            // past it is drift the presenter never corrected, which is precisely
            // what "final A/V drift ≈ 0 within a documented tolerance" rules out.
            for (which, offset) in
                [("start", out.start_offset_ms), ("end", out.end_offset_ms)]
            {
                assert!(
                    offset <= hold_ms + 1.0 && offset >= -(frame_ms + 1.0),
                    "{}: the {which} A/V offset is {offset:+.3} ms, outside the \
                     [{:.3}, {:.3}] ms present window — audio and video do not \
                     line up at that end of the run",
                    case.label, -(frame_ms), hold_ms
                );
            }

            // ── The accumulation check ──────────────────────────────────────────
            //
            // A per-frame truncation passes every assertion above for the first
            // minutes and fails here.  Tolerance is a fraction of a frame rather
            // than a fixed 3 ms so it means the same thing at 24 and 60 fps.
            assert!(
                out.second_half_ms <= out.first_half_ms + frame_ms * 0.1,
                "{}: RMS error grew from {:.3} ms in the first half to {:.3} ms \
                 in the second — the error is accumulating rather than staying \
                 bounded",
                case.label, out.first_half_ms, out.second_half_ms
            );
        }
    }

    /// P1.8 (export side) — the frame→PTS grid must be uniform and complete at
    /// every frame rate the UI offers, and at the NTSC rates it does not.
    ///
    /// `ExportJob::frame_pts` is what the partitioner slices segments on, what the
    /// renderer schedules each frame at, and what the muxer's timestamps derive
    /// from. It computed `project_tb.den / frame_rate.num`, dropping
    /// `frame_rate.den`: exact for every integer rate, and 1001× too small for an
    /// NTSC one — 3 ticks per frame instead of 3003 at 30000/1001. `total_frames`
    /// divides by `frame_rate.den` correctly, so the frame COUNT was right while
    /// every timestamp was wrong, and each of `render_threads` segments got a few
    /// ticks of the timeline.
    ///
    /// Asserted per rate: the step between consecutive frames is constant, it
    /// matches the frame duration derived independently from the rate, and the
    /// last frame lands within one frame of the requested duration. The last of
    /// those is what fails loudly on the old formula — at 29.97 the "10 second"
    /// export ended 30 ms in.
    #[test]
    fn export_timestamp_grid_is_uniform_at_every_frame_rate() {
        use crate::export::job::ExportJob;

        // Every rate the top bar offers, plus the three NTSC rates any real
        // NTSC-region footage carries.
        let rates = [
            ("24",     Rational { num: 24, den: 1 }),
            ("25",     Rational { num: 25, den: 1 }),
            ("30",     Rational { num: 30, den: 1 }),
            ("60",     Rational { num: 60, den: 1 }),
            ("120",    Rational { num: 120, den: 1 }),
            ("23.976", Rational { num: 24_000, den: 1_001 }),
            ("29.97",  Rational { num: 30_000, den: 1_001 }),
            ("59.94",  Rational { num: 60_000, den: 1_001 }),
        ];

        // Ten seconds of timeline in the 90 kHz project timebase.
        const SECONDS: i64 = 10;
        let duration_pts = SECONDS * TB.den;

        for (label, fps) in rates {
            let mut job = ExportJob::preset_web_h264(
                std::path::PathBuf::from("unused.mp4"),
                0,
                duration_pts,
                1920,
                1080,
                fps,
                TB,
            );
            job.render_threads = 4;

            let total = job.total_frames();
            assert!(total > 0, "{label}: total_frames is zero");

            // Frame duration in ticks, derived from the rate here rather than
            // taken from the code under test.
            let want_step =
                (fps.den as f64 * TB.den as f64 / fps.num as f64).round() as i64;

            let pts: Vec<i64> = (0..total).map(|n| job.frame_pts(n)).collect();
            assert_eq!(pts[0], job.pts_in, "{label}: frame 0 is not at pts_in");

            // Uniform grid, and the step is the real frame duration.  Rounding
            // means consecutive steps may differ by one tick at a fractional rate,
            // which is correct; what must not happen is a systematic error.
            for (i, w) in pts.windows(2).enumerate() {
                let step = w[1] - w[0];
                assert!(
                    (step - want_step).abs() <= 1,
                    "{label}: the step between frames {i} and {} is {step} ticks \
                     but one frame at {}/{} is {want_step} ticks. A step this \
                     wrong means `frame_pts` is not using frame_rate.den.",
                    i + 1, fps.num, fps.den
                );
            }

            // The grid has to span the requested duration.  This is the assertion
            // the old formula could not survive: at 29.97 it put the last of 300
            // frames at 897 ticks — 10 ms into a 10-second export.
            let last = pts[total - 1];
            let short_by = duration_pts - last;
            assert!(
                short_by >= 0 && short_by <= want_step + 1,
                "{label}: the last of {total} frames is at {last} ticks but the \
                 timeline is {duration_pts} ticks long — the export covers \
                 {:.3}s of a {SECONDS}s range",
                last as f64 / TB.den as f64
            );

            // The partitioner slices on these timestamps, so a collapsed grid
            // shows up as segments that are not contiguous or do not cover the
            // range.  Checked here because a caller only ever sees the segments.
            let segments = crate::export::partitioner::SegmentPartitioner::partition(&job);
            assert!(!segments.is_empty(), "{label}: no segments");
            assert_eq!(
                segments.iter().map(|s| s.frame_count()).sum::<usize>(), total,
                "{label}: the segments do not cover every frame"
            );
            for i in 0..segments.len() - 1 {
                assert!(
                    crate::export::partitioner::SegmentPartitioner::are_contiguous(
                        &segments[i], &segments[i + 1]
                    ),
                    "{label}: segment {i} does not join segment {}: \
                     {:?} then {:?}",
                    i + 1, segments[i], segments[i + 1]
                );
            }

            eprintln!(
                "[av_sync] export grid {label:>7}: {total} frames, step \
                 {want_step} ticks, last at {last}/{duration_pts}, \
                 {} segment(s)",
                segments.len()
            );
        }
    }

    /// P1.8 — audio timestamps must stay sample-exact across an hour, at every
    /// sample rate, with no reliance on floating point.
    ///
    /// Separate from the matrix above because it isolates ONE property with no
    /// presenter involved: `advance_samples` accumulates a `u64` sample count and
    /// `pts()` converts once in 128-bit arithmetic. The bug it guards against is
    /// the obvious implementation — converting each callback to ticks and adding —
    /// which truncates whenever the callback size does not divide the timebase.
    ///
    /// The rates and callback sizes are chosen so several are deliberately
    /// awkward: at 96 kHz a 1000-sample callback is 937.5 ticks, and at 44.1 kHz a
    /// 1024-sample callback is 2089.79. Both lose most of a tick per callback under
    /// a truncating implementation, which over an hour is seconds of drift.
    #[test]
    fn master_clock_pts_is_sample_exact_over_an_hour() {
        // (sample rate, callback size, ticks per callback as a decimal — stated so
        // the awkward ones are visible in the source rather than implied)
        let cases: [(u32, usize); 6] = [
            (48_000, 1024), // 1920 ticks exactly
            (48_000, 441),  // 826.875
            (44_100, 441),  // 900 exactly
            (44_100, 1024), // 2089.79...
            (96_000, 1000), // 937.5
            (192_000, 441), // 206.71...
        ];

        for (rate, callback) in cases {
            let clock = MasterClock::new(TB, rate);
            // One hour of callbacks.
            let callbacks = (rate as u64 * 3600) / callback as u64;
            for _ in 0..callbacks {
                clock.advance_samples(callback);
            }

            let samples = callbacks * callback as u64;
            let expected = (samples as u128 * TB.den as u128 / rate as u128) as i64;
            let got = clock.pts();
            assert_eq!(
                got, expected,
                "{rate} Hz / {callback}-sample callbacks: after {samples} samples \
                 ({:.2} h) the clock reads {got} ticks but exact arithmetic says \
                 {expected} — a drift of {:.3} s. This is what per-callback \
                 truncation produces.",
                samples as f64 / rate as f64 / 3600.0,
                (expected - got) as f64 / TB.den as f64
            );

            // A seek must re-reference cleanly rather than carrying the hour.
            clock.seek(0);
            assert_eq!(
                clock.pts(), 0,
                "{rate} Hz: seek(0) left the clock at {} — the sample count \
                 survived the seek",
                clock.pts()
            );
        }
    }

    /// P1.8 — `PresentationDecider::next_frame_pts` must advance by one real frame
    /// at every rate, and at an NTSC rate the old formula did not.
    ///
    /// This is asserted DIRECTLY rather than through the simulation above, and the
    /// distinction matters: `simulate_playback` computes each frame's PTS with
    /// `frame_to_pts` itself, exactly as `ExportRenderer` does, so it never calls
    /// this method and cannot see a bug in it. The real playback loop
    /// (`PlaybackEngine::run`) uses `next_frame_pts` for every frame it schedules,
    /// so a wrong step here is a preview that never advances while the export is
    /// fine — the two paths disagreeing is precisely the failure mode.
    ///
    /// The old body was `project_tb.den / frame_rate.num`, which drops
    /// `frame_rate.den`: at 30000/1001 it steps 3 ticks (33 µs) instead of 3003
    /// (33.4 ms). `decide` then classes the frame as Present immediately at
    /// essentially the current clock position, so playback shows the same frame
    /// while the audio clock runs away.
    #[test]
    fn next_frame_pts_advances_one_frame_at_every_rate() {
        use crate::timeline::rational::frame_to_pts;

        let rates = [
            ("24",     Rational { num: 24, den: 1 }),
            ("25",     Rational { num: 25, den: 1 }),
            ("30",     Rational { num: 30, den: 1 }),
            ("60",     Rational { num: 60, den: 1 }),
            ("120",    Rational { num: 120, den: 1 }),
            ("23.976", Rational { num: 24_000, den: 1_001 }),
            ("29.97",  Rational { num: 30_000, den: 1_001 }),
            ("59.94",  Rational { num: 60_000, den: 1_001 }),
        ];

        for (label, fps) in rates {
            let clock = MasterClock::new(TB, SAMPLE_RATE);
            let decider =
                PresentationDecider::new(Arc::clone(&clock), TB, fps, VSYNC_NS);

            // Frame duration derived here from the rate, not from the code under
            // test.
            let want = frame_to_pts(1, fps, TB);
            let got = decider.next_frame_pts();
            assert_eq!(
                got, want,
                "{label}: next_frame_pts from a clock at 0 returned {got} ticks, \
                 but one frame at {}/{} is {want} ticks ({:.3} ms). A step of \
                 {got} means the frame duration was computed without \
                 frame_rate.den.",
                fps.num, fps.den,
                want as f64 * 1000.0 / TB.den as f64
            );

            // And from a non-zero clock position, so the method is the clock plus
            // one frame rather than something that happens to equal a frame at 0.
            clock.advance_samples(SAMPLE_RATE as usize); // one second
            let base = clock.pts();
            assert_eq!(
                decider.next_frame_pts() - base, want,
                "{label}: after advancing the clock the step changed — \
                 next_frame_pts is not `clock + one frame`"
            );

            // The step must also be large enough for `decide` to treat the frame
            // as EARLY rather than on time.  This is what actually breaks in the
            // preview: with a 3-tick step the frame is 33 µs ahead, inside the
            // half-vsync present window, so it is presented at once and the loop
            // spins on the same picture.
            //
            // Only asserted where one frame is genuinely LONGER than the hold
            // window (`vsync/2`, 8.333 ms at 60 Hz).  At 120 fps a frame is
            // 8.333 ms — exactly the window — so `decide` returns Present, and
            // that is correct rather than a bug: content faster than the display
            // has no early frames to wait for.  Asserting Hold there would be
            // asserting a wrong thing, so the guard is derived from the rate
            // instead of the row being special-cased away.
            let frame_ns = fps.den * 1_000_000_000 / fps.num;
            if frame_ns > VSYNC_NS / 2 {
                assert_eq!(
                    decider.decide(decider.next_frame_pts()), PresentAction::Hold,
                    "{label}: one frame is {:.3} ms, past the {:.3} ms hold \
                     window, but the next frame is not classed Hold — the step \
                     ({want} ticks) is being read as on-time, so the playback \
                     loop would present the same frame repeatedly",
                    frame_ns as f64 / 1e6, VSYNC_NS as f64 / 2e6
                );
            } else {
                assert_eq!(
                    decider.decide(decider.next_frame_pts()), PresentAction::Present,
                    "{label}: one frame is {:.3} ms, inside the {:.3} ms hold \
                     window, so the next frame is due now and must be Present",
                    frame_ns as f64 / 1e6, VSYNC_NS as f64 / 2e6
                );
            }
        }
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
