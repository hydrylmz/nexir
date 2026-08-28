// src/audio/audio_decoder.rs

use crate::audio::ffi::avresample::{
    swr_alloc_set_opts, swr_convert, swr_free, swr_get_delay, swr_init, swr_output_sample_count,
    SwrContext, AV_CH_LAYOUT_STEREO, AV_SAMPLE_FMT_FLTP,
};
use crate::audio::ring_buffer::AudioRingBuffer;
use crate::io::decoder::Decoder;
use crate::io::demuxer::Demuxer;
use crate::io::ffi::avcodec::{avcodec_receive_frame, avcodec_send_packet};
use crate::io::ffi::avutil::{
    av_frame_alloc, av_frame_free, av_frame_get_data, av_frame_get_nb_samples,
    av_frame_get_sample_rate, AVFrame, AVERROR_EAGAIN, AVERROR_EOF,
};
use crate::timeline::rational::Rational;
use std::ptr::null_mut;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

pub const OUT_SAMPLE_RATE: u32 = 48_000;
pub const OUT_CHANNELS: u32 = 2;
pub const LOOKAHEAD_MS: u64 = 512;

pub const RING_TARGET_SAMPLES: usize = (OUT_SAMPLE_RATE as usize * LOOKAHEAD_MS as usize) / 1000;

pub struct AudioDecoder {
    demuxer: Demuxer,
    decoder: Decoder,
    swr: *mut SwrContext,
    frame: *mut AVFrame,
    ring: Arc<AudioRingBuffer>,
    project_tb: Rational,
    shutdown: Arc<AtomicBool>,
    seek_request: Arc<Mutex<Option<(i64, i64)>>>, // (source_pts, timeline_pts)
    volume: f32,
    pan: f32,
    muted: bool,
    samples_written: u64,
    stream_tb: Rational,
    path: std::path::PathBuf,
    speed: f32,
    source_in_pts: i64,
    source_out_pts: i64,
    fade_in_pts: i64,
    fade_out_pts: i64,
    current_seek_pts: i64,
    current_timeline_pts: i64,
    current_timeline_out_pts: i64,
}

unsafe impl Send for AudioDecoder {}

impl AudioDecoder {
    pub fn new(
        path: &std::path::Path,
        ring: Arc<AudioRingBuffer>,
        project_tb: Rational,
        shutdown: Arc<AtomicBool>,
        seek_request: Arc<Mutex<Option<(i64, i64)>>>,
        volume: f32,
        pan: f32,
        muted: bool,
        fade_in_pts: i64,
        fade_out_pts: i64,
        speed: f32,
        pitch: f32,
        source_in_pts: i64,
        source_out_pts: i64,
    ) -> Result<Self, AudioError> {
        let demuxer = Demuxer::open(path).map_err(AudioError::Demux)?;

        let audio_stream = demuxer
            .audio_stream()
            .cloned()
            .ok_or(AudioError::NoAudioStream)?;
        let stream_tb = audio_stream.time_base;

        let decoder = Decoder::open(&audio_stream, audio_stream.codecpar, false)
            .map_err(AudioError::Decode)?;

        // Need FFI for codec parameters to set up SwrContext
        let (in_ch_layout, in_sample_fmt, mut in_sample_rate) = unsafe {
            let ctx = decoder.ctx();
            let sr = crate::io::ffi::avcodec::avcodec_ctx_get_sample_rate(ctx);
            let mut cl = crate::io::ffi::avcodec::avcodec_ctx_get_channel_layout(ctx);
            let channels = crate::io::ffi::avcodec::avcodec_ctx_get_channels(ctx);
            let fmt = crate::io::ffi::avcodec::avcodec_ctx_get_sample_fmt(ctx);

            if cl == 0 {
                cl = if channels == 1 { 4 } else { 3 }; // 4 = MONO, 3 = STEREO
            }

            (cl as i64, fmt as i32, sr as i32)
        };

        // Varispeed: adjusting sample rate stretches both speed and pitch
        let pitch_factor = 2.0_f32.powf(pitch / 12.0);
        let varispeed = speed * pitch_factor;
        in_sample_rate = (in_sample_rate as f32 * varispeed).round() as i32;

        let swr = unsafe {
            swr_alloc_set_opts(
                null_mut(),
                AV_CH_LAYOUT_STEREO as i64,
                AV_SAMPLE_FMT_FLTP,
                OUT_SAMPLE_RATE as i32,
                in_ch_layout,
                in_sample_fmt,
                in_sample_rate,
                0,
                null_mut(),
            )
        };

        if swr.is_null() {
            return Err(AudioError::SwrAlloc);
        }

        unsafe {
            let ret = swr_init(swr);
            if ret < 0 {
                let mut p = swr;
                swr_free(&mut p);
                return Err(AudioError::SwrInit(format!("swr_init failed: {}", ret)));
            }
        }

        let frame = unsafe { av_frame_alloc() };
        if frame.is_null() {
            unsafe {
                let mut p = swr;
                swr_free(&mut p);
            }
            return Err(AudioError::Alloc);
        }

        Ok(Self {
            demuxer,
            decoder,
            swr,
            frame,
            ring,
            project_tb,
            shutdown,
            seek_request,
            volume,
            pan,
            muted,
            fade_in_pts,
            fade_out_pts,
            samples_written: 0,
            stream_tb,
            path: path.to_path_buf(),
            speed,
            source_in_pts,
            source_out_pts,
            current_seek_pts: source_in_pts,
            current_timeline_pts: 0,
            current_timeline_out_pts: source_out_pts,
        })
    }

    pub fn run(mut self) {
        log::debug!("[audio] decoder run() started");
        while !self.shutdown.load(Ordering::Relaxed) {
            let seek_req = self.seek_request.lock().unwrap().take();
            if let Some((source_pts, timeline_pts)) = seek_req {
                self.flush_swr();
                self.ring.clear();
                // Seek demuxer to the file PTS (demuxer.seek converts project_tb -> stream_tb internally)
                let mut result = self.demuxer.seek(source_pts, self.project_tb);
                if result.is_err() {
                    log::warn!(
                        "[audio] seek failed, attempting to reopen demuxer: {:?}",
                        result
                    );
                    if let Ok(new_demuxer) = Demuxer::open(&self.path) {
                        self.demuxer = new_demuxer;
                        if source_pts == 0 {
                            result = Ok(0); // For source_pts == 0, reopening is the seek
                        }
                    }
                }

                // ALWAYS discard audio packets until we reach the target PTS
                if source_pts > 0 {
                    let target_stream_pts = self.project_tb.rescale_pts(source_pts, self.stream_tb);
                    let mut last_pts = 0;
                    while let Ok(Some(pkt)) = self.demuxer.next_audio_packet() {
                        if pkt.pts >= target_stream_pts {
                            // Found the packet we need. We've unfortunately consumed it,
                            // but this is standard for simple FFmpeg seek implementations.
                            break;
                        }
                        last_pts = pkt.pts;
                    }
                    result = Ok(last_pts);
                }

                if let Ok(_stream_pts) = result {
                    self.decoder.flush();
                    self.samples_written = 0;
                    self.current_seek_pts = source_pts;
                    self.current_timeline_pts = timeline_pts;
                    self.current_timeline_out_pts =
                        timeline_pts + self.timeline_pts_until_source_out(source_pts);
                }
            }

            let current_pts = self.current_timeline_pts
                + (self.samples_written as f64 * 90000.0 / OUT_SAMPLE_RATE as f64) as i64;
            if current_pts >= self.current_timeline_out_pts {
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }

            let available = self.ring.available_write();
            if available < RING_TARGET_SAMPLES / 4 {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }

            match self.demuxer.next_audio_packet() {
                Ok(Some(pkt)) => unsafe {
                    let ret = avcodec_send_packet(self.decoder.ctx(), pkt.as_ptr());
                    if ret >= 0 {
                        loop {
                            let r = avcodec_receive_frame(self.decoder.ctx(), self.frame);
                            if r == AVERROR_EAGAIN || r == AVERROR_EOF {
                                break;
                            }
                            if r < 0 {
                                log::error!("[audio] avcodec_receive_frame error: {}", r);
                                break;
                            }
                            if let Err(e) = self.resample_and_push() {
                                log::error!("[audio] resample_and_push error: {:?}", e);
                            }
                        }
                    } else {
                        log::error!("[audio] avcodec_send_packet error: {}", ret);
                    }
                },
                Ok(None) => {
                    self.flush_swr();
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    log::error!("[audio] dec: demux error: {:?}", e);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
    }

    fn resample_and_push(&mut self) -> Result<(), AudioError> {
        let (in_count, in_data, in_rate) = unsafe {
            let _f = &*self.frame;
            let count = av_frame_get_nb_samples(self.frame) as usize;
            let data = av_frame_get_data(self.frame);
            let rate = av_frame_get_sample_rate(self.frame) as u32;
            (count, data, rate)
        };

        if in_count == 0 {
            return Ok(());
        }

        let out_count = swr_output_sample_count(in_count, in_rate, OUT_SAMPLE_RATE);

        let mut out_buf = vec![vec![0.0f32; out_count]; OUT_CHANNELS as usize];
        let mut out_ptrs: Vec<*mut u8> = out_buf
            .iter_mut()
            .map(|ch| ch.as_mut_ptr() as *mut u8)
            .collect();

        let written = unsafe {
            swr_convert(
                self.swr,
                out_ptrs.as_mut_ptr(),
                out_count as i32,
                in_data as *const *const u8,
                in_count as i32,
            )
        };

        if written <= 0 {
            return Ok(());
        }

        let written_usize = written as usize;
        let mut interleaved = Vec::with_capacity(written_usize * 2);
        for (&left, &right) in out_buf[0].iter().zip(&out_buf[1]).take(written_usize) {
            interleaved.push(left);
            interleaved.push(right);
        }

        // ── Apply volume, pan, mute DSP ──────────────────────────────────
        if self.muted {
            interleaved.fill(0.0);
            let mut valid_samples = 0;
            for sample_idx in 0..written_usize {
                let current_sample_index = self.samples_written + sample_idx as u64;
                let current_timeline_pts = self.current_timeline_pts
                    + (current_sample_index as f64 * 90000.0 / OUT_SAMPLE_RATE as f64) as i64;
                if current_timeline_pts >= self.current_timeline_out_pts {
                    break;
                }
                valid_samples += 1;
            }
            interleaved.truncate(valid_samples * 2);
        } else {
            let (c_pan_l, c_pan_r) = crate::audio::audio_mixer::constant_power_pan(self.pan);
            let base_left_gain = self.volume * c_pan_l;
            let base_right_gain = self.volume * c_pan_r;
            let mut valid_samples = 0;
            for (sample_idx, frame) in interleaved.chunks_exact_mut(2).enumerate() {
                let current_sample_index = self.samples_written + sample_idx as u64;
                let current_timeline_pts = self.current_timeline_pts
                    + (current_sample_index as f64 * 90000.0 / OUT_SAMPLE_RATE as f64) as i64;

                let clip_pts_in = self.current_timeline_pts
                    - self.timeline_pts_since_source_in(self.current_seek_pts);
                let clip_pts_out = self.current_timeline_out_pts;

                if current_timeline_pts >= clip_pts_out {
                    break;
                }
                valid_samples += 1;

                let fade_factor = crate::audio::audio_mixer::compute_fade_multiplier(
                    current_timeline_pts,
                    clip_pts_in,
                    clip_pts_out,
                    self.fade_in_pts,
                    self.fade_out_pts,
                );

                frame[0] *= base_left_gain * fade_factor;
                frame[1] *= base_right_gain * fade_factor;
            }
            interleaved.truncate(valid_samples * 2);
        }

        if !interleaved.is_empty() {
            let samples_added = (interleaved.len() / 2) as u64;
            self.ring.write(&interleaved);
            self.samples_written += samples_added;
        }

        Ok(())
    }

    fn flush_swr(&mut self) {
        loop {
            let current_pts = self.current_timeline_pts
                + (self.samples_written as f64 * 90000.0 / OUT_SAMPLE_RATE as f64) as i64;
            if current_pts >= self.current_timeline_out_pts {
                break;
            }

            let delay = unsafe { swr_get_delay(self.swr, OUT_SAMPLE_RATE as i64) };
            if delay <= 0 {
                break;
            }
            let out_count = delay as usize + 16;
            let mut out_buf = vec![vec![0.0f32; out_count]; OUT_CHANNELS as usize];
            let mut out_ptrs: Vec<*mut u8> = out_buf
                .iter_mut()
                .map(|ch| ch.as_mut_ptr() as *mut u8)
                .collect();

            let written = unsafe {
                swr_convert(
                    self.swr,
                    out_ptrs.as_mut_ptr(),
                    out_count as i32,
                    null_mut(),
                    0,
                )
            };

            if written <= 0 {
                break;
            }

            let written_usize = written as usize;
            let mut interleaved = Vec::with_capacity(written_usize * 2);
            for (&left, &right) in out_buf[0].iter().zip(&out_buf[1]).take(written_usize) {
                interleaved.push(left);
                interleaved.push(right);
            }
            // Apply volume/pan/mute DSP
            if self.muted {
                interleaved.fill(0.0);
                let mut valid_samples = 0;
                for sample_idx in 0..written_usize {
                    let current_sample_index = self.samples_written + sample_idx as u64;
                    let current_timeline_pts = self.current_timeline_pts
                        + (current_sample_index as f64 * 90000.0 / OUT_SAMPLE_RATE as f64) as i64;
                    if current_timeline_pts >= self.current_timeline_out_pts {
                        break;
                    }
                    valid_samples += 1;
                }
                interleaved.truncate(valid_samples * 2);
            } else {
                let (c_pan_l, c_pan_r) = crate::audio::audio_mixer::constant_power_pan(self.pan);
                let left_gain = self.volume * c_pan_l;
                let right_gain = self.volume * c_pan_r;
                let mut valid_samples = 0;
                for (sample_idx, frame) in interleaved.chunks_exact_mut(2).enumerate() {
                    let current_sample_index = self.samples_written + sample_idx as u64;
                    let current_timeline_pts = self.current_timeline_pts
                        + (current_sample_index as f64 * 90000.0 / OUT_SAMPLE_RATE as f64) as i64;
                    if current_timeline_pts >= self.current_timeline_out_pts {
                        break;
                    }
                    valid_samples += 1;
                    frame[0] *= left_gain;
                    frame[1] *= right_gain;
                }
                interleaved.truncate(valid_samples * 2);
            }
            if !interleaved.is_empty() {
                let samples_added = (interleaved.len() / 2) as u64;
                self.ring.write(&interleaved);
                self.samples_written += samples_added;
            }
        }
    }

    fn timeline_pts_since_source_in(&self, source_pts: i64) -> i64 {
        self.source_delta_to_timeline_pts(source_pts - self.source_in_pts)
    }

    fn timeline_pts_until_source_out(&self, source_pts: i64) -> i64 {
        self.source_delta_to_timeline_pts(self.source_out_pts - source_pts)
    }

    fn source_delta_to_timeline_pts(&self, source_delta: i64) -> i64 {
        if self.speed.abs() < f32::EPSILON {
            return source_delta;
        }

        let den = (self.speed * 10_000.0).round() as i64;
        if den == 0 {
            return source_delta;
        }

        let numer = source_delta as i128 * 10_000_i128;
        let den = den as i128;
        if numer >= 0 {
            ((numer + den / 2) / den) as i64
        } else {
            ((numer - den / 2) / den) as i64
        }
    }
}

impl Drop for AudioDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.frame.is_null() {
                av_frame_free(&mut self.frame);
            }
            if !self.swr.is_null() {
                swr_free(&mut self.swr);
            }
        }
    }
}

#[derive(Debug)]
pub enum AudioError {
    NoAudioStream,
    SwrAlloc,
    SwrInit(String),
    Alloc,
    Demux(crate::io::demuxer::DemuxError),
    Decode(crate::io::decoder::DecodeError),
}

// Debug added at end of file to trace
