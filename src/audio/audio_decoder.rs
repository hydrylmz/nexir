// src/audio/audio_decoder.rs

use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};
use std::ptr::null_mut;
use crate::audio::ffi::avresample::{
    SwrContext, swr_alloc_set_opts, swr_init, swr_free, swr_convert,
    swr_get_delay, swr_output_sample_count,
    AV_SAMPLE_FMT_FLTP, AV_CH_LAYOUT_STEREO,
};
use crate::io::ffi::avutil::{
    AVFrame, av_frame_alloc, av_frame_free, av_frame_get_data,
    av_frame_get_nb_samples, av_frame_get_sample_rate,
    AVERROR_EOF, AVERROR_EAGAIN
};
use crate::io::ffi::avcodec::{
    avcodec_send_packet, avcodec_receive_frame,

};
use crate::io::demuxer::Demuxer;
use crate::io::decoder::Decoder;
use crate::audio::ring_buffer::AudioRingBuffer;
use crate::timeline::rational::Rational;

pub const OUT_SAMPLE_RATE: u32 = 48_000;
pub const OUT_CHANNELS:    u32 = 2;
pub const LOOKAHEAD_MS:    u64 = 512;

pub const RING_TARGET_SAMPLES: usize =
    (OUT_SAMPLE_RATE as usize * LOOKAHEAD_MS as usize) / 1000;

pub struct AudioDecoder {
    demuxer:    Demuxer,
    decoder:    Decoder,
    swr:        *mut SwrContext,
    frame:      *mut AVFrame,
    ring:       Arc<AudioRingBuffer>,
    clock:      Arc<crate::sync::master_clock::MasterClock>,
    project_tb: Rational,
    shutdown:   Arc<AtomicBool>,
    seek_request: Arc<Mutex<Option<(i64, i64)>>>, // (source_pts, timeline_pts)
    samples_written: u64,
    stream_tb:  Rational,
    path:       std::path::PathBuf,
}

unsafe impl Send for AudioDecoder {}

impl AudioDecoder {
    pub fn new(
        path:       &std::path::Path,
        ring:       Arc<AudioRingBuffer>,
        clock:      Arc<crate::sync::master_clock::MasterClock>,
        project_tb: Rational,
        shutdown:   Arc<AtomicBool>,
        seek_request: Arc<Mutex<Option<(i64, i64)>>>,
    ) -> Result<Self, AudioError> {
        let mut demuxer = Demuxer::open(path).map_err(AudioError::Demux)?;
        
        let audio_stream = demuxer.audio_stream().cloned().ok_or(AudioError::NoAudioStream)?;
        let stream_tb = audio_stream.time_base;

        let decoder = Decoder::open(&audio_stream, audio_stream.codecpar, false).map_err(AudioError::Decode)?;

        // Need FFI for codec parameters to set up SwrContext
        let (in_ch_layout, in_sample_fmt, in_sample_rate) = unsafe {
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
            clock,
            project_tb,
            shutdown,
            seek_request,
            samples_written: 0,
            stream_tb,
            path: path.to_path_buf(),
        })
    }

    pub fn run(mut self) {
        eprintln!("[audio] decoder run() started");
        while !self.shutdown.load(Ordering::Relaxed) {
            let seek_req = self.seek_request.lock().unwrap().take();
            if let Some((source_pts, timeline_pts)) = seek_req {
                self.flush_swr();
                self.ring.clear();
                // Seek demuxer to the file PTS (demuxer.seek converts project_tb -> stream_tb internally)
                let mut result = self.demuxer.seek(source_pts, self.project_tb);
                if result.is_err() {
                    eprintln!("[audio] seek failed, attempting to reopen demuxer: {:?}", result);
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
                    
                    // Reset the master clock to the timeline PTS!
                    self.clock.seek(timeline_pts);
                    eprintln!("[audio] seek done, clock reset to pts={}", timeline_pts);
                }
            }

            let available = self.ring.available_write();
            eprintln!("[audio] dec: available_write={} threshold={}", available, RING_TARGET_SAMPLES / 4);
            if available < RING_TARGET_SAMPLES / 4 {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }

            eprintln!("[audio] dec: calling next_audio_packet");
            match self.demuxer.next_audio_packet() {
                Ok(Some(pkt)) => {
                    eprintln!("[audio] dec: got packet pts={}", pkt.pts);
                    unsafe {
                        let ret = avcodec_send_packet(self.decoder.ctx(), pkt.as_ptr());
                        if ret >= 0 {
                            loop {
                                let r = avcodec_receive_frame(self.decoder.ctx(), self.frame);
                                if r == AVERROR_EAGAIN || r == AVERROR_EOF {
                                    break;
                                }
                                if r < 0 {
                                    eprintln!("[audio] avcodec_receive_frame error: {}", r);
                                    break;
                                }
                                if let Err(e) = self.resample_and_push() {
                                    eprintln!("[audio] resample_and_push error: {:?}", e);
                                }
                            }
                        } else {
                            eprintln!("[audio] avcodec_send_packet error: {}", ret);
                        }
                    }
                }
                Ok(None) => {
                    println!("[audio] dec: Ok(None) EOF");
                    self.flush_swr();
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    eprintln!("[audio] dec: demux error: {:?}", e);
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
            println!("[audio] resample_and_push: in_count == 0");
            return Ok(());
        }

        let out_count = swr_output_sample_count(in_count, in_rate, OUT_SAMPLE_RATE);
        
        let mut out_buf = vec![vec![0.0f32; out_count]; OUT_CHANNELS as usize];
        let mut out_ptrs: Vec<*mut u8> = out_buf.iter_mut()
            .map(|ch| ch.as_mut_ptr() as *mut u8).collect();

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
            println!("[audio] resample_and_push: written == {}", written);
            return Ok(());
        }

        let written_usize = written as usize;
        let mut interleaved = Vec::with_capacity(written_usize * 2);
        for i in 0..written_usize {
            interleaved.push(out_buf[0][i]);
            interleaved.push(out_buf[1][i]);
        }

        self.ring.write(&interleaved);
        self.samples_written += written_usize as u64;
        println!("[audio] dec: pushed {} frames (total: {} elements), ring available: {}", written_usize, interleaved.len(), self.ring.available_read());

        Ok(())
    }

    fn flush_swr(&mut self) {
        loop {
            let delay = unsafe { swr_get_delay(self.swr, OUT_SAMPLE_RATE as i64) };
            if delay <= 0 {
                break;
            }
            let out_count = delay as usize + 16;
            let mut out_buf = vec![vec![0.0f32; out_count]; OUT_CHANNELS as usize];
            let mut out_ptrs: Vec<*mut u8> = out_buf.iter_mut()
                .map(|ch| ch.as_mut_ptr() as *mut u8).collect();

            let written = unsafe {
                swr_convert(self.swr, out_ptrs.as_mut_ptr(), out_count as i32, null_mut(), 0)
            };

            if written <= 0 {
                break;
            }

            let written_usize = written as usize;
            let mut interleaved = Vec::with_capacity(written_usize * 2);
            for i in 0..written_usize {
                interleaved.push(out_buf[0][i]);
                interleaved.push(out_buf[1][i]);
            }
            self.ring.write(&interleaved);
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

