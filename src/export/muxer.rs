use std::sync::Mutex;
use std::ffi::CString;
use crate::export::job::ExportJob;
use crate::export::ffi::muxer_ffi::*;
use crate::export::video_encoder::VideoEncoderBackend;
use crate::io::ffi::avutil::{AVPacket, AVRational, av_packet_set_stream_index};
use crate::io::ffi::avformat::{AVFormatContext, avformat_get_stream, avstream_get_time_base};

pub struct Muxer {
    inner: Mutex<MuxerInner>,
}

struct MuxerInner {
    ctx:                   *mut AVFormatContext,
    video_stream_idx:      i32,
    audio_stream_idx:      i32,
    video_tb:              AVRational,
    audio_tb:              AVRational,
    finalised:             bool,
    video_packets_written: usize,
    audio_packets_written: usize,
}

unsafe impl Send for Muxer {}
unsafe impl Sync for Muxer {}

impl Muxer {
    pub fn open(
        job:           &ExportJob,
        video_encoder: &VideoEncoderBackend,
        audio_encoder: &crate::export::audio_encoder::AudioMuxEncoder,
        video_tb:      AVRational,
        audio_tb:      AVRational,
    ) -> Result<Self, MuxError> {
        unsafe {
            let mut ctx: *mut AVFormatContext = std::ptr::null_mut();
            let format_name = CString::new(job.container.format_name()).unwrap();
            let output_path = CString::new(job.output_path.to_str().unwrap()).unwrap();
            
            let ret = avformat_alloc_output_context2(
                &mut ctx,
                std::ptr::null(),
                format_name.as_ptr(),
                output_path.as_ptr(),
            );
            if ret < 0 {
                return Err(MuxError::AllocContext);
            }

            let v_stream = avformat_new_stream(ctx, std::ptr::null());
            if v_stream.is_null() {
                return Err(MuxError::AddStream);
            }
            avcodec_parameters_from_context(avstream_get_codecpar_mut(v_stream), video_encoder.codec_ctx());
            avstream_set_time_base(v_stream, video_tb);
            let video_stream_idx = av_stream_get_index(v_stream);

            let a_stream = avformat_new_stream(ctx, std::ptr::null());
            if a_stream.is_null() {
                return Err(MuxError::AddStream);
            }
            avcodec_parameters_from_context(avstream_get_codecpar_mut(a_stream), audio_encoder.codec_ctx());
            avstream_set_time_base(a_stream, audio_tb);
            let audio_stream_idx = av_stream_get_index(a_stream);

            let ret = avformat_open_output_pb(ctx, output_path.as_ptr(), AVIO_FLAG_WRITE);
            if ret < 0 {
                return Err(MuxError::OpenFile("Failed to open avio".into()));
            }

            let ret = crate::export::ffi::muxer_ffi::avformat_write_header_shim(ctx, std::ptr::null_mut());
            if ret < 0 {
                return Err(MuxError::WriteHeader(format!("Failed to write header: error {}", ret)));
            }

            if job.container == crate::export::job::Container::Mp4 {
                use crate::export::ffi::encoder_ffi::av_opt_set;
                let key = CString::new("movflags").unwrap();
                let val = CString::new("faststart").unwrap();
                av_opt_set(ctx as *mut std::ffi::c_void, key.as_ptr(), val.as_ptr(), 1);
            }

            Ok(Self {
                inner: Mutex::new(MuxerInner {
                    ctx,
                    video_stream_idx,
                    audio_stream_idx,
                    video_tb,
                    audio_tb,
                    finalised: false,
                    video_packets_written: 0,
                    audio_packets_written: 0,
                }),
            })
        }
    }

    pub fn write_packet(
        &self,
        pkt:      *mut AVPacket,
        is_video: bool,
    ) -> Result<(), MuxError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.finalised {
            return Err(MuxError::Write("Cannot write to finalized muxer".into()));
        }
        unsafe {
            let stream_idx = if is_video {
                inner.video_stream_idx
            } else {
                inner.audio_stream_idx
            };
            av_packet_set_stream_index(pkt, stream_idx);

            let src_tb = if is_video { inner.video_tb } else { inner.audio_tb };
            let stream = avformat_get_stream(inner.ctx, stream_idx as std::ffi::c_uint);
            let dst_tb = avstream_get_time_base(stream);
            av_packet_rescale_ts(pkt, src_tb, dst_tb);

            let ret = av_interleaved_write_frame(inner.ctx, pkt);
            if ret < 0 {
                return Err(MuxError::Write(format!("Write frame failed: {}", ret)));
            }

            if is_video {
                inner.video_packets_written += 1;
            } else {
                inner.audio_packets_written += 1;
            }
        }
        Ok(())
    }

    /// Thread-safe synchronous finalisation through shared Arc reference.
    pub fn finalise_sync(&self) -> Result<(), MuxError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.finalised {
            return Ok(());
        }
        inner.finalised = true;
        unsafe {
            let ret = av_write_trailer(inner.ctx);
            avformat_free_context(inner.ctx);
            inner.ctx = std::ptr::null_mut();
            if ret < 0 {
                return Err(MuxError::Trailer(format!("Write trailer failed: {}", ret)));
            }
        }
        Ok(())
    }

    pub fn finalise(self) -> Result<(), MuxError> {
        self.finalise_sync()
    }

    /// Returns the count of (video_packets, audio_packets) written to the container.
    pub fn packet_stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.video_packets_written, inner.audio_packets_written)
    }
}

impl Drop for MuxerInner {
    fn drop(&mut self) {
        if !self.finalised && !self.ctx.is_null() {
            unsafe {
                let _ = av_write_trailer(self.ctx);
                avformat_free_context(self.ctx);
                self.ctx = std::ptr::null_mut();
            }
        }
    }
}

#[derive(Debug)]
pub enum MuxError {
    AllocContext,
    AddStream,
    OpenFile(String),
    WriteHeader(String),
    Write(String),
    Trailer(String),
}
