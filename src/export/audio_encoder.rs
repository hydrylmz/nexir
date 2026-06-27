use crate::export::job::ExportJob;
use crate::export::video_encoder::EncodeError;
use crate::io::ffi::avutil::{AVFrame, AVPacket, AVRational, av_frame_alloc,
                               av_frame_free, av_packet_alloc, av_packet_free,
                               av_packet_unref};
use crate::io::ffi::avcodec::{avcodec_alloc_context3,
                                avcodec_free_context, avcodec_open2};
use crate::audio::ffi::avresample::{swr_alloc_set_opts, swr_free,
                                      AV_SAMPLE_FMT_FLTP, AV_CH_LAYOUT_STEREO,
                                      SwrContext};
use crate::export::ffi::encoder_ffi::*;

pub struct AudioMuxEncoder {
    ctx:         *mut crate::io::ffi::avcodec::AVCodecContext,
    swr:         *mut SwrContext,
    in_frame:    *mut AVFrame,
    enc_frame:   *mut AVFrame,
    packet:      *mut AVPacket,
    frame_size:  usize,
    enc_tb:      AVRational,
    frame_count: i64,
}

unsafe impl Send for AudioMuxEncoder {}

impl AudioMuxEncoder {
    pub fn open(job: &ExportJob) -> Result<Self, EncodeError> {
        unsafe {
            let codec = avcodec_find_encoder(job.audio_codec.ffmpeg_id());
            if codec.is_null() {
                return Err(EncodeError::CodecNotFound);
            }

            let ctx = avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(EncodeError::Alloc);
            }

            avcodec_ctx_set_bit_rate(ctx, job.audio_bitrate as i64);

            let sr_key = std::ffi::CString::new("sample_rate").unwrap();
            av_opt_set_int(ctx as *mut _, sr_key.as_ptr(), 48000, 1);
            
            let cl_key = std::ffi::CString::new("channel_layout").unwrap();
            let cl_str = std::ffi::CString::new("stereo").unwrap();
            av_opt_set(ctx as *mut _, cl_key.as_ptr(), cl_str.as_ptr(), 1);

            let fmt_key = std::ffi::CString::new("sample_fmt").unwrap();
            let fmt_str = std::ffi::CString::new("fltp").unwrap();
            av_opt_set(ctx as *mut _, fmt_key.as_ptr(), fmt_str.as_ptr(), 1);

            let enc_tb = AVRational { num: 1, den: 48000 };
            avcodec_ctx_set_time_base(ctx, enc_tb);

            if matches!(job.container, crate::export::job::Container::Mp4 | crate::export::job::Container::Mov) {
                avcodec_ctx_set_flags(ctx, AV_CODEC_FLAG_GLOBAL_HEADER);
            }

            if avcodec_open2(ctx, codec, std::ptr::null_mut()) < 0 {
                return Err(EncodeError::Open("Failed to open audio codec".to_string()));
            }

            let mut frame_size: i64 = 0;
            let fs_key = std::ffi::CString::new("frame_size").unwrap();
            av_opt_get_int(ctx as *mut _, fs_key.as_ptr(), 0, &mut frame_size);
            
            if frame_size == 0 {
                frame_size = 1024;
            }

            let swr = swr_alloc_set_opts(
                std::ptr::null_mut(),
                AV_CH_LAYOUT_STEREO as i64, AV_SAMPLE_FMT_FLTP, 48000,
                AV_CH_LAYOUT_STEREO as i64, AV_SAMPLE_FMT_FLTP, 48000,
                0, std::ptr::null_mut()
            );

            let in_frame = av_frame_alloc();
            let enc_frame = av_frame_alloc();
            let packet = av_packet_alloc();

            (*enc_frame).nb_samples = frame_size as i32;
            (*enc_frame).format = AV_SAMPLE_FMT_FLTP;
            (*enc_frame).channel_layout = AV_CH_LAYOUT_STEREO as u64;
            (*enc_frame).sample_rate = 48000;
            av_frame_get_buffer(enc_frame, 0);

            Ok(Self {
                ctx,
                swr,
                in_frame,
                enc_frame,
                packet,
                frame_size: frame_size as usize,
                enc_tb,
                frame_count: 0,
            })
        }
    }

    pub fn encode_all(
        &mut self,
        _job:          &ExportJob,
        packet_sink:  &mut dyn FnMut(*mut AVPacket),
    ) -> Result<(), EncodeError> {
        unsafe {
            avcodec_send_frame(self.ctx, std::ptr::null());
            loop {
                let ret = avcodec_receive_packet(self.ctx, self.packet);
                if ret < 0 {
                    break;
                }
                packet_sink(self.packet);
                av_packet_unref(self.packet);
            }
        }
        Ok(())
    }

    pub fn codec_ctx(&self) -> *const crate::io::ffi::avcodec::AVCodecContext {
        self.ctx as *const _
    }
}

impl Drop for AudioMuxEncoder {
    fn drop(&mut self) {
        unsafe {
            av_packet_free(&mut self.packet);
            av_frame_free(&mut self.enc_frame);
            av_frame_free(&mut self.in_frame);
            swr_free(&mut self.swr);
            avcodec_free_context(&mut self.ctx);
        }
    }
}
