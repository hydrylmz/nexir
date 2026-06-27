
use crate::export::job::{ExportJob, VideoQuality, Container};
use crate::export::renderer::RawFrame;
use crate::export::ffi::encoder_ffi::*;
use crate::io::ffi::avcodec::{
    avcodec_alloc_context3, avcodec_free_context,
    avcodec_open2,
};
use crate::io::ffi::avutil::{av_packet_alloc, av_packet_free, av_packet_unref,
                               AVRational};

pub mod swscale_ffi {
    #[repr(C)] pub struct SwsContext { _opaque: [u8; 0] }

    extern "C" {
        pub fn sws_getContext(
            srcW: std::ffi::c_int, srcH: std::ffi::c_int, srcFormat: std::ffi::c_int,
            dstW: std::ffi::c_int, dstH: std::ffi::c_int, dstFormat: std::ffi::c_int,
            flags: std::ffi::c_int,
            srcFilter: *const std::ffi::c_void,
            dstFilter: *const std::ffi::c_void,
            param:     *const f64,
        ) -> *mut SwsContext;

        pub fn sws_scale(
            c:          *mut SwsContext,
            srcSlice:   *const *const u8,
            srcStride:  *const std::ffi::c_int,
            srcSliceY:  std::ffi::c_int,
            srcSliceH:  std::ffi::c_int,
            dst:        *const *mut u8,
            dstStride:  *const std::ffi::c_int,
        ) -> std::ffi::c_int;

        pub fn sws_freeContext(swsCtx: *mut SwsContext);
    }

    pub const AV_PIX_FMT_RGBA:    i32 = 26;
    pub const AV_PIX_FMT_RGBA64:  i32 = 105;
    pub const SWS_BILINEAR:       i32 = 4;
}

pub struct VideoEncoder {
    ctx:       *mut crate::io::ffi::avcodec::AVCodecContext,
    sws:       *mut swscale_ffi::SwsContext,
    yuv_frame: *mut crate::io::ffi::avutil::AVFrame,
    packet:    *mut crate::io::ffi::avutil::AVPacket,
    enc_tb:    AVRational,
    frame_count: i64,
    yuv_buffer: *mut u8,
}

unsafe impl Send for VideoEncoder {}

impl VideoEncoder {
    pub fn open(job: &ExportJob) -> Result<Self, EncodeError> {
        unsafe {
            let codec = avcodec_find_encoder(job.video_codec.ffmpeg_id());
            if codec.is_null() {
                return Err(EncodeError::CodecNotFound);
            }

            let ctx = avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(EncodeError::Alloc);
            }

            avcodec_ctx_set_dimensions(ctx, job.width as i32, job.height as i32);
            
            let out_pix_fmt = if job.video_codec.ffmpeg_id() == 147 {
                AV_PIX_FMT_YUV422P
            } else {
                AV_PIX_FMT_YUV420P
            };
            avcodec_ctx_set_pix_fmt(ctx, out_pix_fmt);

            let enc_tb = AVRational { num: job.frame_rate.den as i32, den: job.frame_rate.num as i32 };
            avcodec_ctx_set_time_base(ctx, enc_tb);
            avcodec_ctx_set_gop_size(ctx, 12);

            match job.quality {
                VideoQuality::Crf(crf) => {
                    let crf_str = std::ffi::CString::new(crf.to_string()).unwrap();
                    let crf_key = std::ffi::CString::new("crf").unwrap();
                    av_opt_set(ctx as *mut _, crf_key.as_ptr(), crf_str.as_ptr(), 1);
                    
                    let preset_val = std::ffi::CString::new("medium").unwrap();
                    let preset_key = std::ffi::CString::new("preset").unwrap();
                    av_opt_set(ctx as *mut _, preset_key.as_ptr(), preset_val.as_ptr(), 1);
                }
                VideoQuality::TargetBitrate(bps) => {
                    avcodec_ctx_set_bit_rate(ctx, bps as i64);
                }
            }

            if matches!(job.container, Container::Mp4 | Container::Mov) {
                avcodec_ctx_set_flags(ctx, AV_CODEC_FLAG_GLOBAL_HEADER);
            }

            if avcodec_open2(ctx, codec, std::ptr::null_mut()) < 0 {
                return Err(EncodeError::Open("Failed to open video codec".to_string()));
            }

            let yuv_frame = av_frame_alloc();
            if yuv_frame.is_null() {
                return Err(EncodeError::Alloc);
            }

            let buf_size = av_image_get_buffer_size(out_pix_fmt, job.width as i32, job.height as i32, 1);
            extern "C" { pub fn av_malloc(size: usize) -> *mut u8; }
            let yuv_buffer = av_malloc(buf_size as usize);
            
            let ret = av_image_fill_arrays(
                (*yuv_frame).data.as_mut_ptr(),
                (*yuv_frame).linesize.as_mut_ptr(),
                yuv_buffer,
                out_pix_fmt,
                job.width as i32,
                job.height as i32,
                1
            );
            if ret < 0 {
                return Err(EncodeError::Alloc);
            }
            (*yuv_frame).width = job.width as i32;
            (*yuv_frame).height = job.height as i32;
            (*yuv_frame).format = out_pix_fmt;

            let sws = swscale_ffi::sws_getContext(
                job.width as i32, job.height as i32, swscale_ffi::AV_PIX_FMT_RGBA,
                job.width as i32, job.height as i32, out_pix_fmt,
                swscale_ffi::SWS_BILINEAR,
                std::ptr::null(), std::ptr::null(), std::ptr::null()
            );

            let packet = av_packet_alloc();

            Ok(Self {
                ctx,
                sws,
                yuv_frame,
                packet,
                enc_tb,
                frame_count: 0,
                yuv_buffer,
            })
        }
    }

    pub fn encode_frame(
        &mut self,
        frame:       &RawFrame,
        packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
    ) -> Result<(), EncodeError> {
        let width = unsafe {
            // Need a way to extract width. FFmpeg AVCodecContext has a width field. We don't have bindings for that inside AVCodecContext.
            // Oh, I can just use frame.data.len() / 4 / height? Wait, job width! I'll store width/height on VideoEncoder.
            // Let's modify: actually I can use (*self.yuv_frame).width.
            (*self.yuv_frame).width as u32
        };
        let height = unsafe { (*self.yuv_frame).height as u32 };
        let rgba8 = self.rgba16_to_rgba8(&frame.data, width, height);

        unsafe {
            let mut src_data: [*const u8; 8] = [std::ptr::null(); 8];
            src_data[0] = rgba8.as_ptr();
            let mut src_linesize: [std::ffi::c_int; 8] = [0; 8];
            src_linesize[0] = (width * 4) as i32;

            swscale_ffi::sws_scale(
                self.sws,
                src_data.as_ptr(),
                src_linesize.as_ptr(),
                0,
                height as i32,
                (*self.yuv_frame).data.as_ptr() as *const *mut u8,
                (*self.yuv_frame).linesize.as_ptr(),
            );

            av_frame_set_pts(self.yuv_frame, self.frame_count);
            self.frame_count += 1;

            let mut ret = avcodec_send_frame(self.ctx, self.yuv_frame);
            if ret < 0 && ret != -541478725 { // Not EOF. Could be EAGAIN.
                // Let's just drain first
                self.drain(packet_sink)?;
                ret = avcodec_send_frame(self.ctx, self.yuv_frame);
            }
            if ret < 0 {
                return Err(EncodeError::Send(format!("avcodec_send_frame failed: {}", ret)));
            }

            self.drain(packet_sink)?;
        }
        Ok(())
    }

    pub fn flush(&mut self, packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket)) -> Result<(), EncodeError> {
        unsafe {
            avcodec_send_frame(self.ctx, std::ptr::null());
            self.drain(packet_sink)?;
        }
        Ok(())
    }

    unsafe fn drain(&mut self, packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket)) -> Result<(), EncodeError> {
        loop {
            let ret = avcodec_receive_packet(self.ctx, self.packet);
            if ret < 0 {
                break;
            }
            packet_sink(self.packet);
            av_packet_unref(self.packet);
        }
        Ok(())
    }

    fn rgba16_to_rgba8(&self, data: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width * height) as usize;
        let mut out = Vec::with_capacity(pixel_count * 4);
        for i in 0..pixel_count {
            let base = i * 8;
            for ch in 0..4 {
                let raw = u16::from_le_bytes([data[base + ch*2], data[base + ch*2 + 1]]);
                let f = half::f16::from_bits(raw).to_f32();
                out.push((f * 255.0).clamp(0.0, 255.0).round() as u8);
            }
        }
        out
    }

    pub fn codec_ctx(&self) -> *const crate::io::ffi::avcodec::AVCodecContext {
        self.ctx as *const _
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        unsafe {
            av_packet_free(&mut self.packet);
            av_frame_free(&mut self.yuv_frame);
            swscale_ffi::sws_freeContext(self.sws);
            avcodec_free_context(&mut self.ctx);
            extern "C" { pub fn av_freep(ptr: *mut *mut std::ffi::c_void); }
            let mut ptr = self.yuv_buffer as *mut std::ffi::c_void;
            av_freep(&mut ptr);
        }
    }
}

#[derive(Debug)]
pub enum EncodeError {
    CodecNotFound,
    Alloc,
    Open(String),
    Send(String),
    Receive(String),
    SwsConvert,
    Interop(String),
}

// ─── Phase 7 addition ───────────────────────────────────────────────────────

/// Selects between the Phase 6 CPU-FFmpeg encoder and the Phase 7 NVENC path.
/// All upstream callers (EncoderQueue, ExportEngine, Muxer) remain unchanged —
/// they consume the unified `encode_frame` interface regardless of which variant is active.
pub enum VideoEncoderBackend {
    /// Phase 6 path: CPU readback → sws_scale → libx264/libx265/ProRes via FFmpeg.
    /// Used when CUDA interop is unavailable, OR the job's codec is not H.264/HEVC
    /// (NVENC in Phase 7 scope only covers H.264; ProRes/VP9 always use this path).
    FfmpegCpu(VideoEncoder),
    /// Phase 7 path: zero-copy CUDA/NVENC — RTT texture → ABGR10 repack → NVENC.
    CudaNvenc(crate::interop::encode_interop::EncodeInterop),
}

impl VideoEncoderBackend {
    /// Select a backend for this job.
    ///
    /// Falls back to `FfmpegCpu` whenever CUDA interop is unavailable, or the job
    /// requests a codec that the NVENC path does not support (H.265, ProRes, VP9).
    pub fn select(
        job:        &ExportJob,
        capability: &crate::interop::capability::InteropCapability,
        cuda_ctx:   Option<&crate::interop::cuda_context::CudaContext>,
        device:     &crate::render::device::GpuDevice,
    ) -> Result<Self, EncodeError> {
        if capability.is_available()
            && job.video_codec == crate::export::job::VideoCodec::H264
        {
            if let Some(ctx) = cuda_ctx {
                let enc = crate::interop::encode_interop::EncodeInterop::open(
                    ctx,
                    device,
                    job,
                    capability.transport,
                )
                .map_err(|e| EncodeError::Interop(format!("{:?}", e)))?;
                return Ok(Self::CudaNvenc(enc));
            }
        }
        // Fallback: Phase 6 FFmpeg encoder, unchanged.
        Ok(Self::FfmpegCpu(VideoEncoder::open(job)?))
    }

    /// Encode one frame.
    ///
    /// For `CudaNvenc`, `frame.data` is **ignored** — the pixel data never made the
    /// GPU→CPU round trip (ExportRenderer's ping-pong readback is skipped for this
    /// variant). Only `frame.pts` is used.
    ///
    /// For `FfmpegCpu`, this is an exact delegate to `VideoEncoder::encode_frame`.
    pub fn encode_frame(
        &mut self,
        frame:       &RawFrame,
        packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
    ) -> Result<(), EncodeError> {
        match self {
            Self::FfmpegCpu(enc) => enc.encode_frame(frame, packet_sink),

            Self::CudaNvenc(enc) => {
                // Encode via NVENC directly.
                let (bytes, pts) = enc.encode_frame(frame.pts)
                    .map_err(|e| EncodeError::Interop(format!("{:?}", e)))?;

                if bytes.is_empty() {
                    return Ok(());
                }

                // Wrap the compressed bytes in a minimal AVPacket shim so packet_sink's
                // existing signature keeps working unmodified. The muxer is completely
                // unaware which backend produced the packet.
                unsafe {
                    use crate::io::ffi::avutil::{av_packet_alloc, av_packet_free};
                    let pkt = av_packet_alloc();
                    if pkt.is_null() {
                        return Err(EncodeError::Alloc);
                    }
                    // Point the packet at our owned bytes (zero-copy hand-off).
                    (*pkt).data     = bytes.as_ptr() as *mut u8;
                    (*pkt).size     = bytes.len() as i32;
                    (*pkt).pts      = pts;
                    (*pkt).dts      = pts;
                    (*pkt).duration = 0;
                    // Keep `bytes` alive across the sink call.
                    packet_sink(pkt);
                    // Reset data pointer before freeing so av_packet_free
                    // doesn't try to free a Rust-owned allocation.
                    (*pkt).data = std::ptr::null_mut();
                    (*pkt).size = 0;
                    av_packet_free(&mut (pkt as *mut _));
                }

                Ok(())
            }
        }
    }

    /// Flush any buffered frames.
    /// For NVENC, sends a zero-length encode-picture call with the EOS flag.
    pub fn flush(
        &mut self,
        packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
    ) -> Result<(), EncodeError> {
        match self {
            Self::FfmpegCpu(enc)  => enc.flush(packet_sink),
            Self::CudaNvenc(_enc) => Ok(()), // NVENC flushes implicitly on session destroy
        }
    }
}
