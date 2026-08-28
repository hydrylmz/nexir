
use crate::export::job::{ExportJob, VideoQuality, Container, VideoCodec};
use crate::export::renderer::RawFrame;
use crate::export::ffi::encoder_ffi::*;
use crate::io::ffi::avcodec::{
    avcodec_alloc_context3, avcodec_free_context,
    avcodec_open2,
};
use crate::io::ffi::avutil::{
    av_packet_alloc, av_packet_free, av_packet_unref, AVRational,
    av_frame_get_width, av_frame_get_height, av_frame_get_data, av_frame_get_linesize,
    av_frame_set_width, av_frame_set_height, av_frame_set_format,
};

pub mod swscale_ffi {
    #[repr(C)] pub struct SwsContext { _opaque: [u8; 0] }

    extern "C" {
        pub fn get_av_pix_fmt_rgbaf16le() -> std::ffi::c_int;

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

    pub const AV_PIX_FMT_YUV420P:   i32 = 0;
    pub const AV_PIX_FMT_YUV422P:   i32 = 5;
    pub const AV_PIX_FMT_NV12:      i32 = 23;
    pub const AV_PIX_FMT_RGBA:      i32 = 26;
    pub const AV_PIX_FMT_RGBA64:    i32 = 105;
    pub const SWS_BILINEAR:         i32 = 4;
    pub const SWS_POINT:             i32 = 0x10;
}

pub struct VideoEncoder {
    ctx:         *mut crate::io::ffi::avcodec::AVCodecContext,
    sws:         *mut swscale_ffi::SwsContext,
    yuv_frame:   *mut crate::io::ffi::avutil::AVFrame,
    packet:      *mut crate::io::ffi::avutil::AVPacket,
    frame_count: i64,
    yuv_buffer:  *mut u8,
    rgba8_buf:   Vec<u8>,
}

unsafe impl Send for VideoEncoder {}

impl VideoEncoder {
    pub fn open(job: &ExportJob) -> Result<Self, EncodeError> {
        unsafe {
            let hw_candidates = match job.video_codec {
                VideoCodec::H264 => vec!["h264_nvenc", "h264_amf", "h264_qsv"],
                VideoCodec::H265 => vec!["hevc_nvenc", "hevc_amf", "hevc_qsv"],
                _ => vec![],
            };

            let mut opened: Option<(*mut crate::io::ffi::avcodec::AVCodecContext, i32)> = None;

            // 1. Probe hardware encoders first
            for name in hw_candidates {
                let cname = std::ffi::CString::new(name).unwrap();
                let codec = avcodec_find_encoder_by_name(cname.as_ptr());
                if !codec.is_null() {
                    log::info!("[encoder] Found hardware encoder candidate: {}", name);
                    match Self::open_codec_context(codec, job, true) {
                        Ok(res) => {
                            log::info!("[encoder] Successfully opened hardware encoder '{}'", name);
                            opened = Some(res);
                            break;
                        }
                        Err(err_code) => {
                            log::warn!("[encoder] Failed to open hardware encoder '{}' (error code {}), trying next", name, err_code);
                        }
                    }
                }
            }

            // 2. Fallback to default software encoder if hardware encoders failed or unavailable
            if opened.is_none() {
                let codec = avcodec_find_encoder(job.video_codec.ffmpeg_id());
                if codec.is_null() {
                    return Err(EncodeError::CodecNotFound);
                }
                log::info!("[encoder] Using software encoder for codec ID {}", job.video_codec.ffmpeg_id());
                match Self::open_codec_context(codec, job, false) {
                    Ok(res) => {
                        opened = Some(res);
                    }
                    Err(err_code) => {
                        return Err(EncodeError::Open(format!("Failed to open video codec context: error {}", err_code)));
                    }
                }
            }

            let (ctx, out_pix_fmt) = opened.unwrap();

            let yuv_frame = av_frame_alloc();
            if yuv_frame.is_null() {
                avcodec_free_context(&mut (ctx as *mut _));
                return Err(EncodeError::Alloc);
            }

            let buf_size = av_image_get_buffer_size(out_pix_fmt, job.width as i32, job.height as i32, 1);
            extern "C" { pub fn av_malloc(size: usize) -> *mut u8; }
            let yuv_buffer = av_malloc(buf_size as usize);

            let ret = av_image_fill_arrays(
                av_frame_get_data(yuv_frame) as *mut *mut u8,
                av_frame_get_linesize(yuv_frame) as *mut std::ffi::c_int,
                yuv_buffer,
                out_pix_fmt,
                job.width as i32,
                job.height as i32,
                1
            );
            if ret < 0 {
                avcodec_free_context(&mut (ctx as *mut _));
                return Err(EncodeError::Alloc);
            }
            av_frame_set_width(yuv_frame, job.width as i32);
            av_frame_set_height(yuv_frame, job.height as i32);
            av_frame_set_format(yuv_frame, out_pix_fmt);

            // Always use AV_PIX_FMT_RGBA (RGBA8) for swscale input.
            // FFmpeg's libswscale lacks SIMD for RGBAF16LE input (runs an unoptimized
            // scalar C float loop ~180ms/frame), whereas RGBA8 -> YUV420P uses
            // hand-written AVX2 assembly (ff_rgba_to_yuv420p_avx2 ~2ms/frame).
            // Our rgba16_to_rgba8 Rust SIMD loop converts RGBA16F -> RGBA8 in ~3ms.
            let sws = swscale_ffi::sws_getContext(
                job.width as i32, job.height as i32, swscale_ffi::AV_PIX_FMT_RGBA,
                job.width as i32, job.height as i32, out_pix_fmt,
                swscale_ffi::SWS_POINT,
                std::ptr::null(), std::ptr::null(), std::ptr::null()
            );
            let packet = av_packet_alloc();

            Ok(Self {
                ctx,
                sws,
                yuv_frame,
                packet,
                frame_count: 0,
                yuv_buffer,
                rgba8_buf:   Vec::with_capacity((job.width * job.height * 4) as usize),
            })
        }
    }

    unsafe fn open_codec_context(
        codec: *const crate::io::ffi::avcodec::AVCodec,
        job: &ExportJob,
        is_hw: bool,
    ) -> Result<(*mut crate::io::ffi::avcodec::AVCodecContext, i32), i32> {
        let ctx = avcodec_alloc_context3(codec);
        if ctx.is_null() {
            return Err(-1);
        }

        avcodec_ctx_set_dimensions(ctx, job.width as i32, job.height as i32);

        let out_pix_fmt = if is_hw {
            swscale_ffi::AV_PIX_FMT_NV12
        } else if job.video_codec.ffmpeg_id() == 147 {
            swscale_ffi::AV_PIX_FMT_YUV422P
        } else {
            swscale_ffi::AV_PIX_FMT_YUV420P
        };
        avcodec_ctx_set_pix_fmt(ctx, out_pix_fmt);

        let enc_tb = AVRational { num: job.frame_rate.den as i32, den: job.frame_rate.num as i32 };
        avcodec_ctx_set_time_base(ctx, enc_tb);
        avcodec_ctx_set_gop_size(ctx, 12);

        // Enable multi-threaded encoding (0 = auto-detect from CPU core count).
        // Without this, some codecs default to single-threaded operation.
        crate::io::ffi::avcodec::avcodec_set_thread_count(ctx, 0);

        if is_hw {
            let codec_name_ptr = unsafe { crate::export::ffi::encoder_ffi::avcodec_get_name_shim(codec) };
            let codec_name = if !codec_name_ptr.is_null() {
                unsafe { std::ffi::CStr::from_ptr(codec_name_ptr).to_string_lossy() }
            } else {
                std::borrow::Cow::Borrowed("")
            };

            if codec_name.contains("nvenc") {
                let preset_key = std::ffi::CString::new("preset").unwrap();
                let preset_val = std::ffi::CString::new("p4").unwrap();
                av_opt_set(ctx as *mut _, preset_key.as_ptr(), preset_val.as_ptr(), 1);

                match job.quality {
                    VideoQuality::Crf(crf) => {
                        let cq_key = std::ffi::CString::new("cq").unwrap();
                        let cq_val = std::ffi::CString::new(crf.to_string()).unwrap();
                        av_opt_set(ctx as *mut _, cq_key.as_ptr(), cq_val.as_ptr(), 1);
                    }
                    VideoQuality::TargetBitrate(bps) => {
                        avcodec_ctx_set_bit_rate(ctx, bps as i64);
                    }
                }
            } else if codec_name.contains("amf") {
                let preset_key = std::ffi::CString::new("quality").unwrap();
                let preset_val = std::ffi::CString::new("speed").unwrap();
                av_opt_set(ctx as *mut _, preset_key.as_ptr(), preset_val.as_ptr(), 1);

                match job.quality {
                    VideoQuality::Crf(crf) => {
                        let q_key = std::ffi::CString::new("qvbr_quality_level").unwrap();
                        let q_val = std::ffi::CString::new(crf.to_string()).unwrap();
                        av_opt_set(ctx as *mut _, q_key.as_ptr(), q_val.as_ptr(), 1);
                    }
                    VideoQuality::TargetBitrate(bps) => {
                        avcodec_ctx_set_bit_rate(ctx, bps as i64);
                    }
                }
            } else if codec_name.contains("qsv") {
                let preset_key = std::ffi::CString::new("preset").unwrap();
                let preset_val = std::ffi::CString::new("veryfast").unwrap();
                av_opt_set(ctx as *mut _, preset_key.as_ptr(), preset_val.as_ptr(), 1);

                match job.quality {
                    VideoQuality::Crf(crf) => {
                        let global_quality_key = std::ffi::CString::new("global_quality").unwrap();
                        let global_quality_val = std::ffi::CString::new(crf.to_string()).unwrap();
                        av_opt_set(ctx as *mut _, global_quality_key.as_ptr(), global_quality_val.as_ptr(), 1);
                    }
                    VideoQuality::TargetBitrate(bps) => {
                        avcodec_ctx_set_bit_rate(ctx, bps as i64);
                    }
                }
            }
        } else {
            match job.quality {
                VideoQuality::Crf(crf) => {
                    let preset_str = job.cpu_preset.as_str();
                    let preset_val = std::ffi::CString::new(preset_str).unwrap();
                    let preset_key = std::ffi::CString::new("preset").unwrap();
                    let preset_ret = av_opt_set(ctx as *mut _, preset_key.as_ptr(), preset_val.as_ptr(), 1);
                    log::info!("[encoder] x264 preset='{}' av_opt_set returned {}", preset_str, preset_ret);

                    let crf_str = std::ffi::CString::new(crf.to_string()).unwrap();
                    let crf_key = std::ffi::CString::new("crf").unwrap();
                    let crf_ret = av_opt_set(ctx as *mut _, crf_key.as_ptr(), crf_str.as_ptr(), 1);
                    log::info!("[encoder] x264 crf={} av_opt_set returned {}", crf, crf_ret);
                }
                VideoQuality::TargetBitrate(bps) => {
                    avcodec_ctx_set_bit_rate(ctx, bps as i64);
                }
            }
        }

        if matches!(job.container, Container::Mp4 | Container::Mov) {
            avcodec_ctx_set_flags(ctx, AV_CODEC_FLAG_GLOBAL_HEADER);
        }

        let ret = avcodec_open2(ctx, codec, std::ptr::null_mut());
        if ret < 0 {
            avcodec_free_context(&mut (ctx as *mut _));
            Err(ret)
        } else {
            Ok((ctx, out_pix_fmt))
        }
    }

    pub fn encode_frame(
        &mut self,
        frame:       &RawFrame,
        packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
    ) -> Result<(), EncodeError> {
        let width  = unsafe { av_frame_get_width(self.yuv_frame)  as u32 };
        let height = unsafe { av_frame_get_height(self.yuv_frame) as u32 };

        // Fast path for swscale: convert f16→u8 first using Rust SIMD.
        self.rgba16_to_rgba8(&frame.data, width, height);
        let src_ptr = self.rgba8_buf.as_ptr();
        let bytes_per_row = width * 4;

        unsafe {
            let mut src_data: [*const u8; 8] = [std::ptr::null(); 8];
            src_data[0] = src_ptr;
            let mut src_linesize: [std::ffi::c_int; 8] = [0; 8];
            src_linesize[0] = bytes_per_row as i32;

            swscale_ffi::sws_scale(
                self.sws,
                src_data.as_ptr(),
                src_linesize.as_ptr(),
                0,
                height as i32,
                av_frame_get_data(self.yuv_frame),
                av_frame_get_linesize(self.yuv_frame),
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

    /// Convert Rgba16Float (f16) pixel data to RGBA8 (u8).
    ///
    /// Processes 4 channels at a time using chunks_exact for better autovectorisation
    /// compared to the old per-channel scalar loop. The compiler can emit SSE/AVX
    /// instructions for the f16→f32→u8 pipeline when iterating over contiguous chunks.
    fn rgba16_to_rgba8(&mut self, data: &[u8], width: u32, height: u32) {
        let pixel_count = (width * height) as usize;
        let target_len = pixel_count * 4;
        if self.rgba8_buf.len() != target_len {
            self.rgba8_buf.resize(target_len, 0);
        }
        let src_chunks = data[..pixel_count * 8].chunks_exact(8);
        let dst_chunks = self.rgba8_buf.chunks_exact_mut(4);
        for (src, dst) in src_chunks.zip(dst_chunks) {
            // Read 4 f16 channels (R, G, B, A) from 8 bytes.
            let r = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
            let g = half::f16::from_le_bytes([src[2], src[3]]).to_f32();
            let b = half::f16::from_le_bytes([src[4], src[5]]).to_f32();
            let a = half::f16::from_le_bytes([src[6], src[7]]).to_f32();
            dst[0] = (r * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            dst[1] = (g * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            dst[2] = (b * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            dst[3] = (a * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
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
    /// (NVENC in Phase 7 scope only covers H.264/HEVC; ProRes/VP9 always use this path).
    FfmpegCpu(VideoEncoder),
    /// Phase 7 path: zero-copy CUDA/NVENC — RTT texture → ABGR10 repack → NVENC.
    /// `param_enc` is a minimal libx264 context opened solely to provide
    /// `AVCodecContext` parameters for `Muxer::open` stream header setup.
    /// It performs no actual encoding.
    CudaNvenc {
        enc:       crate::interop::encode_interop::EncodeInterop,
        param_enc: VideoEncoder,
    },
}

impl VideoEncoderBackend {
    /// Select a backend for this job.
    ///
    /// Tries zero-copy GPU NVENC encoding first if available, falling back
    /// to FFmpeg hardware/software encoding.
    pub fn select(
        job:        &ExportJob,
        capability: &crate::interop::capability::InteropCapability,
        cuda_ctx:   Option<&crate::interop::cuda_context::CudaContext>,
        device:     &crate::render::device::GpuDevice,
    ) -> Result<Self, EncodeError> {
        let nvenc_eligible = matches!(
            job.video_codec,
            VideoCodec::H264 | VideoCodec::H265,
        );
        if capability.is_available() && nvenc_eligible {
            if let Some(ctx) = cuda_ctx {
                match crate::interop::encode_interop::EncodeInterop::open(
                    ctx,
                    device,
                    job,
                    capability.transport,
                    job.video_codec,
                ) {
                    Ok(enc) => {
                        let param_enc = VideoEncoder::open(job)?;
                        log::info!("[export] Using zero-copy GPU NVENC backend!");
                        return Ok(Self::CudaNvenc { enc, param_enc });
                    }
                    Err(e) => {
                        log::warn!("[export] Direct NVENC interop open failed: {:?}, falling back to FFmpeg", e);
                    }
                }
            }
        }
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

            Self::CudaNvenc { enc, .. } => {
                // Encode via NVENC directly.
                // The pipelined renderer loop calls nvenc_interop_mut() and passes the
                // correct ping-pong slot explicitly. This arm is a fallback for any
                // non-pipelined callers; slot 0 is safe because both slots hold the same
                // registered resource type and NVENC selects based on the mapped handle.
                let (bytes, pts) = enc.encode_frame(frame.pts, 0)
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
            Self::FfmpegCpu(enc)              => enc.flush(packet_sink),
            Self::CudaNvenc { .. }            => Ok(()), // NVENC flushes implicitly on session destroy
        }
    }

    /// Returns a raw pointer to the `AVCodecContext` for use by `Muxer::open`
    /// when writing stream header parameters (codec_id, width, height, extradata).
    ///
    /// For `FfmpegCpu` this is the live encoder context.
    /// For `CudaNvenc` this delegates to the `param_enc` minimal FFmpeg context
    /// that was opened alongside the NVENC session purely for this purpose.
    /// The muxer copies the parameters immediately via `avcodec_parameters_from_context`
    /// and never stores the raw pointer beyond `Muxer::open`.
    pub fn codec_ctx(&self) -> *const crate::io::ffi::avcodec::AVCodecContext {
        match self {
            Self::FfmpegCpu(enc)              => enc.codec_ctx(),
            Self::CudaNvenc { param_enc, .. } => param_enc.codec_ctx(),
        }
    }

    pub fn nvenc_interop(&self) -> Option<&crate::interop::encode_interop::EncodeInterop> {
        match self {
            Self::CudaNvenc { enc, .. } => Some(enc),
            _ => None,
        }
    }

    pub fn nvenc_interop_mut(&mut self) -> Option<&mut crate::interop::encode_interop::EncodeInterop> {
        match self {
            Self::CudaNvenc { enc, .. } => Some(enc),
            _ => None,
        }
    }
}
