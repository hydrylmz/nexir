
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

        /// Coefficient table for one of the `SWS_CS_*` colour spaces.
        pub fn sws_getCoefficients(colorspace: std::ffi::c_int) -> *const std::ffi::c_int;

        /// Pin the matrix and range swscale converts with.
        ///
        /// P1.7 — without this call swscale uses its default (BT.601) matrix no
        /// matter what the stream is tagged as, so a BT.2020 export would carry
        /// BT.601-converted samples: a real hue shift, not just wrong metadata.
        pub fn sws_setColorspaceDetails(
            c:          *mut SwsContext,
            inv_table:  *const std::ffi::c_int,
            srcRange:   std::ffi::c_int,
            table:      *const std::ffi::c_int,
            dstRange:   std::ffi::c_int,
            brightness: std::ffi::c_int,
            contrast:   std::ffi::c_int,
            saturation: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }

    pub const AV_PIX_FMT_YUV420P:   i32 = 0;
    pub const AV_PIX_FMT_YUV422P:   i32 = 5;
    pub const AV_PIX_FMT_NV12:      i32 = 23;
    pub const AV_PIX_FMT_RGBA:      i32 = 26;
    /// Packed 16-bit-per-channel RGBA, little-endian — the swscale input for a
    /// 10-bit encode, where RGBA8 would throw away the precision the whole HDR
    /// path exists to keep.
    pub const AV_PIX_FMT_RGBA64:    i32 = 105;
    pub const SWS_BILINEAR:         i32 = 4;
    pub const SWS_POINT:             i32 = 0x10;

    /// swscale's 16.16 fixed-point identity for brightness/contrast/saturation.
    pub const SWS_ONE_16_16: i32 = 1 << 16;

    /// Whether an encoder output format stores more than 8 bits per component.
    ///
    /// Asked of the format the encoder was actually opened with rather than of the
    /// job, so a hardware encoder that fell back to a different format cannot end
    /// up with a mismatched swscale input.  `av_pix_fmt_bit_depth` is FFmpeg's own
    /// descriptor lookup, so this cannot drift from the enum.
    pub fn pix_fmt_is_high_depth(fmt: i32) -> bool {
        unsafe { crate::io::ffi::avutil::av_pix_fmt_bit_depth(fmt) > 8 }
    }
}

pub struct VideoEncoder {
    ctx:         *mut crate::io::ffi::avcodec::AVCodecContext,
    sws:         *mut swscale_ffi::SwsContext,
    yuv_frame:   *mut crate::io::ffi::avutil::AVFrame,
    packet:      *mut crate::io::ffi::avutil::AVPacket,
    frame_count: i64,
    yuv_buffer:  *mut u8,
    /// Scratch buffer holding the swscale input for one frame.
    ///
    /// 8-bit jobs pack RGBA8 here (4 bytes/px); 10-bit jobs pack RGBA64LE
    /// (8 bytes/px), because narrowing to 8 bits first would discard exactly the
    /// precision a 10-bit encode exists to carry.
    rgb_buf:     Vec<u8>,
    /// True when the encoder is fed 10-bit-or-higher pixels — i.e. an HDR job or
    /// ProRes.  Selects which packer `encode_frame` runs and which swscale input
    /// format `open` configured.
    high_depth:  bool,
}

unsafe impl Send for VideoEncoder {}

impl VideoEncoder {
    pub fn open(job: &ExportJob) -> Result<Self, EncodeError> {
        unsafe {
            // NVENC/AMF/QSV are only probed when the job's depth is one they can
            // take.  For a 10-bit HDR job that means HEVC only: no vendor H.264
            // encoder accepts P010, and asking for it is a guaranteed open
            // failure that would just log noise before falling back.
            let hw_candidates = match (job.video_codec, job.encode_bit_depth() >= 10) {
                (VideoCodec::H264, false) => vec!["h264_nvenc", "h264_amf", "h264_qsv"],
                (VideoCodec::H264, true)  => vec![],
                (VideoCodec::H265, _)     => vec!["hevc_nvenc", "hevc_amf", "hevc_qsv"],
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

            // Feed swscale RGBA8 for an 8-bit encode and RGBA64LE for a 10-bit one.
            //
            // FFmpeg's libswscale lacks SIMD for RGBAF16LE input (an unoptimised
            // scalar float loop, ~180 ms/frame), whereas RGBA8 -> YUV420P uses
            // hand-written AVX2 assembly (~2 ms/frame) and RGBA64 -> 10-bit planar
            // has a fast integer path.  Our own f16 -> u8/u16 packers below run in
            // ~3 ms, so both cases stay well clear of the float path.
            //
            // RGBA8 is NOT usable for the 10-bit encode: quantising to 8 bits here
            // would throw away exactly the precision the HDR path exists to keep,
            // and PQ at 8 bits bands visibly in every gradient.
            let high_depth = swscale_ffi::pix_fmt_is_high_depth(out_pix_fmt);
            let src_pix_fmt = if high_depth {
                swscale_ffi::AV_PIX_FMT_RGBA64
            } else {
                swscale_ffi::AV_PIX_FMT_RGBA
            };

            let sws = swscale_ffi::sws_getContext(
                job.width as i32, job.height as i32, src_pix_fmt,
                job.width as i32, job.height as i32, out_pix_fmt,
                swscale_ffi::SWS_POINT,
                std::ptr::null(), std::ptr::null(), std::ptr::null()
            );
            if sws.is_null() {
                avcodec_free_context(&mut (ctx as *mut _));
                return Err(EncodeError::SwsConvert);
            }

            // P1.7 — pin the RGB->YUV matrix and range to what the stream is
            // TAGGED as.  swscale otherwise applies its BT.601 default, so a
            // BT.2020 HDR export would be tagged BT.2020 while carrying BT.601
            // samples.  The source is full-range RGB either way (srcRange = 1).
            let dst_coeffs = swscale_ffi::sws_getCoefficients(job.sws_colorspace());
            let src_coeffs = swscale_ffi::sws_getCoefficients(1); // SWS_CS_ITU709
            let cs_ret = swscale_ffi::sws_setColorspaceDetails(
                sws,
                src_coeffs, 1,                    // RGB input is always full range
                dst_coeffs, job.sws_dst_range(),
                0,
                swscale_ffi::SWS_ONE_16_16,
                swscale_ffi::SWS_ONE_16_16,
            );
            if cs_ret < 0 {
                // Not fatal — the conversion still runs, just with swscale's
                // default matrix.  Log loudly: it means the file's colour tags and
                // its samples disagree.
                log::warn!(
                    "[encoder] sws_setColorspaceDetails failed ({cs_ret}) — output \
                     will be tagged colorspace {} but converted with swscale's \
                     default matrix",
                    job.sws_colorspace()
                );
            } else {
                log::info!(
                    "[encoder] swscale: {}-bit {} -> pix_fmt {}, colorspace {}, dst_range {}",
                    if high_depth { 16 } else { 8 },
                    if high_depth { "RGBA64" } else { "RGBA8" },
                    out_pix_fmt,
                    job.sws_colorspace(),
                    job.sws_dst_range(),
                );
            }

            let packet = av_packet_alloc();

            let bytes_per_px = if high_depth { 8usize } else { 4usize };

            Ok(Self {
                ctx,
                sws,
                yuv_frame,
                packet,
                frame_count: 0,
                yuv_buffer,
                rgb_buf:    Vec::with_capacity(
                    job.width as usize * job.height as usize * bytes_per_px,
                ),
                high_depth,
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

        // P1.7 — the encoder's pixel format now comes from the job rather than
        // being hardcoded 8-bit.  `ExportJob::encoder_pix_fmt` returns P010 /
        // YUV420P10LE for an HDR job, NV12 / YUV420P for SDR, and always
        // YUV422P10LE for ProRes (which has no 8-bit mode).  Feeding an HDR job
        // through yuv420p was the reason a file could be tagged HDR10 and still
        // carry SDR pixels.
        let out_pix_fmt = job.encoder_pix_fmt(is_hw);
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

        // P1.7 — colour description. Set BEFORE avcodec_open2 so libavcodec bakes
        // it into the bitstream's VUI; setting it afterwards only reaches the
        // container. `job.output_color` describes what the encoder is fed: Rec.709
        // after the renderer's tone-map for an SDR job, BT.2020 + PQ for an HDR one.
        let color = &job.output_color;
        avcodec_ctx_set_color_space(ctx, color.av_color_space());
        avcodec_ctx_set_color_range(ctx, color.av_color_range());
        avcodec_ctx_set_color_trc(ctx, color.av_color_trc());
        avcodec_ctx_set_color_primaries(ctx, color.av_color_primaries());
        // AVCHROMA_LOC_LEFT (1) — MPEG-2/H.264/HEVC 4:2:0 siting, which is what
        // both swscale and NVENC produce here.
        avcodec_ctx_set_chroma_location(ctx, 1);

        // P1.7 — HDR10 static metadata, also before open2: libavcodec reads
        // `decoded_side_data` there, and that is what makes libx265 emit the
        // mastering-display and content-light-level SEI into the bitstream.
        // Without it a BT.2020/PQ file carries no grade information and displays
        // fall back to their own guesses.
        if let Some(hdr) = job.hdr10.as_ref() {
            let ret = crate::export::ffi::encoder_ffi::avcodec_ctx_set_hdr10_metadata(
                ctx,
                hdr.primaries.as_ptr(),
                hdr.white_point.as_ptr(),
                hdr.min_luminance,
                hdr.max_luminance,
                hdr.max_cll,
                hdr.max_fall,
            );
            if ret < 0 {
                // Non-fatal: the file is still valid HDR10 by its colour tags,
                // just unmastered.  Worth a warning rather than failing an export
                // that is otherwise correct.
                log::warn!(
                    "[encoder] failed to attach HDR10 static metadata to the encoder \
                     context — the bitstream will carry no mastering-display SEI"
                );
            } else {
                log::info!(
                    "[encoder] HDR10 metadata: max_luminance={} cd/m², MaxCLL={}, MaxFALL={}",
                    hdr.peak_nits(), hdr.max_cll, hdr.max_fall
                );
            }
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

        // Pack the RGBA16Float render target into whichever integer RGB format
        // swscale was configured for, then let swscale's optimised path do the
        // RGB->YUV conversion.  See `open` for why the f16 texture is not handed
        // to swscale directly.
        let bytes_per_row = if self.high_depth {
            self.rgba16f_to_rgba64(&frame.data, width, height);
            width * 8
        } else {
            self.rgba16f_to_rgba8(&frame.data, width, height);
            width * 4
        };
        let src_ptr = self.rgb_buf.as_ptr();

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
            // One tick in the encoder timebase (which is 1/frame_rate), so the
            // muxer can size the final `stts` entry.  With duration = 0 the mp4
            // track's `mdhd` duration ends exactly at the last frame's PTS and
            // decoders flag that frame AV_PKT_FLAG_DISCARD, silently losing it.
            av_frame_set_duration(self.yuv_frame, 1);
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
            // Guarantee a non-zero packet duration.  Not every encoder wrapper
            // propagates AVFrame::duration onto its output packets, and the mp4/mov
            // muxer sizes the final `stts` entry from the LAST packet's duration:
            // when that is 0 the track's `mdhd` duration ends exactly at the final
            // frame's PTS, and decoders flag the frame AV_PKT_FLAG_DISCARD — which
            // silently drops the last frame of every export.
            if (*self.packet).duration == 0 {
                (*self.packet).duration = 1; // one tick of the encoder timebase
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
    fn rgba16f_to_rgba8(&mut self, data: &[u8], width: u32, height: u32) {
        let pixel_count = (width * height) as usize;
        let target_len = pixel_count * 4;
        if self.rgb_buf.len() != target_len {
            self.rgb_buf.resize(target_len, 0);
        }
        let src_chunks = data[..pixel_count * 8].chunks_exact(8);
        let dst_chunks = self.rgb_buf.chunks_exact_mut(4);
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

    /// Convert Rgba16Float (f16) pixel data to RGBA64LE (u16 per channel).
    ///
    /// P1.7 — the 10-bit path's equivalent of `rgba16f_to_rgba8`. Scaling to the
    /// full 16-bit range rather than to 1023 matters: swscale's RGBA64 -> 10-bit
    /// YUV conversion assumes 16-bit input and shifts down itself, so pre-scaling
    /// to 10 bits here would lose a factor of 64 in brightness.
    ///
    /// f16 has an 11-bit significand, so it represents every 10-bit code exactly
    /// and the 16-bit intermediate is lossless with respect to the 10-bit output.
    fn rgba16f_to_rgba64(&mut self, data: &[u8], width: u32, height: u32) {
        let pixel_count = (width * height) as usize;
        let target_len = pixel_count * 8;
        if self.rgb_buf.len() != target_len {
            self.rgb_buf.resize(target_len, 0);
        }
        let src_chunks = data[..pixel_count * 8].chunks_exact(8);
        let dst_chunks = self.rgb_buf.chunks_exact_mut(8);
        for (src, dst) in src_chunks.zip(dst_chunks) {
            for c in 0..4 {
                let v = half::f16::from_le_bytes([src[c * 2], src[c * 2 + 1]]).to_f32();
                let scaled = (v * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                let bytes = scaled.to_le_bytes();
                dst[c * 2]     = bytes[0];
                dst[c * 2 + 1] = bytes[1];
            }
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

/// Hand one NVENC bitstream to a `packet_sink` as a borrowed `AVPacket`.
///
/// The packet points directly at `packet.bytes` (no copy) and its data pointer
/// is cleared before `av_packet_free` so FFmpeg never tries to free a
/// Rust-owned allocation.  `bytes` must outlive the sink call, which it does:
/// the caller owns the `EncodedPacket` for the whole call.
///
/// PTS and DTS are written separately — see `DtsQueue` in `encode_interop.rs`
/// for why `dts = pts` is wrong for a reordering encoder — and
/// `AV_PKT_FLAG_KEY` is set from the picture type NVENC reported so the muxer
/// can build a correct sync-sample table.
pub(crate) fn write_nvenc_packet(
    packet:      &crate::interop::encode_interop::EncodedPacket,
    packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
) -> Result<(), EncodeError> {
    use crate::io::ffi::avutil::{av_packet_alloc, av_packet_free, AV_PKT_FLAG_KEY};

    unsafe {
        let pkt = av_packet_alloc();
        if pkt.is_null() {
            return Err(EncodeError::Alloc);
        }
        (*pkt).data     = packet.bytes.as_ptr() as *mut u8;
        (*pkt).size     = packet.bytes.len() as i32;
        (*pkt).pts      = packet.pts;
        (*pkt).dts      = packet.dts;
        // One tick in the muxer's video timebase (1/frame_rate).  libavcodec sets
        // this itself on the CPU path; here the packet is built by hand, and a
        // zero duration on the LAST packet makes the mp4 track's `mdhd` duration
        // stop at that frame's PTS — decoders then flag it AV_PKT_FLAG_DISCARD
        // and the final frame silently vanishes.
        (*pkt).duration = 1;
        (*pkt).flags    = if packet.is_keyframe { AV_PKT_FLAG_KEY } else { 0 };

        packet_sink(pkt);

        // Detach the Rust-owned buffer before freeing the packet shell.
        (*pkt).data = std::ptr::null_mut();
        (*pkt).size = 0;
        av_packet_free(&mut (pkt as *mut _));
    }
    Ok(())
}

/// Selects between the libavcodec encoder and the direct zero-copy NVENC path.
/// All upstream callers (EncoderQueue, ExportEngine, Muxer) remain unchanged —
/// they consume the unified `encode_frame` interface regardless of which variant is active.
pub enum VideoEncoderBackend {
    /// libavcodec path: CPU readback → sws_scale → FFmpeg encoder.
    ///
    /// **Not necessarily a CPU encode.** `VideoEncoder::open` probes
    /// `h264_nvenc`/`hevc_nvenc` (then AMF, then QSV) before falling back to
    /// libx264/libx265, so for H.264/HEVC jobs this usually still encodes on the
    /// GPU — what it gives up versus `CudaNvenc` is the zero-copy upload, not
    /// hardware acceleration. Genuine software encoding only happens when no
    /// vendor encoder opens, or for codecs NVENC does not cover (ProRes, VP9).
    ///
    /// Chosen when CUDA interop is unavailable, the codec is outside NVENC's
    /// H.264/HEVC scope, or the output's bit depth / range cannot be signalled by
    /// the direct interop session (see `select`).
    FfmpegEncoder(VideoEncoder),
    /// Phase 7 path: zero-copy CUDA/NVENC — RTT texture → NV12 convert → NVENC.
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
        cuda_ctx:   Option<&std::sync::Arc<crate::interop::cuda_context::CudaContext>>,
        device:     &crate::render::device::GpuDevice,
    ) -> Result<Self, EncodeError> {
        let nvenc_eligible = matches!(
            job.video_codec,
            VideoCodec::H264 | VideoCodec::H265,
        );

        // P1.7 — an HDR job never takes the zero-copy NVENC path.
        //
        // `EncodeInterop` feeds NVENC ABGR10 and initialises the session from the
        // driver's preset config, patching only `frameIntervalP` at a byte offset
        // it verifies. Producing a Main10 HEVC stream additionally requires
        // `profileGUID` and the codec-specific `pixelBitDepthMinus8`, both of
        // which live inside NV_ENC_CONFIG's per-codec union — territory this code
        // deliberately does not poke at guessed offsets (see the offset discussion
        // in `interop/ffi/nvenc.rs`). Left alone, the session would encode a
        // Main-profile 8-bit stream from the 10-bit input, i.e. exactly the
        // tagged-HDR-with-SDR-pixels file this work exists to eliminate.
        //
        // The FFmpeg path is not a downgrade in speed: it still resolves
        // `hevc_nvenc` first, so the encode remains on the GPU. What it gives up
        // is the zero-copy readback, and it gains a libavcodec wrapper that
        // configures Main10 + P010 correctly.
        if job.is_hdr() {
            log::info!(
                "[export] HDR job — using the FFmpeg encoder path (hevc_nvenc if \
                 available) rather than zero-copy NVENC: the direct interop \
                 session cannot be configured for Main10 10-bit output"
            );
            return Ok(Self::FfmpegEncoder(VideoEncoder::open(job)?));
        }

        if capability.is_available() && nvenc_eligible {
            // P1.9 — the zero-copy path now converts RGB→YUV in our own shader
            // (`Nv12EncodeNode`) and hands NVENC NV12, so the driver applies no
            // matrix of its own and the stream's tags describe the samples. The
            // gate that remains covers 10-bit and full-range output, whose
            // signalling lives in the NV_ENC_CONFIG VUI union this code will not
            // write at guessed offsets. See
            // `ExportJob::nvenc_zero_copy_is_colour_safe`.
            if !job.nvenc_zero_copy_is_colour_safe() {
                log::info!(
                    "[export] output is {:?} {}-bit {:?} — the zero-copy NVENC path \
                     cannot signal that (its NV12 shader is 8-bit, and the range flag \
                     lives in the VUI union), so using the FFmpeg encoder path \
                     (h264_nvenc/hevc_nvenc if available). The encode stays on the \
                     GPU; only the zero-copy readback is given up.",
                    job.output_color.matrix,
                    job.output_color.bit_depth,
                    job.output_color.effective_range()
                );
                return Ok(Self::FfmpegEncoder(VideoEncoder::open(job)?));
            }

            if let Some(ctx) = cuda_ctx {
                // Clone the Arc: EncodeInterop must own a share of the CUDA
                // primary context for as long as its NVENC session lives.
                match crate::interop::encode_interop::EncodeInterop::open(
                    std::sync::Arc::clone(ctx),
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
                        log::warn!("[export] Direct NVENC interop open failed: {e}, falling back to FFmpeg");
                    }
                }
            }
        }
        Ok(Self::FfmpegEncoder(VideoEncoder::open(job)?))
    }

    /// Encode one frame.
    ///
    /// For `CudaNvenc`, `frame.data` is **ignored** — the pixel data never made the
    /// GPU→CPU round trip (ExportRenderer's ping-pong readback is skipped for this
    /// variant). Only `frame.pts` is used.
    ///
    /// For `FfmpegEncoder`, this is an exact delegate to `VideoEncoder::encode_frame`.
    pub fn encode_frame(
        &mut self,
        frame:       &RawFrame,
        packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
    ) -> Result<(), EncodeError> {
        match self {
            Self::FfmpegEncoder(enc) => enc.encode_frame(frame, packet_sink),

            Self::CudaNvenc { enc, .. } => {
                // Encode via NVENC directly.
                // The pipelined renderer loop calls nvenc_interop_mut() and passes the
                // correct pipeline slot explicitly. This arm is a fallback for any
                // non-pipelined callers; slot 0 is safe because `encode_frame`
                // reclaims the slot itself before reusing it.
                let packets = enc.encode_frame(frame.pts, 0)
                    .map_err(|e| EncodeError::Interop(format!("{:?}", e)))?;

                // An empty Vec means the encoder accepted the picture and is still
                // working — the bitstream arrives from a later call or from flush().
                for packet in &packets {
                    if packet.bytes.is_empty() {
                        continue;
                    }
                    write_nvenc_packet(packet, packet_sink)?;
                }
                Ok(())
            }
        }
    }

    /// Flush any buffered frames.
    ///
    /// For NVENC this submits the EOS picture and writes out every packet the
    /// encoder was still holding; skipping it silently truncates the tail of the
    /// export whenever the driver kept reordering enabled.
    pub fn flush(
        &mut self,
        packet_sink: &mut dyn FnMut(*mut crate::io::ffi::avutil::AVPacket),
    ) -> Result<(), EncodeError> {
        match self {
            Self::FfmpegEncoder(enc) => enc.flush(packet_sink),
            Self::CudaNvenc { enc, .. } => {
                let packets = enc
                    .flush()
                    .map_err(|e| EncodeError::Interop(format!("NVENC flush failed: {e:?}")))?;
                for packet in &packets {
                    if packet.bytes.is_empty() {
                        continue;
                    }
                    write_nvenc_packet(packet, packet_sink)?;
                }
                Ok(())
            }
        }
    }

    /// Returns a raw pointer to the `AVCodecContext` for use by `Muxer::open`
    /// when writing stream header parameters (codec_id, width, height, extradata).
    ///
    /// For `FfmpegEncoder` this is the live encoder context.
    /// For `CudaNvenc` this delegates to the `param_enc` minimal FFmpeg context
    /// that was opened alongside the NVENC session purely for this purpose.
    /// The muxer copies the parameters immediately via `avcodec_parameters_from_context`
    /// and never stores the raw pointer beyond `Muxer::open`.
    pub fn codec_ctx(&self) -> *const crate::io::ffi::avcodec::AVCodecContext {
        match self {
            Self::FfmpegEncoder(enc)              => enc.codec_ctx(),
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
