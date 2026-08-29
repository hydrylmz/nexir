// src/io/decoder.rs
// Safe wrapper around AVCodecContext. Provides decode_into() for the hot path.

use std::ptr;
use crate::io::ffi::avcodec::*;
use crate::io::ffi::avutil::{
    AVFrame, av_frame_alloc, av_frame_free, av_frame_unref,
    av_frame_get_pts, av_frame_get_data, av_frame_get_linesize,
    av_frame_get_format, av_frame_get_width, av_frame_get_height,
    av_frame_get_color_space, av_frame_get_color_range,
    av_frame_get_color_trc, av_frame_get_color_primaries,
    av_frame_set_color_space, av_frame_set_color_range,
    av_frame_set_color_trc, av_frame_set_color_primaries,
    av_pix_fmt_bit_depth,
    av_image_copy_to_buffer, av_image_fill_arrays, AVERROR_EOF, AVERROR_EAGAIN, AV_NOPTS_VALUE,
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE,
    frame_layout_for_pix_fmt,
    av_err_to_string,
};
use crate::timeline::source::{ColorInfo, DecodedFrameMeta, FrameLayout};
use crate::io::ffi::hw_accel::{HwDeviceType, probe_hardware_device, av_hwframe_transfer_data};
use crate::io::ffi::swscale::{sws_freeContext, sws_getContext, sws_scale, SWS_BILINEAR};
use crate::io::demuxer::{Demuxer, Packet, StreamInfo};

/// What one successful decode produced: presentation timestamp, the pixel layout
/// and colour metadata of the bytes now in the destination buffer, and the frame's
/// real dimensions.
///
/// P1.6 — this replaces the old `(i64, bool, u32, u32)` tuple whose `bool` meant
/// "is NV12".  That single bit could not express bit depth, code alignment or any
/// colour property, so the render path had to guess them from the container's
/// `VideoStreamInfo` — which describes the SOURCE, not necessarily what the
/// decoder emitted after a swscale conversion.
pub struct DecodedFrame {
    pub pts:    i64,
    pub meta:   DecodedFrameMeta,
    pub width:  u32,
    pub height: u32,
}

impl DecodedFrame {
    /// Whether the buffer holds semi-planar chroma (NV12 / P010).  Kept because
    /// the upload path still branches on it.
    pub fn is_semi_planar(&self) -> bool {
        self.meta.layout.semi_planar
    }
}


pub struct Decoder {
    ctx:       *mut AVCodecContext,
    frame:     *mut AVFrame,
    /// Transfer-target software frame (only used when hardware decode is active).
    sw_frame:  *mut AVFrame,
    hw_type:   HwDeviceType,
    /// True once the NULL end-of-stream packet has been sent, i.e. the decoder is
    /// in draining mode and will accept no further input until it is flushed.
    draining:  bool,
}

// SAFETY: Decoder owns all pointers exclusively; not shared across threads.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Get the raw AVCodecContext pointer.
    pub fn ctx(&self) -> *mut AVCodecContext {
        self.ctx
    }

    /// Flush the decoder (called on seek).
    ///
    /// Also clears the draining flag: after `avcodec_flush_buffers` the decoder
    /// accepts input again, so a later `drain_into` must re-send the EOS packet.
    pub fn flush(&mut self) {
        unsafe { avcodec_flush_buffers(self.ctx); }
        self.draining = false;
    }

    /// Open a decoder for the given stream.
    /// Set `enable_hw` to false for audio decoders (skips slow GPU device probing).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn open(
        stream_info:     &StreamInfo,
        stream_codecpar: *mut AVCodecParameters,
        enable_hw:       bool,
    ) -> Result<Self, DecodeError> {
        // SAFETY: `stream_codecpar` is sourced from the owning Demuxer stream
        // metadata and is copied into this decoder context during open.
        // Step 1 — Find the codec
        let codec = unsafe { avcodec_find_decoder(stream_info.codec_id) };
        if codec.is_null() {
            return Err(DecodeError::CodecNotFound(stream_info.codec_id));
        }

        // Step 2 — Allocate codec context
        let ctx = unsafe { avcodec_alloc_context3(codec) };
        if ctx.is_null() {
            return Err(DecodeError::Alloc);
        }

        // Step 3 — Copy codec parameters from stream
        let ret = unsafe { avcodec_parameters_to_context(ctx, stream_codecpar) };
        if ret < 0 {
            let mut ctx_ptr = ctx;
            unsafe { avcodec_free_context(&mut ctx_ptr); }
            return Err(DecodeError::Open(av_err_to_string(ret)));
        }

        // Step 4 — Probe and attach hardware device (best-effort; falls back to SW)
        // Skip for audio decoders — hardware video acceleration doesn't apply to audio.
        let hw_type = if enable_hw {
            match probe_hardware_device() {
                Ok(Some((hw_type, hw_ctx))) => {
                    unsafe {
                        avcodec_set_hw_device_ctx(ctx, hw_ctx);
                        avcodec_enable_hw_get_format(ctx);
                    }
                    log::info!("[decoder] Attached hardware decoder: {:?}", hw_type);
                    hw_type
                }
                Ok(None) => HwDeviceType::None,
                Err(e) => {
                    eprintln!("[decoder] hw probe failed: {:?} — using software decode", e);
                    HwDeviceType::None
                }
            }
        } else {
            HwDeviceType::None
        };

        // Step 5 — Set thread count for software decode
        if hw_type == HwDeviceType::None {
            unsafe { avcodec_set_thread_count(ctx, 0); }
        }

        // Step 6 — Open the codec
        let ret = unsafe { avcodec_open2(ctx, codec, ptr::null_mut()) };
        if ret < 0 {
            let mut ctx_ptr = ctx;
            unsafe { avcodec_free_context(&mut ctx_ptr); }
            return Err(DecodeError::Open(av_err_to_string(ret)));
        }

        // Step 7 — Allocate frame buffers
        let frame    = unsafe { av_frame_alloc() };
        let sw_frame = unsafe { av_frame_alloc() };
        if frame.is_null() || sw_frame.is_null() {
            let mut ctx_ptr = ctx;
            unsafe { avcodec_free_context(&mut ctx_ptr); }
            return Err(DecodeError::Alloc);
        }

        Ok(Decoder { ctx, frame, sw_frame, hw_type, draining: false })
    }

    /// Open a software-only decoder (no hardware acceleration).
    /// Use this for image codecs (MJPEG, PNG, etc.) that CUDA cannot handle.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn open_sw(
        stream_info:     &StreamInfo,
        stream_codecpar: *mut AVCodecParameters,
    ) -> Result<Self, DecodeError> {
        // SAFETY: `stream_codecpar` is sourced from the owning Demuxer stream
        // metadata and is copied into this decoder context during open.
        // Step 1 — Find the codec
        let codec = unsafe { avcodec_find_decoder(stream_info.codec_id) };
        if codec.is_null() {
            return Err(DecodeError::CodecNotFound(stream_info.codec_id));
        }

        // Step 2 — Allocate codec context
        let ctx = unsafe { avcodec_alloc_context3(codec) };
        if ctx.is_null() {
            return Err(DecodeError::Alloc);
        }

        // Step 3 — Copy codec parameters from stream
        let ret = unsafe { avcodec_parameters_to_context(ctx, stream_codecpar) };
        if ret < 0 {
            let mut ctx_ptr = ctx;
            unsafe { avcodec_free_context(&mut ctx_ptr); }
            return Err(DecodeError::Open(av_err_to_string(ret)));
        }

        // Step 4 — Software decode: set thread count and skip hardware
        unsafe { avcodec_set_thread_count(ctx, 0); }

        // Step 5 — Open the codec
        let ret = unsafe { avcodec_open2(ctx, codec, ptr::null_mut()) };
        if ret < 0 {
            let mut ctx_ptr = ctx;
            unsafe { avcodec_free_context(&mut ctx_ptr); }
            return Err(DecodeError::Open(av_err_to_string(ret)));
        }

        // Step 6 — Allocate frame buffers
        let frame    = unsafe { av_frame_alloc() };
        let sw_frame = unsafe { av_frame_alloc() };
        if frame.is_null() || sw_frame.is_null() {
            let mut ctx_ptr = ctx;
            unsafe { avcodec_free_context(&mut ctx_ptr); }
            return Err(DecodeError::Alloc);
        }

        Ok(Decoder { ctx, frame, sw_frame, hw_type: HwDeviceType::None, draining: false })
    }

    /// Decode one frame and write its pixel data into `dst` (a mapped staging buffer slice).
    ///
    /// Phase 7 update: accepts an optional `interop` parameter. When provided and the
    /// active hardware type is CUDA, this function routes through `DecodeInteropTarget`
    /// (GPU-to-GPU copy via cuMemcpy2DAsync) and does NOT populate `dst` at all —
    /// the DecodeInteropTarget's textures ARE the final Y/UV textures. The calling
    /// IoLayer must skip the map_write/copy_buffer_to_texture steps in that case.
    ///
    /// When `interop` is None (or CUDA interop is not available), behaviour is
    /// byte-for-byte identical to the original Phase 4 implementation.
    ///
    /// Returns `Ok(Some(frame))` on success — carrying the PTS plus the pixel
    /// layout and colour metadata of what was actually written — `Ok(None)` if the
    /// decoder needs more input, or `Err` on a hard failure.
    pub fn decode_into(
        &mut self,
        packet: &Packet,
        dst:    &mut [u8],
        // None → CPU path (Phase 4 behaviour unchanged).
        // Some((cuda_ctx, target, capability)) → GPU-to-GPU interop path (Phase 7).
        interop: Option<(
            &crate::interop::cuda_context::CudaContext,
            &crate::interop::decode_interop::DecodeInteropTarget,
            &crate::interop::capability::InteropCapability,
        )>,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        // Step 1 & 2 — Send packet and receive frame
        let mut got_frame = false;
        loop {
            // Try sending the packet
            let send_ret = unsafe { avcodec_send_packet(self.ctx, packet.as_ptr()) };
            if send_ret < 0 && send_ret != AVERROR_EAGAIN {
                return Err(DecodeError::Send(av_err_to_string(send_ret)));
            }

            // Try receiving a frame
            let recv_ret = unsafe { avcodec_receive_frame(self.ctx, self.frame) };
            if recv_ret == 0 {
                // Successfully got a frame!
                got_frame = true;
                // If send_ret was EAGAIN, it means the packet wasn't sent yet,
                // but since we freed up space by receiving a frame, the NEXT loop iteration will send it.
                if send_ret == 0 {
                    break;
                }
            } else if recv_ret == AVERROR_EAGAIN || recv_ret == AVERROR_EOF {
                if send_ret == 0 {
                    // Packet sent, but no frame ready yet
                    break;
                } else {
                    // Send was EAGAIN, Receive is EAGAIN (should never happen)
                    break;
                }
            } else {
                return Err(DecodeError::Receive(av_err_to_string(recv_ret)));
            }
        }

        if !got_frame {
            return Ok(None);
        }

        self.emit_frame(dst, interop)
    }

    /// Signal end-of-stream and pull one buffered frame out of the decoder.
    ///
    /// A decoder holds frames back for two reasons: B-frame reordering, and
    /// frame-level threading (`avcodec_set_thread_count(ctx, 0)` lets FFmpeg use
    /// one thread per core, each adding a frame of delay).  On this machine that
    /// is ~10 frames, so a file shorter than the delay decodes to NOTHING through
    /// `decode_into` alone — every call returns `Ok(None)`.
    ///
    /// Call this repeatedly after the demuxer reports EOF until it returns
    /// `Ok(None)`, which means the decoder is fully drained.  The first call sends
    /// the NULL packet that puts the decoder into draining mode; later calls only
    /// receive.
    ///
    /// `dst` has the same meaning as in [`Self::decode_into`].
    pub fn drain_into(
        &mut self,
        dst: &mut [u8],
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        if !self.draining {
            // NULL packet = "no more input".  Sending it twice is not an error but
            // there is no reason to, so the flag keeps it to once.
            let ret = unsafe { avcodec_send_packet(self.ctx, ptr::null()) };
            if ret < 0 && ret != AVERROR_EOF {
                return Err(DecodeError::Send(av_err_to_string(ret)));
            }
            self.draining = true;
        }

        let recv_ret = unsafe { avcodec_receive_frame(self.ctx, self.frame) };
        if recv_ret == AVERROR_EOF || recv_ret == AVERROR_EAGAIN {
            return Ok(None); // fully drained
        }
        if recv_ret < 0 {
            return Err(DecodeError::Receive(av_err_to_string(recv_ret)));
        }

        // Draining never goes through the CUDA interop path: `interop = None`
        // routes it down the CPU copy, which is what a caller reading a finished
        // file wants.
        self.emit_frame(dst, None)
    }

    /// Turn the frame currently held in `self.frame` into pixels in `dst`.
    ///
    /// Shared by [`Self::decode_into`] and [`Self::drain_into`] so both paths
    /// perform the same hardware transfer, format normalisation and PTS handling.
    /// Always unrefs the frame(s) before returning, including on error.
    ///
    /// P1.6 — this is where colour metadata enters the pipeline.  It is read from
    /// the AVFrame (not the container) after any hardware transfer, and the layout
    /// it reports describes what was actually written to `dst`: if the frame had to
    /// go through swscale, the reported layout is the CONVERSION TARGET, because
    /// that is what the buffer holds.
    fn emit_frame(
        &mut self,
        dst: &mut [u8],
        interop: Option<(
            &crate::interop::cuda_context::CudaContext,
            &crate::interop::decode_interop::DecodeInteropTarget,
            &crate::interop::capability::InteropCapability,
        )>,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        // Step 3 — Phase 7 branch: if CUDA interop is available and hw_type is Cuda,
        // copy NVDEC's device pointer directly into the interop target's CUDA arrays
        // and return WITHOUT touching `dst` (the CPU staging buffer is skipped entirely).
        if self.hw_type == HwDeviceType::Cuda {
            if let Some((cuda_ctx, target, capability)) = interop {
                if capability.is_available() {
                    let w = unsafe { av_frame_get_width(self.frame) } as u32;
                    let h = unsafe { av_frame_get_height(self.frame) } as u32;
                    let pts_raw = unsafe { av_frame_get_pts(self.frame) };
                    let pts = if pts_raw == AV_NOPTS_VALUE { 0i64 } else { pts_raw };
                    // The hardware surface's own pixel format is CUDA/opaque, so its
                    // real layout comes from what NVDEC produces: NV12 for 8-bit,
                    // P010 for 10-bit.  Colour is read from the frame, which carries
                    // it even for hardware surfaces; the depth it reports is what
                    // decides between the two layouts, so read colour once at the
                    // source's depth and then re-read pinned to the layout's.
                    let probe_depth = Self::pix_fmt_bit_depth(unsafe {
                        av_frame_get_format(self.frame)
                    });
                    let probe = unsafe {
                        Self::read_frame_color(self.frame, probe_depth.max(8), w, h)
                    };
                    let layout = if probe.bit_depth >= 10 || probe.is_hdr() {
                        FrameLayout::P010
                    } else {
                        FrameLayout::NV12
                    };
                    let mut color = unsafe {
                        Self::read_frame_color(self.frame, layout.bit_depth, w, h)
                    };
                    color.bit_depth = layout.bit_depth;

                    target.copy_from_nvdec_frame(cuda_ctx, self.frame, w, h)
                        .map_err(|e| DecodeError::HwTransfer(format!("{:?}", e)))?;

                    unsafe {
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                    }
                    return Ok(Some(DecodedFrame {
                        pts,
                        meta: DecodedFrameMeta { layout, color },
                        width: w,
                        height: h,
                    }));
                }
            }
        }

        // Step 3 (CPU path) — Hardware frame transfer via av_hwframe_transfer_data
        let src_frame = if self.hw_type != HwDeviceType::None {
            let ret = unsafe {
                av_hwframe_transfer_data(self.sw_frame, self.frame, 0)
            };
            if ret < 0 {
                unsafe {
                    av_frame_unref(self.frame);
                    av_frame_unref(self.sw_frame);
                }
                return Err(DecodeError::HwTransfer(av_err_to_string(ret)));
            }
            // av_hwframe_transfer_data copies pixels but NOT the colour properties,
            // so carry them over from the hardware frame before it is unref'd.
            unsafe { Self::copy_color_props(self.frame, self.sw_frame) };
            self.sw_frame
        } else {
            self.frame
        };

        // Step 4 — Copy pixel data into the staging buffer.
        //
        // The upload path can consume YUV420P, NV12, P010 and planar 10/12-bit
        // directly; `frame_layout_for_pix_fmt` is the authority on which those are.
        // Anything else (RGB stills, 4:2:2/4:4:4, exotic depths) is converted with
        // swscale, and the layout reported back is the conversion TARGET rather than
        // the source format — the buffer holds the target's bytes.
        //
        // The target depth follows the source: converting 10-bit input down to 8-bit
        // here would throw away exactly the precision an HDR export needs.
        let out_layout;
        unsafe {
            let data = av_frame_get_data(src_frame);
            let linesize = av_frame_get_linesize(src_frame);
            let fmt = av_frame_get_format(src_frame);
            let w = av_frame_get_width(src_frame);
            let h = av_frame_get_height(src_frame);

            match frame_layout_for_pix_fmt(fmt) {
                // Direct copy: the decoder's format is one the GPU path understands.
                Some(layout) => {
                    out_layout = layout;
                    let ret = av_image_copy_to_buffer(
                        dst.as_mut_ptr(),
                        dst.len() as i32,
                        data as *const *const u8,
                        linesize,
                        fmt,
                        w,
                        h,
                        1, // align=1: exact row widths, no extra padding
                    );
                    if ret < 0 {
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                        return Err(DecodeError::Receive(av_err_to_string(ret)));
                    }
                }
                // Conversion needed.  Pick the target from the source's depth so
                // high-bit-depth sources stay high-bit-depth.
                None => {
                    let src_depth = Self::pix_fmt_bit_depth(fmt);
                    let (dst_fmt, dst_layout) = if src_depth > 8 {
                        (AV_PIX_FMT_YUV420P10LE, FrameLayout::YUV420P10)
                    } else {
                        (AV_PIX_FMT_YUV420P, FrameLayout::YUV420P8)
                    };
                    out_layout = dst_layout;

                    let sws = sws_getContext(
                        w,
                        h,
                        fmt,
                        w,
                        h,
                        dst_fmt,
                        SWS_BILINEAR,
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                    );
                    if sws.is_null() {
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                        return Err(DecodeError::Convert("sws_getContext failed".into()));
                    }

                    let mut dst_data = [std::ptr::null_mut::<u8>(); 8];
                    let mut dst_linesize = [0i32; 8];
                    let fill_ret = av_image_fill_arrays(
                        dst_data.as_mut_ptr(),
                        dst_linesize.as_mut_ptr(),
                        dst.as_mut_ptr(),
                        dst_fmt,
                        w,
                        h,
                        1,
                    );
                    if fill_ret < 0 || fill_ret as usize > dst.len() {
                        sws_freeContext(sws);
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                        return Err(DecodeError::Receive(av_err_to_string(fill_ret)));
                    }

                    let scaled = sws_scale(
                        sws,
                        data as *const *const u8,
                        linesize,
                        0,
                        h,
                        dst_data.as_ptr(),
                        dst_linesize.as_ptr(),
                    );
                    sws_freeContext(sws);
                    if scaled <= 0 {
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                        return Err(DecodeError::Convert(format!("sws_scale returned {scaled}")));
                    }
                }
            }
        }

        // Step 5 — Read colour metadata and PTS, then clean up frame references.
        let (pts, actual_w, actual_h, color) = unsafe {
            let raw_pts = av_frame_get_pts(src_frame);
            let pts = if raw_pts == AV_NOPTS_VALUE { 0i64 } else { raw_pts };
            let aw = av_frame_get_width(src_frame) as u32;
            let ah = av_frame_get_height(src_frame) as u32;
            // Colour comes from the frame, with the depth of the buffer we just
            // wrote — not the source's — so range maths downstream matches the data.
            let color = Self::read_frame_color(src_frame, out_layout.bit_depth, aw, ah);
            av_frame_unref(self.frame);
            if self.hw_type != HwDeviceType::None {
                av_frame_unref(self.sw_frame);
            }
            (pts, aw, ah, color)
        };

        // A source that signalled a higher depth than the buffer can hold would make
        // the shader's range maths disagree with the data; `out_layout` is
        // authoritative, so pin the ColorInfo depth to it.
        let mut color = color;
        color.bit_depth = out_layout.bit_depth;

        Ok(Some(DecodedFrame {
            pts,
            meta: DecodedFrameMeta { layout: out_layout, color },
            width: actual_w,
            height: actual_h,
        }))
    }

    /// Read an AVFrame's colour properties into a `ColorInfo`, resolving anything
    /// the encoder left unspecified through `ColorInfo::from_ffmpeg`'s heuristics.
    ///
    /// `bit_depth` is the depth of the pixels the caller is about to hand
    /// downstream, which is not always the source's own depth (see `emit_frame`).
    ///
    /// # Safety
    /// `frame` must be a valid, currently-referenced AVFrame.
    unsafe fn read_frame_color(
        frame: *const AVFrame,
        bit_depth: u8,
        width: u32,
        height: u32,
    ) -> ColorInfo {
        let matrix_raw = av_frame_get_color_space(frame);
        let range_raw  = av_frame_get_color_range(frame);
        let trc_raw    = av_frame_get_color_trc(frame);
        let pri_raw    = av_frame_get_color_primaries(frame);
        ColorInfo::from_ffmpeg(matrix_raw, range_raw, trc_raw, pri_raw, bit_depth, width, height)
    }

    /// Copy colour properties from `src` to `dst`.
    ///
    /// `av_hwframe_transfer_data` moves pixels only, leaving the destination's
    /// colour fields at their defaults (all "unspecified"), which would send every
    /// hardware-decoded HDR frame down the SDR path.
    ///
    /// # Safety
    /// Both pointers must be valid AVFrames.
    unsafe fn copy_color_props(src: *const AVFrame, dst: *mut AVFrame) {
        av_frame_set_color_space(dst, av_frame_get_color_space(src));
        av_frame_set_color_range(dst, av_frame_get_color_range(src));
        av_frame_set_color_trc(dst, av_frame_get_color_trc(src));
        av_frame_set_color_primaries(dst, av_frame_get_color_primaries(src));
    }

    /// Bits per component of an AVPixelFormat, for choosing a conversion target.
    ///
    /// Uses FFmpeg's own descriptor table rather than a local list, so a format
    /// this code has never heard of still gets the right depth.
    fn pix_fmt_bit_depth(fmt: i32) -> u8 {
        let depth = unsafe { av_pix_fmt_bit_depth(fmt) };
        if depth <= 0 { 8 } else { depth as u8 }
    }

    /// Seek decoder to a target PTS by flushing then running the discard loop.
    ///
    /// Returns the PTS of the first frame that is >= `target_stream_pts`.
    pub fn seek_to(
        &mut self,
        demuxer:           &mut Demuxer,
        target_stream_pts: i64,
    ) -> Result<i64, DecodeError> {
        // Step 1 — Flush the decoder's internal buffer
        unsafe { avcodec_flush_buffers(self.ctx); }

        // Step 2 — Discard loop: decode and throw away until target PTS
        loop {
            let pkt = demuxer.next_video_packet()
                .map_err(|_| DecodeError::NoFrame)?;
            let pkt = match pkt {
                Some(p) => p,
                None    => return Err(DecodeError::NoFrame), // EOF before target
            };

            let ret = unsafe { avcodec_send_packet(self.ctx, pkt.as_ptr()) };
            if ret < 0 && ret != AVERROR_EAGAIN {
                continue; // skip bad packets during seek
            }

            loop {
                let ret2 = unsafe { avcodec_receive_frame(self.ctx, self.frame) };
                if ret2 == AVERROR_EAGAIN || ret2 == AVERROR_EOF {
                    break;
                }
                if ret2 < 0 {
                    break;
                }

                let frame_pts = unsafe { av_frame_get_pts(self.frame) };
                unsafe { av_frame_unref(self.frame); }

                let frame_pts = if frame_pts == AV_NOPTS_VALUE { 0i64 } else { frame_pts };
                if frame_pts >= target_stream_pts {
                    return Ok(frame_pts);
                }
            }
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            // Free frames before the context they reference
            if !self.frame.is_null()    { av_frame_free(&mut self.frame); }
            if !self.sw_frame.is_null() { av_frame_free(&mut self.sw_frame); }
            if !self.ctx.is_null()      { avcodec_free_context(&mut self.ctx); }
        }
    }
}

#[derive(Debug)]
pub enum DecodeError {
    CodecNotFound(u32),
    Alloc,
    Open(String),
    Send(String),
    Receive(String),
    HwTransfer(String),
    Convert(String),
    NoFrame,
}
