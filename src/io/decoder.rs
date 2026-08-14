// src/io/decoder.rs
// Safe wrapper around AVCodecContext. Provides decode_into() for the hot path.

use std::ptr;
use crate::io::ffi::avcodec::*;
use crate::io::ffi::avutil::{
    AVFrame, av_frame_alloc, av_frame_free, av_frame_unref,
    av_frame_get_pts, av_frame_get_data, av_frame_get_linesize,
    av_frame_get_format, av_frame_get_width, av_frame_get_height,
    av_image_copy_to_buffer, av_image_fill_arrays, AVERROR_EOF, AVERROR_EAGAIN, AV_NOPTS_VALUE,
    AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P,
    av_err_to_string,
};
use crate::io::ffi::hw_accel::{HwDeviceType, probe_hardware_device, av_hwframe_transfer_data};
use crate::io::ffi::swscale::{sws_freeContext, sws_getContext, sws_scale, SWS_BILINEAR};
use crate::io::demuxer::{Demuxer, Packet, StreamInfo};

pub struct Decoder {
    ctx:       *mut AVCodecContext,
    frame:     *mut AVFrame,
    /// Transfer-target software frame (only used when hardware decode is active).
    sw_frame:  *mut AVFrame,
    hw_type:   HwDeviceType,
}

// SAFETY: Decoder owns all pointers exclusively; not shared across threads.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Get the raw AVCodecContext pointer.
    pub fn ctx(&self) -> *mut AVCodecContext {
        self.ctx
    }

    /// Flush the decoder (called on seek).
    pub fn flush(&mut self) {
        unsafe { avcodec_flush_buffers(self.ctx); }
    }

    /// Open a decoder for the given stream.
    /// Set `enable_hw` to false for audio decoders (skips slow GPU device probing).
    pub fn open(
        stream_info:     &StreamInfo,
        stream_codecpar: *mut AVCodecParameters,
        enable_hw:       bool,
    ) -> Result<Self, DecodeError> {
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

        Ok(Decoder { ctx, frame, sw_frame, hw_type })
    }

    /// Open a software-only decoder (no hardware acceleration).
    /// Use this for image codecs (MJPEG, PNG, etc.) that CUDA cannot handle.
    pub fn open_sw(
        stream_info:     &StreamInfo,
        stream_codecpar: *mut AVCodecParameters,
    ) -> Result<Self, DecodeError> {
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

        Ok(Decoder { ctx, frame, sw_frame, hw_type: HwDeviceType::None })
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
    /// Returns `Ok(Some(pts))` on success, `Ok(None)` if the decoder needs more input,
    /// or `Err` on a hard failure.
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
    ) -> Result<Option<(i64, bool, u32, u32)>, DecodeError> {
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

        // Check if format is NV12 before phase 7 branch
        let is_nv12 = unsafe {
            let fmt = av_frame_get_format(self.frame);
            fmt == 23 || fmt == 24 || self.hw_type != HwDeviceType::None
        };

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

                    target.copy_from_nvdec_frame(cuda_ctx, self.frame, w, h)
                        .map_err(|e| DecodeError::HwTransfer(format!("{:?}", e)))?;

                    unsafe {
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                    }
                    return Ok(Some((pts, is_nv12, w, h)));
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
            self.sw_frame
        } else {
            self.frame
        };

        // Step 4 — Copy pixel data into staging buffer. The render upload path
        // accepts YUV420p and NV12, so normalize RGB/RGBA image codecs here.
        let mut output_is_nv12 = is_nv12;
        unsafe {
            let data = av_frame_get_data(src_frame);
            let linesize = av_frame_get_linesize(src_frame);
            let fmt = av_frame_get_format(src_frame);
            let w = av_frame_get_width(src_frame);
            let h = av_frame_get_height(src_frame);

            if fmt == AV_PIX_FMT_YUV420P || fmt == AV_PIX_FMT_NV12 {
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
            } else {
                let sws = sws_getContext(
                    w,
                    h,
                    fmt,
                    w,
                    h,
                    AV_PIX_FMT_YUV420P,
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
                    AV_PIX_FMT_YUV420P,
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
                output_is_nv12 = false;
            }
        }

        // Step 5 — Extract PTS and clean up frame references
        let (pts, actual_w, actual_h) = unsafe {
            let raw_pts = av_frame_get_pts(src_frame);
            let pts = if raw_pts == AV_NOPTS_VALUE { 0i64 } else { raw_pts };
            let aw = av_frame_get_width(src_frame) as u32;
            let ah = av_frame_get_height(src_frame) as u32;
            av_frame_unref(self.frame);
            if self.hw_type != HwDeviceType::None {
                av_frame_unref(self.sw_frame);
            }
            (pts, aw, ah)
        };

        Ok(Some((pts, output_is_nv12, actual_w, actual_h)))
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
