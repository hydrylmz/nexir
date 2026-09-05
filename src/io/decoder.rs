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
    av_pix_fmt_is_known,
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
    /// Packets `avcodec_send_packet` refused with EAGAIN, oldest first.
    ///
    /// P2.3 — a refused packet must not be dropped: every caller treats
    /// `Ok(None)`/`Ok(Some(..))` as "I am done with that packet, here is the next
    /// one", so a packet that never got in is gone. For an intra-only codec that
    /// loses a frame; for AV1 it breaks the OBU sequence and the following send
    /// fails with "Invalid data found when processing input".
    ///
    /// A QUEUE rather than one slot, because the re-send can itself be refused —
    /// measured on a 2 s AV1 fixture, where packet 13's send AND its retry both
    /// returned EAGAIN while packet 14 was already being offered. With a single
    /// slot one of the two is always lost.
    ///
    /// `Packet` owns its `AVPacket` (`av_packet_ref`), so holding these across
    /// calls is sound; the buffer is shared, not copied.
    pending:   std::collections::VecDeque<Packet>,
}

// SAFETY: Decoder owns all pointers exclusively; not shared across threads.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Get the raw AVCodecContext pointer.
    pub fn ctx(&self) -> *mut AVCodecContext {
        self.ctx
    }

    /// Which hardware decoder, if any, this decoder actually attached.
    ///
    /// G2c reads this rather than assuming: `Decoder::open` *probes* for hardware
    /// and silently falls back to software, and `open_sw` never probes at all. The
    /// interop decode path is only reachable for `Cuda`, so a caller that allocated
    /// a `DecodeInteropTarget` for a software decoder would hold 12.4 MB of VRAM
    /// per source that nothing ever writes into — and every frame would come back
    /// from the CPU path with the graph still bound to an empty texture.
    pub fn hw_type(&self) -> HwDeviceType {
        self.hw_type
    }

    /// Flush the decoder (called on seek).
    ///
    /// Also clears the draining flag: after `avcodec_flush_buffers` the decoder
    /// accepts input again, so a later `drain_into` must re-send the EOS packet.
    pub fn flush(&mut self) {
        unsafe { avcodec_flush_buffers(self.ctx); }
        self.draining = false;
        // Any held packets belong to the position being seeked away from, so they
        // must go — re-sending them after a flush would decode frames from before
        // the seek and hand them back as if they were at the new position.
        self.pending.clear();
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

        Ok(Decoder {
            ctx,
            frame,
            sw_frame,
            hw_type,
            draining: false,
            pending: std::collections::VecDeque::new(),
        })
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

        Ok(Decoder {
            ctx,
            frame,
            sw_frame,
            hw_type: HwDeviceType::None,
            draining: false,
            pending: std::collections::VecDeque::new(),
        })
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
        // Feed the decoder, then take at most ONE frame back out.
        //
        // P2.3 — two coupled bugs lived here, and the second is only reachable
        // once the first is fixed. Both are now contained by the two halves this
        // delegates to, and the reasoning has to stay because the shape of those
        // halves is the fix:
        //
        // 1. The old loop latched a `got_frame` flag and kept iterating. With
        //    `avcodec_send_packet` returning EAGAIN (input queue full, which
        //    frame-threading reaches within a few packets) it would receive a
        //    frame, set the flag, iterate, re-send, and call
        //    `avcodec_receive_frame` AGAIN. That second receive returns EAGAIN —
        //    and receive **unrefs its destination frame before doing anything
        //    else** — so the frame just obtained was wiped while the flag still
        //    claimed it. `emit_frame` read `format` off an empty AVFrame, got
        //    AV_PIX_FMT_NONE (-1), missed `frame_layout_for_pix_fmt`, and handed
        //    -1 to `sws_getContext`, which aborts the process inside libswscale
        //    (`Assertion desc failed at swscale_internal.h:778`,
        //    STATUS_STACK_BUFFER_OVERRUN). Every codec can reach it; AV1 got there
        //    first because libdav1d fills its input queue fastest. Hence
        //    [`Self::receive_into`] receives EXACTLY ONCE, and `emit_frame` reads
        //    `self.frame` immediately afterwards.
        // 2. Returning without keeping the refused packet DROPS it, because the
        //    caller moves on to the next one. For AV1 that is a gap in the OBU
        //    sequence and the next send fails with "Invalid data found when
        //    processing input". Hence [`Self::send_packet_only`]'s backlog, and its
        //    front/back ordering.
        self.send_packet_only(packet)?;
        self.receive_into(dst, interop)
    }

    /// Offer one packet to the decoder and return WITHOUT receiving.
    ///
    /// Split out of [`Self::decode_into`], which is now this plus
    /// [`Self::receive_into`] — the two halves in the same order, so the ordinary
    /// caller's behaviour is unchanged.
    ///
    /// **Why a caller would want the halves apart: input queue depth.** The
    /// one-packet-one-receive pattern gives NVDEC an input queue of exactly one, so
    /// every `avcodec_receive_frame` waits out the hardware's whole latency for the
    /// picture it is asking for, and nothing is decoding while the caller copies the
    /// result. Measured on `examples/b_decode_split_probe.rs`, one 4K60 source, 90
    /// frames: the per-frame decode series alternates 0.6 ms / 5-15 ms with the
    /// cheap frames being the receives that found a picture already finished. A
    /// caller that sends several packets before receiving keeps the hardware fed.
    ///
    /// **The caller MUST bound how far it runs ahead.** A send the decoder refuses
    /// with EAGAIN is queued in `self.pending` and retried by the next send, so
    /// sending without ever receiving grows that queue without limit — one
    /// `av_packet_ref` per packet, which is a reference to the demuxer's buffer
    /// rather than a copy, but unbounded all the same.
    ///
    /// The packet order this preserves is the same one [`Self::decode_into`]
    /// preserves and for the same reason (P2.3): a refused packet goes to the FRONT
    /// of the backlog and the new one behind it, because for AV1 a gap in the OBU
    /// sequence makes the following send fail outright.
    pub fn send_packet_only(&mut self, packet: &Packet) -> Result<(), DecodeError> {
        while let Some(front) = self.pending.pop_front() {
            let ret = unsafe { avcodec_send_packet(self.ctx, front.as_ptr()) };
            if ret == AVERROR_EAGAIN {
                // Still no room. Put it back at the FRONT to preserve order, and
                // queue the caller's packet behind it rather than dropping it.
                self.pending.push_front(front);
                self.pending.push_back(packet.clone_ref());
                return Ok(());
            }
            if ret < 0 {
                return Err(DecodeError::Send(av_err_to_string(ret)));
            }
        }

        let send_ret = unsafe { avcodec_send_packet(self.ctx, packet.as_ptr()) };
        if send_ret == AVERROR_EAGAIN {
            self.pending.push_back(packet.clone_ref());
        } else if send_ret < 0 {
            return Err(DecodeError::Send(av_err_to_string(send_ret)));
        }
        Ok(())
    }

    /// Take at most one frame out of the decoder, without offering a packet first.
    ///
    /// The other half of [`Self::decode_into`] — see [`Self::send_packet_only`] for
    /// why the two are separable. `Ok(None)` means the decoder has nothing ready and
    /// wants more input; it is not end-of-stream (that is [`Self::drain_into`]).
    pub fn receive_into(
        &mut self,
        dst: &mut [u8],
        interop: Option<(
            &crate::interop::cuda_context::CudaContext,
            &crate::interop::decode_interop::DecodeInteropTarget,
            &crate::interop::capability::InteropCapability,
        )>,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        self.receive_one(dst, interop)
    }

    /// Take at most one frame out of the decoder and emit it.
    ///
    /// Split out of [`Self::decode_into`] so the receive happens exactly once per
    /// call: the frame `avcodec_receive_frame` fills must be read by `emit_frame`
    /// before anything can call receive again, because receive unrefs it first.
    fn receive_one(
        &mut self,
        dst: &mut [u8],
        interop: Option<(
            &crate::interop::cuda_context::CudaContext,
            &crate::interop::decode_interop::DecodeInteropTarget,
            &crate::interop::capability::InteropCapability,
        )>,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        let recv_ret = unsafe { avcodec_receive_frame(self.ctx, self.frame) };
        if recv_ret == 0 {
            return self.emit_frame(dst, interop);
        }
        if recv_ret == AVERROR_EAGAIN || recv_ret == AVERROR_EOF {
            return Ok(None);
        }
        Err(DecodeError::Receive(av_err_to_string(recv_ret)))
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
        // Any packets still held from an EAGAIN must go in BEFORE the EOS packet,
        // or their frames are lost: `avcodec_send_packet(NULL)` closes the input,
        // so a backlog offered afterwards is refused outright.
        //
        // P2.3 — measured: without this the AV1 fixture decodes 55 of 60 frames
        // even with the backlog queue in place, because the queue still held
        // packets when the demuxer hit EOF and `drain_into` sent EOS past them.
        // The five missing frames are exactly the queue's depth at that moment.
        //
        // A loop rather than a single send because the decoder may still be full
        // here; each `receive_one` takes one frame out and makes room, and the
        // caller calls again until `Ok(None)`.
        while let Some(front) = self.pending.pop_front() {
            let ret = unsafe { avcodec_send_packet(self.ctx, front.as_ptr()) };
            if ret == AVERROR_EAGAIN {
                self.pending.push_front(front);
                // Hand back the frame that frees the slot; the backlog is retried
                // on the next call.
                return self.receive_one(dst, None);
            }
            if ret < 0 && ret != AVERROR_EOF {
                return Err(DecodeError::Send(av_err_to_string(ret)));
            }
        }

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
                    // P2.3 — refuse a format libavutil does not describe BEFORE
                    // handing it to swscale.  `sws_getContext` does not validate
                    // its arguments: it calls `av_pix_fmt_desc_get(fmt)` and
                    // dereferences the result, so an unknown format trips
                    // `Assertion desc failed at libswscale/swscale_internal.h:778`
                    // and takes the process down with STATUS_STACK_BUFFER_OVERRUN
                    // rather than returning NULL.  An unreadable input file has to
                    // be an `Err` the caller can report, not a crash that loses
                    // the user's session.
                    if av_pix_fmt_is_known(fmt) == 0 {
                        av_frame_unref(self.frame);
                        av_frame_unref(self.sw_frame);
                        return Err(DecodeError::Convert(format!(
                            "the decoder emitted pixel format {fmt}, which this \
                             build of libavutil does not describe — it cannot be \
                             converted, and passing it to swscale would abort the \
                             process"
                        )));
                    }
                    let src_depth = Self::pix_fmt_bit_depth(fmt);
                    log::debug!(
                        "[decoder] converting pix_fmt {fmt} ({src_depth}-bit, \
                         {w}x{h}) — not a layout the upload path takes directly"
                    );
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
            // PTS COMES FROM `self.frame`, NOT FROM `src_frame`, AND THAT IS THE
            // WHOLE POINT OF THIS LINE.
            //
            // `av_hwframe_transfer_data` copies PIXELS ONLY — the same reason
            // `copy_color_props` exists two functions down. It sets `format`,
            // `width` and `height` on the destination and touches nothing else, so
            // `sw_frame.pts` is whatever it was initialised to: **zero, on every
            // frame of every hardware decode.** Measured with
            // `examples/g3_drain_probe.rs` on a 60-frame H.264 file with NVDEC
            // attached: `decode_into` produced 58 frames whose pts were
            // `[0, 0, 0, 0, 0, 0]`, and the two drained frames were `[0, 0]` too.
            //
            // The read-forward loops in `IoLayer` hid it: both have an
            // `if frame.pts == 0 { pkt_pts }` fallback, so a decoder that timestamps
            // nothing is indistinguishable from one that does — until something asks
            // for a frame with NO packet beside it. That is exactly the G3 drain:
            // `drain_into` compares `frame.pts >= target_stream_pts`, which with a
            // zeroed pts is `0 >= target`, so every drained frame was discarded and
            // the last frames of every clip stayed unreachable *after* the drain
            // landed. `playback_smoke::decode_30_frames_monotonic_pts` asserts
            // strictly increasing pts on this same path and would have caught it, but
            // it needs `VE_TEST_FILE` and does not run by default.
            //
            // `self.frame` is what `avcodec_receive_frame` filled, so it carries the
            // decoder's own timestamp on both paths — and on the software path the
            // two pointers are the same frame, so this is a no-op there.
            let raw_pts = av_frame_get_pts(self.frame);
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
        // Step 1 — Flush the decoder's internal buffer.
        //
        // Via `self.flush()` rather than `avcodec_flush_buffers` directly, so the
        // held EAGAIN packet and the draining flag are cleared too: a packet from
        // before the seek must not be re-sent afterwards.
        self.flush();

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
