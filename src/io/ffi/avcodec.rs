// src/io/ffi/avcodec.rs
// Raw FFI bindings for libavcodec — find, open, send/receive.

use super::avutil::{AVFrame, AVPacket, AVDictionary};

#[repr(C)] pub struct AVCodec           { _opaque: [u8; 0] }
#[repr(C)] pub struct AVCodecContext    { _opaque: [u8; 0] }
#[repr(C)] pub struct AVCodecParameters { _opaque: [u8; 0] }

#[link(name = "avcodec")]
unsafe extern "C" {

    /// Find a decoder by codec ID. Returns NULL if not available.
    /// The returned pointer is static and must NOT be freed.
    pub fn avcodec_find_decoder(id: std::ffi::c_uint) -> *const AVCodec;

    /// Allocate a codec context for the given codec. Returns NULL on allocation failure.
    pub fn avcodec_alloc_context3(codec: *const AVCodec) -> *mut AVCodecContext;

    /// Free a codec context. Sets `*avctx = NULL`.
    pub fn avcodec_free_context(avctx: *mut *mut AVCodecContext);

    /// Copy codec parameters from a stream into a codec context.
    /// Must be called before `avcodec_open2`.
    pub fn avcodec_parameters_to_context(
        codec: *mut AVCodecContext,
        par:   *const AVCodecParameters,
    ) -> std::ffi::c_int;

    /// Open a codec context for decoding. Call after `avcodec_parameters_to_context`.
    pub fn avcodec_open2(
        avctx:   *mut AVCodecContext,
        codec:   *const AVCodec,
        options: *mut *mut AVDictionary,
    ) -> std::ffi::c_int;

    /// Send a compressed packet to the decoder.
    ///
    /// Returns 0: success. AVERROR_EAGAIN: decoder needs output drained first.
    /// Pass NULL to flush (end of stream).
    pub fn avcodec_send_packet(
        avctx: *mut AVCodecContext,
        avpkt: *const AVPacket,
    ) -> std::ffi::c_int;

    /// Receive a decoded frame from the decoder.
    ///
    /// Returns 0: success. AVERROR_EAGAIN: need more packets. AVERROR_EOF: fully flushed.
    pub fn avcodec_receive_frame(
        avctx: *mut AVCodecContext,
        frame: *mut AVFrame,
    ) -> std::ffi::c_int;

    /// Flush all buffered frames. Call after seeking.
    pub fn avcodec_flush_buffers(avctx: *mut AVCodecContext);
}

extern "C" {
    /// Get the codec parameters from an AVStream.
    pub fn avstream_get_codecpar(
        stream: *const super::avformat::AVStream,
    ) -> *mut AVCodecParameters;

    /// Get the codec_id from codec parameters.
    pub fn avcodecpar_get_codec_id(par: *const AVCodecParameters) -> std::ffi::c_uint;

    /// Get width from codec parameters.
    pub fn avcodecpar_get_width(par: *const AVCodecParameters) -> std::ffi::c_int;

    /// Get height from codec parameters.
    pub fn avcodecpar_get_height(par: *const AVCodecParameters) -> std::ffi::c_int;

    /// Set the hardware device context before `avcodec_open2`.
    pub fn avcodec_set_hw_device_ctx(
        avctx:      *mut AVCodecContext,
        device_ctx: *mut super::avutil::AVBufferRef,
    );

    /// Set decoder thread count. 0 = let FFmpeg choose (= num_cpus).
    pub fn avcodec_set_thread_count(avctx: *mut AVCodecContext, count: std::ffi::c_int);
    pub fn avcodec_enable_hw_get_format(avctx: *mut AVCodecContext);
    pub fn avcodec_ctx_get_sample_rate(ctx: *const AVCodecContext) -> std::ffi::c_int;
    pub fn avcodec_ctx_get_channels(ctx: *const AVCodecContext) -> std::ffi::c_int;
    pub fn avcodec_ctx_get_sample_fmt(ctx: *const AVCodecContext) -> std::ffi::c_int;
    pub fn avcodec_ctx_get_channel_layout(ctx: *const AVCodecContext) -> u64;
}
