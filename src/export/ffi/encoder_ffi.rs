use crate::io::ffi::avcodec::AVCodecContext;
use crate::io::ffi::avutil::{AVFrame, AVPacket, AVRational};

extern "C" {
    pub fn avcodec_find_encoder(id: std::ffi::c_uint) -> *const crate::io::ffi::avcodec::AVCodec;
    pub fn avcodec_find_encoder_by_name(name: *const std::ffi::c_char) -> *const crate::io::ffi::avcodec::AVCodec;
    pub fn avcodec_get_name_shim(codec: *const crate::io::ffi::avcodec::AVCodec) -> *const std::ffi::c_char;
    pub fn avcodec_ctx_set_bit_rate(ctx: *mut AVCodecContext, bitrate: i64);
    pub fn avcodec_ctx_set_sample_rate(ctx: *mut AVCodecContext, sample_rate: i32);
    pub fn avcodec_ctx_set_ch_layout(ctx: *mut AVCodecContext, ch_mask: u64);
    pub fn avcodec_ctx_set_sample_fmt(ctx: *mut AVCodecContext, fmt: i32);
    pub fn avcodec_ctx_set_dimensions(ctx: *mut AVCodecContext, width: i32, height: i32);
    pub fn avcodec_ctx_set_pix_fmt(ctx: *mut AVCodecContext, pix_fmt: i32);
    pub fn avcodec_ctx_set_time_base(ctx: *mut AVCodecContext, tb: AVRational);
    pub fn avcodec_ctx_set_gop_size(ctx: *mut AVCodecContext, gop: i32);

    // Colour description, set on the ENCODER context before `avcodec_open2` so
    // libavcodec writes it into the bitstream (H.264/HEVC VUI) and the muxer
    // copies it into the container. Without these an HDR export is decoded as
    // SDR regardless of what the pixels contain.
    pub fn avcodec_ctx_set_color_space(ctx: *mut AVCodecContext, v: i32);
    pub fn avcodec_ctx_set_color_range(ctx: *mut AVCodecContext, v: i32);
    pub fn avcodec_ctx_set_color_trc(ctx: *mut AVCodecContext, v: i32);
    pub fn avcodec_ctx_set_color_primaries(ctx: *mut AVCodecContext, v: i32);
    pub fn avcodec_ctx_set_chroma_location(ctx: *mut AVCodecContext, v: i32);

    pub fn avcodec_ctx_get_color_space(ctx: *const AVCodecContext) -> i32;
    pub fn avcodec_ctx_get_color_range(ctx: *const AVCodecContext) -> i32;
    pub fn avcodec_ctx_get_color_trc(ctx: *const AVCodecContext) -> i32;
    pub fn avcodec_ctx_get_color_primaries(ctx: *const AVCodecContext) -> i32;

    /// Attach HDR10 static metadata (SMPTE ST 2086 mastering display + CTA-861.3
    /// content light level) to an encoder context.
    ///
    /// MUST be called before `avcodec_open2`: libavcodec reads
    /// `AVCodecContext::decoded_side_data` there and that is what makes
    /// libx265/libx264 emit the corresponding SEI messages into the bitstream.
    /// Afterwards the array is owned by the encoder.
    ///
    /// `prim` is 6 chromaticity numerators over 50000 (`R.x R.y G.x G.y B.x B.y`),
    /// `wp` is 2 over 50000, luminances are numerators over 10000.
    /// Returns 0 on success, -1 on allocation failure.
    pub fn avcodec_ctx_set_hdr10_metadata(
        ctx:           *mut AVCodecContext,
        prim:          *const i32,
        wp:            *const i32,
        min_luminance: i32,
        max_luminance: i32,
        max_cll:       u32,
        max_fall:      u32,
    ) -> std::ffi::c_int;

    
    pub fn av_opt_set(
        obj:   *mut std::ffi::c_void,
        name:  *const std::ffi::c_char,
        val:   *const std::ffi::c_char,
        flags: std::ffi::c_int,
    ) -> std::ffi::c_int;

    pub fn av_opt_set_int(
        obj:   *mut std::ffi::c_void,
        name:  *const std::ffi::c_char,
        val:   i64,
        flags: std::ffi::c_int,
    ) -> std::ffi::c_int;

    pub fn av_opt_get_int(
        obj:   *mut std::ffi::c_void,
        name:  *const std::ffi::c_char,
        flags: std::ffi::c_int,
        out_val: *mut i64,
    ) -> std::ffi::c_int;

    pub fn avcodec_ctx_set_flags(ctx: *mut AVCodecContext, flags: i32);

    pub fn avcodec_send_frame(
        avctx: *mut AVCodecContext,
        frame: *const AVFrame,
    ) -> std::ffi::c_int;

    pub fn avcodec_receive_packet(
        avctx: *mut AVCodecContext,
        avpkt: *mut AVPacket,
    ) -> std::ffi::c_int;

    pub fn av_image_fill_arrays(
        dst_data:     *mut *mut u8,
        dst_linesize: *mut std::ffi::c_int,
        src:          *const u8,
        pix_fmt:      std::ffi::c_int,
        width:        std::ffi::c_int,
        height:       std::ffi::c_int,
        align:        std::ffi::c_int,
    ) -> std::ffi::c_int;

    pub fn av_image_get_buffer_size(
        pix_fmt: std::ffi::c_int,
        width:   std::ffi::c_int,
        height:  std::ffi::c_int,
        align:   std::ffi::c_int,
    ) -> std::ffi::c_int;

    pub fn av_frame_alloc() -> *mut AVFrame;
    pub fn av_frame_free(frame: *mut *mut AVFrame);
    pub fn av_frame_set_pts(frame: *mut AVFrame, pts: i64);
    /// Set `AVFrame::duration`, in the frame's own timebase.
    ///
    /// libavcodec propagates it to the encoded `AVPacket`, and the mp4 muxer
    /// uses the last packet's duration to close the final `stts` entry.  A zero
    /// duration there makes the track's `mdhd` duration end exactly at the last
    /// frame's PTS, and decoders then flag that frame `AV_PKT_FLAG_DISCARD`.
    pub fn av_frame_set_duration(frame: *mut AVFrame, duration: i64);
    pub fn av_frame_get_buffer(frame: *mut AVFrame, align: std::ffi::c_int) -> std::ffi::c_int;
}

pub const AV_CODEC_FLAG_GLOBAL_HEADER: i32 = 1 << 22;

pub const AV_PIX_FMT_YUV420P: i32 = 0;
pub const AV_PIX_FMT_GBRP:    i32 = 44;
pub const AV_PIX_FMT_YUV422P: i32 = 4;
