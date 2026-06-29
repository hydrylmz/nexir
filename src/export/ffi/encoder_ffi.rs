use crate::io::ffi::avcodec::AVCodecContext;
use crate::io::ffi::avutil::{AVFrame, AVPacket, AVRational};

extern "C" {
    pub fn avcodec_find_encoder(id: std::ffi::c_uint) -> *const crate::io::ffi::avcodec::AVCodec;
    pub fn avcodec_ctx_set_bit_rate(ctx: *mut AVCodecContext, bitrate: i64);
    pub fn avcodec_ctx_set_sample_rate(ctx: *mut AVCodecContext, sample_rate: i32);
    pub fn avcodec_ctx_set_ch_layout(ctx: *mut AVCodecContext, ch_mask: u64);
    pub fn avcodec_ctx_set_sample_fmt(ctx: *mut AVCodecContext, fmt: i32);
    pub fn avcodec_ctx_set_dimensions(ctx: *mut AVCodecContext, width: i32, height: i32);
    pub fn avcodec_ctx_set_pix_fmt(ctx: *mut AVCodecContext, pix_fmt: i32);
    pub fn avcodec_ctx_set_time_base(ctx: *mut AVCodecContext, tb: AVRational);
    pub fn avcodec_ctx_set_gop_size(ctx: *mut AVCodecContext, gop: i32);
    
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
    pub fn av_frame_get_buffer(frame: *mut AVFrame, align: std::ffi::c_int) -> std::ffi::c_int;
}

pub const AV_CODEC_FLAG_GLOBAL_HEADER: i32 = 1 << 22;

pub const AV_PIX_FMT_YUV420P: i32 = 0;
pub const AV_PIX_FMT_GBRP:    i32 = 44;
pub const AV_PIX_FMT_YUV422P: i32 = 4;
