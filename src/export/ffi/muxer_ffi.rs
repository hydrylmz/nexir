use crate::io::ffi::avformat::{AVFormatContext, AVStream};
use crate::io::ffi::avutil::{AVPacket, AVRational, AVDictionary};
use crate::io::ffi::avcodec::{AVCodecContext, AVCodecParameters};

extern "C" {
    pub fn avformat_alloc_output_context2(
        ctx:         *mut *mut AVFormatContext,
        oformat:     *const std::ffi::c_void,
        format_name: *const std::ffi::c_char,
        filename:    *const std::ffi::c_char,
    ) -> std::ffi::c_int;

    pub fn avformat_new_stream(
        s:     *mut AVFormatContext,
        codec: *const std::ffi::c_void,
    ) -> *mut AVStream;

    pub fn avcodec_parameters_from_context(
        par:  *mut AVCodecParameters,
        codec: *const AVCodecContext,
    ) -> std::ffi::c_int;

    pub fn avstream_get_codecpar_mut(stream: *mut AVStream) -> *mut AVCodecParameters;

    pub fn avstream_set_time_base(stream: *mut AVStream, tb: AVRational);

    pub fn avformat_write_header(
        s:       *mut AVFormatContext,
        options: *mut *mut AVDictionary,
    ) -> std::ffi::c_int;

    pub fn avio_open(
        pb:    *mut *mut std::ffi::c_void,
        url:   *const std::ffi::c_char,
        flags: std::ffi::c_int,
    ) -> std::ffi::c_int;

    pub fn av_stream_get_index(stream: *mut AVStream) -> std::ffi::c_int;
    pub fn avformat_open_output_pb(s: *mut AVFormatContext, url: *const std::ffi::c_char, flags: std::ffi::c_int) -> std::ffi::c_int;
    pub fn av_packet_rescale_ts(pkt: *mut AVPacket, tb_src: AVRational, tb_dst: AVRational);


    pub fn av_interleaved_write_frame(
        s:   *mut AVFormatContext,
        pkt: *mut AVPacket,
    ) -> std::ffi::c_int;

    pub fn av_write_trailer(s: *mut AVFormatContext) -> std::ffi::c_int;

    pub fn avformat_free_context(s: *mut AVFormatContext);
}

pub const AVIO_FLAG_WRITE: i32 = 2;
