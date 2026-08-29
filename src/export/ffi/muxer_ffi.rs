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

    // Colour description written straight onto the stream's AVCodecParameters.
    // `avcodec_parameters_from_context` already copies these from a live encoder
    // context, but the NVENC path describes its stream with a parameter-only
    // context, so the muxer sets them explicitly to keep both paths identical.
    pub fn avcodecpar_set_color_space(par: *mut AVCodecParameters, v: i32);
    pub fn avcodecpar_set_color_range(par: *mut AVCodecParameters, v: i32);
    pub fn avcodecpar_set_color_trc(par: *mut AVCodecParameters, v: i32);
    pub fn avcodecpar_set_color_primaries(par: *mut AVCodecParameters, v: i32);
    pub fn avcodecpar_set_chroma_location(par: *mut AVCodecParameters, v: i32);

    /// Attach HDR10 static metadata to a muxer stream's `AVCodecParameters`.
    ///
    /// MUST be called before `avformat_write_header`: the muxer reads
    /// `coded_side_data` there to write the mp4 `mdcv` / `clli` boxes (and the
    /// equivalent Matroska colour elements).
    ///
    /// Needed in addition to the encoder-side call because the NVENC path builds
    /// its bitstream outside libavcodec, and because container boxes survive a
    /// stream copy that would drop bitstream SEI.
    /// Returns 0 on success, -1 on failure.
    pub fn avcodecpar_set_hdr10_metadata(
        par:           *mut AVCodecParameters,
        prim:          *const i32,
        wp:            *const i32,
        min_luminance: i32,
        max_luminance: i32,
        max_cll:       u32,
        max_fall:      u32,
    ) -> std::ffi::c_int;

    /// Read HDR10 static metadata back off a demuxed stream.
    ///
    /// `out` must point at 8 `i64`s, filled as
    /// `[has_primaries, has_luminance, min_num, min_den, max_num, max_den,
    /// MaxCLL, MaxFALL]`.
    ///
    /// Returns a bitmask: 1 = mastering display present, 2 = content light level
    /// present, 0 = neither.
    pub fn avcodecpar_get_hdr10_metadata(
        par: *const AVCodecParameters,
        out: *mut i64,
    ) -> std::ffi::c_int;

    pub fn avformat_write_header_shim(
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
