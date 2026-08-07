// src/io/ffi/avformat.rs
// Raw FFI bindings for libavformat — open, seek, read packet, close.

/// Opaque FFmpeg types — never dereferenced from Rust; all access is via accessor FFI.
#[repr(C)] pub struct AVFormatContext {
    pub av_class: *const std::ffi::c_void,
    pub iformat: *const std::ffi::c_void,
    pub oformat: *const std::ffi::c_void,
    pub priv_data: *mut std::ffi::c_void,
    pub pb: *mut std::ffi::c_void,
}
#[repr(C)] pub struct AVStream {
    pub index: std::ffi::c_int,
    pub id: std::ffi::c_int,
}
#[repr(C)] pub struct AVInputFormat   { _opaque: [u8; 0] }

use super::avutil::{AVPacket, AVRational, AVDictionary};

#[link(name = "avformat")]
unsafe extern "C" {

    /// Open a media file and read its header.
    /// On success: `*ps` points to a newly allocated `AVFormatContext`, returns 0.
    /// On failure: `*ps = NULL`, returns negative AVERROR code.
    ///
    /// # Safety
    /// - `filename` must be a valid null-terminated string.
    /// - Call `avformat_close_input(ps)` to free on both success AND failure.
    pub fn avformat_open_input(
        ps:       *mut *mut AVFormatContext,
        filename: *const std::ffi::c_char,
        fmt:      *const AVInputFormat,
        options:  *mut *mut AVDictionary,
    ) -> std::ffi::c_int;

    /// Read stream info (duration, frame rate, codec params) by decoding a few frames.
    /// Must be called after `avformat_open_input` and before reading packets.
    pub fn avformat_find_stream_info(
        ic:      *mut AVFormatContext,
        options: *mut *mut AVDictionary,
    ) -> std::ffi::c_int;

    /// Close a demuxer and free all resources. Sets `*ps = NULL`.
    pub fn avformat_close_input(ps: *mut *mut AVFormatContext);

    /// Find the "best" stream of the given type.
    /// Returns stream index (≥ 0) on success, negative AVERROR on failure.
    pub fn av_find_best_stream(
        ic:               *mut AVFormatContext,
        media_type:       std::ffi::c_int,
        wanted_stream_nb: std::ffi::c_int,
        related_stream:   std::ffi::c_int,
        decoder_ret:      *mut *const super::avcodec::AVCodec,
        flags:            std::ffi::c_int,
    ) -> std::ffi::c_int;

    /// Read the next compressed packet from the file.
    /// Returns 0 on success, AVERROR_EOF at end of file, negative on error.
    /// Caller MUST call `av_packet_unref()` after use.
    pub fn av_read_frame(s: *mut AVFormatContext, pkt: *mut AVPacket) -> std::ffi::c_int;

    /// Seek to the keyframe at or before the given timestamp.
    /// `flags = AVSEEK_FLAG_BACKWARD` seeks to keyframe AT OR BEFORE timestamp.
    pub fn av_seek_frame(
        s:            *mut AVFormatContext,
        stream_index: std::ffi::c_int,
        timestamp:    i64,
        flags:            std::ffi::c_int,
    ) -> std::ffi::c_int;
}

extern "C" {
    /// Access the stream array at the given index.
    pub fn avformat_get_stream(
        ctx:   *const AVFormatContext,
        index: std::ffi::c_uint,
    ) -> *mut AVStream;

    /// Get the number of streams in a format context.
    pub fn avformat_nb_streams(ctx: *const AVFormatContext) -> std::ffi::c_uint;

    /// Get the time_base of a stream (ticks per second as a rational).
    pub fn avstream_get_time_base(stream: *const AVStream) -> AVRational;
    pub fn avstream_set_time_base(stream: *mut AVStream, tb: AVRational);
    /// Get the average frame rate of a video stream.
    /// Returns `{0, 0}` if unknown — call `avformat_find_stream_info` first.
    pub fn avstream_get_avg_frame_rate(stream: *const AVStream) -> AVRational;
    pub fn avstream_get_r_frame_rate(stream: *const AVStream) -> AVRational;

    /// Get the duration of the format context in AV_TIME_BASE (µs) units.
    pub fn avformat_get_duration(ctx: *const AVFormatContext) -> i64;
}

// AVERROR constants
pub const AVERROR_EOF:    i32 = -541_478_725;
pub const AVERROR_EAGAIN: i32 = -11;

// Media type constants
pub const AVMEDIA_TYPE_VIDEO: i32 = 0;
pub const AVMEDIA_TYPE_AUDIO: i32 = 1;

/// Seek flag: seek to keyframe AT OR BEFORE the timestamp.
pub const AVSEEK_FLAG_BACKWARD: i32 = 1;

/// FFmpeg internal time base: 1 µs = AV_TIME_BASE ticks per second.
pub const AV_TIME_BASE: i64 = 1_000_000;
