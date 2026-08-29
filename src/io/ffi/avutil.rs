#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct AVRational {
    pub num: std::ffi::c_int,
    pub den: std::ffi::c_int,
}

impl AVRational {
    /// Convert to the engine's `Rational` type.
    pub fn to_rational(self) -> crate::timeline::rational::Rational {
        crate::timeline::rational::Rational {
            num: self.num as i64,
            den: self.den as i64,
        }
    }

    /// True if the rational has both num and den non-zero.
    pub fn is_valid(self) -> bool {
        self.num != 0 && self.den != 0
    }
}

/// Opaque FFmpeg dictionary (used for codec/format options).
#[repr(C)]
pub struct AVDictionary {
    _opaque: [u8; 0],
}

/// Opaque hardware frame context (ref-counted via AVBufferRef).
#[repr(C)]
pub struct AVBufferRef {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct AVFrame {
    pub data: [*mut u8; 8],
    pub linesize: [std::ffi::c_int; 8],
    pub extended_data: *mut *mut u8,
    pub width: std::ffi::c_int,
    pub height: std::ffi::c_int,
    pub nb_samples: std::ffi::c_int,
    pub format: std::ffi::c_int,
    pub key_frame: std::ffi::c_int,
    pub pict_type: std::ffi::c_int,
    pub sample_aspect_ratio: AVRational,
    pub pts: i64,
    pub pkt_pts: i64,
    pub pkt_dts: i64,
    pub coded_picture_number: std::ffi::c_int,
    pub display_picture_number: std::ffi::c_int,
    pub quality: std::ffi::c_int,
    pub opaque: *mut std::ffi::c_void,
    pub error: [u64; 8],
    pub repeat_pict: std::ffi::c_int,
    pub interlaced_frame: std::ffi::c_int,
    pub top_field_first: std::ffi::c_int,
    pub palette_has_changed: std::ffi::c_int,
    pub reordered_opaque: i64,
    pub sample_rate: std::ffi::c_int,
    pub channel_layout: u64,
}

#[repr(C)]
pub struct AVPacket {
    pub buf: *mut AVBufferRef,
    pub pts: i64,
    pub dts: i64,
    pub data: *mut u8,
    pub size: std::ffi::c_int,
    pub stream_index: std::ffi::c_int,
    pub flags: std::ffi::c_int,
    pub side_data: *mut std::ffi::c_void,
    pub side_data_elems: std::ffi::c_int,
    pub duration: i64,
    pub pos: i64,
}

#[link(name = "avutil")]
unsafe extern "C" {
    /// Allocate an AVFrame. Must be freed with `av_frame_free`.
    pub fn av_frame_alloc() -> *mut AVFrame;

    /// Free an AVFrame and set the pointer to NULL.
    pub fn av_frame_free(frame: *mut *mut AVFrame);

    /// Unreference frame data (reduce ref count, possibly free buffer).
    /// After this call the frame is in a "blank" state and can be reused.
    pub fn av_frame_unref(frame: *mut AVFrame);

    /// Allocate an AVPacket. Must be freed with `av_packet_free`.
    pub fn av_packet_alloc() -> *mut AVPacket;

    /// Free an AVPacket and set the pointer to NULL.
    pub fn av_packet_free(pkt: *mut *mut AVPacket);

    /// Unreference packet data (for packets obtained from `av_read_frame`).
    pub fn av_packet_unref(pkt: *mut AVPacket);
}

extern "C" {
    pub fn av_packet_set_stream_index(pkt: *mut AVPacket, idx: std::ffi::c_int);

    /// Get the PTS of an AVFrame (in stream timebase ticks).
    /// Returns `AV_NOPTS_VALUE` (i64::MIN) if PTS is unknown.
    pub fn av_frame_get_pts(frame: *const AVFrame) -> i64;

    /// Get the data plane pointer array.
    /// For YUV420p: data[0]=Y, data[1]=U, data[2]=V.
    /// For NV12:    data[0]=Y, data[1]=UV interleaved.
    pub fn av_frame_get_data(frame: *const AVFrame) -> *const *mut u8;

    /// Get the linesize (stride) array.
    /// linesize[0] = bytes per row of the first plane.
    /// This is NOT equal to the width — the decoder may pad rows.
    pub fn av_frame_get_linesize(frame: *const AVFrame) -> *const std::ffi::c_int;

    /// Get the pixel format (AVPixelFormat enum value).
    pub fn av_frame_get_format(frame: *const AVFrame) -> std::ffi::c_int;

    /// Get frame width in pixels.
    pub fn av_frame_get_width(frame: *const AVFrame) -> std::ffi::c_int;

    /// Get frame height in pixels.
    pub fn av_frame_get_height(frame: *const AVFrame) -> std::ffi::c_int;

    /// Get number of audio samples (per channel) in the frame.
    pub fn av_frame_get_nb_samples(frame: *const AVFrame) -> std::ffi::c_int;

    /// Get sample rate of the audio frame.
    pub fn av_frame_get_sample_rate(frame: *const AVFrame) -> std::ffi::c_int;

    pub fn av_frame_get_color_space(frame: *const AVFrame) -> std::ffi::c_int;
    pub fn av_frame_get_color_range(frame: *const AVFrame) -> std::ffi::c_int;
    pub fn av_frame_get_color_trc(frame: *const AVFrame) -> std::ffi::c_int;
    pub fn av_frame_get_color_primaries(frame: *const AVFrame) -> std::ffi::c_int;

    /// Colour-property setters.  `av_hwframe_transfer_data` copies pixels only,
    /// so the decoder copies these across manually — see
    /// `Decoder::copy_color_props`.
    pub fn av_frame_set_color_space(frame: *mut AVFrame, v: std::ffi::c_int);
    pub fn av_frame_set_color_range(frame: *mut AVFrame, v: std::ffi::c_int);
    pub fn av_frame_set_color_trc(frame: *mut AVFrame, v: std::ffi::c_int);
    pub fn av_frame_set_color_primaries(frame: *mut AVFrame, v: std::ffi::c_int);

    /// Bits per component of an AVPixelFormat, from FFmpeg's descriptor table.
    /// 0 when the format has no descriptor (hardware surfaces).
    pub fn av_pix_fmt_bit_depth(fmt: std::ffi::c_int) -> std::ffi::c_int;
    /// Number of planes in an AVPixelFormat; 0 when it has no descriptor.
    pub fn av_pix_fmt_plane_count(fmt: std::ffi::c_int) -> std::ffi::c_int;
    /// Bit shift of component 0 inside its storage word: 6 for P010 (MSB-aligned
    /// 10-bit), 0 for LSB-aligned planar formats.
    pub fn av_pix_fmt_component_shift(fmt: std::ffi::c_int) -> std::ffi::c_int;

    pub fn av_frame_set_width(frame: *mut AVFrame, width: std::ffi::c_int);
    pub fn av_frame_set_height(frame: *mut AVFrame, height: std::ffi::c_int);
    pub fn av_frame_set_format(frame: *mut AVFrame, format: std::ffi::c_int);
    pub fn av_frame_set_nb_samples(frame: *mut AVFrame, samples: std::ffi::c_int);
    pub fn av_frame_set_sample_rate(frame: *mut AVFrame, sample_rate: std::ffi::c_int);
    pub fn av_frame_set_ch_layout(frame: *mut AVFrame, ch_mask: u64);
}

#[link(name = "avutil")]
unsafe extern "C" {

    /// Rescale a PTS from one timebase to another using 64-bit integer arithmetic.
    ///
    /// result = a * bq.num * cq.den / (bq.den * cq.num)
    /// Uses internal i128 to avoid overflow.
    pub fn av_rescale_q(a: i64, bq: AVRational, cq: AVRational) -> i64;

    /// Copy decoded pixel data into a flat buffer.
    ///
    /// When `dst` is the mapped wgpu staging buffer pointer this is a zero-copy path:
    /// FFmpeg writes directly into GPU-visible memory. `align=1` means no extra row padding.
    pub fn av_image_copy_to_buffer(
        dst:          *mut u8,
        dst_size:     std::ffi::c_int,
        src_data:     *const *const u8,
        src_linesize: *const std::ffi::c_int,
        pix_fmt:      std::ffi::c_int,
        width:        std::ffi::c_int,
        height:       std::ffi::c_int,
        align:        std::ffi::c_int,
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

    /// Get the error string for an AVERROR code into a caller-supplied buffer.
    pub fn av_strerror(
        errnum:      std::ffi::c_int,
        errbuf:      *mut std::ffi::c_char,
        errbuf_size: usize,
    ) -> std::ffi::c_int;
}

pub const AV_NOPTS_VALUE:        i64 = i64::MIN;
pub const AV_NUM_DATA_POINTERS:  usize = 8;

/// AV_PKT_FLAG_KEY — the packet belongs to a keyframe.  Muxers use it to build
/// the container's sync-sample table (mp4 `stss`), so a stream written without
/// it is not seekable even when its bitstream contains IDRs.
pub const AV_PKT_FLAG_KEY: i32 = 0x0001;

// FFmpeg AVERROR sentinel codes
pub const AVERROR_EOF:    i32 = -541_478_725; // AVERROR(EOF)
pub const AVERROR_EAGAIN: i32 = -11;          // EAGAIN / AVERROR(EAGAIN)

// AVPixelFormat values for formats we handle.
//
// Verified against the installed FFmpeg headers (C:/ffmpeg/include) with a
// standalone probe rather than transcribed: the enum is dense and shifts between
// major versions, and AV_PIX_FMT_CUDA in particular was wrong here (119) for this
// build, where it is 117.
pub const AV_PIX_FMT_YUV420P:    i32 = 0;
pub const AV_PIX_FMT_YUV422P:    i32 = 4;
pub const AV_PIX_FMT_YUV444P:    i32 = 5;
pub const AV_PIX_FMT_NV12:       i32 = 23;
/// 10-bit planar 4:2:0, codes LSB-aligned in 16-bit little-endian words.
pub const AV_PIX_FMT_YUV420P10LE: i32 = 62;
/// 10-bit planar 4:2:2.
pub const AV_PIX_FMT_YUV422P10LE: i32 = 64;
/// 10-bit planar 4:4:4.
pub const AV_PIX_FMT_YUV444P10LE: i32 = 68;
/// 12-bit planar 4:2:0.
pub const AV_PIX_FMT_YUV420P12LE: i32 = 123;
/// 10-bit semi-planar 4:2:0, codes MSB-aligned (`code << 6`) — NVDEC's HDR output.
pub const AV_PIX_FMT_P010LE:     i32 = 158;
/// 16-bit semi-planar 4:2:0.
pub const AV_PIX_FMT_P016LE:     i32 = 169;
pub const AV_PIX_FMT_CUDA:       i32 = 117;
pub const AV_PIX_FMT_VIDEOTOOLBOX: i32 = 140;
pub const AV_PIX_FMT_NONE:       i32 = -1;

// AVColorRange / AVColorSpace / AVColorTransferCharacteristic / AVColorPrimaries
// values, as read from an AVFrame by the shim accessors.  Also probed against the
// installed headers.  `ColorInfo::from_ffmpeg` consumes these raw ints.
pub const AVCOL_RANGE_UNSPECIFIED: i32 = 0;
pub const AVCOL_RANGE_MPEG:        i32 = 1; // limited / broadcast
pub const AVCOL_RANGE_JPEG:        i32 = 2; // full
pub const AVCOL_SPC_UNSPECIFIED:   i32 = 2;
pub const AVCOL_TRC_UNSPECIFIED:   i32 = 2;
pub const AVCOL_PRI_UNSPECIFIED:   i32 = 2;

/// Bit depth and chroma layout of a decoded AVFrame's pixel format.
///
/// Returns `None` for a format the render upload path cannot consume directly
/// (RGB, 4:2:2/4:4:4, hardware surfaces), which is the decoder's signal to run it
/// through swscale first.
pub fn frame_layout_for_pix_fmt(
    fmt: i32,
) -> Option<crate::timeline::source::FrameLayout> {
    use crate::timeline::source::FrameLayout;
    Some(match fmt {
        AV_PIX_FMT_YUV420P       => FrameLayout::YUV420P8,
        AV_PIX_FMT_NV12          => FrameLayout::NV12,
        AV_PIX_FMT_P010LE        => FrameLayout::P010,
        AV_PIX_FMT_YUV420P10LE   => FrameLayout::YUV420P10,
        AV_PIX_FMT_YUV420P12LE   => FrameLayout {
            bit_depth: 12, semi_planar: false, msb_aligned: false,
        },
        AV_PIX_FMT_P016LE        => FrameLayout {
            bit_depth: 16, semi_planar: true, msb_aligned: false,
        },
        _ => return None,
    })
}

/// Format an AVERROR code as a Rust String for error reporting.
pub fn av_err_to_string(errnum: i32) -> String {
    let mut buf = [0i8; 64];
    unsafe {
        av_strerror(errnum, buf.as_mut_ptr(), buf.len());
        std::ffi::CStr::from_ptr(buf.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}
