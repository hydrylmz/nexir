// src/audio/ffi/avresample.rs

/// Opaque libswresample context.
#[repr(C)] pub struct SwrContext { _opaque: [u8; 0] }

// AVSampleFormat values for the formats we use
pub const AV_SAMPLE_FMT_FLTP: i32 = 8;  // F32 planar (our canonical format)
pub const AV_SAMPLE_FMT_FLT:  i32 = 3;  // F32 interleaved
pub const AV_SAMPLE_FMT_S16:  i32 = 1;  // signed 16-bit interleaved
pub const AV_SAMPLE_FMT_S16P: i32 = 6;  // signed 16-bit planar

// Channel layout masks (AV_CH_LAYOUT_*)
pub const AV_CH_LAYOUT_MONO:   u64 = 0x0000_0004;
pub const AV_CH_LAYOUT_STEREO: u64 = 0x0000_0003;
pub const AV_CH_LAYOUT_5_1:    u64 = 0x0000_003F;

#[link(name = "swresample")]
unsafe extern "C" {
    /// Allocate and configure a resampler context.
    pub fn swr_alloc_set_opts(
        s:               *mut SwrContext,  // NULL = allocate new
        out_ch_layout:   i64,
        out_sample_fmt:  i32,
        out_sample_rate: i32,
        in_ch_layout:    i64,
        in_sample_fmt:   i32,
        in_sample_rate:  i32,
        log_offset:      i32,
        log_ctx:         *mut std::ffi::c_void,
    ) -> *mut SwrContext;

    /// Initialise the resampler (must be called before swr_convert).
    pub fn swr_init(s: *mut SwrContext) -> std::ffi::c_int;

    /// Free a SwrContext and set pointer to NULL.
    pub fn swr_free(s: *mut *mut SwrContext);

    /// Convert audio samples from input format to output format.
    pub fn swr_convert(
        s:        *mut SwrContext,
        out:      *mut *mut u8,
        out_count: std::ffi::c_int,
        in_data:  *const *const u8,
        in_count:  std::ffi::c_int,
    ) -> std::ffi::c_int;

    /// Get the delay (in output samples) currently buffered in the resampler.
    pub fn swr_get_delay(s: *const SwrContext, base: i64) -> i64;
}

/// Safe calculation of the output buffer size for swr_convert.
pub fn swr_output_sample_count(
    input_samples: usize,
    in_rate:       u32,
    out_rate:      u32,
) -> usize {
    // Math: ceil(in_count * out_rate / in_rate) + SWR_DELAY_MARGIN
    // Delay margin = 16 (for polyphase filter internal delay)
    ((input_samples as i64 * out_rate as i64 + in_rate as i64 - 1)
     / in_rate as i64 + 16) as usize
}
