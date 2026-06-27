// src/io/ffi/hw_accel.rs
// Hardware acceleration probe and device initialisation.

use super::avutil::AVBufferRef;
#[allow(unused_imports)]
use super::avcodec::AVCodecContext;

#[repr(C)]
pub struct AVHWDeviceContext {
    _opaque: [u8; 0],
}

/// Hardware device types we support, in preference order.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HwDeviceType {
    Cuda,          // NVIDIA — AV_HWDEVICE_TYPE_CUDA = 2
    D3D12Va,       // Windows — AV_HWDEVICE_TYPE_D3D12VA = 13
    VideoToolbox,  // macOS — AV_HWDEVICE_TYPE_VIDEOTOOLBOX = 9
    Vaapi,         // Linux — AV_HWDEVICE_TYPE_VAAPI = 6
    None,          // software fallback (no hardware)
}

impl HwDeviceType {
    /// FFmpeg AV_HWDEVICE_TYPE_* integer value for this type.
    pub fn ffi_value(self) -> i32 {
        match self {
            HwDeviceType::Cuda         => 2,
            HwDeviceType::D3D12Va      => 13,
            HwDeviceType::VideoToolbox => 9,
            HwDeviceType::Vaapi        => 6,
            HwDeviceType::None         => 0,
        }
    }
}

#[link(name = "avutil")]
unsafe extern "C" {
    /// Create a hardware device context of the given type.
    /// On success: `*device_ctx` points to a ref-counted `AVBufferRef`, returns 0.
    /// On failure: `*device_ctx = NULL`, returns negative AVERROR.
    pub fn av_hwdevice_ctx_create(
        device_ctx:  *mut *mut AVBufferRef,
        device_type: std::ffi::c_int,
        device:      *const std::ffi::c_char,
        opts:        *mut super::avutil::AVDictionary,
        flags:       std::ffi::c_int,
    ) -> std::ffi::c_int;

    /// Transfer a hardware frame to a software frame.
    /// `dst` must be a pre-allocated software `AVFrame`.
    pub fn av_hwframe_transfer_data(
        dst:   *mut super::avutil::AVFrame,
        src:   *const super::avutil::AVFrame,
        flags: std::ffi::c_int,
    ) -> std::ffi::c_int;

    /// Unref a buffer reference (decrement ref count; may free the buffer).
    pub fn av_buffer_unref(buf: *mut *mut AVBufferRef);
}

/// Probe hardware device types in preference order and return the first that works.
///
/// Returns `Ok(Some((type, ctx)))` if a hardware device was successfully created,
/// `Ok(None)` if all types failed (software fallback), or `Err` on a hard failure.
pub fn probe_hardware_device()
    -> Result<Option<(HwDeviceType, *mut AVBufferRef)>, HwAccelError>
{
    // Step 1 — Build platform-specific preference list
    #[cfg(target_os = "windows")]
    let preferred: &[HwDeviceType] = &[HwDeviceType::D3D12Va, HwDeviceType::Cuda];
    #[cfg(target_os = "macos")]
    let preferred: &[HwDeviceType] = &[HwDeviceType::VideoToolbox];
    #[cfg(target_os = "linux")]
    let preferred: &[HwDeviceType] = &[HwDeviceType::Cuda, HwDeviceType::Vaapi];
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let preferred: &[HwDeviceType] = &[];

    // Step 2 — Try each type in order
    for &hw_type in preferred {
        let mut ctx: *mut AVBufferRef = std::ptr::null_mut();
        let ret = unsafe {
            av_hwdevice_ctx_create(
                &mut ctx,
                hw_type.ffi_value(),
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 {
            return Ok(Some((hw_type, ctx)));
        }
        eprintln!(
            "[hw_accel] {:?} unavailable: {}",
            hw_type,
            super::avutil::av_err_to_string(ret)
        );
    }

    // Step 3 — All failed → software fallback
    Ok(None)
}

#[derive(Debug)]
pub enum HwAccelError {
    #[allow(dead_code)]
    AllDevicesFailed,
}
