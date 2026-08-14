#[repr(C)]
pub struct SwsContext {
    _opaque: [u8; 0],
}

#[link(name = "swscale")]
unsafe extern "C" {
    pub fn sws_getContext(
        srcW: i32,
        srcH: i32,
        srcFormat: i32,
        dstW: i32,
        dstH: i32,
        dstFormat: i32,
        flags: i32,
        srcFilter: *const std::ffi::c_void,
        dstFilter: *const std::ffi::c_void,
        param: *const f64,
    ) -> *mut SwsContext;

    pub fn sws_scale(
        c: *mut SwsContext,
        srcSlice: *const *const u8,
        srcStride: *const i32,
        srcSliceY: i32,
        srcSliceH: i32,
        dst: *const *mut u8,
        dstStride: *const i32,
    ) -> i32;

    pub fn sws_freeContext(swsContext: *mut SwsContext);
}

pub const SWS_BILINEAR: i32 = 2;
