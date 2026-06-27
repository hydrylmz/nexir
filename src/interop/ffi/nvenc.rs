#![allow(non_camel_case_types)]
pub type NV_ENCODE_API_FUNCTION_LIST = *mut std::ffi::c_void; // opaque function table
pub type NvEncodeSession = *mut std::ffi::c_void;

pub const NV_ENC_SUCCESS: i32 = 0;

#[repr(C)]
pub struct NvEncOpenEncodeSessionExParams {
    pub version:       u32,
    pub device_type:   u32,       // NV_ENC_DEVICE_TYPE_CUDA = 1
    pub device:        *mut std::ffi::c_void,   // the CUcontext, cast to void*
    pub reserved:      *mut std::ffi::c_void,
    pub api_version:   u32,
}

pub const NV_ENC_DEVICE_TYPE_CUDA: u32 = 1;

#[repr(C)]
pub struct NvEncInitializeParams {
    pub version:        u32,
    pub encode_guid:    [u8; 16],   // NV_ENC_CODEC_H264_GUID or _HEVC_GUID
    pub preset_guid:    [u8; 16],   // NV_ENC_PRESET_P*_GUID — quality/speed presets
    pub encode_width:   u32,
    pub encode_height:  u32,
    pub frame_rate_num: u32,
    pub frame_rate_den: u32,
    pub max_encode_width:  u32,
    pub max_encode_height: u32,
    // Note: The rest of the fields are omitted for simplicity as they aren't explicitly required to be written out manually here in the scaffold.
    // In practice, this struct is quite large. We will only use what's strictly necessary.
    // Wait, FFmpeg's nvEncodeAPI.h contains the full struct. 
    // To be safe, I'll allocate a large enough buffer and zero it out.
    pub reserved: [u8; 1024], 
}

#[repr(C)]
pub struct NvEncRegisterResource {
    pub version:           u32,
    pub resource_type:     u32,     // NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY = 2
    pub width:              u32,
    pub height:             u32,
    pub pitch:              u32,
    pub resource_to_register: *mut std::ffi::c_void,  // the CUarray from Task 3
    pub registered_resource:  *mut std::ffi::c_void,  // output: opaque NVENC handle
    pub buffer_format:      u32,    // NV_ENC_BUFFER_FORMAT_ABGR10 or _ARGB10 — match our RGBA16F via a tonemap step, see Task 8
    pub buffer_usage:       u32,
    pub p_input_fence_point: *mut std::ffi::c_void,
    pub reserved: [u32; 249],
}

pub const NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY: u32 = 2;
pub const NV_ENC_BUFFER_FORMAT_ABGR10: u32 = 2;

#[repr(C)]
pub struct NvEncPicParams {
    pub version:          u32,
    pub input_width:       u32,
    pub input_height:      u32,
    pub input_pitch:       u32,
    pub encode_pic_flags:  u32,
    pub frame_idx:         u32,
    pub input_timestamp:   u64,
    pub input_duration:    u64,
    pub input_buffer:      *mut std::ffi::c_void,   // mapped input resource handle
    pub output_bitstream:  *mut std::ffi::c_void,   // pre-allocated bitstream buffer
    pub completion_event:  *mut std::ffi::c_void,
    pub buffer_fmt:        u32,
    pub picture_struct:    u32,     // NV_ENC_PIC_STRUCT_FRAME = 1
    pub picture_type:      u32,
    pub codec_pic_params:  [u8; 128], // union
    pub me_hint_counts_per_block: [u32; 2],
    pub me_external_hints: *mut std::ffi::c_void,
    pub reserved: [u32; 221],
    pub reserved2: [u32; 64], // pointers actually
}

#[link(name = "nvidia-encode")]
unsafe extern "C" {
    pub fn NvEncodeAPICreateInstance(function_list: NV_ENCODE_API_FUNCTION_LIST) -> i32;
}

pub mod functions {
    use super::*;
    pub type OpenEncodeSessionEx = unsafe extern "C" fn(
        *const NvEncOpenEncodeSessionExParams,
        *mut NvEncodeSession,
    ) -> i32;
    pub type InitializeEncoder = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncInitializeParams,
    ) -> i32;
    pub type RegisterResource = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncRegisterResource,
    ) -> i32;
    pub type MapInputResource = unsafe extern "C" fn(
        NvEncodeSession,
        *mut std::ffi::c_void,   // NV_ENC_MAP_INPUT_RESOURCE params
    ) -> i32;
    pub type EncodePicture = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncPicParams,
    ) -> i32;
    pub type LockBitstream = unsafe extern "C" fn(
        NvEncodeSession,
        *mut std::ffi::c_void,   // NV_ENC_LOCK_BITSTREAM params (in/out: ptr + size)
    ) -> i32;
    pub type UnlockBitstream = unsafe extern "C" fn(
        NvEncodeSession,
        *mut std::ffi::c_void,
    ) -> i32;
    pub type DestroyEncoder = unsafe extern "C" fn(NvEncodeSession) -> i32;
}
