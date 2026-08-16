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
    pub reserved1:     [u32; 253],
    pub reserved2:     [*mut std::ffi::c_void; 64],
}

pub const NV_ENC_DEVICE_TYPE_CUDA: u32 = 1;

#[repr(C)]
pub struct NvEncInitializeParams {
    pub version:                  u32,
    pub encode_guid:               [u8; 16],
    pub preset_guid:               [u8; 16],
    pub encode_width:              u32,
    pub encode_height:             u32,
    pub dar_width:                 u32,
    pub dar_height:                u32,
    pub frame_rate_num:            u32,
    pub frame_rate_den:            u32,
    /// enableEncodeAsync — set to 0 (sync mode)
    pub enable_encode_async:       u32,
    /// enablePTD — set to 1 to let NVENC pick frame types
    pub enable_ptd:                u32,
    /// Packed bitfields: reportSliceOffsets:1, enableSubFrameWrite:1,
    /// enableExternalMEHints:1, enableMEOnlyMode:1, enableWeightedPrediction:1,
    /// splitEncodeMode:4, enableOutputInVidmem:1, enableReconFrameOutput:1,
    /// enableOutputStats:1, enableUniDirectionalB:1, reservedBitFields:19
    pub flags:                     u32,
    pub priv_data_size:            u32,
    /// Reserved u32 immediately after privDataSize (not a pointer, not an array)
    pub reserved_u32:              u32,
    pub priv_data:                 *mut std::ffi::c_void,
    pub encode_config:             *mut std::ffi::c_void,
    pub max_encode_width:          u32,
    pub max_encode_height:         u32,
    /// maxMEHintCountsPerBlock[2] — each is one u32 of bitfields, zero them
    pub max_me_hint_counts_per_block: [u32; 2],
    pub tuning_info:               u32,
    pub buffer_format:             u32,
    pub num_state_buffers:         u32,
    pub output_stats_level:        u32,
    pub reserved1:                 [u32; 284],
    pub reserved2:                 [*mut std::ffi::c_void; 64],
}
// Fixed layout (x86-64, repr(C)):
//   version(4) encode_guid(16) preset_guid(16) encode_width(4) encode_height(4)
//   dar_width(4) dar_height(4) frame_rate_num(4) frame_rate_den(4)
//   enable_encode_async(4) enable_ptd(4) flags(4) priv_data_size(4) reserved_u32(4)
//   => 80 bytes, *mut void alignment satisfied at offset 80
//   priv_data(8@80) encode_config(8@88)
//   max_encode_width(4@96) max_encode_height(4@100) max_me_hint_counts_per_block(8@104)
//   tuning_info(4@112) buffer_format(4@116) num_state_buffers(4@120) output_stats_level(4@124)
//   => offset 128, reserved1[284] u32 = 1136 bytes => offset 1264
//   reserved2[64] *mut void = 512 bytes => total 1776
const _: () = {
    assert!(
        std::mem::size_of::<NvEncInitializeParams>() == 1776,
        "NvEncInitializeParams size mismatch — update to match nvEncodeAPI.h"
    );
};

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
    pub type GetPresetConfig = unsafe extern "C" fn(
        NvEncodeSession,
        [u8; 16],
        [u8; 16],
        *mut std::ffi::c_void,
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
