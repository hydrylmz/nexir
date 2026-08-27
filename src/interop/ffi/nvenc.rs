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
    /// maxMEHintCountsPerBlock[2]: each element is NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE
    /// which is { bitfield_u32 + reserved1[3]: u32 } = 16 bytes. Two of them = 32 bytes = 8 u32s.
    pub max_me_hint_counts_per_block: [u32; 8],
    pub tuning_info:               u32,
    pub buffer_format:             u32,
    pub num_state_buffers:         u32,
    pub output_stats_level:        u32,
    pub reserved1:                 [u32; 284],
    pub reserved2:                 [*mut std::ffi::c_void; 64],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncInitializeParams>() == 1800,
        "NvEncInitializeParams size mismatch — update to match nvEncodeAPI.h"
    );
};

#[repr(C)]
pub struct NvEncRegisterResource {
    pub version:              u32,
    pub resource_type:        u32,     // NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY = 2
    pub width:                 u32,
    pub height:                u32,
    pub pitch:                 u32,
    pub sub_resource_index:    u32,
    pub resource_to_register:  *mut std::ffi::c_void,  // the CUarray from Task 3
    pub registered_resource:   *mut std::ffi::c_void,  // output: opaque NVENC handle
    pub buffer_format:         u32,    // NV_ENC_BUFFER_FORMAT_ABGR10
    pub buffer_usage:          u32,
    pub p_input_fence_point:   *mut std::ffi::c_void,
    pub reserved1:             [u32; 248],
    pub reserved2:             [*mut std::ffi::c_void; 61],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncRegisterResource>() == 1536,
        "NvEncRegisterResource size mismatch — update to match nvEncodeAPI.h"
    );
};

impl Default for NvEncRegisterResource {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
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
    pub tail:              [u8; 3284],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncPicParams>() == 3360,
        "NvEncPicParams size mismatch — update to match nvEncodeAPI.h"
    );
};

impl Default for NvEncPicParams {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

/// NV_ENC_MAP_INPUT_RESOURCE
#[repr(C)]
pub struct NvEncMapInputResource {
    pub version:             u32,
    pub sub_resource_index:  u32,
    pub input_resource:      *mut std::ffi::c_void,
    pub registered_resource: *mut std::ffi::c_void,
    pub mapped_resource:     *mut std::ffi::c_void,
    pub mapped_buffer_fmt:   u32,
    pub reserved1:           [u32; 251],
    pub reserved2:           [*mut std::ffi::c_void; 63],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncMapInputResource>() == 1544,
        "NvEncMapInputResource size mismatch — update to match nvEncodeAPI.h"
    );
};

impl Default for NvEncMapInputResource {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

/// NV_ENC_LOCK_BITSTREAM
#[repr(C)]
pub struct NvEncLockBitstream {
    pub version:                   u32,
    pub flags:                     u32,
    pub output_bitstream:          *mut std::ffi::c_void,
    pub slice_offsets:             *mut u32,
    pub frame_idx:                 u32,
    pub hw_encode_status:          u32,
    pub num_slices:                u32,
    pub bitstream_size_in_bytes:   u32,
    pub output_timestamp:          u64,
    pub output_duration:           u64,
    pub bitstream_buffer_ptr:      *mut std::ffi::c_void,
    pub picture_type:              u32,
    pub picture_struct:            u32,
    pub frame_avg_qp:              u32,
    pub frame_satd:                u32,
    pub ltr_frame_idx:             u32,
    pub ltr_frame_bitmap:          u32,
    pub temporal_id:               u32,
    pub intra_mb_count:            u32,
    pub inter_mb_count:            u32,
    pub average_mvx:               i32,
    pub average_mvy:               i32,
    pub alpha_layer_size_in_bytes: u32,
    pub output_stats_ptr_size:     u32,
    pub reserved:                  u32,
    pub output_stats_ptr:          *mut std::ffi::c_void,
    pub frame_idx_display:         u32,
    pub reserved1:                 [u32; 219],
    pub reserved2:                 [*mut std::ffi::c_void; 63],
    pub reserved_internal:         [u32; 8],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncLockBitstream>() == 1544,
        "NvEncLockBitstream size mismatch — update to match nvEncodeAPI.h"
    );
};

impl Default for NvEncLockBitstream {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

/// NV_ENC_CREATE_BITSTREAM_BUFFER
#[repr(C)]
pub struct NvEncCreateBitstreamBuffer {
    pub version:              u32,           // NVENCAPI_STRUCT_VERSION(1)
    pub size:                 u32,           // deprecated, must be 0
    pub memory_heap:          u32,           // deprecated, must be 0
    pub reserved:             u32,
    pub bitstream_buffer:     *mut std::ffi::c_void,  // [out] opaque handle
    pub bitstream_buffer_ptr: *mut std::ffi::c_void,  // deprecated
    pub reserved1:            [u32; 58],
    pub reserved2:            [*mut std::ffi::c_void; 64],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncCreateBitstreamBuffer>() == 776,
        "NvEncCreateBitstreamBuffer size mismatch — update to match nvEncodeAPI.h"
    );
};

impl Default for NvEncCreateBitstreamBuffer {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
}

/// NV_ENC_EVENT_PARAMS — passed to nvEncRegisterAsyncEvent / nvEncUnregisterAsyncEvent.
/// Layout mirrors nvEncodeAPI.h: version(u32) + reserved(u32) + completion_event(*void)
/// + reserved1([u32; 254]) + reserved2([*void; 64]) = 4+4+8+1016+512 = 1544 bytes.
#[repr(C)]
pub struct NvEncEventParams {
    pub version:          u32,
    pub reserved:         u32,
    pub completion_event: *mut std::ffi::c_void,
    pub reserved1:        [u32; 254],
    pub reserved2:        [*mut std::ffi::c_void; 64],
}

const _: () = {
    assert!(
        std::mem::size_of::<NvEncEventParams>() == 1544,
        "NvEncEventParams size mismatch — update to match nvEncodeAPI.h"
    );
};

impl Default for NvEncEventParams {
    fn default() -> Self { unsafe { std::mem::zeroed() } }
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
    pub type CreateBitstreamBuffer = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncCreateBitstreamBuffer,
    ) -> i32;
    pub type DestroyBitstreamBuffer = unsafe extern "C" fn(
        NvEncodeSession,
        *mut std::ffi::c_void,
    ) -> i32;
    pub type RegisterResource = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncRegisterResource,
    ) -> i32;
    pub type MapInputResource = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncMapInputResource,
    ) -> i32;
    pub type UnmapInputResource = unsafe extern "C" fn(
        NvEncodeSession,
        *mut std::ffi::c_void,
    ) -> i32;
    pub type EncodePicture = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncPicParams,
    ) -> i32;
    pub type LockBitstream = unsafe extern "C" fn(
        NvEncodeSession,
        *mut NvEncLockBitstream,
    ) -> i32;
    pub type UnlockBitstream = unsafe extern "C" fn(
        NvEncodeSession,
        *mut std::ffi::c_void,
    ) -> i32;
    pub type DestroyEncoder = unsafe extern "C" fn(NvEncodeSession) -> i32;
    pub type RegisterAsyncEvent = unsafe extern "C" fn(
        NvEncodeSession,
        *mut super::NvEncEventParams,
    ) -> i32;
    pub type UnregisterAsyncEvent = unsafe extern "C" fn(
        NvEncodeSession,
        *mut super::NvEncEventParams,
    ) -> i32;
}
