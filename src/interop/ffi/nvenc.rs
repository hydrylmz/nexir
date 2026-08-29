#![allow(non_camel_case_types)]
pub type NV_ENCODE_API_FUNCTION_LIST = *mut std::ffi::c_void; // opaque function table
pub type NvEncodeSession = *mut std::ffi::c_void;

pub const NV_ENC_SUCCESS: i32 = 0;

/// NV_ENC_ERR_NEED_MORE_INPUT — `nvEncEncodePicture` accepted the picture but
/// cannot produce a bitstream yet because the encoder is reordering (B-frames
/// or lookahead).  This is NOT a failure: the caller must submit more pictures
/// (or an EOS) and retrieve the output later.
pub const NV_ENC_ERR_NEED_MORE_INPUT: i32 = 17;

// ---------------------------------------------------------------------------
// NVENCSTATUS values used by name below.  Verified by preprocessing the vendor
// header (see the PROVENANCE note further down): the enum is a dense counter
// starting at NV_ENC_SUCCESS = 0.
//
// Worth spelling out because two of these were mixed up while debugging P1.5:
// INVALID_PARAM is 8, and 10 is OUT_OF_MEMORY — which is what a session whose
// CUDA context has been destroyed underneath it reports from EncodePicture.
// ---------------------------------------------------------------------------

/// NV_ENC_ERR_INVALID_PTR
pub const NV_ENC_ERR_INVALID_PTR: i32 = 6;
/// NV_ENC_ERR_INVALID_PARAM — a field of the passed struct is rejected.
pub const NV_ENC_ERR_INVALID_PARAM: i32 = 8;
/// NV_ENC_ERR_INVALID_CALL
pub const NV_ENC_ERR_INVALID_CALL: i32 = 9;
/// NV_ENC_ERR_OUT_OF_MEMORY — also what the driver returns when the session's
/// backing CUDA context is no longer valid.
pub const NV_ENC_ERR_OUT_OF_MEMORY: i32 = 10;
/// NV_ENC_ERR_ENCODER_NOT_INITIALIZED
pub const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: i32 = 11;
/// NV_ENC_ERR_UNSUPPORTED_PARAM
pub const NV_ENC_ERR_UNSUPPORTED_PARAM: i32 = 12;
/// NV_ENC_ERR_INVALID_VERSION — a struct `version` word the driver rejects.
pub const NV_ENC_ERR_INVALID_VERSION: i32 = 15;
/// NV_ENC_ERR_MAP_FAILED
pub const NV_ENC_ERR_MAP_FAILED: i32 = 16;
/// NV_ENC_ERR_GENERIC
pub const NV_ENC_ERR_GENERIC: i32 = 20;

/// Human-readable NVENCSTATUS, so a failure reports a name instead of a bare
/// integer the reader has to look up (and can mis-look-up).
pub fn nvenc_status_str(status: i32) -> &'static str {
    match status {
        0  => "NV_ENC_SUCCESS",
        1  => "NV_ENC_ERR_NO_ENCODE_DEVICE",
        2  => "NV_ENC_ERR_UNSUPPORTED_DEVICE",
        3  => "NV_ENC_ERR_INVALID_ENCODERDEVICE",
        4  => "NV_ENC_ERR_INVALID_DEVICE",
        5  => "NV_ENC_ERR_DEVICE_NOT_EXIST",
        6  => "NV_ENC_ERR_INVALID_PTR",
        7  => "NV_ENC_ERR_INVALID_EVENT",
        8  => "NV_ENC_ERR_INVALID_PARAM",
        9  => "NV_ENC_ERR_INVALID_CALL",
        10 => "NV_ENC_ERR_OUT_OF_MEMORY",
        11 => "NV_ENC_ERR_ENCODER_NOT_INITIALIZED",
        12 => "NV_ENC_ERR_UNSUPPORTED_PARAM",
        13 => "NV_ENC_ERR_LOCK_BUSY",
        14 => "NV_ENC_ERR_NOT_ENOUGH_BUFFER",
        15 => "NV_ENC_ERR_INVALID_VERSION",
        16 => "NV_ENC_ERR_MAP_FAILED",
        17 => "NV_ENC_ERR_NEED_MORE_INPUT",
        18 => "NV_ENC_ERR_ENCODER_BUSY",
        19 => "NV_ENC_ERR_EVENT_NOT_REGISTERD",
        20 => "NV_ENC_ERR_GENERIC",
        21 => "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY",
        22 => "NV_ENC_ERR_UNIMPLEMENTED",
        _  => "NV_ENC_ERR_<unknown>",
    }
}

// ---------------------------------------------------------------------------
// NV_ENC_PIC_FLAGS — a BIT FIELD, not a counter (P1.5).
//
//     NV_ENC_PIC_FLAG_FORCEINTRA     = 0x1
//     NV_ENC_PIC_FLAG_FORCEIDR       = 0x2
//     NV_ENC_PIC_FLAG_OUTPUT_SPSPPS  = 0x4
//     NV_ENC_PIC_FLAG_EOS            = 0x8
//
// This block previously defined EOS as 0x1, i.e. FORCEINTRA.  The flush path
// therefore asked the encoder to code an intra picture out of a NULL input
// buffer, which the driver rejects with NV_ENC_ERR_INVALID_PARAM (8).
// ---------------------------------------------------------------------------

/// NV_ENC_PIC_FLAG_FORCEINTRA — code this picture as intra.
pub const NV_ENC_PIC_FLAG_FORCEINTRA: u32 = 0x0000_0001;
/// NV_ENC_PIC_FLAG_FORCEIDR — code this picture as an IDR.
pub const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x0000_0002;
/// NV_ENC_PIC_FLAG_OUTPUT_SPSPPS — emit SPS/PPS with this picture.
pub const NV_ENC_PIC_FLAG_OUTPUT_SPSPPS: u32 = 0x0000_0004;
/// NV_ENC_PIC_FLAG_EOS — end-of-stream picture.  Submitted with a NULL input
/// buffer and NULL output bitstream to make NVENC flush every frame still held
/// in its reorder/lookahead queue before the session is destroyed.
pub const NV_ENC_PIC_FLAG_EOS: u32 = 0x0000_0008;

// NV_ENC_PIC_TYPE — reported back in NV_ENC_LOCK_BITSTREAM::pictureType.
// Only the two intra types mark a packet as a container-level sync sample.
pub const NV_ENC_PIC_TYPE_P: u32 = 0;
pub const NV_ENC_PIC_TYPE_B: u32 = 1;
pub const NV_ENC_PIC_TYPE_I: u32 = 2;
pub const NV_ENC_PIC_TYPE_IDR: u32 = 3;

/// True when a picture type produced by NVENC is a keyframe (I or IDR), i.e.
/// the packet must carry `AV_PKT_FLAG_KEY` so the muxer records a sync sample.
#[inline(always)]
pub const fn nv_enc_pic_type_is_keyframe(picture_type: u32) -> bool {
    picture_type == NV_ENC_PIC_TYPE_I || picture_type == NV_ENC_PIC_TYPE_IDR
}

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

// ---------------------------------------------------------------------------
// Enumerants verified against ffnvcodec `nvEncodeAPI.h` (P3.2).
//
// PROVENANCE: compiled a C probe against the real vendor headers shipped by
// FFmpeg/nv-codec-headers tags n12.0.16.0, n12.2.72.0 and n13.0.19.0 and printed
// the enumerants with the preprocessor.  All three agree on every value below,
// so these are stable across the API versions this code probes for.
//
// NV_ENC_BUFFER_FORMAT is a sparse BIT-FLAG enum, not a dense counter:
//     UNDEFINED       = 0x00000000      YUV420_10BIT = 0x00010000
//     NV12            = 0x00000001      YUV444_10BIT = 0x00100000
//     YV12            = 0x00000010      ARGB         = 0x01000000
//     IYUV            = 0x00000100      ARGB10       = 0x02000000
//     YUV444          = 0x00001000      AYUV         = 0x04000000
//                                       ABGR         = 0x10000000
//                                       ABGR10       = 0x20000000
// `2` is NOT a member of this enum at all — NVENC would have rejected it.
// ---------------------------------------------------------------------------

/// NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY — verified 0x2.
pub const NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY: u32 = 0x0000_0002;

/// NV_ENC_BUFFER_FORMAT_ABGR10 — 10-bit packed A2B10G10R10, word-ordered with R
/// in the low 10 bits, then G, then B, alpha in the top 2 bits.  That matches the
/// packing `Abgr10RepackNode` produces (see `src/interop/encode_interop.rs:15`).
pub const NV_ENC_BUFFER_FORMAT_ABGR10: u32 = 0x2000_0000;

/// NV_ENC_BUFFER_FORMAT_ABGR — 8-bit packed A8B8G8R8, same channel order.
pub const NV_ENC_BUFFER_FORMAT_ABGR: u32 = 0x1000_0000;

/// NV_ENC_BUFFER_FORMAT_ARGB10 — 10-bit packed A2R10G10B10 (B in the low bits).
/// Kept only to document the value ABGR10 must not be confused with.
pub const NV_ENC_BUFFER_FORMAT_ARGB10: u32 = 0x0200_0000;

/// NV_ENC_BUFFER_FORMAT_UNDEFINED — what NVENC treats as "no format given".
pub const NV_ENC_BUFFER_FORMAT_UNDEFINED: u32 = 0x0000_0000;

// ---------------------------------------------------------------------------
// Struct version words (P3.2).
//
// Every NVENC struct carries a `version` field the driver validates before it
// looks at anything else; a wrong word fails the call with
// NV_ENC_ERR_INVALID_VERSION (15) and nothing else is reported.  The header
// builds these words as:
//
//   NVENCAPI_VERSION           = major | (minor << 24)
//   NVENCAPI_STRUCT_VERSION(n) = NVENCAPI_VERSION | (n << 16) | (0x7 << 28)
//
// Two things are easy to get wrong and both were, here:
//   1. The API version word carries the MINOR version in bits 24..31.  Passing
//      the bare major (e.g. 12 instead of 0x0200000C for 12.2) is rejected.
//   2. A few structs additionally OR in `1u << 31`.  Those are marked below.
//
// The per-struct `n` values are NOT interchangeable between structs and they
// change between API versions.  These are the API 12.2 / 13.x values, verified
// by preprocessing the vendor header and by calling the installed driver: every
// word below was accepted, and every previously used word was rejected with
// NV_ENC_ERR_INVALID_VERSION.
// ---------------------------------------------------------------------------

/// NVENCAPI_VERSION — `major | (minor << 24)`.
#[inline(always)]
pub const fn nvenc_api_version(major: u32, minor: u32) -> u32 {
    major | (minor << 24)
}

/// NVENCAPI_STRUCT_VERSION(n) for a given API version word.
///
/// `api_version` must be a full NVENCAPI_VERSION word (see
/// [`nvenc_api_version`]), not a bare major number.
#[inline(always)]
pub const fn nvenc_struct_version(api_version: u32, struct_ver: u32) -> u32 {
    api_version | (struct_ver << 16) | (0x7 << 28)
}

/// Structs whose `_VER` macro ORs in `1u << 31` on top of
/// NVENCAPI_STRUCT_VERSION: NV_ENC_INITIALIZE_PARAMS, NV_ENC_PIC_PARAMS,
/// NV_ENC_LOCK_BITSTREAM, NV_ENC_CONFIG and NV_ENC_PRESET_CONFIG.
#[inline(always)]
const fn with_high_bit(v: u32) -> u32 { v | (1 << 31) }

/// NV_ENC_ENCODE_API_FUNCTION_LIST_VER — STRUCT_VERSION(2).
#[inline(always)]
pub const fn nv_encode_api_function_list_ver(api: u32) -> u32 { nvenc_struct_version(api, 2) }

/// NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER — STRUCT_VERSION(1).
#[inline(always)]
pub const fn nv_enc_open_encode_session_ex_params_ver(api: u32) -> u32 { nvenc_struct_version(api, 1) }

/// NV_ENC_INITIALIZE_PARAMS_VER — STRUCT_VERSION(7) | 1<<31.
#[inline(always)]
pub const fn nv_enc_initialize_params_ver(api: u32) -> u32 { with_high_bit(nvenc_struct_version(api, 7)) }

/// NV_ENC_REGISTER_RESOURCE_VER — STRUCT_VERSION(5).
#[inline(always)]
pub const fn nv_enc_register_resource_ver(api: u32) -> u32 { nvenc_struct_version(api, 5) }

/// NV_ENC_MAP_INPUT_RESOURCE_VER — STRUCT_VERSION(4).
#[inline(always)]
pub const fn nv_enc_map_input_resource_ver(api: u32) -> u32 { nvenc_struct_version(api, 4) }

/// NV_ENC_PIC_PARAMS_VER — STRUCT_VERSION(7) | 1<<31.
#[inline(always)]
pub const fn nv_enc_pic_params_ver(api: u32) -> u32 { with_high_bit(nvenc_struct_version(api, 7)) }

/// NV_ENC_LOCK_BITSTREAM_VER — STRUCT_VERSION(2) | 1<<31.
#[inline(always)]
pub const fn nv_enc_lock_bitstream_ver(api: u32) -> u32 { with_high_bit(nvenc_struct_version(api, 2)) }

/// NV_ENC_CREATE_BITSTREAM_BUFFER_VER — STRUCT_VERSION(1).
#[inline(always)]
pub const fn nv_enc_create_bitstream_buffer_ver(api: u32) -> u32 { nvenc_struct_version(api, 1) }

/// NV_ENC_EVENT_PARAMS_VER — STRUCT_VERSION(2).
#[inline(always)]
pub const fn nv_enc_event_params_ver(api: u32) -> u32 { nvenc_struct_version(api, 2) }

/// NV_ENC_CONFIG_VER — STRUCT_VERSION(9) | 1<<31.
#[inline(always)]
pub const fn nv_enc_config_ver(api: u32) -> u32 { with_high_bit(nvenc_struct_version(api, 9)) }

/// NV_ENC_PRESET_CONFIG_VER — STRUCT_VERSION(5) | 1<<31.
#[inline(always)]
pub const fn nv_enc_preset_config_ver(api: u32) -> u32 { with_high_bit(nvenc_struct_version(api, 5)) }

// ---------------------------------------------------------------------------
// Preset GUIDs (P3.2).
//
// These are the real GUIDs from the vendor header, cross-checked against the
// list `nvEncGetEncodePresetGUIDs` returns from the installed driver.  Presets
// run P1 (fastest) to P7 (best quality).
//
// The previous P4/DEFAULT GUIDs in encode_interop.rs did not exist in any
// header and the driver rejected both with NV_ENC_ERR_INVALID_PARAM.
// ---------------------------------------------------------------------------

/// NV_ENC_PRESET_P1_GUID {FC0A8D3E-45F8-4CF8-80C7-298871590EBF} — fastest.
pub const NV_ENC_PRESET_P1_GUID: [u8; 16] = [
    0x3E, 0x8D, 0x0A, 0xFC, 0xF8, 0x45, 0xF8, 0x4C,
    0x80, 0xC7, 0x29, 0x88, 0x71, 0x59, 0x0E, 0xBF,
];

/// NV_ENC_PRESET_P4_GUID {90A7B826-DF06-4862-B9D2-CD6D73A08681} — balanced.
pub const NV_ENC_PRESET_P4_GUID: [u8; 16] = [
    0x26, 0xB8, 0xA7, 0x90, 0x06, 0xDF, 0x62, 0x48,
    0xB9, 0xD2, 0xCD, 0x6D, 0x73, 0xA0, 0x86, 0x81,
];

/// NV_ENC_PRESET_P7_GUID {84848C12-6F71-4C13-931B-53E283F57974} — best quality.
pub const NV_ENC_PRESET_P7_GUID: [u8; 16] = [
    0x12, 0x8C, 0x84, 0x84, 0x71, 0x6F, 0x13, 0x4C,
    0x93, 0x1B, 0x53, 0xE2, 0x83, 0xF5, 0x79, 0x74,
];

// ---------------------------------------------------------------------------
// NV_ENC_TUNING_INFO (P3.2).
//
// `tuningInfo` is NOT optional for the P1..P7 presets.  Leaving it at 0
// (NV_ENC_TUNING_INFO_UNDEFINED) makes nvEncInitializeEncoder fail with
// NV_ENC_ERR_UNSUPPORTED_PARAM (12), verified against the installed driver.
// ---------------------------------------------------------------------------

/// Invalid for encoding — the driver rejects it.
pub const NV_ENC_TUNING_INFO_UNDEFINED: u32 = 0;
/// Latency-tolerant encoding; the right choice for file export.
pub const NV_ENC_TUNING_INFO_HIGH_QUALITY: u32 = 1;
pub const NV_ENC_TUNING_INFO_LOW_LATENCY: u32 = 2;
pub const NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY: u32 = 3;
pub const NV_ENC_TUNING_INFO_LOSSLESS: u32 = 4;

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

// ---------------------------------------------------------------------------
// NV_ENC_PRESET_CONFIG / NV_ENC_CONFIG (P1.3)
//
// The only reason this code needs NV_ENC_CONFIG at all is to pin the reorder
// behaviour: with `encodeConfig = NULL` the driver applies the preset's own
// defaults, which for the P-presets MAY enable B-frames and/or lookahead.  A
// reordering encoder returns bitstreams in decode order and holds input
// surfaces past the EncodePicture call, neither of which the two-slot
// ping-pong export pipeline in `encode_interop.rs` can absorb.
//
// Rather than transcribing the whole NV_ENC_CONFIG tree (it embeds
// NV_ENC_RC_PARAMS plus a union of the per-codec configs, all of which move
// between API versions), this keeps the driver's own preset config as an opaque
// blob and patches the two scalars it needs, at fixed offsets near the front of
// the struct:
//
//   typedef struct _NV_ENC_PRESET_CONFIG {
//       uint32_t         version;      // +0
//       NV_ENC_CONFIG    presetCfg;    // +8  (NV_ENC_CONFIG is 8-byte aligned:
//       ...                            //      it ends in void* reserved2[64])
//   } NV_ENC_PRESET_CONFIG;
//
//   typedef struct _NV_ENC_CONFIG {
//       uint32_t version;              // +0   -> +8  absolute
//       GUID     profileGUID;          // +4   (16 bytes)
//       uint32_t gopLength;            // +20  -> +28 absolute
//       int32_t  frameIntervalP;       // +24  -> +32 absolute
//       ...
//   } NV_ENC_CONFIG;
//
// The offsets are ASSUMED, not verified against a local header (the vendor
// headers are not installed on the build machine), so every user of this buffer
// MUST call `preset_cfg_version_matches` after the driver fills it in and fall
// back to `encodeConfig = NULL` when the check fails.  A wrong offset shows up
// as a version word that is not NV_ENC_CONFIG_VER, which is exactly what that
// check tests.
// ---------------------------------------------------------------------------

/// Byte offset of `NV_ENC_PRESET_CONFIG::presetCfg`.
const PRESET_CFG_OFFSET: usize = 8;
/// Byte offset of `NV_ENC_CONFIG::gopLength` inside NV_ENC_PRESET_CONFIG.
const CFG_GOP_LENGTH_OFFSET: usize = PRESET_CFG_OFFSET + 20;
/// Byte offset of `NV_ENC_CONFIG::frameIntervalP` inside NV_ENC_PRESET_CONFIG.
const CFG_FRAME_INTERVAL_P_OFFSET: usize = PRESET_CFG_OFFSET + 24;

/// Oversized zeroed backing store for NV_ENC_PRESET_CONFIG.
///
/// NV_ENC_PRESET_CONFIG is a little over 3 KB in API 12.x; 16 KB leaves room
/// for any later version to grow without this code having to track its size.
/// Passing a larger buffer than the driver expects is harmless — it writes only
/// as far as its own `version` word says.
const PRESET_CONFIG_BUF_BYTES: usize = 16_384;

/// `NV_ENC_INFINITE_GOPLENGTH` — intra-only-on-demand; not used here, kept so
/// the meaning of a very large gopLength is not mistaken for a bug.
pub const NV_ENC_INFINITE_GOPLENGTH: u32 = 0xFFFF_FFFF;

/// A driver-filled NV_ENC_PRESET_CONFIG, held as bytes.
///
/// Use [`PresetConfig::query`] to have the driver populate it, then
/// [`PresetConfig::config_ptr`] as `NV_ENC_INITIALIZE_PARAMS::encodeConfig`.
/// The buffer must outlive the `nvEncInitializeEncoder` call.
pub struct PresetConfig {
    raw: Vec<u8>,
    api: u32,
}

impl PresetConfig {
    /// Allocate a zeroed buffer with both version words stamped, ready to hand
    /// to `nvEncGetEncodePresetConfigEx`.
    pub fn new(api_version: u32) -> Self {
        let mut raw = vec![0u8; PRESET_CONFIG_BUF_BYTES];
        // NV_ENC_PRESET_CONFIG::version
        raw[0..4].copy_from_slice(&nv_enc_preset_config_ver(api_version).to_le_bytes());
        // NV_ENC_PRESET_CONFIG::presetCfg.version
        raw[PRESET_CFG_OFFSET..PRESET_CFG_OFFSET + 4]
            .copy_from_slice(&nv_enc_config_ver(api_version).to_le_bytes());
        Self { raw, api: api_version }
    }

    /// Pointer to the whole NV_ENC_PRESET_CONFIG, for
    /// `nvEncGetEncodePresetConfigEx`.
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.raw.as_mut_ptr() as *mut std::ffi::c_void
    }

    /// Pointer to the embedded NV_ENC_CONFIG, for
    /// `NV_ENC_INITIALIZE_PARAMS::encodeConfig`.
    pub fn config_ptr(&mut self) -> *mut std::ffi::c_void {
        unsafe { self.raw.as_mut_ptr().add(PRESET_CFG_OFFSET) as *mut std::ffi::c_void }
    }

    fn read_u32(&self, offset: usize) -> u32 {
        u32::from_le_bytes([
            self.raw[offset],
            self.raw[offset + 1],
            self.raw[offset + 2],
            self.raw[offset + 3],
        ])
    }

    fn write_u32(&mut self, offset: usize, value: u32) {
        self.raw[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// True when the driver-written `presetCfg.version` is the NV_ENC_CONFIG
    /// version word for this API — i.e. the offsets above are correct and the
    /// scalar patches below will land where intended.
    ///
    /// A `false` here means the layout assumption is wrong: the caller MUST
    /// discard this buffer and initialise with `encodeConfig = NULL` instead of
    /// writing to guessed offsets.
    pub fn preset_cfg_version_matches(&self) -> bool {
        self.read_u32(PRESET_CFG_OFFSET) == nv_enc_config_ver(self.api)
    }

    /// The `presetCfg.version` word the driver actually wrote — useful only for
    /// diagnosing a failed [`Self::preset_cfg_version_matches`].
    pub fn preset_cfg_version(&self) -> u32 {
        self.read_u32(PRESET_CFG_OFFSET)
    }

    pub fn gop_length(&self) -> u32 {
        self.read_u32(CFG_GOP_LENGTH_OFFSET)
    }

    pub fn frame_interval_p(&self) -> u32 {
        self.read_u32(CFG_FRAME_INTERVAL_P_OFFSET)
    }

    /// Distance between P-frames.  1 = IPPP (no B-frames), 2 = one B-frame
    /// between references, and so on.  Setting 1 is what keeps output order
    /// equal to input order.
    pub fn set_frame_interval_p(&mut self, interval: u32) {
        self.write_u32(CFG_FRAME_INTERVAL_P_OFFSET, interval);
    }

    /// Frames between IDRs.
    pub fn set_gop_length(&mut self, gop: u32) {
        self.write_u32(CFG_GOP_LENGTH_OFFSET, gop);
    }
}

#[link(name = "nvidia-encode")]
unsafe extern "C" {
    pub fn NvEncodeAPICreateInstance(function_list: NV_ENCODE_API_FUNCTION_LIST) -> i32;
}

// ---------------------------------------------------------------------------
// NvEncodeAPIGetMaxSupportedVersion (P3.2)
//
// This entry point is NOT in the import library FFmpeg ships as
// `nvidia-encode.lib` (that stub exports only NvEncodeAPICreateInstance — check
// with `nm -g C:/ffmpeg/lib/nvidia-encode.lib`), so it cannot be declared in the
// `#[link]` block above: doing so fails at link time with LNK2019.  It is
// resolved from the driver DLL at runtime instead.
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
extern "system" {
    // Signature kept byte-compatible with the declaration in
    // `crate::interop::capability` (*const u8) so rustc does not warn about
    // clashing extern declarations for the same symbol.
    fn LoadLibraryA(name: *const u8) -> *mut std::ffi::c_void;
    fn GetProcAddress(
        module: *mut std::ffi::c_void,
        name:   *const u8,
    ) -> *mut std::ffi::c_void;
}

/// Ask the driver for its maximum supported NVENC API version.
///
/// Returns `(major, minor)`.  The driver packs the answer as
/// `major | (minor << 4)` — note the 4-bit shift, which is NOT the
/// NVENCAPI_VERSION layout (that one shifts the minor by 24), so the two must
/// not be conflated.  See [`nvenc_api_version`] for the word the structs want.
///
/// Returns `None` when the symbol cannot be resolved (no driver, or a platform
/// where it is unavailable); callers should fall back to probing versions with
/// `NvEncodeAPICreateInstance`.
#[cfg(target_os = "windows")]
pub fn query_max_supported_api_version() -> Option<(u32, u32)> {
    type GetMaxVer = unsafe extern "C" fn(*mut u32) -> i32;

    let sym = unsafe {
        let module = LoadLibraryA(c"nvEncodeAPI64.dll".as_ptr() as *const u8);
        if module.is_null() {
            return None;
        }
        let sym = GetProcAddress(
            module,
            c"NvEncodeAPIGetMaxSupportedVersion".as_ptr() as *const u8,
        );
        if sym.is_null() {
            return None;
        }
        std::mem::transmute::<*mut std::ffi::c_void, GetMaxVer>(sym)
    };

    let mut packed: u32 = 0;
    let ret = unsafe { sym(&mut packed) };
    if ret != 0 {
        return None;
    }
    // major in bits 4..11, minor in the low 4 bits: the driver on this machine
    // reports 0x000000C2 for API 12.2.
    Some(((packed >> 4) & 0xFF, packed & 0xF))
}

#[cfg(not(target_os = "windows"))]
pub fn query_max_supported_api_version() -> Option<(u32, u32)> { None }

#[cfg(test)]
mod version_word_tests {
    use super::*;

    /// API 12.2 — the version the installed driver reports and the version every
    /// struct word below was checked against by calling the driver directly
    /// (P3.2).  A word the driver rejects fails with NV_ENC_ERR_INVALID_VERSION
    /// (15) and reports nothing else, so these are worth pinning.
    const API_12_2: u32 = 0x0200_000C;

    #[test]
    fn api_version_word_carries_minor_in_bits_24_31() {
        assert_eq!(nvenc_api_version(12, 2), API_12_2);
        // The bare major is what the old code sent; it is NOT a valid API word.
        assert_ne!(nvenc_api_version(12, 2), 12);
    }

    #[test]
    fn struct_version_words_match_the_vendor_header() {
        // Values taken from ffnvcodec nvEncodeAPI.h (n12.2.72.0) and confirmed
        // accepted by the driver.
        assert_eq!(nv_encode_api_function_list_ver(API_12_2),        0x7202_000C);
        assert_eq!(nv_enc_open_encode_session_ex_params_ver(API_12_2), 0x7201_000C);
        assert_eq!(nv_enc_initialize_params_ver(API_12_2),           0xF207_000C);
        assert_eq!(nv_enc_register_resource_ver(API_12_2),           0x7205_000C);
        assert_eq!(nv_enc_map_input_resource_ver(API_12_2),          0x7204_000C);
        assert_eq!(nv_enc_pic_params_ver(API_12_2),                  0xF207_000C);
        assert_eq!(nv_enc_lock_bitstream_ver(API_12_2),              0xF202_000C);
        assert_eq!(nv_enc_create_bitstream_buffer_ver(API_12_2),     0x7201_000C);
        assert_eq!(nv_enc_event_params_ver(API_12_2),                0x7202_000C);
        assert_eq!(nv_enc_config_ver(API_12_2),                      0xF209_000C);
        assert_eq!(nv_enc_preset_config_ver(API_12_2),               0xF205_000C);
    }

    #[test]
    fn buffer_format_abgr10_is_the_bit_flag_not_a_counter() {
        // NV_ENC_BUFFER_FORMAT is a sparse bit-flag enum; the driver echoes
        // 0x20000000 back in NV_ENC_MAP_INPUT_RESOURCE::mappedBufferFmt.
        assert_eq!(NV_ENC_BUFFER_FORMAT_ABGR10, 0x2000_0000);
        assert_eq!(NV_ENC_BUFFER_FORMAT_ABGR,   0x1000_0000);
        assert_eq!(NV_ENC_BUFFER_FORMAT_ARGB10, 0x0200_0000);
        assert_ne!(NV_ENC_BUFFER_FORMAT_ABGR10, 2);
    }
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
    /// nvEncGetEncodePresetConfigEx — the tuning-info-aware variant, at
    /// vtable slot 40.  The P1..P7 presets only have a defined config in
    /// combination with a tuningInfo, so the non-Ex call cannot be used for them.
    pub type GetPresetConfigEx = unsafe extern "C" fn(
        NvEncodeSession,
        [u8; 16],   // encodeGUID
        [u8; 16],   // presetGUID
        u32,        // NV_ENC_TUNING_INFO
        *mut std::ffi::c_void, // NV_ENC_PRESET_CONFIG*
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
