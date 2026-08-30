// src/interop/encode_interop.rs
// Replaces Phase 6's FrameReadback CPU round-trip with a direct NVENC encode path.
// A shared D3D12 buffer holding NV12 is registered with NVENC as a CUdeviceptr,
// and encoding is entirely GPU-side.
//
// P1.9 (step 3) — WHAT CHANGED AND WHY.  This used to hand NVENC packed RGB
// (`Abgr10RepackNode` → NV_ENC_BUFFER_FORMAT_ABGR10 on a CUarray) and let the
// DRIVER convert RGB→YUV.  The driver applies BT.601 and there is no supported
// way to tell it otherwise: selecting the matrix means writing
// NV_ENC_CONFIG's per-codec VUI union, i.e. guessed struct offsets.  Meanwhile
// `Muxer::open` tags the stream from `job.output_color`, BT.709 for every normal
// export — so samples and tags disagreed (red decoding as `[255, 25, 0]`) and
// `ExportJob::nvenc_zero_copy_is_colour_safe` had to route every non-BT.601 job
// away from zero-copy entirely.
//
// Now `Nv12EncodeNode` performs the conversion in our own shader, writing NV12
// into a `SharedBuffer` that NVENC reads directly.  The encoder performs no
// matrix conversion at all, so our tags are authoritative by construction and the
// gate opens for every matrix.
//
// THE THREE MEASURED FACTS THIS FILE DEPENDS ON (RTX 3050, NVENC API 12.2,
// headers n12.2.72.0; probes `nvchk/d3d12_buf_probe.c`, `nvchk/nv12_probe.c`):
//
//   * A D3D12 shared BUFFER imported via `cuExternalMemoryGetMappedBuffer` and
//     registered as NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR with
//     `bufferFormat = NV12` encodes and decodes to the right pixels.
//   * The chroma plane is read at `pitch * height`, NOT `width * height`.
//     Verified with pitch 320 on a 256-wide frame, with the inter-row padding
//     filled 0xAA so a misread would have decoded as garbage.
//   * `pitch` is NOT ignored for this resource type.  It was for the old packed
//     single-plane ABGR10 CUarray — {0, W, W*4} all produced byte-identical
//     bitstreams — and generalising that measurement to the two-plane case is
//     what a `pitch = 0` NV12 registration turns into: a process kill inside
//     `nvEncEncodePicture` with no error return.  Both `pitch` here and
//     `NV_ENC_PIC_PARAMS::inputPitch` carry the real stride.

use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::external_buffer::SharedBuffer;
use crate::interop::nv12_encode::Nv12EncodeNode;
use crate::interop::ffi::nvenc::*;
use crate::interop::ffi::cuda_driver::{cuCtxPushCurrent, cuCtxPopCurrent, CUcontext};
use crate::render::device::GpuDevice;
use crate::export::job::{ExportJob, VideoCodec};
use std::sync::Arc;

/// A wgpu compute node that repacks RGBA16Float (already tone-mapped)
/// into ABGR10 format that NVENC accepts directly.
///
/// ABGR10 packs: packed = (a2 << 30) | (b10 << 20) | (g10 << 10) | r10
/// where each 10-bit channel = round(clamp(f32, 0.0, 1.0) * 1023.0) as u32,
/// and the 2-bit alpha is fixed at 0b11 (fully opaque).
///
/// **No longer on the export path.**  Handing NVENC packed RGB meant the DRIVER
/// performed the RGB→YUV conversion, always with BT.601 and with no supported way
/// to say otherwise, so every BT.709 job had to be routed away from zero-copy to
/// avoid shipping a file whose samples and tags disagreed.  `Nv12EncodeNode`
/// (`src/interop/nv12_encode.rs`) now does the conversion in our own shader and
/// NVENC receives NV12, which is what let `nvenc_zero_copy_is_colour_safe` open
/// up for every matrix.
///
/// Kept because it is still the only 10-bit-per-channel packing in the tree and
/// `src/tests/abgr10_repack.rs` still proves its arithmetic — a future P010 HDR
/// path is the obvious reuse.  It is NOT dead-code-allowed on a guess: if it ends
/// up genuinely unused, delete it and its test together rather than leaving a
/// second encode path that nothing exercises.
pub struct Abgr10RepackNode {
    pub in_rgba:       crate::render::resource::ResourceId,
    pub out_abgr10:    crate::render::resource::ResourceId,
    pipeline:          std::sync::Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
}

// WGSL shader for ABGR10 repack
pub const ABGR10_REPACK_WGSL: &str = r#"
@group(0) @binding(0) var in_rgba: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_packed: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(in_rgba);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let rgba = textureLoad(in_rgba, vec2<i32>(gid.xy));
    let r10 = u32(clamp(rgba.r, 0.0, 1.0) * 1023.0 + 0.5);
    let g10 = u32(clamp(rgba.g, 0.0, 1.0) * 1023.0 + 0.5);
    let b10 = u32(clamp(rgba.b, 0.0, 1.0) * 1023.0 + 0.5);
    let a2  = 3u; // fully opaque

    let packed = (a2 << 30u) | (b10 << 20u) | (g10 << 10u) | r10;
    textureStore(out_packed, vec2<i32>(gid.xy), vec4<u32>(packed, 0u, 0u, 0u));
}
"#;

impl Abgr10RepackNode {
    pub fn new(
        device:    &GpuDevice,
        in_rgba:   crate::render::resource::ResourceId,
        out_abgr10: crate::render::resource::ResourceId,
    ) -> Self {
        let bgl = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Abgr10Repack BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Abgr10Repack"),
            source: wgpu::ShaderSource::Wgsl(ABGR10_REPACK_WGSL.into()),
        });
        let pl = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Abgr10Repack PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = std::sync::Arc::new(device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Abgr10Repack Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        }));

        Self { in_rgba, out_abgr10, pipeline, bind_group_layout: bgl }
    }

    pub fn record(
        &self,
        encoder:  &mut wgpu::CommandEncoder,
        device:   &GpuDevice,
        in_view:  &wgpu::TextureView,
        out_view: &wgpu::TextureView,
        width:    u32,
        height:   u32,
    ) {
        let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Abgr10Repack BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(in_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Abgr10Repack"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
}

#[derive(Debug)]
pub enum EncodeInteropError {
    ApiLoad,
    SessionOpen(i32),
    Initialize(i32),
    Register(i32),
    Map(i32),
    Encode(i32),
    Lock(i32),
    Cuda(CudaError),
}

impl std::fmt::Display for EncodeInteropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A bare integer is what made an OUT_OF_MEMORY(10) get read as a
        // parameter problem for a whole session; always print the name.
        let named = |f: &mut std::fmt::Formatter<'_>, call: &str, s: i32| {
            write!(f, "{call} failed: {} ({s})", nvenc_status_str(s))
        };
        match self {
            Self::ApiLoad         => write!(f, "NvEncodeAPICreateInstance failed"),
            Self::SessionOpen(s)  => named(f, "nvEncOpenEncodeSessionEx", *s),
            Self::Initialize(s)   => named(f, "nvEncInitializeEncoder", *s),
            Self::Register(s)     => named(f, "nvEncRegisterResource", *s),
            Self::Map(s)          => named(f, "nvEncMapInputResource", *s),
            Self::Encode(s)       => named(f, "nvEncEncodePicture", *s),
            Self::Lock(s)         => named(f, "nvEncLockBitstream", *s),
            Self::Cuda(e)         => write!(f, "CUDA interop failed: {e:?}"),
        }
    }
}

impl std::error::Error for EncodeInteropError {}

/// One encoded picture handed back by NVENC.
///
/// `pts` is the presentation timestamp NVENC echoed back for the picture
/// (NV_ENC_LOCK_BITSTREAM::outputTimestamp, i.e. the value passed as
/// NV_ENC_PIC_PARAMS::inputTimestamp), while `dts` is a decode-order timestamp
/// generated by [`DtsQueue`] — the two differ as soon as the encoder reorders.
pub struct EncodedPacket {
    pub bytes:       Vec<u8>,
    pub pts:         i64,
    pub dts:         i64,
    /// True when NVENC reported NV_ENC_PIC_TYPE_I / _IDR for this picture, so
    /// the muxer must mark the packet as a sync sample (AV_PKT_FLAG_KEY).
    pub is_keyframe: bool,
}

/// Generates monotonically increasing DTS values for a reordering encoder.
///
/// NVENC is fed pictures in presentation order and returns bitstreams in
/// **decode** order, so the Nth packet out is not the Nth picture in.  Writing
/// `dts = pts` (what this code did before) produces a non-monotonic DTS stream
/// the moment B-frames appear, which mp4/matroska reject or mis-index.
///
/// The fix is the same one libavcodec's nvenc wrapper uses: keep the submitted
/// timestamps in a FIFO and pop one per emitted packet.  Because the inputs
/// arrive in presentation order, popping in output order yields a
/// non-decreasing sequence that is always ≤ the packet's own PTS.
#[derive(Default)]
pub struct DtsQueue {
    submitted: std::collections::VecDeque<i64>,
    last_dts:  Option<i64>,
}

impl DtsQueue {
    /// Record the timestamp of a picture handed to `nvEncEncodePicture`.
    pub fn push_input(&mut self, pts: i64) {
        self.submitted.push_back(pts);
    }

    /// Take the DTS for the next emitted packet.
    ///
    /// Falls back to `fallback_pts` when the FIFO has run dry (which would mean
    /// NVENC emitted more packets than pictures were submitted), and clamps the
    /// result so the returned sequence can never step backwards.
    pub fn pop_output(&mut self, fallback_pts: i64) -> i64 {
        let mut dts = self.submitted.pop_front().unwrap_or(fallback_pts);
        if let Some(last) = self.last_dts {
            if dts <= last {
                dts = last + 1;
            }
        }
        self.last_dts = Some(dts);
        dts
    }

    pub fn pending(&self) -> usize {
        self.submitted.len()
    }
}

/// NVENC function table loaded from the driver.
struct NvencFunctions {
    open_session:          functions::OpenEncodeSessionEx,
    initialize:            functions::InitializeEncoder,
    create_bitstream:      functions::CreateBitstreamBuffer,
    destroy_bitstream:     functions::DestroyBitstreamBuffer,
    register_resource:     functions::RegisterResource,
    map_input:             functions::MapInputResource,
    unmap_input:           functions::UnmapInputResource,
    encode_picture:        functions::EncodePicture,
    lock_bitstream:        functions::LockBitstream,
    unlock_bitstream:      functions::UnlockBitstream,
    destroy_encoder:       functions::DestroyEncoder,
    register_async_event:  functions::RegisterAsyncEvent,
    unregister_async_event: functions::UnregisterAsyncEvent,
    /// nvEncGetEncodePresetConfigEx (vtable slot 40).  `None` when the driver's
    /// function table has a null there, in which case the encoder falls back to
    /// initialising with `encodeConfig = NULL` (preset defaults).
    get_preset_config_ex:  Option<functions::GetPresetConfigEx>,
}

/// How many frames may be in flight inside NVENC at once (P1.1).
///
/// One slot is one ABGR10 interop texture + one NVENC registered resource + one
/// bitstream buffer + one completion event, so the pipeline can hold this many
/// *independent* frames: the GPU renders frame N into slot N % SLOTS while NVENC
/// is still encoding frames N-1 … N-SLOTS+1 out of the other slots.
///
/// The old value was effectively 1 — the encoder waited for each frame's
/// completion event before returning from `encode_frame` — which serialised
/// render → wait → encode.  4 is enough to keep NVENC saturated on a single
/// encoder engine while costing only `4 * width * height * 4` bytes of VRAM
/// (33 MB at 1080p, 133 MB at 4K).
pub const NVENC_INFLIGHT_SLOTS: usize = 4;

/// One pipeline slot: everything one in-flight frame needs, owned per-slot so
/// two frames never share a resource.
struct EncodeSlot {
    /// Shared allocation: the NV12 buffer (written by `Nv12EncodeNode`'s two
    /// compute passes) together with its CUDA import.  A single
    /// `D3D12_HEAP_FLAG_SHARED` resource seen from both sides — see
    /// [`SharedBuffer`].
    shared:     SharedBuffer,
    /// NVENC registered resource handle for `shared`'s CUDA device pointer.
    registered: *mut std::ffi::c_void,
    /// Pre-allocated bitstream output buffer for this slot.
    bitstream:  *mut std::ffi::c_void,
    /// Win32 completion event for this slot (null when async mode is off).
    event:      *mut std::ffi::c_void,
}

/// Holds the NVENC session and its [`NVENC_INFLIGHT_SLOTS`] interop-imported
/// NV12 buffers.
pub struct EncodeInterop {
    session:               NvEncodeSession,
    /// The pipeline slots, indexed by the `slot` argument of
    /// [`EncodeInterop::encode_frame`].
    slots:                 Vec<EncodeSlot>,
    funcs:                 NvencFunctions,
    width:                 u32,
    height:                u32,
    /// Row stride of both NV12 planes, in bytes: what the shader was told to
    /// write, what the registration declared, and what every
    /// `NV_ENC_PIC_PARAMS::inputPitch` carries.  One field so the three can never
    /// disagree — a chroma plane written at one stride and read at another is the
    /// failure mode this whole step exists to avoid.
    pitch:                 u32,
    /// The NVENC API major version we probed successfully; used to construct
    /// per-struct version fields for encode-time calls.
    api_version:           u32,
    /// Dedicated completion event for the EOS picture, so `flush` never has to
    /// borrow a slot event that a still-in-flight frame owns.
    eos_event:             *mut std::ffi::c_void,
    /// Whether asynchronous NVENC encoding is active.  False means sync mode
    /// (every event handle is null and no WaitForSingleObject calls are made;
    /// `nvEncLockBitstream` blocks instead).
    is_async:              bool,
    /// DTS generator: NVENC returns packets in decode order, so a decode
    /// timestamp cannot be derived from the picture's own PTS.  See [`DtsQueue`].
    dts_queue:             DtsQueue,
    /// True when the session was initialised with an explicit NV_ENC_CONFIG
    /// pinning `frameIntervalP = 1` (no B-frames), so output order is guaranteed
    /// to match input order.  False means the driver's preset defaults are in
    /// effect and reordering is possible — the DTS queue handles both.
    reorder_disabled:      bool,
    /// Backing store for the NV_ENC_CONFIG handed to `nvEncInitializeEncoder`.
    ///
    /// The driver copies the config during initialisation, so this is not read
    /// after `open` returns; it is retained only so the pointer stayed valid for
    /// the whole init call and so `reorder_disabled` can be explained in logs.
    #[allow(dead_code)]
    preset_config:         Option<PresetConfig>,
    /// Bitstream buffers holding output NVENC has accepted but this code has not
    /// locked yet, oldest first, together with the input mapping that produced
    /// each one and the completion event that signals it.
    ///
    /// NVENC retrieval is FIFO, and an input surface must stay mapped until its
    /// bitstream has been read, so all three travel together.  With
    /// [`NVENC_INFLIGHT_SLOTS`] slots this holds up to that many entries; a slot
    /// is only reusable once its entry has been drained, which is what
    /// [`EncodeInterop::wait_for_slot`] enforces.
    pending:               std::collections::VecDeque<PendingOutput>,
    /// True once the EOS picture has been submitted.  No further pictures may be
    /// submitted afterwards, and a second flush is a no-op.
    eos_sent:              bool,
    /// True once an NVENC call failed in a way that makes the session unusable.
    ///
    /// A poisoned session accepts no further pictures, flushes nothing, and —
    /// crucially — has its teardown reduced to `nvEncDestroyEncoder` alone: the
    /// unmap/unlock/destroy-bitstream calls `Drop` would otherwise make are what
    /// turned a plain NVENC error into STATUS_ACCESS_VIOLATION, because they run
    /// against a session the driver has already torn down internally.  The
    /// export then fails cleanly instead of killing the process.
    poisoned:              bool,
    // -----------------------------------------------------------------------
    // KEEP THIS FIELD LAST.  Rust drops struct fields in declaration order, and
    // everything above it — the NVENC session and every `SharedBuffer` with its
    // CUDA import — is only valid while this context lives.
    // -----------------------------------------------------------------------
    /// A share of the CUDA primary context this session was opened against.
    ///
    /// Load-bearing in two separate ways, both learned the hard way:
    ///
    /// 1. **Lifetime.**  `nvEncOpenEncodeSessionEx` stores the CUcontext inside
    ///    the session; every later call (register / map / EncodePicture /
    ///    LockBitstream) runs against it.  When `open` took `&CudaContext` and
    ///    kept nothing, the last `Arc` lived in `ExportEngine` — which
    ///    `start(self)` consumes — so `cuDevicePrimaryCtxRelease` destroyed the
    ///    context mid-export and the next `nvEncEncodePicture` returned
    ///    NV_ENC_ERR_OUT_OF_MEMORY (10).  That 10 was misread as a pitch/format
    ///    problem for a whole session.  Reproduced standalone: releasing the
    ///    primary context makes even `nvEncOpenEncodeSessionEx` fail with
    ///    NV_ENC_ERR_UNSUPPORTED_DEVICE (2).
    /// 2. **Drop order.**  Declared second (right after `session`) it dropped
    ///    *before* the shared allocations, so `cuMemFree` /
    ///    `cuDestroyExternalMemory` ran against a dead context and the crash
    ///    simply moved from frame 0 to end-of-teardown.  `ExternalBuffer` now
    ///    holds its own `Arc` as well, so correctness no longer *depends* on this
    ///    position — but keeping it last is still the honest expression of the
    ///    invariant, and costs nothing.
    #[allow(dead_code)]
    cuda_ctx:              Arc<CudaContext>,
}

/// One in-flight NVENC output: the bitstream buffer to lock, the mapped input
/// resource to release once it has been read, and (async mode) the Win32 event
/// NVENC signals when the picture is done.
struct PendingOutput {
    bitstream:    *mut std::ffi::c_void,
    mapped_input: *mut std::ffi::c_void,
    /// Completion event for this picture, or null in sync mode.  Owned by the
    /// slot, not by this entry — never closed from here.
    event:        *mut std::ffi::c_void,
    /// Which pipeline slot produced this output, so `wait_for_slot` can tell
    /// whether draining one entry frees the slot it wants.
    slot:         usize,
}

unsafe impl Send for EncodeInterop {}
/// WAIT_OBJECT_0: WaitForSingleObject returned because the object was signalled.
#[cfg(target_os = "windows")]
const WAIT_OBJECT_0: u32 = 0x00000000;

/// How long to wait for a single picture's completion event before declaring the
/// encoder wedged.
///
/// Generous on purpose: with [`NVENC_INFLIGHT_SLOTS`] frames in flight the wait
/// covers a queue of pictures, not one, and a 4K B-frame GOP on a busy GPU can
/// legitimately take well over a second.  A real hang still fails within 10s
/// rather than blocking the export thread forever.
#[cfg(target_os = "windows")]
const ENCODE_WAIT_TIMEOUT_MS: u32 = 10_000;

// Win32 kernel32 functions needed for asynchronous NVENC completion events.
// These are the standard Windows synchronization APIs; we declare them here
// rather than pulling in the `windows-sys` crate to avoid a new dependency.
#[cfg(target_os = "windows")]
extern "system" {
    fn CreateEventA(
        lp_event_attributes: *mut std::ffi::c_void,
        b_manual_reset:      i32,
        b_initial_state:     i32,
        lp_name:             *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;

    fn WaitForSingleObject(
        h_handle:         *mut std::ffi::c_void,
        dw_milliseconds:  u32,
    ) -> u32;

    /// Signal an event.  Used only to put back a signal that a zero-timeout
    /// readiness probe consumed — see `EncodeInterop::head_is_ready`.
    fn SetEvent(h_event: *mut std::ffi::c_void) -> i32;

    fn CloseHandle(h_object: *mut std::ffi::c_void) -> i32;
}

// NV_ENC_PIC_STRUCT_FRAME — all exported frames are progressive frames.
const NV_ENC_PIC_STRUCT_FRAME: u32 = 1;

// Struct version words are built by the helpers in
// `crate::interop::ffi::nvenc` (nv_enc_*_ver), which encode the per-struct layout
// numbers verified against the vendor header in P3.2.  Do not hand-roll them:
// the numbers differ per struct, some ORs in `1 << 31`, and the API version word
// must carry the minor version in bits 24..31.
// H264 GUID: {6BC82762-4E63-4ca4-AA85-1E50F321F6BF}
const NV_ENC_CODEC_H264_GUID: [u8; 16] = [
    0x62, 0x27, 0xC8, 0x6B, 0x63, 0x4E, 0xa4, 0x4c,
    0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF,
];
// HEVC GUID: {790CDC88-4522-4d7b-9425-BDA9975F7603}
//
// The first four bytes are `Data1` in LITTLE-ENDIAN order, so 0x790CDC88 is
// `88 DC 0C 79`.  This constant read `88 CD 0C 79` until P1.2 — one transposed
// nibble, and the only symptom was `nvEncInitializeEncoder` returning
// NV_ENC_ERR_UNSUPPORTED_PARAM (12) for every H.265 job at every resolution,
// which reads exactly like "this GPU cannot do HEVC" and silently routed every
// H.265 export to the FFmpeg encoder.
//
// Both GUIDs are checked by `nvchk/nv12_probe.c --codec h264|hevc`, which diffs
// these exact bytes against the vendor header's own `static const GUID` and then
// initialises a session with THESE bytes rather than the header's — so a typo
// here reproduces the driver failure in the probe instead of being masked by it.
// Verified 2026-08-30: match, NV_ENC_SUCCESS, and the decoded bars exact; with
// one nibble flipped back, UNSUPPORTED_PARAM (12).  See `nvchk/README.md`.
const NV_ENC_CODEC_HEVC_GUID: [u8; 16] = [
    0x88, 0xDC, 0x0C, 0x79, 0x22, 0x45, 0x7B, 0x4d,
    0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03,
];
// Preset GUIDs live in `crate::interop::ffi::nvenc` (NV_ENC_PRESET_P1/P4/P7_GUID),
// verified against the vendor header and against the list the installed driver
// returns from nvEncGetEncodePresetGUIDs.  P1 is the fastest preset and is used
// as the fallback when P4 is rejected.

impl EncodeInterop {
    /// Open an NVENC session against the shared CUDA context and register one
    /// NV12 interop buffer per pipeline slot as NVENC's input resource.
    ///
    /// `codec` controls whether to initialise an H.264 or HEVC session;
    /// both codecs use the same NV12 input format and P4 preset.
    ///
    /// Takes the context by `Arc` and keeps a clone: the NVENC session is only
    /// valid while the CUDA primary context it was opened against is alive (see
    /// [`EncodeInterop::cuda_ctx`]).  A `&CudaContext` would let the caller drop
    /// the last owner mid-export, which is exactly the bug this signature
    /// prevents from recurring.
    pub fn open(
        cuda_ctx_arc: Arc<CudaContext>,
        device:       &GpuDevice,
        job:          &ExportJob,
        transport:    crate::interop::capability::InteropTransport,
        codec:        VideoCodec,
    ) -> Result<Self, EncodeInteropError> {
        let cuda_ctx: &CudaContext = &cuda_ctx_arc;
        let fn_table_size = std::mem::size_of::<usize>() * 512;
        let mut fn_table_raw = vec![0u8; fn_table_size];

        // Ask the driver which API version it supports instead of probing blind.
        // The driver packs the answer as `major | (minor << 4)`, which is NOT the
        // NVENCAPI_VERSION layout (that shifts minor by 24), so it has to be
        // unpacked and rebuilt — see `query_max_supported_api_version`.
        //
        // Falling back to 12.2 keeps this working against an import stub that
        // lacks the symbol; 12.2 is what the driver on this machine reports and
        // is the version the struct layout numbers were verified against.
        let (driver_major, driver_minor) = match query_max_supported_api_version() {
            Some(v) => v,
            None => {
                log::warn!(
                    "[export] NvEncodeAPIGetMaxSupportedVersion unavailable — assuming API 12.2"
                );
                (12, 2)
            }
        };

        // The API version word every struct's version field is built from.
        let probed_api_version = nvenc_api_version(driver_major, driver_minor);
        log::info!(
            "[export] NVENC driver supports API {}.{} (NVENCAPI_VERSION=0x{:08x})",
            driver_major, driver_minor, probed_api_version
        );

        unsafe {
            let ptr = fn_table_raw.as_mut_ptr() as *mut u32;
            *ptr = nv_encode_api_function_list_ver(probed_api_version);
        }
        let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;
        let ret = unsafe { NvEncodeAPICreateInstance(function_list) };
        if ret != NV_ENC_SUCCESS {
            log::error!("[export] NvEncodeAPICreateInstance failed with error code {}", ret);
            return Err(EncodeInteropError::ApiLoad);
        }

        // Pointer offsets (base.add(N)) into NV_ENCODE_API_FUNCTION_LIST.
        // Verified (P3.2) by taking offsetof() on the real vendor struct from
        // ffnvcodec and dividing by sizeof(void*):
        //   base.add(0)  = version (u32) + reserved (u32)
        //   base.add(1)  = nvEncOpenEncodeSession
        //   base.add(9)  = nvEncGetEncodePresetCount
        //   base.add(10) = nvEncGetEncodePresetGUIDs
        //   base.add(11) = nvEncGetEncodePresetConfig   (NOT 10)
        //   base.add(12) = nvEncInitializeEncoder
        //   base.add(15) = nvEncCreateBitstreamBuffer
        //   base.add(16) = nvEncDestroyBitstreamBuffer
        //   base.add(17) = nvEncEncodePicture
        //   base.add(18) = nvEncLockBitstream
        //   base.add(19) = nvEncUnlockBitstream
        //   base.add(24) = nvEncRegisterAsyncEvent
        //   base.add(25) = nvEncUnregisterAsyncEvent
        //   base.add(26) = nvEncMapInputResource
        //   base.add(27) = nvEncUnmapInputResource
        //   base.add(28) = nvEncDestroyEncoder
        //   base.add(30) = nvEncOpenEncodeSessionEx
        //   base.add(31) = nvEncRegisterResource
        //   base.add(40) = nvEncGetEncodePresetConfigEx
        let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;
        let base = function_list as *const usize;
        let funcs = unsafe {
            let open_off              = 30usize; // nvEncOpenEncodeSessionEx
            let init_off              = 12usize; // nvEncInitializeEncoder
            let create_bs_off         = 15usize; // nvEncCreateBitstreamBuffer
            let destroy_bs_off        = 16usize; // nvEncDestroyBitstreamBuffer
            let enc_off               = 17usize; // nvEncEncodePicture
            let lock_off              = 18usize; // nvEncLockBitstream
            let unlock_off            = 19usize; // nvEncUnlockBitstream
            let reg_async_off         = 24usize; // nvEncRegisterAsyncEvent
            let unreg_async_off       = 25usize; // nvEncUnregisterAsyncEvent
            let map_off               = 26usize; // nvEncMapInputResource
            let unmap_off             = 27usize; // nvEncUnmapInputResource
            let destroy_off           = 28usize; // nvEncDestroyEncoder
            let reg_off               = 31usize; // nvEncRegisterResource
            let preset_cfg_ex_off     = 40usize; // nvEncGetEncodePresetConfigEx

            let preset_cfg_ex_raw = *base.add(preset_cfg_ex_off);
            let get_preset_config_ex = if preset_cfg_ex_raw == 0 {
                log::warn!(
                    "[export] nvEncGetEncodePresetConfigEx is NULL in the function table — \
                     falling back to preset default encode config"
                );
                None
            } else {
                Some(std::mem::transmute::<usize, functions::GetPresetConfigEx>(preset_cfg_ex_raw))
            };

            NvencFunctions {
                open_session:           std::mem::transmute(*base.add(open_off)),
                initialize:             std::mem::transmute(*base.add(init_off)),
                create_bitstream:       std::mem::transmute(*base.add(create_bs_off)),
                destroy_bitstream:      std::mem::transmute(*base.add(destroy_bs_off)),
                register_resource:      std::mem::transmute(*base.add(reg_off)),
                map_input:              std::mem::transmute(*base.add(map_off)),
                unmap_input:            std::mem::transmute(*base.add(unmap_off)),
                encode_picture:         std::mem::transmute(*base.add(enc_off)),
                lock_bitstream:         std::mem::transmute(*base.add(lock_off)),
                unlock_bitstream:       std::mem::transmute(*base.add(unlock_off)),
                destroy_encoder:        std::mem::transmute(*base.add(destroy_off)),
                register_async_event:   std::mem::transmute(*base.add(reg_async_off)),
                unregister_async_event: std::mem::transmute(*base.add(unreg_async_off)),
                get_preset_config_ex,
            }
        };

        let fn_ptr_open    = unsafe { *base.add(30) };
        let fn_ptr_init    = unsafe { *base.add(12) };
        let fn_ptr_reg     = unsafe { *base.add(31) };
        let fn_ptr_destroy = unsafe { *base.add(28) };
        log::info!("[export] vtable[30]=0x{:x}  vtable[12]=0x{:x}  vtable[31]=0x{:x}  vtable[28]=0x{:x}",
            fn_ptr_open, fn_ptr_init, fn_ptr_reg, fn_ptr_destroy);
        if fn_ptr_open == 0 {
            log::error!("[export] nvEncOpenEncodeSessionEx is NULL in function table — driver too old?");
            return Err(EncodeInteropError::ApiLoad);
        }

        // Step 2 — Open the encode session against our shared CUDA context.
        let open_params_ver = nv_enc_open_encode_session_ex_params_ver(probed_api_version);
        let mut session: NvEncodeSession = std::ptr::null_mut();
        let params = NvEncOpenEncodeSessionExParams {
            version:     open_params_ver,
            device_type: NV_ENC_DEVICE_TYPE_CUDA,
            device:      cuda_ctx.raw_context() as *mut _,
            reserved:    std::ptr::null_mut(),
            api_version: probed_api_version,
            reserved1:   [0u32; 253],
            reserved2:   [std::ptr::null_mut(); 64],
        };
        log::info!("[export] Opening NVENC session (params.version=0x{:08x}, api_version={})",
            open_params_ver, probed_api_version);
        // Push CUDA context onto this thread's stack before calling into NVENC driver.
        let mut _popped_ctx: CUcontext = std::ptr::null_mut();
        unsafe { cuCtxPushCurrent(cuda_ctx.raw_context()); }
        let ret = unsafe { (funcs.open_session)(&params, &mut session) };
        unsafe { cuCtxPopCurrent(&mut _popped_ctx); }
        if ret != NV_ENC_SUCCESS {
            log::error!("[export] nvEncOpenEncodeSessionEx returned error {}", ret);
            return Err(EncodeInteropError::SessionOpen(ret));
        }
        log::info!("[export] NVENC session opened successfully");

        // Step 3 — Initialise the encoder (codec, preset, dimensions, frame rate).
        let encode_guid = match codec {
            VideoCodec::H265 => NV_ENC_CODEC_HEVC_GUID,
            _ => NV_ENC_CODEC_H264_GUID,
        };
        let init_params_ver = nv_enc_initialize_params_ver(probed_api_version);
        let mut init_params = NvEncInitializeParams {
            version:                      init_params_ver,
            encode_guid,
            preset_guid:                  NV_ENC_PRESET_P4_GUID,
            encode_width:                 job.width,
            encode_height:                job.height,
            dar_width:                    job.width,
            dar_height:                   job.height,
            frame_rate_num:               job.frame_rate.num as u32,
            frame_rate_den:               job.frame_rate.den as u32,
            enable_encode_async:          0, // set to 1 below on Windows if driver allows it
            enable_ptd:                   1, // let NVENC pick frame types
            flags:                        0,
            priv_data_size:               0,
            reserved_u32:                 0,
            priv_data:                    std::ptr::null_mut(),
            encode_config:                std::ptr::null_mut(),
            max_encode_width:             job.width,
            max_encode_height:            job.height,
            max_me_hint_counts_per_block: [0u32; 8],
            // tuningInfo is mandatory for the P1..P7 presets: leaving it
            // UNDEFINED makes nvEncInitializeEncoder fail with
            // NV_ENC_ERR_UNSUPPORTED_PARAM (verified against the driver in P3.2).
            tuning_info:                  NV_ENC_TUNING_INFO_HIGH_QUALITY,
            buffer_format:                0,
            num_state_buffers:            0,
            output_stats_level:           0,
            reserved1:                    [0u32; 284],
            reserved2:                    [std::ptr::null_mut(); 64],
        };

        // --- Initialization ---
        // Try the P4 preset first and fall back to P1 (both verified against the
        // driver's own nvEncGetEncodePresetGUIDs list in P3.2).  A failed
        // nvEncInitializeEncoder leaves the session un-initialised, so retrying on
        // the same session is legal; only a SUCCESSFUL init is one-shot.
        // `session` is passed in explicitly rather than captured: the async→sync
        // fallback below may destroy and reopen the session, which needs a mutable
        // borrow the closure would otherwise hold hostage.
        //
        // P1.3 — the encode config is no longer left at NULL.  With NULL the
        // driver applies the preset's own defaults, which may enable B-frames and
        // lookahead; a reordering encoder both returns bitstreams out of
        // presentation order and holds input surfaces past the EncodePicture
        // call, and the two-slot ping-pong pipeline in `render_segment` reuses
        // slot N-2 as soon as its submission completes.  So the driver's preset
        // config is fetched and `frameIntervalP` pinned to 1 (IPPP).  Every step
        // of that is best-effort: if the Ex entry point is missing, the query
        // fails, the struct layout check fails, or init is rejected with the
        // explicit config, it falls back to NULL and `DtsQueue` still keeps DTS
        // monotonic.
        let query_preset_config = |session: NvEncodeSession,
                                   preset_guid: [u8; 16]| -> Option<PresetConfig> {
            let get_ex = funcs.get_preset_config_ex?;
            let mut cfg = PresetConfig::new(probed_api_version);
            let ret = unsafe {
                get_ex(
                    session,
                    encode_guid,
                    preset_guid,
                    NV_ENC_TUNING_INFO_HIGH_QUALITY,
                    cfg.as_mut_ptr(),
                )
            };
            if ret != NV_ENC_SUCCESS {
                log::warn!(
                    "[export] nvEncGetEncodePresetConfigEx failed ({ret}) — \
                     using preset default encode config"
                );
                return None;
            }
            if !cfg.preset_cfg_version_matches() {
                // The NV_ENC_CONFIG offsets this code assumes do not match what
                // the driver wrote.  Patching at guessed offsets from here would
                // corrupt unrelated fields, so bail out to the NULL-config path.
                log::warn!(
                    "[export] NV_ENC_PRESET_CONFIG layout check failed \
                     (presetCfg.version=0x{:08x}, expected 0x{:08x}) — \
                     using preset default encode config",
                    cfg.preset_cfg_version(),
                    nv_enc_config_ver(probed_api_version)
                );
                return None;
            }
            log::info!(
                "[export] preset config: gopLength={}, frameIntervalP={} → pinning frameIntervalP=1",
                cfg.gop_length(),
                cfg.frame_interval_p()
            );
            // frameIntervalP = 1 means IPPP: no B-frames, so NVENC emits one
            // bitstream per submitted picture, in submission order.
            cfg.set_frame_interval_p(1);
            Some(cfg)
        };

        let try_init = |session: NvEncodeSession,
                        init_params: &mut NvEncInitializeParams,
                        async_mode: u32,
                        cfg_holder: &mut Option<PresetConfig>| -> i32 {
            init_params.enable_encode_async = async_mode;
            let mut last_ret = NV_ENC_SUCCESS;

            for (label, preset) in [
                ("P4", NV_ENC_PRESET_P4_GUID),
                ("P1", NV_ENC_PRESET_P1_GUID),
            ] {
                init_params.preset_guid = preset;

                // Attempt 1: explicit config with reordering disabled.
                *cfg_holder = query_preset_config(session, preset);
                init_params.encode_config = cfg_holder
                    .as_mut()
                    .map(|c| c.config_ptr())
                    .unwrap_or(std::ptr::null_mut());
                last_ret = unsafe { (funcs.initialize)(session, init_params) };
                if last_ret == NV_ENC_SUCCESS {
                    return last_ret;
                }

                // Attempt 2: same preset, preset-default config.  Distinguishes
                // "this preset is unsupported" from "our patched config is".
                if cfg_holder.is_some() {
                    log::warn!(
                        "[export] nvEncInitializeEncoder(async={async_mode}, preset={label}) \
                         rejected the explicit encode config ({last_ret}) — retrying with \
                         preset defaults (B-frames may be enabled)"
                    );
                    *cfg_holder = None;
                    init_params.encode_config = std::ptr::null_mut();
                    last_ret = unsafe { (funcs.initialize)(session, init_params) };
                    if last_ret == NV_ENC_SUCCESS {
                        return last_ret;
                    }
                }

                log::warn!(
                    "[export] nvEncInitializeEncoder(async={async_mode}) with {label} preset \
                     failed ({last_ret})"
                );
            }

            last_ret
        };

        // Holds the NV_ENC_CONFIG for whichever init attempt succeeded.  It must
        // stay alive for the whole `nvEncInitializeEncoder` call and is moved
        // into the returned struct afterwards.
        let mut preset_config: Option<PresetConfig> = None;

        // On Windows: attempt async NVENC mode.  The completion events are created
        // BEFORE initialising, so a failure to create them costs nothing — there is
        // no successfully-initialised session to unwind.  On non-Windows: always
        // synchronous (no Win32 event primitives available).
        //
        // P1.1 — one event PER PIPELINE SLOT plus one for the EOS picture, rather
        // than two shared between everything.  An event is an auto-reset object:
        // if two in-flight frames shared one, the wait for frame N could consume
        // the signal belonging to frame N+1 and the pipeline would deadlock on the
        // next wait.  `EVENT_COUNT` events make each wait unambiguous.
        const EVENT_COUNT: usize = NVENC_INFLIGHT_SLOTS + 1;

        #[cfg(target_os = "windows")]
        let (is_async, events) = unsafe {
            // Win32 auto-reset events (bManualReset=0, bInitialState=0).
            let mut evs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(EVENT_COUNT);
            for _ in 0..EVENT_COUNT {
                evs.push(CreateEventA(std::ptr::null_mut(), 0, 0, std::ptr::null()));
            }
            let close_all = |evs: &[*mut std::ffi::c_void]| {
                for &e in evs {
                    if !e.is_null() { CloseHandle(e); }
                }
            };

            let all_created = evs.iter().all(|e| !e.is_null());
            if !all_created {
                close_all(&evs);
                log::warn!("[export] CreateEventA failed — using synchronous NVENC mode");
                let ret = try_init(session, &mut init_params, 0, &mut preset_config);
                if ret != NV_ENC_SUCCESS {
                    log::error!("[export] nvEncInitializeEncoder (sync) failed with error {}", ret);
                    (funcs.destroy_encoder)(session);
                    return Err(EncodeInteropError::Initialize(ret));
                }
                (false, vec![std::ptr::null_mut(); EVENT_COUNT])
            } else {
                let async_ret = try_init(session, &mut init_params, 1, &mut preset_config);
                if async_ret != NV_ENC_SUCCESS {
                    // Async rejected; the session is still un-initialised, so a
                    // synchronous init on it is valid.
                    log::warn!(
                        "[export] NVENC async mode rejected by driver ({}) — falling back to sync",
                        async_ret
                    );
                    close_all(&evs);
                    let sync_ret = try_init(session, &mut init_params, 0, &mut preset_config);
                    if sync_ret != NV_ENC_SUCCESS {
                        log::error!("[export] nvEncInitializeEncoder (sync) failed with error {}", sync_ret);
                        (funcs.destroy_encoder)(session);
                        return Err(EncodeInteropError::Initialize(sync_ret));
                    }
                    (false, vec![std::ptr::null_mut(); EVENT_COUNT])
                } else {
                    // Register every event with the now-async session.
                    let event_params_ver = nv_enc_event_params_ver(probed_api_version);
                    let mut register_failure: Option<i32> = None;
                    let mut registered_count = 0usize;
                    for &ev in &evs {
                        let mut ep = NvEncEventParams {
                            version: event_params_ver,
                            completion_event: ev,
                            ..Default::default()
                        };
                        let r = (funcs.register_async_event)(session, &mut ep);
                        if r != NV_ENC_SUCCESS {
                            register_failure = Some(r);
                            break;
                        }
                        registered_count += 1;
                    }

                    if let Some(r) = register_failure {
                        // The session is already initialised in async mode and
                        // cannot be re-initialised, so it must be destroyed and a
                        // fresh one opened for the synchronous path.  Unregister
                        // whatever did get registered first: those handles belong
                        // to the session that is about to be destroyed, and
                        // destroy_encoder reclaims them, but unregistering keeps
                        // the driver's bookkeeping honest.
                        log::warn!(
                            "[export] nvEncRegisterAsyncEvent failed after {registered_count} \
                             event(s) ({r}) — reopening the session in sync mode"
                        );
                        for &ev in evs.iter().take(registered_count) {
                            let mut ep = NvEncEventParams {
                                version: event_params_ver,
                                completion_event: ev,
                                ..Default::default()
                            };
                            let _ = (funcs.unregister_async_event)(session, &mut ep);
                        }
                        close_all(&evs);
                        (funcs.destroy_encoder)(session);

                        let mut popped: CUcontext = std::ptr::null_mut();
                        cuCtxPushCurrent(cuda_ctx.raw_context());
                        let reopen = (funcs.open_session)(&params, &mut session);
                        cuCtxPopCurrent(&mut popped);
                        if reopen != NV_ENC_SUCCESS {
                            log::error!("[export] reopening NVENC session failed with error {}", reopen);
                            return Err(EncodeInteropError::SessionOpen(reopen));
                        }
                        let sync_ret = try_init(session, &mut init_params, 0, &mut preset_config);
                        if sync_ret != NV_ENC_SUCCESS {
                            log::error!("[export] nvEncInitializeEncoder (sync) failed with error {}", sync_ret);
                            (funcs.destroy_encoder)(session);
                            return Err(EncodeInteropError::Initialize(sync_ret));
                        }
                        (false, vec![std::ptr::null_mut(); EVENT_COUNT])
                    } else {
                        log::info!(
                            "[export] NVENC asynchronous encoding active \
                             ({} Win32 event(s) registered, {} in-flight slot(s))",
                            EVENT_COUNT, NVENC_INFLIGHT_SLOTS
                        );
                        (true, evs)
                    }
                }
            }
        };

        // Non-Windows: always synchronous.
        #[cfg(not(target_os = "windows"))]
        let (is_async, events) = {
            let ret = try_init(session, &mut init_params, 0, &mut preset_config);
            if ret != NV_ENC_SUCCESS {
                log::error!("[export] nvEncInitializeEncoder failed with error {}", ret);
                unsafe { (funcs.destroy_encoder)(session) };
                return Err(EncodeInteropError::Initialize(ret));
            }
            (
                false,
                vec![std::ptr::null_mut::<std::ffi::c_void>(); EVENT_COUNT],
            )
        };

        // Step 4 — Allocate the pipeline's NV12 interop buffers, register each
        // imported CUdeviceptr with NVENC, and give each slot its own bitstream
        // buffer.
        //
        // These are NOT plain `device.create_buffer` allocations: CUDA can only
        // import memory that was allocated with export-compatible flags, so
        // `SharedBuffer::new` creates the D3D12 resource itself with
        // `D3D12_HEAP_FLAG_SHARED` and then hands it to wgpu.  Substituting a
        // plain `create_buffer` here fails `cuImportExternalMemory` outright — and
        // when the test suite forced exactly that substitution, five tests failed
        // including the NV12 readback reading back the 0xAA pre-fill.
        //
        // `NvEncRegisterResource::pitch` CARRIES THE REAL STRIDE and must not be
        // 0.  The earlier note here said the driver ignores this field, which was
        // measured on the packed single-plane ABGR10 CUarray this path used to
        // register: pitch ∈ {0, width, width*4} all encoded byte-identical
        // bitstreams.  That result does NOT generalise to a two-plane NV12
        // resource — with `pitch = 0` a standalone probe had
        // `nvEncEncodePicture` kill the process outright, no error return
        // (`nvchk/nv12_probe.c` rung B0).  For CUDADEVICEPTR the header documents
        // pitch as the row stride in bytes, "must be a multiple of 4", and the
        // chroma plane is then read at `pitch * height`.
        //
        // `NV_ENC_PIC_PARAMS::inputPitch` is a separate field carrying the same
        // number — see `encode_frame`.  Both come from `self.pitch` so they cannot
        // drift apart.
        let pitch = Nv12EncodeNode::aligned_pitch(job.width);
        let nv12_size = Nv12EncodeNode::buffer_size(job.height, pitch);

        // STORAGE only: the compute passes write it and NVENC reads it through
        // CUDA.  Nothing copies it, which is the entire point.
        const NV12_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;

        // Labels are `&'static str` in SharedBuffer::new, so they cannot be
        // formatted per slot; a fixed table keeps the per-slot naming.
        const SLOT_LABELS: [&str; NVENC_INFLIGHT_SLOTS] = [
            "EncodeInterop NV12 slot-0",
            "EncodeInterop NV12 slot-1",
            "EncodeInterop NV12 slot-2",
            "EncodeInterop NV12 slot-3",
        ];

        let register_resource_ver = nv_enc_register_resource_ver(probed_api_version);
        let create_bs_ver = nv_enc_create_bitstream_buffer_ver(probed_api_version);

        let mut slots: Vec<EncodeSlot> = Vec::with_capacity(NVENC_INFLIGHT_SLOTS);

        log::info!(
            "[export] NVENC NV12 zero-copy input: {}x{}, pitch {} bytes, \
             {} bytes/slot, chroma plane at {} (pitch * height)",
            job.width, job.height, pitch, nv12_size, pitch * job.height
        );

        // Any failure part-way through must not leak the bitstream buffers and
        // registered resources already handed out; destroying the encoder does
        // reclaim them, so unwinding is just "destroy what we built, then the
        // session".  `slots` drops its SharedBuffers on the way out.
        for slot_idx in 0..NVENC_INFLIGHT_SLOTS {
            let shared = match SharedBuffer::new(
                Arc::clone(&cuda_ctx_arc), device, transport,
                SLOT_LABELS[slot_idx],
                nv12_size,
                NV12_USAGE,
            ) {
                Ok(b) => b,
                Err(e) => {
                    drop(slots);
                    unsafe { (funcs.destroy_encoder)(session) };
                    return Err(EncodeInteropError::Cuda(e));
                }
            };

            let mut register = NvEncRegisterResource {
                version:              register_resource_ver,
                resource_type:        NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                width:                job.width,
                height:               job.height,
                pitch,
                // A CUdeviceptr is a 64-bit handle, not a pointer to a pointer:
                // NVENC wants the VALUE in `resourceToRegister`, which is what
                // the C probe passed as `(void *)(uintptr_t)dptr`.
                resource_to_register: shared.external.device_ptr() as *mut _,
                registered_resource:  std::ptr::null_mut(),
                buffer_format:        NV_ENC_BUFFER_FORMAT_NV12,
                buffer_usage:         NV_ENC_BUFFER_USAGE_INPUT_IMAGE,
                p_input_fence_point:  std::ptr::null_mut(),
                ..Default::default()
            };
            let ret = unsafe { (funcs.register_resource)(session, &mut register) };
            if ret != NV_ENC_SUCCESS {
                log::error!(
                    "[export] nvEncRegisterResource(CUDADEVICEPTR, NV12, pitch={pitch}) \
                     failed for slot {slot_idx}: {} ({ret})",
                    nvenc_status_str(ret)
                );
                drop(shared);
                drop(slots);
                unsafe { (funcs.destroy_encoder)(session) };
                return Err(EncodeInteropError::Register(ret));
            }

            let mut bs_params = NvEncCreateBitstreamBuffer {
                version: create_bs_ver,
                ..Default::default()
            };
            let ret = unsafe { (funcs.create_bitstream)(session, &mut bs_params) };
            if ret != NV_ENC_SUCCESS {
                log::error!(
                    "[export] nvEncCreateBitstreamBuffer failed for slot {slot_idx}: {} ({ret})",
                    nvenc_status_str(ret)
                );
                drop(shared);
                drop(slots);
                unsafe { (funcs.destroy_encoder)(session) };
                return Err(EncodeInteropError::Initialize(ret));
            }

            slots.push(EncodeSlot {
                shared,
                registered: register.registered_resource,
                bitstream:  bs_params.bitstream_buffer,
                event:      events[slot_idx],
            });
        }

        Ok(Self {
            session,
            slots,
            funcs,
            width: job.width,
            height: job.height,
            pitch,
            api_version: probed_api_version,
            eos_event: events[NVENC_INFLIGHT_SLOTS],
            is_async,
            dts_queue: DtsQueue::default(),
            reorder_disabled: preset_config.is_some(),
            preset_config,
            pending: std::collections::VecDeque::new(),
            eos_sent: false,
            poisoned: false,
            // Declared (and therefore dropped) last — see the field's comment.
            cuda_ctx: cuda_ctx_arc,
        })
    }

    /// How many frames this session can keep in flight — the number of pipeline
    /// slots, i.e. the modulus the caller should use when choosing a slot index.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Return the NV12 wgpu buffer for the given pipeline slot.
    ///
    /// This is the buffer `Nv12EncodeNode::record` must be given as its
    /// destination, at [`Self::pitch`].
    pub fn nv12_buffer_for_slot(&self, slot: usize) -> &wgpu::Buffer {
        &self.slots[slot % self.slots.len()].shared.buffer
    }

    /// Row stride, in bytes, of both NV12 planes in every slot's buffer.
    ///
    /// The caller MUST pass this to `Nv12EncodeNode::record`.  Writing at a
    /// different stride than the registration declared puts the chroma plane
    /// somewhere NVENC does not read it, and the failure is a hue shift or a
    /// sheared image, not an error.
    pub fn pitch(&self) -> u32 {
        self.pitch
    }

    /// Return the NVENC registered resource handle for the given pipeline slot.
    fn registered_resource_for_slot(&self, slot: usize) -> *mut std::ffi::c_void {
        self.slots[slot % self.slots.len()].registered
    }

    /// Return the pre-allocated bitstream output buffer for the given pipeline slot.
    fn bitstream_buffer_for_slot(&self, slot: usize) -> *mut std::ffi::c_void {
        self.slots[slot % self.slots.len()].bitstream
    }

    /// Return the Win32 completion event handle for the given pipeline slot.
    /// Returns null when async mode is inactive.
    fn completion_event_for_slot(&self, slot: usize) -> *mut std::ffi::c_void {
        self.slots[slot % self.slots.len()].event
    }

    /// True when this session was initialised with B-frames explicitly disabled,
    /// so NVENC emits exactly one packet per submitted picture and output order
    /// equals submission order.
    pub fn reorder_disabled(&self) -> bool {
        self.reorder_disabled
    }

    /// True once an NVENC call has failed hard enough that this session must not
    /// be used again.  Callers should fall back rather than retry.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Mark the session unusable and abandon every queued handle WITHOUT calling
    /// back into the driver.
    ///
    /// Abandoning is the point.  Once NVENC has reported a hard failure the
    /// session's internal state is gone, and unmapping or unlocking against it is
    /// an access violation rather than an error return — the crash that turned
    /// `Encode(10)` into exit code 0xC0000005.  The handles belong to the session
    /// and are reclaimed by `nvEncDestroyEncoder`, so dropping them here leaks
    /// nothing that outlives the encoder.
    fn poison(&mut self) {
        self.poisoned = true;
        self.pending.clear();
    }

    /// Lock the oldest pending output, copy the bitstream out, then unlock it and
    /// unmap the input surface that produced it.
    ///
    /// In async mode this first waits on that picture's own completion event —
    /// which is why every slot owns one.  The wait happens HERE, at the moment
    /// the result is actually needed, instead of inside `encode_frame` right
    /// after submission: that is the whole of P1.1.  Submission no longer blocks,
    /// so NVENC keeps working on frames N-1 … N-3 while the GPU renders frame N.
    ///
    /// The entry is popped from `pending` before any fallible work so a lock
    /// failure cannot leave the same entry queued twice.
    fn take_oldest_output(&mut self) -> Result<Option<EncodedPacket>, EncodeInteropError> {
        if self.poisoned {
            return Ok(None);
        }
        let Some(entry) = self.pending.pop_front() else {
            return Ok(None);
        };

        // Wait for NVENC to signal that this picture is encoded (async mode only).
        // In sync mode nvEncLockBitstream blocks by itself, so there is nothing to
        // wait on and this block compiles away on non-Windows.
        #[cfg(target_os = "windows")]
        if self.is_async && !entry.event.is_null() {
            let wait_result = unsafe { WaitForSingleObject(entry.event, ENCODE_WAIT_TIMEOUT_MS) };
            if wait_result != WAIT_OBJECT_0 {
                log::error!(
                    "[export] WaitForSingleObject timed out or failed (result=0x{:x}) \
                     for slot {} — the encoder is wedged, poisoning the session",
                    wait_result, entry.slot
                );
                // A timeout means NVENC never finished this picture, so its
                // bitstream cannot be trusted and the session state is unknown.
                // Abandon the session rather than unmapping into a driver that may
                // be mid-failure.
                self.poison();
                return Err(EncodeInteropError::Encode(NV_ENC_ERR_GENERIC));
            }
        }

        let lock_ver = nv_enc_lock_bitstream_ver(self.api_version);
        let mut lock_params = NvEncLockBitstream {
            version:          lock_ver,
            output_bitstream: entry.bitstream,
            ..Default::default()
        };
        let ret = unsafe { (self.funcs.lock_bitstream)(self.session, &mut lock_params) };
        if ret != NV_ENC_SUCCESS {
            log::error!(
                "[export] nvEncLockBitstream failed: {} ({ret}) — poisoning the session",
                nvenc_status_str(ret)
            );
            // Release the input mapping while the session is still believed sane,
            // then poison: leaking the mapping would wedge the next
            // nvEncMapInputResource for the same registered resource, but calling
            // into a session that has already failed twice is worse.
            if !entry.mapped_input.is_null() {
                unsafe { (self.funcs.unmap_input)(self.session, entry.mapped_input); }
            }
            self.poison();
            return Err(EncodeInteropError::Lock(ret));
        }

        let bytes = unsafe {
            std::slice::from_raw_parts(
                lock_params.bitstream_buffer_ptr as *const u8,
                lock_params.bitstream_size_in_bytes as usize,
            )
            .to_vec()
        };
        // NVENC echoes the inputTimestamp of the picture this bitstream belongs
        // to, which is what makes PTS correct even when output order != input order.
        let pts = lock_params.output_timestamp as i64;
        let is_keyframe = nv_enc_pic_type_is_keyframe(lock_params.picture_type);

        unsafe {
            (self.funcs.unlock_bitstream)(self.session, entry.bitstream);
            if !entry.mapped_input.is_null() {
                (self.funcs.unmap_input)(self.session, entry.mapped_input);
            }
        }

        let dts = self.dts_queue.pop_output(pts);
        Ok(Some(EncodedPacket { bytes, pts, dts, is_keyframe }))
    }

    /// Drain outputs until no in-flight picture is still using `slot`, returning
    /// every packet that came out on the way.
    ///
    /// This is the backpressure primitive of the async pipeline, and the answer to
    /// "ensure a frame resource is not reused before NVENC has finished using it":
    /// the caller must call it BEFORE recording GPU work into `slot`'s ABGR10
    /// texture.  When nothing in flight owns the slot it is free and this returns
    /// an empty Vec without touching the driver.
    ///
    /// Draining is FIFO because NVENC bitstream retrieval is FIFO, so freeing a
    /// slot may also produce the packets of older frames — they are all returned,
    /// oldest first, and the caller must mux them in that order.
    pub fn reclaim_slot(&mut self, slot: usize) -> Result<Vec<EncodedPacket>, EncodeInteropError> {
        if self.poisoned || self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let slot = slot % self.slots.len();
        let mut packets = Vec::new();
        while self.pending.iter().any(|e| e.slot == slot) {
            match self.take_oldest_output()? {
                Some(pkt) => packets.push(pkt),
                // `pending` is non-empty but produced nothing: only reachable via
                // the poisoned early-return inside take_oldest_output.  Stop
                // rather than spin.
                None => break,
            }
        }
        Ok(packets)
    }

    /// Every packet NVENC has already finished, without waiting for anything.
    ///
    /// Optional: the pipeline is correct without it (`reclaim_slot` and
    /// [`Self::flush`] between them retrieve everything), but calling it each
    /// frame keeps the muxer fed and `pending` short.
    ///
    /// Only the head of the FIFO can be tested, because that is the only output
    /// NVENC is allowed to hand back next.
    pub fn drain_completed(&mut self) -> Result<Vec<EncodedPacket>, EncodeInteropError> {
        let mut packets = Vec::new();
        while self.head_is_ready() {
            match self.take_oldest_output()? {
                Some(pkt) => packets.push(pkt),
                None => break,
            }
        }
        Ok(packets)
    }

    /// True when the oldest in-flight picture's bitstream is ready to lock right
    /// now.  Always false in sync mode (there is no event to test, and locking is
    /// what blocks), and false when nothing is in flight.
    fn head_is_ready(&self) -> bool {
        if self.poisoned || !self.is_async {
            return false;
        }
        let Some(head) = self.pending.front() else {
            return false;
        };
        if head.event.is_null() {
            return false;
        }
        // A zero timeout is a non-blocking test.  The event is auto-reset, so a
        // successful test CONSUMES the signal — which is exactly right here,
        // because the caller immediately drains that entry and
        // `take_oldest_output` would otherwise wait on an already-consumed
        // signal.  The waits are therefore never doubled: this function is only
        // ever called from `drain_completed`, which drains on true.
        #[cfg(target_os = "windows")]
        {
            let r = unsafe { WaitForSingleObject(head.event, 0) };
            if r == WAIT_OBJECT_0 {
                // Re-signal so the drain's own wait still succeeds; SetEvent on an
                // auto-reset event is the documented way to hand the signal back.
                unsafe { SetEvent(head.event) };
                return true;
            }
            false
        }
        #[cfg(not(target_os = "windows"))]
        false
    }

    /// Submit one frame to NVENC **without waiting for it to finish**.
    ///
    /// The NV12 buffer for `slot` must already have been written by
    /// `Nv12EncodeNode` at [`Self::pitch`] and its GPU submission must have
    /// completed (the caller does `poll(WaitForSubmissionIndex)` first), and the
    /// slot must have been freed with [`Self::reclaim_slot`] before that GPU work
    /// was recorded.
    ///
    /// P1.1: this returns as soon as the driver has accepted the picture.  Any
    /// packets it hands back are ones that became available on the way — either
    /// because the pipeline was full and a slot had to be reclaimed, or because an
    /// older picture had already finished.  They are in decode order, oldest
    /// first, and must be muxed in that order.  An empty Vec is normal and means
    /// "accepted, still encoding".
    ///
    /// Nothing is lost by an empty return: every remaining packet comes out of a
    /// later call, out of [`Self::reclaim_slot`], or out of [`Self::flush`].
    pub fn encode_frame(
        &mut self,
        pts:  i64,
        slot: usize,
    ) -> Result<Vec<EncodedPacket>, EncodeInteropError> {
        if self.poisoned {
            // Report the same error class rather than pretending success: the
            // caller must fail the export, not silently drop frames.
            return Err(EncodeInteropError::Encode(NV_ENC_ERR_INVALID_CALL));
        }
        if self.eos_sent {
            // The session has been flushed; submitting another picture after EOS
            // is invalid per the NVENC programming guide.
            log::error!("[export] encode_frame called after EOS — ignoring the picture");
            return Ok(Vec::new());
        }

        let slot = slot % self.slots.len();

        // Backpressure.  If this slot's resources are still owned by an in-flight
        // picture, drain until they are not — that is the only thing this pipeline
        // ever blocks on, and it blocks on the OLDEST frame rather than the one
        // just submitted.  Normally a no-op: the caller reclaims the slot before
        // recording the GPU work that overwrites its texture.
        let mut packets = self.reclaim_slot(slot)?;

        let bitstream_buffer = self.bitstream_buffer_for_slot(slot);

        // Step 1 — Map the registered input resource for this encode call.
        let map_ver = nv_enc_map_input_resource_ver(self.api_version);
        let mut map_params = NvEncMapInputResource {
            version:             map_ver,
            sub_resource_index:  0,
            input_resource:      std::ptr::null_mut(),
            registered_resource: self.registered_resource_for_slot(slot),
            mapped_resource:     std::ptr::null_mut(),
            mapped_buffer_fmt:   NV_ENC_BUFFER_FORMAT_NV12,
            reserved1:           [0u32; 251],
            reserved2:           [std::ptr::null_mut(); 63],
        };
        let ret = unsafe {
            (self.funcs.map_input)(self.session, &mut map_params)
        };
        if ret != NV_ENC_SUCCESS {
            log::error!(
                "[export] nvEncMapInputResource failed for slot {slot}: {} ({ret})",
                nvenc_status_str(ret)
            );
            self.poison();
            return Err(EncodeInteropError::Map(ret));
        }
        let mapped_buffer = map_params.mapped_resource;

        // Step 2 — Submit the picture to the NVENC hardware encoder.
        // In async mode (Windows only): attach the slot's own Win32 completion
        // event so NVENC signals it when THIS frame finishes.  nvEncEncodePicture
        // returns immediately and the CPU thread goes straight back to rendering
        // the next frame; the wait for this event happens later, in
        // `take_oldest_output`, when the bitstream is actually needed.
        let pic_ver = nv_enc_pic_params_ver(self.api_version);
        let completion_event = if self.is_async {
            self.completion_event_for_slot(slot)
        } else {
            std::ptr::null_mut()
        };
        // `input_pitch` is the input buffer's row stride in BYTES.  It carries the
        // same number as the registration's `pitch` — see `open`, and note that
        // the header's "if pitch value is not known, set this to inputWidth"
        // escape hatch is NOT usable here: for a two-plane NV12 layout the stride
        // is what places the chroma plane, and getting it wrong is silent bad
        // colour rather than an error.
        //
        // P1.5 measured that for the OLD packed ABGR10 CUarray input the driver
        // ignored this field entirely — inputPitch ∈ {width, width*4, 0} all
        // produced byte-identical bitstreams, because a CUarray carries its own
        // descriptor.  That measurement does not transfer to a CUdeviceptr NV12
        // input, which is why this is `self.pitch` and not a guess.
        let input_pitch = self.pitch;
        let mut pic_params = NvEncPicParams {
            version:          pic_ver,
            input_width:      self.width,
            input_height:     self.height,
            input_pitch,
            encode_pic_flags: 0,
            frame_idx:        0,
            input_timestamp:  pts as u64,
            input_duration:   0,
            input_buffer:     mapped_buffer,
            output_bitstream: bitstream_buffer,
            completion_event,
            buffer_fmt:       NV_ENC_BUFFER_FORMAT_NV12,
            picture_struct:   NV_ENC_PIC_STRUCT_FRAME,
            picture_type:     0,
            ..Default::default()
        };
        let ret = unsafe { (self.funcs.encode_picture)(self.session, &mut pic_params) };
        if ret != NV_ENC_SUCCESS && ret != NV_ENC_ERR_NEED_MORE_INPUT {
            // The picture was REJECTED: NVENC holds nothing for it, so the input
            // mapping must be released here and the session poisoned.
            //
            // Poisoning matters more than the unmap.  Before this, a failed
            // encode left `pending` and the mapped surfaces in a state that
            // `Drop` then unwound against a session the driver had already
            // invalidated (e.g. after its CUDA context died), turning any NVENC
            // error into STATUS_ACCESS_VIOLATION — a process kill instead of the
            // graceful fallback the audit asks for.
            unsafe { (self.funcs.unmap_input)(self.session, mapped_buffer); }
            log::error!(
                "[export] nvEncEncodePicture rejected frame at pts {pts}: {} ({ret}) — \
                 the NVENC session is now poisoned and will not be used again",
                nvenc_status_str(ret)
            );
            self.poison();
            return Err(EncodeInteropError::Encode(ret));
        }

        // The picture was accepted either way (NV_ENC_ERR_NEED_MORE_INPUT means
        // "accepted, buffered for reordering"): record its timestamp for DTS
        // generation and queue its resources for retrieval.  The input surface
        // stays mapped, and the slot stays busy, until its bitstream is read.
        self.dts_queue.push_input(pts);
        self.pending.push_back(PendingOutput {
            bitstream:    bitstream_buffer,
            mapped_input: mapped_buffer,
            event:        completion_event,
            slot,
        });

        // Step 3 — Hand over anything NVENC has already finished, WITHOUT
        // blocking.  This is what makes the pipeline asynchronous: the previous
        // implementation waited on this frame's own event here, so the CPU never
        // got ahead of the encoder.
        packets.extend(self.drain_completed()?);
        Ok(packets)
    }

    /// Submit an End-of-Stream picture and drain every packet NVENC is still
    /// holding.
    ///
    /// Without this, any frame left in the encoder's reorder/lookahead queue is
    /// destroyed along with the session — silently truncating the tail of the
    /// export.  Safe to call more than once; later calls return an empty Vec.
    ///
    /// Per the NVENC programming guide the EOS picture carries
    /// `NV_ENC_PIC_FLAG_EOS` with a NULL input buffer and NULL output bitstream,
    /// and in async mode must still carry a registered completion event.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>, EncodeInteropError> {
        if self.poisoned {
            log::warn!(
                "[export] NVENC flush skipped: the session is poisoned, so any frames \
                 it still held are lost"
            );
            return Ok(Vec::new());
        }
        if self.eos_sent {
            return Ok(Vec::new());
        }
        self.eos_sent = true;

        let pic_ver = nv_enc_pic_params_ver(self.api_version);
        // Async mode: NVENC signals this event once the EOS has been processed.
        // It is a DEDICATED event, not a slot's: with several frames in flight,
        // every slot event may still belong to an unfinished picture, and waiting
        // on one of those here would consume the signal its own frame needs.
        let completion_event = if self.is_async {
            self.eos_event
        } else {
            std::ptr::null_mut()
        };
        // The EOS picture carries NO input surface, so it must NOT describe one:
        // inputWidth/Height/Pitch and bufferFmt stay zero.  Passing the frame
        // geometry here alongside a NULL inputBuffer is a contradiction the
        // driver is entitled to reject.
        let mut eos = NvEncPicParams {
            version:          pic_ver,
            input_width:      0,
            input_height:     0,
            input_pitch:      0,
            encode_pic_flags: NV_ENC_PIC_FLAG_EOS,
            frame_idx:        0,
            input_timestamp:  0,
            input_duration:   0,
            input_buffer:     std::ptr::null_mut(),
            output_bitstream: std::ptr::null_mut(),
            completion_event,
            buffer_fmt:       NV_ENC_BUFFER_FORMAT_UNDEFINED,
            picture_struct:   NV_ENC_PIC_STRUCT_FRAME,
            picture_type:     0,
            ..Default::default()
        };
        let ret = unsafe { (self.funcs.encode_picture)(self.session, &mut eos) };
        if ret != NV_ENC_SUCCESS {
            log::error!(
                "[export] NVENC EOS nvEncEncodePicture failed: {} ({ret}) — \
                 poisoning the session; buffered frames are lost",
                nvenc_status_str(ret)
            );
            self.poison();
            return Err(EncodeInteropError::Encode(ret));
        }

        #[cfg(target_os = "windows")]
        if self.is_async && !completion_event.is_null() {
            let wait_result = unsafe { WaitForSingleObject(completion_event, ENCODE_WAIT_TIMEOUT_MS) };
            if wait_result != WAIT_OBJECT_0 {
                log::error!(
                    "[export] WaitForSingleObject on the EOS event timed out or failed \
                     (result=0x{:x}) — the tail of the stream may be incomplete",
                    wait_result
                );
            }
        }

        // Drain every queued output.  With several frames in flight this is where
        // the tail of the pipeline comes out — up to NVENC_INFLIGHT_SLOTS pictures
        // that were submitted but never waited on, plus anything the driver was
        // holding for reordering.  `take_oldest_output` waits on each picture's own
        // event, so this is the point at which the export blocks for the encoder to
        // finish, exactly once, at the end.
        let mut packets = Vec::with_capacity(self.pending.len());
        while !self.pending.is_empty() {
            match self.take_oldest_output()? {
                Some(pkt) => packets.push(pkt),
                None => break,
            }
        }

        if !packets.is_empty() {
            log::info!("[export] NVENC EOS flush recovered {} buffered packet(s)", packets.len());
        }
        Ok(packets)
    }
}

impl Drop for EncodeInterop {
    fn drop(&mut self) {
        unsafe {
            // A poisoned session has already failed inside the driver.  Calling
            // unmap / destroy-bitstream / unregister-event against it is an
            // access violation, not an error return — that is precisely how a
            // recoverable `Encode` error became exit code 0xC0000005.  So on the
            // poisoned path do nothing but destroy the encoder, which the driver
            // handles for a failed session, and let it reclaim the handles.
            if self.poisoned {
                log::warn!(
                    "[export] dropping a poisoned EncodeInterop — releasing only the \
                     session; NVENC reclaims its own buffers and mappings"
                );
                let _ = (self.funcs.destroy_encoder)(self.session);
                #[cfg(target_os = "windows")]
                {
                    // The Win32 events are OURS, not the driver's, so they are
                    // still safe (and necessary) to close.  Skip
                    // nvEncUnregisterAsyncEvent: that one goes through the
                    // session.
                    for slot in &mut self.slots {
                        if !slot.event.is_null() {
                            CloseHandle(slot.event);
                            slot.event = std::ptr::null_mut();
                        }
                    }
                    if !self.eos_event.is_null() {
                        CloseHandle(self.eos_event);
                        self.eos_event = std::ptr::null_mut();
                    }
                }
                return;
            }

            // Release any input resources still mapped for outputs that were
            // never retrieved.  `flush` normally drains these; this covers the
            // error paths that drop the encoder mid-export.
            if !self.pending.is_empty() {
                log::warn!(
                    "[export] dropping EncodeInterop with {} un-retrieved output(s) — \
                     their frames are lost (flush() was not called or failed)",
                    self.pending.len()
                );
            }
            while let Some(entry) = self.pending.pop_front() {
                if !entry.mapped_input.is_null() {
                    (self.funcs.unmap_input)(self.session, entry.mapped_input);
                }
            }

            // Unregister async completion events before destroying the encoder.
            // Must happen before destroy_encoder; guards for null handles ensure
            // this is a no-op when async mode was never activated or init failed.
            #[cfg(target_os = "windows")]
            if self.is_async {
                let event_params_ver = nv_enc_event_params_ver(self.api_version);
                let unregister = |ev: &mut *mut std::ffi::c_void| {
                    if ev.is_null() {
                        return;
                    }
                    let mut ep = NvEncEventParams {
                        version:          event_params_ver,
                        completion_event: *ev,
                        ..Default::default()
                    };
                    let _ = (self.funcs.unregister_async_event)(self.session, &mut ep);
                    CloseHandle(*ev);
                    *ev = std::ptr::null_mut();
                };
                for slot in &mut self.slots {
                    unregister(&mut slot.event);
                }
                unregister(&mut self.eos_event);
            }

            for slot in &self.slots {
                if !slot.bitstream.is_null() {
                    (self.funcs.destroy_bitstream)(self.session, slot.bitstream);
                }
            }
            let _ = (self.funcs.destroy_encoder)(self.session);
        }
    }
}
