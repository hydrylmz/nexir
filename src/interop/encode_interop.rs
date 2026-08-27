// src/interop/encode_interop.rs
// Replaces Phase 6's FrameReadback CPU round-trip with a direct NVENC encode path.
// The RTT texture's CUDA array is registered with NVENC, and encoding is entirely GPU-side.

use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::external_texture::ExternalTexture;
use crate::interop::ffi::nvenc::*;
use crate::interop::ffi::cuda_driver::{cuCtxPushCurrent, cuCtxPopCurrent, CUcontext};
use crate::render::device::GpuDevice;
use crate::export::job::{ExportJob, VideoCodec};

/// A wgpu compute node that repacks RGBA16Float (already tone-mapped)
/// into ABGR10 format that NVENC accepts directly.
///
/// ABGR10 packs: packed = (a2 << 30) | (b10 << 20) | (g10 << 10) | r10
/// where each 10-bit channel = round(clamp(f32, 0.0, 1.0) * 1023.0) as u32,
/// and the 2-bit alpha is fixed at 0b11 (fully opaque).
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

/// NVENC function table loaded from the driver.
struct NvencFunctions {
    open_session:          functions::OpenEncodeSessionEx,
    get_preset_config:     functions::GetPresetConfig,
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
}

/// Holds the NVENC session and two ping-pong interop-imported ABGR10 textures.
///
/// Two textures allow the GPU to render frame N into slot N%2 while NVENC
/// simultaneously encodes the previous frame from slot (N-1)%2, eliminating
/// the serialised render→wait→encode stall of a single-buffer design.
pub struct EncodeInterop {
    session:               NvEncodeSession,
    /// Slot 0 ABGR10 texture (R32Uint packed, written by Abgr10RepackNode).
    abgr10_texture_0:      wgpu::Texture,
    /// Slot 1 ABGR10 texture — the second ping-pong buffer.
    abgr10_texture_1:      wgpu::Texture,
    /// Keeps the CUDA external memory import for slot 0 alive.
    #[allow(dead_code)]
    abgr10_external_0:     ExternalTexture,
    /// Keeps the CUDA external memory import for slot 1 alive.
    #[allow(dead_code)]
    abgr10_external_1:     ExternalTexture,
    /// NVENC registered resource handle for slot 0.
    registered_resource_0: *mut std::ffi::c_void,
    /// NVENC registered resource handle for slot 1.
    registered_resource_1: *mut std::ffi::c_void,
    /// Pre-allocated bitstream output buffer for slot 0.
    bitstream_buffer_0:    *mut std::ffi::c_void,
    /// Pre-allocated bitstream output buffer for slot 1.
    bitstream_buffer_1:    *mut std::ffi::c_void,
    funcs:                 NvencFunctions,
    width:                 u32,
    height:                u32,
    /// The NVENC API major version we probed successfully; used to construct
    /// per-struct version fields for encode-time calls.
    api_version:           u32,
    /// Win32 event handle for slot 0 completion (null when async mode is off).
    /// Only populated on Windows when NVENC async init succeeded.
    completion_event_0:    *mut std::ffi::c_void,
    /// Win32 event handle for slot 1 completion (null when async mode is off).
    completion_event_1:    *mut std::ffi::c_void,
    /// Whether asynchronous NVENC encoding is active.  False means sync mode
    /// (completion_event_* are null and no WaitForSingleObject calls are made).
    is_async:              bool,
}

unsafe impl Send for EncodeInterop {}
/// WAIT_OBJECT_0: WaitForSingleObject returned because the object was signalled.
#[cfg(target_os = "windows")]
const WAIT_OBJECT_0: u32 = 0x00000000;

/// Win32 kernel32 functions needed for asynchronous NVENC completion events.
/// These are the standard Windows synchronization APIs; we declare them here
/// rather than pulling in the `windows-sys` crate to avoid a new dependency.
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

    fn CloseHandle(h_object: *mut std::ffi::c_void) -> i32;
}

// NV_ENC_PIC_STRUCT_FRAME — all exported frames are progressive frames.
const NV_ENC_PIC_STRUCT_FRAME: u32 = 1;

/// Compute NVENCAPI_STRUCT_VERSION(n) from the probed API major version.
///
/// Every NVENC struct's version field must be formed as:
///   (api_version) | (struct_ver << 16) | (0x7 << 28)
/// The `0x7 << 28` magic marker is mandatory — NVENC rejects structs without it.
/// struct_ver is the per-struct layout version from the nvEncodeAPI.h header.
#[inline(always)]
fn nvenc_struct_ver(probed_api_version: u32, struct_ver: u32) -> u32 {
    probed_api_version | (struct_ver << 16) | (0x7u32 << 28)
}
// H264 GUID: {6BC82762-4E63-4ca4-AA85-1E50F321F6BF}
const NV_ENC_CODEC_H264_GUID: [u8; 16] = [
    0x62, 0x27, 0xC8, 0x6B, 0x63, 0x4E, 0xa4, 0x4c,
    0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF,
];
// HEVC GUID: {790CDC88-4522-4d7b-9425-BDA9975F7603}
const NV_ENC_CODEC_HEVC_GUID: [u8; 16] = [
    0x88, 0xCD, 0x0C, 0x79, 0x22, 0x45, 0x7B, 0x4d,
    0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03,
];
// P4 preset GUID: {FC0A8D3E-45F3-4cf8-878F-7B9A9C7A6A97}
// Supported for both H.264 and HEVC.
const NV_ENC_PRESET_P4_GUID: [u8; 16] = [
    0x3E, 0x8D, 0x0A, 0xFC, 0xF3, 0x45, 0xF8, 0x4c,
    0x87, 0x8F, 0x7B, 0x9A, 0x9C, 0x7A, 0x6A, 0x97,
];
// Default preset GUID: {45210760-6936-45e6-B926-2F9E58A3F2BF}
const NV_ENC_PRESET_DEFAULT_GUID: [u8; 16] = [
    0x60, 0x07, 0x21, 0x45, 0x36, 0x69, 0xe6, 0x45,
    0xB9, 0x26, 0x2F, 0x9E, 0x58, 0xA3, 0xF2, 0xBF,
];

impl EncodeInterop {
    /// Open an NVENC session against the shared CUDA context and register the
    /// ABGR10 interop texture as NVENC's input resource.
    ///
    /// `codec` controls whether to initialise an H.264 or HEVC session;
    /// both codecs use the same ABGR10 input format and P4 preset.
    pub fn open(
        cuda_ctx:  &CudaContext,
        device:    &GpuDevice,
        job:       &ExportJob,
        transport: crate::interop::capability::InteropTransport,
        codec:     VideoCodec,
    ) -> Result<Self, EncodeInteropError> {
        let fn_table_size = std::mem::size_of::<usize>() * 512;
        let mut fn_table_raw = vec![0u8; fn_table_size];
        let mut success = false;
        let mut ret = -1;

        // Try major versions 12 to 8 (matching driver API 12.2 on this system).
        let mut probed_api_version: u32 = 0;
        for major_ver in (8u32..=12u32).rev() {
            let version = major_ver | (2 << 16) | (0x7 << 28);
            unsafe {
                let ptr = fn_table_raw.as_mut_ptr() as *mut u32;
                *ptr = version;
            }
            let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;
            ret = unsafe { NvEncodeAPICreateInstance(function_list) };
            if ret == NV_ENC_SUCCESS {
                success = true;
                probed_api_version = major_ver;
                log::info!("[export] NvEncodeAPICreateInstance succeeded with major version {}", major_ver);
                break;
            }
        }

        if !success {
            log::error!("[export] NvEncodeAPICreateInstance failed with error code {}", ret);
            return Err(EncodeInteropError::ApiLoad);
        }

        // Pointer offsets (base.add(N)) into NV_ENCODE_API_FUNCTION_LIST:
        //   base.add(0)  = version (u32) + reserved (u32)
        //   base.add(1)  = nvEncOpenEncodeSession
        //   base.add(10) = nvEncGetEncodePresetConfig
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
        let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;
        let base = function_list as *const usize;
        let funcs = unsafe {
            let open_off              = 30usize; // nvEncOpenEncodeSessionEx
            let preset_cfg_off        = 10usize; // nvEncGetEncodePresetConfig
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

            NvencFunctions {
                open_session:           std::mem::transmute(*base.add(open_off)),
                get_preset_config:      std::mem::transmute(*base.add(preset_cfg_off)),
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
        let open_params_ver = nvenc_struct_ver(probed_api_version, 1);
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
        let init_params_ver = nvenc_struct_ver(probed_api_version, 5);
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
            tuning_info:                  0,
            buffer_format:                0,
            num_state_buffers:            0,
            output_stats_level:           0,
            reserved1:                    [0u32; 284],
            reserved2:                    [std::ptr::null_mut(); 64],
        };

        // --- Async-first initialization ---
        // On Windows: attempt async NVENC mode (enable_encode_async = 1).
        // If the driver accepts it AND Win32 event creation succeeds, use async.
        // Any failure at any step falls back silently to synchronous mode.
        // On non-Windows: always synchronous (no Win32 event primitives available).
        #[cfg(target_os = "windows")]
        let (is_async, completion_event_0, completion_event_1) = unsafe {
            init_params.enable_encode_async = 1;
            let mut async_ret = (funcs.initialize)(session, &mut init_params);
            if async_ret != NV_ENC_SUCCESS {
                log::warn!(
                    "[export] nvEncInitializeEncoder async failed ({}), retrying with DEFAULT preset async",
                    async_ret
                );
                init_params.preset_guid = NV_ENC_PRESET_DEFAULT_GUID;
                async_ret = (funcs.initialize)(session, &mut init_params);
            }

            if async_ret == NV_ENC_SUCCESS {
                // Try to create Win32 auto-reset events (bManualReset=0, bInitialState=0).
                let ev0 = CreateEventA(std::ptr::null_mut(), 0, 0, std::ptr::null());
                let ev1 = CreateEventA(std::ptr::null_mut(), 0, 0, std::ptr::null());
                if ev0.is_null() || ev1.is_null() {
                    // Clean up any handle that was successfully created.
                    if !ev0.is_null() { CloseHandle(ev0); }
                    if !ev1.is_null() { CloseHandle(ev1); }
                    log::warn!("[export] CreateEventA failed — falling back to sync NVENC mode");
                    // Destroy and re-init in sync mode below.
                    (funcs.destroy_encoder)(session);
                    init_params.enable_encode_async = 0;
                    init_params.preset_guid = NV_ENC_PRESET_P4_GUID;
                    let mut sync_ret = (funcs.initialize)(session, &mut init_params);
                    if sync_ret != NV_ENC_SUCCESS {
                        init_params.preset_guid = NV_ENC_PRESET_DEFAULT_GUID;
                        sync_ret = (funcs.initialize)(session, &mut init_params);
                    }
                    if sync_ret != NV_ENC_SUCCESS {
                        log::error!("[export] nvEncInitializeEncoder sync fallback failed ({})", sync_ret);
                        (funcs.destroy_encoder)(session);
                        return Err(EncodeInteropError::Initialize(sync_ret));
                    }
                    (false, std::ptr::null_mut(), std::ptr::null_mut())
                } else {
                    // Register both events with NVENC.
                    let event_params_ver = nvenc_struct_ver(probed_api_version, 1);
                    let mut ep0 = NvEncEventParams {
                        version: event_params_ver,
                        completion_event: ev0,
                        ..Default::default()
                    };
                    let mut ep1 = NvEncEventParams {
                        version: event_params_ver,
                        completion_event: ev1,
                        ..Default::default()
                    };
                    let r0 = (funcs.register_async_event)(session, &mut ep0);
                    let r1 = (funcs.register_async_event)(session, &mut ep1);
                    if r0 != NV_ENC_SUCCESS || r1 != NV_ENC_SUCCESS {
                        log::warn!(
                            "[export] nvEncRegisterAsyncEvent failed (r0={}, r1={}) — \
                             falling back to sync NVENC mode",
                            r0, r1
                        );
                        CloseHandle(ev0);
                        CloseHandle(ev1);
                        // Destroy and re-init in sync mode.
                        (funcs.destroy_encoder)(session);
                        init_params.enable_encode_async = 0;
                        init_params.preset_guid = NV_ENC_PRESET_P4_GUID;
                        let mut sync_ret = (funcs.initialize)(session, &mut init_params);
                        if sync_ret != NV_ENC_SUCCESS {
                            init_params.preset_guid = NV_ENC_PRESET_DEFAULT_GUID;
                            sync_ret = (funcs.initialize)(session, &mut init_params);
                        }
                        if sync_ret != NV_ENC_SUCCESS {
                            log::error!("[export] nvEncInitializeEncoder sync fallback failed ({})", sync_ret);
                            (funcs.destroy_encoder)(session);
                            return Err(EncodeInteropError::Initialize(sync_ret));
                        }
                        (false, std::ptr::null_mut(), std::ptr::null_mut())
                    } else {
                        log::info!("[export] NVENC asynchronous encoding active (Win32 events registered)");
                        (true, ev0, ev1)
                    }
                }
            } else {
                // Async init failed; try sync fallback.
                log::warn!("[export] nvEncInitializeEncoder async mode rejected by driver — falling back to sync");
                init_params.enable_encode_async = 0;
                init_params.preset_guid = NV_ENC_PRESET_P4_GUID;
                let mut sync_ret = (funcs.initialize)(session, &mut init_params);
                if sync_ret != NV_ENC_SUCCESS {
                    init_params.preset_guid = NV_ENC_PRESET_DEFAULT_GUID;
                    sync_ret = (funcs.initialize)(session, &mut init_params);
                }
                if sync_ret != NV_ENC_SUCCESS {
                    log::error!("[export] nvEncInitializeEncoder failed with error {}", sync_ret);
                    (funcs.destroy_encoder)(session);
                    return Err(EncodeInteropError::Initialize(sync_ret));
                }
                (false, std::ptr::null_mut(), std::ptr::null_mut())
            }
        };

        // Non-Windows: always synchronous.
        #[cfg(not(target_os = "windows"))]
        let (is_async, completion_event_0, completion_event_1) = {
            let mut ret = unsafe { (funcs.initialize)(session, &mut init_params) };
            if ret != NV_ENC_SUCCESS {
                log::warn!("[export] nvEncInitializeEncoder with P4 preset failed ({}), retrying with DEFAULT preset", ret);
                init_params.preset_guid = NV_ENC_PRESET_DEFAULT_GUID;
                ret = unsafe { (funcs.initialize)(session, &mut init_params) };
            }
            if ret != NV_ENC_SUCCESS {
                log::error!("[export] nvEncInitializeEncoder failed with error {}", ret);
                unsafe { (funcs.destroy_encoder)(session) };
                return Err(EncodeInteropError::Initialize(ret));
            }
            (false, std::ptr::null_mut::<std::ffi::c_void>(), std::ptr::null_mut::<std::ffi::c_void>())
        };

        // Step 4 — Allocate two ping-pong ABGR10 interop textures (R32Uint packed).
        let make_abgr10_texture = |label: &'static str| wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: job.width, height: job.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let abgr10_texture_0 = device.device.create_texture(&make_abgr10_texture("EncodeInterop ABGR10 slot-0"));
        let abgr10_external_0 = ExternalTexture::import(
            cuda_ctx, device, &abgr10_texture_0, transport, job.width, job.height,
        ).map_err(EncodeInteropError::Cuda)?;

        let abgr10_texture_1 = device.device.create_texture(&make_abgr10_texture("EncodeInterop ABGR10 slot-1"));
        let abgr10_external_1 = ExternalTexture::import(
            cuda_ctx, device, &abgr10_texture_1, transport, job.width, job.height,
        ).map_err(EncodeInteropError::Cuda)?;

        // Step 5 — Register both imported CUarrays as NVENC input resources.
        let register_resource_ver = nvenc_struct_ver(probed_api_version, 4);

        let mut register_0 = NvEncRegisterResource {
            version:              register_resource_ver,
            resource_type:        NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
            width:                job.width,
            height:               job.height,
            pitch:                0,
            resource_to_register: abgr10_external_0.cuda_array() as *mut _,
            registered_resource:  std::ptr::null_mut(),
            buffer_format:        NV_ENC_BUFFER_FORMAT_ABGR10,
            buffer_usage:         0,
            p_input_fence_point:  std::ptr::null_mut(),
            ..Default::default()
        };
        let ret = unsafe { (funcs.register_resource)(session, &mut register_0) };
        if ret != NV_ENC_SUCCESS {
            unsafe { (funcs.destroy_encoder)(session) };
            return Err(EncodeInteropError::Register(ret));
        }

        let mut register_1 = NvEncRegisterResource {
            version:              register_resource_ver,
            resource_type:        NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
            width:                job.width,
            height:               job.height,
            pitch:                0,
            resource_to_register: abgr10_external_1.cuda_array() as *mut _,
            registered_resource:  std::ptr::null_mut(),
            buffer_format:        NV_ENC_BUFFER_FORMAT_ABGR10,
            buffer_usage:         0,
            p_input_fence_point:  std::ptr::null_mut(),
            ..Default::default()
        };
        let ret = unsafe { (funcs.register_resource)(session, &mut register_1) };
        if ret != NV_ENC_SUCCESS {
            unsafe { (funcs.destroy_encoder)(session) };
            return Err(EncodeInteropError::Register(ret));
        }

        // Step 6 — Allocate ping-pong bitstream output buffers.
        let create_bs_ver = nvenc_struct_ver(probed_api_version, 1);
        let mut bs_params_0 = NvEncCreateBitstreamBuffer {
            version: create_bs_ver,
            ..Default::default()
        };
        let ret = unsafe { (funcs.create_bitstream)(session, &mut bs_params_0) };
        if ret != NV_ENC_SUCCESS {
            log::error!("[export] nvEncCreateBitstreamBuffer slot 0 failed with error {}", ret);
            unsafe { (funcs.destroy_encoder)(session) };
            return Err(EncodeInteropError::Initialize(ret));
        }
        let bitstream_buffer_0 = bs_params_0.bitstream_buffer;

        let mut bs_params_1 = NvEncCreateBitstreamBuffer {
            version: create_bs_ver,
            ..Default::default()
        };
        let ret = unsafe { (funcs.create_bitstream)(session, &mut bs_params_1) };
        if ret != NV_ENC_SUCCESS {
            log::error!("[export] nvEncCreateBitstreamBuffer slot 1 failed with error {}", ret);
            unsafe {
                (funcs.destroy_bitstream)(session, bitstream_buffer_0);
                (funcs.destroy_encoder)(session);
            }
            return Err(EncodeInteropError::Initialize(ret));
        }
        let bitstream_buffer_1 = bs_params_1.bitstream_buffer;

        Ok(Self {
            session,
            abgr10_texture_0,
            abgr10_texture_1,
            abgr10_external_0,
            abgr10_external_1,
            registered_resource_0: register_0.registered_resource,
            registered_resource_1: register_1.registered_resource,
            bitstream_buffer_0,
            bitstream_buffer_1,
            funcs,
            width: job.width,
            height: job.height,
            api_version: probed_api_version,
            completion_event_0,
            completion_event_1,
            is_async,
        })
    }

    /// Return the ABGR10 wgpu texture for the given ping-pong slot (0 or 1).
    pub fn abgr10_texture_for_slot(&self, slot: usize) -> &wgpu::Texture {
        if slot == 0 { &self.abgr10_texture_0 } else { &self.abgr10_texture_1 }
    }

    /// Return the NVENC registered resource handle for the given ping-pong slot.
    fn registered_resource_for_slot(&self, slot: usize) -> *mut std::ffi::c_void {
        if slot == 0 { self.registered_resource_0 } else { self.registered_resource_1 }
    }

    /// Return the pre-allocated bitstream output buffer for the given ping-pong slot.
    fn bitstream_buffer_for_slot(&self, slot: usize) -> *mut std::ffi::c_void {
        if slot == 0 { self.bitstream_buffer_0 } else { self.bitstream_buffer_1 }
    }

    /// Return the Win32 completion event handle for the given ping-pong slot.
    /// Returns null when async mode is inactive.
    fn completion_event_for_slot(&self, slot: usize) -> *mut std::ffi::c_void {
        if slot == 0 { self.completion_event_0 } else { self.completion_event_1 }
    }

    /// Encode one frame. The ABGR10 texture for `slot` must have already been
    /// written by `Abgr10RepackNode` and its GPU submission must have completed
    /// (caller uses `poll(WaitForSubmissionIndex)` before calling this).
    /// Returns (bitstream_bytes, pts) ready for Phase 6's Muxer::write_packet.
    pub fn encode_frame(&mut self, pts: i64, slot: usize) -> Result<(Vec<u8>, i64), EncodeInteropError> {
        let bitstream_buffer = self.bitstream_buffer_for_slot(slot);

        // Step 1 — Map the registered input resource for this encode call.
        let map_ver = nvenc_struct_ver(self.api_version, 1);
        let mut map_params = NvEncMapInputResource {
            version:             map_ver,
            sub_resource_index:  0,
            input_resource:      std::ptr::null_mut(),
            registered_resource: self.registered_resource_for_slot(slot),
            mapped_resource:     std::ptr::null_mut(),
            mapped_buffer_fmt:   NV_ENC_BUFFER_FORMAT_ABGR10,
            reserved1:           [0u32; 251],
            reserved2:           [std::ptr::null_mut(); 63],
        };
        let ret = unsafe {
            (self.funcs.map_input)(self.session, &mut map_params)
        };
        if ret != NV_ENC_SUCCESS {
            return Err(EncodeInteropError::Map(ret));
        }
        let mapped_buffer = map_params.mapped_resource;

        // Step 2 — Submit the picture to the NVENC hardware encoder.
        // In async mode (Windows only): attach the slot's Win32 completion event so
        // NVENC signals it when encoding of this frame finishes.  nvEncEncodePicture
        // returns immediately (NV_ENC_SUCCESS) and the CPU thread is free to submit
        // the next GPU render while NVENC works in parallel.
        let pic_ver = nvenc_struct_ver(self.api_version, 4);
        let completion_event = if self.is_async {
            self.completion_event_for_slot(slot)
        } else {
            std::ptr::null_mut()
        };
        let mut pic_params = NvEncPicParams {
            version:          pic_ver,
            input_width:      self.width,
            input_height:     self.height,
            input_pitch:      self.width,
            encode_pic_flags: 0,
            frame_idx:        0,
            input_timestamp:  pts as u64,
            input_duration:   0,
            input_buffer:     mapped_buffer,
            output_bitstream: bitstream_buffer,
            completion_event,
            buffer_fmt:       NV_ENC_BUFFER_FORMAT_ABGR10,
            picture_struct:   NV_ENC_PIC_STRUCT_FRAME,
            picture_type:     0,
            ..Default::default()
        };
        let ret = unsafe { (self.funcs.encode_picture)(self.session, &mut pic_params) };
        if ret != NV_ENC_SUCCESS {
            unsafe { (self.funcs.unmap_input)(self.session, mapped_buffer); }
            return Err(EncodeInteropError::Encode(ret));
        }

        // Step 2b — Wait for NVENC to signal the completion event (async mode only).
        // On Windows with async enabled, nvEncEncodePicture returned immediately; we
        // now block on the event with a 5-second safety timeout before accessing the
        // bitstream.  On non-Windows or in sync mode this block compiles away.
        #[cfg(target_os = "windows")]
        if self.is_async {
            let wait_result = unsafe { WaitForSingleObject(completion_event, 5000) };
            if wait_result != WAIT_OBJECT_0 {
                log::error!(
                    "[export] WaitForSingleObject timed out or failed (result=0x{:x}) \
                     for slot {} — encoder may be wedged",
                    wait_result, slot
                );
                unsafe { (self.funcs.unmap_input)(self.session, mapped_buffer); }
                return Err(EncodeInteropError::Encode(-1));
            }
        }

        // Step 3 — Lock the bitstream to read the encoded bytes out.
        let lock_ver = nvenc_struct_ver(self.api_version, 1);
        let mut lock_params = NvEncLockBitstream {
            version:                 lock_ver,
            output_bitstream:        bitstream_buffer,
            ..Default::default()
        };
        let ret = unsafe {
            (self.funcs.lock_bitstream)(self.session, &mut lock_params)
        };
        if ret != NV_ENC_SUCCESS {
            unsafe { (self.funcs.unmap_input)(self.session, mapped_buffer); }
            return Err(EncodeInteropError::Lock(ret));
        }

        // Copy the bitstream bytes into an owned Vec.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                lock_params.bitstream_buffer_ptr as *const u8,
                lock_params.bitstream_size_in_bytes as usize,
            ).to_vec()
        };

        // Step 4 — Unlock the bitstream buffer and unmap input resource.
        unsafe {
            (self.funcs.unlock_bitstream)(self.session, bitstream_buffer);
            (self.funcs.unmap_input)(self.session, mapped_buffer);
        }

        Ok((bytes, pts))
    }
}

impl Drop for EncodeInterop {
    fn drop(&mut self) {
        unsafe {
            // Unregister async completion events before destroying the encoder.
            // Must happen before destroy_encoder; guards for null handles ensure
            // this is a no-op when async mode was never activated or init failed.
            #[cfg(target_os = "windows")]
            if self.is_async {
                let event_params_ver = nvenc_struct_ver(self.api_version, 1);
                if !self.completion_event_0.is_null() {
                    let mut ep = NvEncEventParams {
                        version:          event_params_ver,
                        completion_event: self.completion_event_0,
                        ..Default::default()
                    };
                    let _ = (self.funcs.unregister_async_event)(self.session, &mut ep);
                    CloseHandle(self.completion_event_0);
                    self.completion_event_0 = std::ptr::null_mut();
                }
                if !self.completion_event_1.is_null() {
                    let mut ep = NvEncEventParams {
                        version:          event_params_ver,
                        completion_event: self.completion_event_1,
                        ..Default::default()
                    };
                    let _ = (self.funcs.unregister_async_event)(self.session, &mut ep);
                    CloseHandle(self.completion_event_1);
                    self.completion_event_1 = std::ptr::null_mut();
                }
            }

            if !self.bitstream_buffer_0.is_null() {
                (self.funcs.destroy_bitstream)(self.session, self.bitstream_buffer_0);
            }
            if !self.bitstream_buffer_1.is_null() {
                (self.funcs.destroy_bitstream)(self.session, self.bitstream_buffer_1);
            }
            let _ = (self.funcs.destroy_encoder)(self.session);
        }
    }
}
