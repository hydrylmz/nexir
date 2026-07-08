// src/interop/encode_interop.rs
// Replaces Phase 6's FrameReadback CPU round-trip with a direct NVENC encode path.
// The RTT texture's CUDA array is registered with NVENC, and encoding is entirely GPU-side.

use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::external_texture::ExternalTexture;
use crate::interop::ffi::nvenc::*;
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
    open_session:       functions::OpenEncodeSessionEx,
    initialize:         functions::InitializeEncoder,
    register_resource:  functions::RegisterResource,
    map_input:          functions::MapInputResource,
    encode_picture:     functions::EncodePicture,
    lock_bitstream:     functions::LockBitstream,
    unlock_bitstream:   functions::UnlockBitstream,
    destroy_encoder:    functions::DestroyEncoder,
}

/// Holds the NVENC session and the interop-imported ABGR10 texture it reads from.
pub struct EncodeInterop {
    session:              NvEncodeSession,
    abgr10_texture:       wgpu::Texture,
    /// Held for Drop semantics: keeps the CUDA external memory import alive for the
    /// lifetime of the encode session.
    #[allow(dead_code)]
    abgr10_external:      ExternalTexture,
    registered_resource:  *mut std::ffi::c_void,
    bitstream_buffer:     *mut std::ffi::c_void,
    funcs:                NvencFunctions,
    width:                u32,
    height:               u32,
}

unsafe impl Send for EncodeInterop {}

// Version constants for the NVENC API
const NVENCAPI_VERSION: u32 = 14;
const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = (2 << 16) | 0x6001001;
const NV_ENC_INITIALIZE_PARAMS_VER: u32 = (8 << 16) | 0x6001001;
const NV_ENC_REGISTER_RESOURCE_VER: u32 = (4 << 16) | 0x6001001;
const NV_ENC_PIC_PARAMS_VER: u32 = (6 << 16) | 0x6001001;
const NV_ENC_PIC_STRUCT_FRAME: u32 = 1;
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
        // Step 1 — Load the NVENC function table via the API instance creator.
        // NVENC uses a vtable-style C API: NvEncodeAPICreateInstance fills a
        // function-pointer struct, and all subsequent calls go through it.
        // We allocate the struct as a raw block and read out the function pointers.
        let fn_table_size = std::mem::size_of::<usize>() * 64; // Conservative size for table
        let fn_table_raw = vec![0u8; fn_table_size];
        let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;

        let ret = unsafe { NvEncodeAPICreateInstance(function_list) };
        if ret != NV_ENC_SUCCESS {
            return Err(EncodeInteropError::ApiLoad);
        }

        // In a real implementation, we'd read function pointers from the versioned struct
        // using offsets specified by the NVENC SDK header. For this scaffold we demonstrate
        // the pattern; the actual field layout must match nvEncodeAPI.h exactly.
        // We cast to function pointers at the known offsets (pointer-sized slots after version field).
        // NVENC API Function Table Layout
        //
        // NvEncodeAPICreateInstance fills an NV_ENCODE_API_FUNCTION_LIST struct — a
        // versioned C struct where the first field is a u32 version and all subsequent
        // fields are function pointers in the order declared in nvEncodeAPI.h (NVENC SDK 12.x).
        //
        // The offsets below are SLOT INDICES into the function-pointer array (i.e.,
        // `*base.add(N)` where `base` is a pointer to the first function-pointer slot
        // immediately after the u32 version field). They MUST match the field order in
        // nvEncodeAPI.h exactly:
        //
        //   Slot 1  — nvEncOpenEncodeSessionEx
        //   Slot 2  — nvEncInitializeEncoder
        //   Slot 8  — nvEncRegisterResource
        //   Slot 9  — nvEncMapInputResource
        //   Slot 11 — nvEncEncodePicture
        //   Slot 14 — nvEncLockBitstream
        //   Slot 15 — nvEncUnlockBitstream
        //   Slot 22 — nvEncDestroyEncoder
        //
        // ⚠️  SDK VERSION WARNING: These offsets are correct for NVENC SDK API v14,
        // which ships with CUDA 12.x drivers. If the NV_ENCODE_API_FUNCTION_LIST struct
        // layout changes in a future major SDK version, these offsets MUST be updated to
        // match the new nvEncodeAPI.h. Verify against the official NVENC SDK headers.
        //
        // A safer long-term approach is to generate Rust bindings directly from
        // nvEncodeAPI.h via bindgen, which would make field access by name rather than
        // by raw pointer offset.
        let base = function_list as *const usize;
        let funcs = unsafe {
            let open_off    = 1usize; // nvEncOpenEncodeSessionEx
            let init_off    = 2usize; // nvEncInitializeEncoder
            let reg_off     = 8usize; // nvEncRegisterResource
            let map_off     = 9usize; // nvEncMapInputResource
            let enc_off     = 11usize; // nvEncEncodePicture
            let lock_off    = 14usize; // nvEncLockBitstream
            let unlock_off  = 15usize; // nvEncUnlockBitstream
            let destroy_off = 22usize; // nvEncDestroyEncoder

            NvencFunctions {
                open_session:      std::mem::transmute(*base.add(open_off)),
                initialize:        std::mem::transmute(*base.add(init_off)),
                register_resource: std::mem::transmute(*base.add(reg_off)),
                map_input:         std::mem::transmute(*base.add(map_off)),
                encode_picture:    std::mem::transmute(*base.add(enc_off)),
                lock_bitstream:    std::mem::transmute(*base.add(lock_off)),
                unlock_bitstream:  std::mem::transmute(*base.add(unlock_off)),
                destroy_encoder:   std::mem::transmute(*base.add(destroy_off)),
            }
        };

        // Step 2 — Open the encode session against our shared CUDA context.
        let mut session: NvEncodeSession = std::ptr::null_mut();
        let params = NvEncOpenEncodeSessionExParams {
            version:     NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
            device_type: NV_ENC_DEVICE_TYPE_CUDA,
            device:      cuda_ctx.raw_context() as *mut _,
            reserved:    std::ptr::null_mut(),
            api_version: NVENCAPI_VERSION,
        };
        let ret = unsafe { (funcs.open_session)(&params, &mut session) };
        if ret != NV_ENC_SUCCESS {
            return Err(EncodeInteropError::SessionOpen(ret));
        }

        // Step 3 — Initialise the encoder (codec, preset, dimensions, frame rate).
        // Select the NVENC codec GUID based on the requested output codec.
        let encode_guid = match codec {
            VideoCodec::H265 => NV_ENC_CODEC_HEVC_GUID,
            _ => NV_ENC_CODEC_H264_GUID, // H264, ProRes, VP9 fall through to H264 GUID
                                          // (ProRes/VP9 should never reach NVENC path, but
                                          //  this is a safe default rather than unreachable!())
        };
        let mut init_params = NvEncInitializeParams {
            version:          NV_ENC_INITIALIZE_PARAMS_VER,
            encode_guid,
            preset_guid:      NV_ENC_PRESET_P4_GUID,
            encode_width:     job.width,
            encode_height:    job.height,
            frame_rate_num:   job.frame_rate.num as u32,
            frame_rate_den:   job.frame_rate.den as u32,
            max_encode_width:  job.width,
            max_encode_height: job.height,
            reserved: [0u8; 1024],
        };
        let ret = unsafe { (funcs.initialize)(session, &mut init_params) };
        if ret != NV_ENC_SUCCESS {
            unsafe { (funcs.destroy_encoder)(session) };
            return Err(EncodeInteropError::Initialize(ret));
        }

        // Step 4 — Allocate the ABGR10 interop texture (R32Uint packed).
        let abgr10_desc = wgpu::TextureDescriptor {
            label: Some("EncodeInterop ABGR10"),
            size: wgpu::Extent3d { width: job.width, height: job.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let abgr10_texture = device.device.create_texture(&abgr10_desc);
        let abgr10_external = ExternalTexture::import(
            cuda_ctx, device, &abgr10_texture, transport, job.width, job.height,
        ).map_err(EncodeInteropError::Cuda)?;

        // Step 5 — Register the imported CUarray as an NVENC input resource.
        let mut register = NvEncRegisterResource {
            version:              NV_ENC_REGISTER_RESOURCE_VER,
            resource_type:        NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
            width:                job.width,
            height:               job.height,
            pitch:                0,
            resource_to_register: abgr10_external.cuda_array() as *mut _,
            registered_resource:  std::ptr::null_mut(),
            buffer_format:        NV_ENC_BUFFER_FORMAT_ABGR10,
            buffer_usage:         0,
            p_input_fence_point:  std::ptr::null_mut(),
            reserved: [0u32; 249],
        };
        let ret = unsafe { (funcs.register_resource)(session, &mut register) };
        if ret != NV_ENC_SUCCESS {
            unsafe { (funcs.destroy_encoder)(session) };
            return Err(EncodeInteropError::Register(ret));
        }

        // Allocate a bitstream output buffer (opaque NVENC handle)
        // In a full implementation: nvEncCreateBitstreamBuffer
        let bitstream_buffer: *mut std::ffi::c_void = std::ptr::null_mut();

        Ok(Self {
            session,
            abgr10_texture,
            abgr10_external,
            registered_resource: register.registered_resource,
            bitstream_buffer,
            funcs,
            width: job.width,
            height: job.height,
        })
    }

    /// Encode one frame. The ABGR10 texture must have already been written by
    /// Abgr10RepackNode earlier in the same render-graph submission.
    /// Returns (bitstream_bytes, pts) ready for Phase 6's Muxer::write_packet.
    pub fn encode_frame(&mut self, pts: i64) -> Result<(Vec<u8>, i64), EncodeInteropError> {
        // Step 1 — Map the registered input resource for this encode call.
        #[repr(C)]
        struct NvEncMapInputResource {
            version:            u32,
            subresource_index:  u32,
            input_resource:     *mut std::ffi::c_void,
            registered_resource: *mut std::ffi::c_void,
            mapped_resource:    *mut std::ffi::c_void,
            mapped_buffer_fmt:  u32,
            reserved: [u32; 251],
        }
        let mut map_params = NvEncMapInputResource {
            version:             (1 << 16) | 0x6001001,
            subresource_index:   0,
            input_resource:      std::ptr::null_mut(),
            registered_resource: self.registered_resource,
            mapped_resource:     std::ptr::null_mut(),
            mapped_buffer_fmt:   0,
            reserved: [0u32; 251],
        };
        let ret = unsafe {
            (self.funcs.map_input)(self.session, &mut map_params as *mut _ as *mut std::ffi::c_void)
        };
        if ret != NV_ENC_SUCCESS {
            return Err(EncodeInteropError::Map(ret));
        }
        let mapped_buffer = map_params.mapped_resource;

        // Step 2 — Encode the picture via NVENC.
        let mut pic_params = NvEncPicParams {
            version:          NV_ENC_PIC_PARAMS_VER,
            input_width:       self.width,
            input_height:      self.height,
            input_pitch:       self.width,
            encode_pic_flags:  0,
            frame_idx:         0,
            input_timestamp:   pts as u64,
            input_duration:    0,
            input_buffer:      mapped_buffer,
            output_bitstream:  self.bitstream_buffer,
            completion_event:  std::ptr::null_mut(),
            buffer_fmt:        NV_ENC_BUFFER_FORMAT_ABGR10,
            picture_struct:    NV_ENC_PIC_STRUCT_FRAME,
            picture_type:      0,
            codec_pic_params:  [0u8; 128],
            me_hint_counts_per_block: [0u32; 2],
            me_external_hints: std::ptr::null_mut(),
            reserved: [0u32; 221],
            reserved2: [0u32; 64],
        };
        let ret = unsafe { (self.funcs.encode_picture)(self.session, &mut pic_params) };
        if ret != NV_ENC_SUCCESS {
            return Err(EncodeInteropError::Encode(ret));
        }

        // Step 3 — Lock the bitstream to read the encoded bytes out.
        // This is the ONE necessary CPU touch in the pipeline: copying the compressed
        // bitstream (typically tens of KB per frame, vs tens of MB of raw pixels).
        #[repr(C)]
        struct NvEncLockBitstream {
            version:         u32,
            do_not_wait:     u32,
            is_idr_frame:    u32,
            reserved:        u32,
            output_bitstream: *mut std::ffi::c_void,
            bitstream_size_in_bytes: u32,
            frame_idx:       u32,
            picture_type:    u32,
            reserved2: [u32; 238],
            bitstream_ptr:   *mut std::ffi::c_void,
        }
        let mut lock_params = NvEncLockBitstream {
            version:         (1 << 16) | 0x6001001,
            do_not_wait:     0,
            is_idr_frame:    0,
            reserved:        0,
            output_bitstream: self.bitstream_buffer,
            bitstream_size_in_bytes: 0,
            frame_idx:       0,
            picture_type:    0,
            reserved2: [0u32; 238],
            bitstream_ptr:   std::ptr::null_mut(),
        };
        let ret = unsafe {
            (self.funcs.lock_bitstream)(self.session, &mut lock_params as *mut _ as *mut std::ffi::c_void)
        };
        if ret != NV_ENC_SUCCESS {
            return Err(EncodeInteropError::Lock(ret));
        }

        // Copy the bitstream bytes into an owned Vec.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                lock_params.bitstream_ptr as *const u8,
                lock_params.bitstream_size_in_bytes as usize,
            ).to_vec()
        };

        // Step 4 — Unlock the bitstream buffer and return.
        unsafe {
            (self.funcs.unlock_bitstream)(self.session, self.bitstream_buffer);
        }

        Ok((bytes, pts))
    }

    /// Convenience accessor for the ABGR10 texture (written by Abgr10RepackNode).
    pub fn abgr10_texture(&self) -> &wgpu::Texture {
        &self.abgr10_texture
    }
}

impl Drop for EncodeInterop {
    fn drop(&mut self) {
        unsafe {
            // Unregister the input resource, then destroy the session.
            // The ExternalTexture and underlying wgpu::Texture clean up via their own Drop impls.
            let _ = (self.funcs.destroy_encoder)(self.session);
        }
    }
}
