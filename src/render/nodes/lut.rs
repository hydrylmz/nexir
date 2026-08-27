// src/render/nodes/lut.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::context::RenderContext;
use std::sync::Mutex;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};
use crate::colour::lut_parser::Lut3D;
use half::f16;

/// Push constants for the LUT node. 16 bytes.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LutParams {
    /// LUT strength: 0.0 = bypass, 1.0 = full LUT.
    pub strength: f32,
    pub width:    u32,
    pub height:   u32,
    pub _pad:     f32,
}

pub struct LutNode {
    pub in_rgba:       ResourceId,
    pub out_rgba:      ResourceId,
    pub params:        LutParams,
    lut_texture:       wgpu::Texture,
    lut_view:          wgpu::TextureView,
    lut_sampler:       wgpu::Sampler,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
    bg_cache:          Mutex<Option<([ViewId; 2], wgpu::BindGroup)>>,
}

impl LutNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        lut:            &Lut3D,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        strength:       f32,
    ) -> Self {
        let n = lut.size;

        // Step 1 — Create 3D texture
        let lut_texture = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lut_3d"),
            size: wgpu::Extent3d {
                width:  n,
                height: n,
                depth_or_array_layers: n,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Step 2 — Convert f32 LUT data to f16 and upload
        let mut lut_bytes: Vec<u8> = Vec::with_capacity((n * n * n * 8) as usize); // 4 x f16 per texel
        for entry in &lut.data {
            lut_bytes.extend_from_slice(&f16::from_f32(entry[0]).to_le_bytes());
            lut_bytes.extend_from_slice(&f16::from_f32(entry[1]).to_le_bytes());
            lut_bytes.extend_from_slice(&f16::from_f32(entry[2]).to_le_bytes());
            lut_bytes.extend_from_slice(&f16::from_f32(1.0).to_le_bytes()); // alpha
        }

        device.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &lut_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row:  Some(n * 8),  // 4 channels × 2 bytes (f16)
                rows_per_image: Some(n),       // height of one 2D slice
            },
            wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n },
        );

        // Step 3 — Create 3D sampler (trilinear, clamp-to-edge)
        let lut_sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lut_sampler"),
            mag_filter:    wgpu::FilterMode::Linear,
            min_filter:    wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let lut_view = lut_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        // Step 4 — Build bind group layout: in (storage read), out (storage write), lut (3D texture), sampler
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("lut_bgl"),
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
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        );

        // Step 5 — Compile pipeline with 16-byte push constants
        let pipeline_layout = device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("lut_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            },
        );

        let shader_mod = shaders.get(BuiltinShader::Lut3D);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::Lut3D, entry_point: "cs_main" },
            &pipeline_layout,
            &shader_mod,
        );

        // Width and height refer to the 2D image being processed (not the LUT)
        // We accept these as part of LutParams at construction time.
        // However we don't know them yet — they'll be set by EffectChainBuilder.
        let params = LutParams { strength, width: 0, height: 0, _pad: 0.0 };

        Self {
            in_rgba,
            out_rgba,
            params,
            lut_texture,
            lut_view,
            lut_sampler,
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
            bg_cache: Mutex::new(None),
        }
    }

    /// Set image dimensions (must be called before the node is used in a graph).
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.params.width  = width;
        self.params.height = height;
    }
}

impl RenderNode for LutNode {
    fn name(&self) -> &str { "Lut3D" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("Lut3D_{}", self.out_rgba.0)),
            size: ResolutionSource::Fixed(self.params.width, self.params.height),
            format: wgpu::TextureFormat::Rgba16Float,
        }));
        builder.read(self.in_rgba, TextureAccess::StorageRead);
        builder.write(self.out_rgba, TextureAccess::StorageWrite);
        // lut_texture is node-owned, not a graph resource
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        let in_res  = ctx.get(self.in_rgba);
        let out_res = ctx.get(self.out_rgba);

        // Views are already native format from pool

        let mut cache = self.bg_cache.lock().unwrap();
        let cache_key = [in_res.view_id, out_res.view_id];

        if cache.is_none() || cache.as_ref().unwrap().0 != cache_key {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lut_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(in_res.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_res.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.lut_view) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
                ],
            });
            *cache = Some((cache_key, bind_group));
        }

        let bind_group = &cache.as_ref().unwrap().1;

        let push_bytes = bytemuck::bytes_of(&self.params);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lut"),
            timestamp_writes: None,
        });
        ComputePassHelper::dispatch(
            &mut pass, &self.pipeline, bind_group,
            Some(push_bytes),
            self.params.width, self.params.height,
        );
        drop(pass);
    }
}
