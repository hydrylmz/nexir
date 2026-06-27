// src/render/nodes/yuv_to_rgb.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};
use crate::timeline::source::ColorSpace;

/// Push constants for the YUV→RGB compute shader. 16 bytes.
/// Must match the WGSL `struct YuvParams` layout exactly.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct YuvParams {
    /// 0 = BT.601, 1 = BT.709, 2 = BT.2020
    pub color_space:   u32,
    /// 0 = full range, 1 = limited range
    pub limited_range: u32,
    /// Texture dimensions (needed for bounds check in shader).
    pub width:         u32,
    pub height:        u32,
}

pub struct YuvToRgbNode {
    /// Y-plane texture (R8Unorm), produced by YuvUploadNode.
    pub in_y:          ResourceId,
    /// UV-plane texture (Rg8Unorm), produced by YuvUploadNode.
    pub in_uv:         ResourceId,
    /// RGBA16Float output texture written by this node.
    pub out_rgba:      ResourceId,
    pub width:         u32,
    pub height:        u32,
    color_space:       ColorSpace,
    limited_range:     bool,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
}

impl YuvToRgbNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_y:           ResourceId,
        in_uv:          ResourceId,
        out_rgba:       ResourceId,
        width:          u32,
        height:         u32,
        color_space:    ColorSpace,
        limited_range:  bool,
    ) -> Self {
        // Step 1 — Build bind group layout: Y(r8), UV(rg8), RGBA(rgba16f)
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("yuv_to_rgb_bgl"),
                entries: &[
                    // binding 0: Y plane (r8unorm, storage read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // binding 1: UV plane (rg8unorm, storage read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rg8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // binding 2: RGBA16Float output (storage write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        );

        // Step 2 — Build pipeline layout with 16-byte push constants
        let pipeline_layout = device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("yuv_to_rgb_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            },
        );

        // Step 3 — Get or compile compute pipeline
        let shader_mod = shaders.get(BuiltinShader::YuvToRgb);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::YuvToRgb, entry_point: "cs_main" },
            &pipeline_layout,
            &shader_mod,
        );

        Self {
            in_y,
            in_uv,
            out_rgba,
            width,
            height,
            color_space,
            limited_range,
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
        }
    }
}

impl RenderNode for YuvToRgbNode {
    fn name(&self) -> &str { "YuvToRgb" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource};
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("YuvToRgb_{}", self.out_rgba.0)),
            size: ResolutionSource::Fixed(self.width, self.height),
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        }));
        builder.read(self.in_y);
        builder.read(self.in_uv);
        builder.write(self.out_rgba);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        // Step 1 — Retrieve resolved texture views
        let y_res    = ctx.get(self.in_y);
        let uv_res   = ctx.get(self.in_uv);
        let rgba_res = ctx.get(self.out_rgba);

        // Step 2 — Create format-specific views for storage textures
        let y_view = y_res.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R8Unorm),
            ..Default::default()
        });
        let uv_view = uv_res.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rg8Unorm),
            ..Default::default()
        });
        let rgba_view = rgba_res.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba16Float),
            ..Default::default()
        });

        // Step 3 — Create per-frame bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv_to_rgb_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&rgba_view) },
            ],
        });

        // Step 4 — Build push constants
        let color_space_u32 = match self.color_space {
            ColorSpace::Bt601  => 0u32,
            ColorSpace::Bt709  => 1u32,
            ColorSpace::Bt2020 => 2u32,
            ColorSpace::Srgb   => 1u32, // fallback to BT.709
        };
        let params = YuvParams {
            color_space:   color_space_u32,
            limited_range: self.limited_range as u32,
            width:         self.width,
            height:        self.height,
        };
        let push_bytes = bytemuck::bytes_of(&params);

        // Step 5 — Begin compute pass and dispatch
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("yuv_to_rgb"),
            timestamp_writes: None,
        });
        ComputePassHelper::dispatch(
            &mut pass, &self.pipeline, &bind_group,
            Some(push_bytes),
            self.width, self.height,
        );
        drop(pass);
    }
}
