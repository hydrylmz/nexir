// src/render/nodes/color_correction.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Push constants for the color correction compute shader. 64 bytes = 1 cache line.
/// Matches WGSL `struct ColorParams` exactly.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ColorCorrectionParams {
    /// Per-channel lift [R, G, B]. Range typically [-0.5, 0.5]. Alpha unused.
    pub lift:       [f32; 4],
    /// Per-channel gamma [R, G, B]. Value 1.0 = no change.
    pub gamma:      [f32; 4],
    /// Per-channel gain [R, G, B]. Value 1.0 = no change.
    pub gain:       [f32; 4],
    /// Scalar saturation. 0.0 = grey, 1.0 = no change.
    pub saturation: f32,
    /// Texture dimensions for bounds check.
    pub width:      u32,
    pub height:     u32,
    pub _pad:       f32,
}

// Compile-time size assertion: must be exactly 64 bytes
const _: () = assert!(std::mem::size_of::<ColorCorrectionParams>() == 64);

impl ColorCorrectionParams {
    /// Identity params — no change to any channel.
    pub fn identity(width: u32, height: u32) -> Self {
        Self {
            lift:       [0.0, 0.0, 0.0, 0.0],
            gamma:      [1.0, 1.0, 1.0, 1.0],
            gain:       [1.0, 1.0, 1.0, 1.0],
            saturation: 1.0,
            width,
            height,
            _pad:       0.0,
        }
    }
}

pub struct ColorCorrectionNode {
    pub in_rgba:       ResourceId,
    pub out_rgba:      ResourceId,
    pub params:        ColorCorrectionParams,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
}

impl ColorCorrectionNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        params:         ColorCorrectionParams,
    ) -> Self {
        // Step 1 — Build bind group layout: in (rgba16f read), out (rgba16f write)
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("color_correction_bgl"),
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
                ],
            },
        );

        // Step 2 — Build pipeline layout with 64-byte push constants
        let pipeline_layout = device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("color_correction_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..64,
                }],
            },
        );

        // Step 3 — Compile pipeline
        let shader_mod = shaders.get(BuiltinShader::ColorCorrection);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::ColorCorrection, entry_point: "cs_main" },
            &pipeline_layout,
            &shader_mod,
        );

        Self {
            in_rgba,
            out_rgba,
            params,
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
        }
    }

    /// Update parameters between frames (no pipeline recompile needed).
    pub fn set_params(&mut self, new_params: ColorCorrectionParams) {
        self.params = new_params;
    }
}

impl RenderNode for ColorCorrectionNode {
    fn name(&self) -> &str { "ColorCorrection" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(self.in_rgba);
        builder.write(self.out_rgba);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        let in_res  = ctx.get(self.in_rgba);
        let out_res = ctx.get(self.out_rgba);

        let in_view = in_res.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba16Float),
            ..Default::default()
        });
        let out_view = out_res.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba16Float),
            ..Default::default()
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color_correction_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&in_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out_view) },
            ],
        });

        let push_bytes = bytemuck::bytes_of(&self.params);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("color_correction"),
            timestamp_writes: None,
        });
        ComputePassHelper::dispatch(
            &mut pass, &self.pipeline, &bind_group,
            Some(push_bytes),
            self.params.width, self.params.height,
        );
        drop(pass);
    }
}
