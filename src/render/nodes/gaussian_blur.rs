// src/render/nodes/gaussian_blur.rs

use std::sync::Arc;
use std::sync::Mutex;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId, ResourceDescriptor, ResolutionSource, TextureAccess};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Push constants for the Gaussian blur compute shader. 32 bytes = 2 vec4s.
/// Matches WGSL `struct BlurParams` exactly.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlurParams {
    /// Direction unit vector: [1.0, 0.0] for horizontal, [0.0, 1.0] for vertical
    pub direction: [f32; 2],
    /// Blur radius in pixels (typically 1.0 .. 64.0)
    pub radius:    f32,
    /// Standard deviation sigma (typically radius / 2.0 or radius / 3.0)
    pub sigma:     f32,
    /// Canvas width
    pub width:     u32,
    /// Canvas height
    pub height:    u32,
    pub _pad0:     f32,
    pub _pad1:     f32,
}

// Compile-time size assertion: must be exactly 32 bytes
const _: () = assert!(std::mem::size_of::<BlurParams>() == 32);

impl BlurParams {
    pub fn horizontal(radius: f32, sigma: f32, width: u32, height: u32) -> Self {
        Self {
            direction: [1.0, 0.0],
            radius,
            sigma: if sigma <= 0.0 { radius * 0.5 } else { sigma },
            width,
            height,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    pub fn vertical(radius: f32, sigma: f32, width: u32, height: u32) -> Self {
        Self {
            direction: [0.0, 1.0],
            radius,
            sigma: if sigma <= 0.0 { radius * 0.5 } else { sigma },
            width,
            height,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}

/// A single 1D directional blur pass (horizontal or vertical).
pub struct BlurPassNode {
    pub in_rgba:           ResourceId,
    pub out_rgba:          ResourceId,
    pub params:            BlurParams,
    label:                 String,
    pipeline:              Arc<wgpu::ComputePipeline>,
    bind_group_layout:     wgpu::BindGroupLayout,
    device:                Arc<wgpu::Device>,
    bg_cache:              Mutex<Option<([ViewId; 2], wgpu::BindGroup)>>,
}

impl BlurPassNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        params:         BlurParams,
        label:          &str,
    ) -> Self {
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("blur_pass_bgl"),
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

        let pipeline_layout = device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("blur_pass_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            },
        );

        let shader_mod = shaders.get(BuiltinShader::GaussianBlur);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::GaussianBlur, entry_point: "cs_main" },
            &pipeline_layout,
            &shader_mod,
        );

        Self {
            in_rgba,
            out_rgba,
            params,
            label: label.to_string(),
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
            bg_cache: Mutex::new(None),
        }
    }

    pub fn set_params(&mut self, new_params: BlurParams) {
        self.params = new_params;
    }
}

impl RenderNode for BlurPassNode {
    fn name(&self) -> &str {
        &self.label
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("{}_{}", self.label, self.out_rgba.0)),
            size: ResolutionSource::Fixed(self.params.width, self.params.height),
            format: wgpu::TextureFormat::Rgba16Float,
        }));
        builder.read(self.in_rgba, TextureAccess::StorageRead);
        builder.write(self.out_rgba, TextureAccess::StorageWrite);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        let in_res  = ctx.get(self.in_rgba);
        let out_res = ctx.get(self.out_rgba);

        let mut cache = self.bg_cache.lock().unwrap();
        let cache_key = [in_res.view_id, out_res.view_id];
        
        if cache.is_none() || cache.as_ref().unwrap().0 != cache_key {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blur_pass_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(in_res.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(out_res.view) },
                ],
            });
            *cache = Some((cache_key, bind_group));
        }

        let bind_group = &cache.as_ref().unwrap().1;
        let push_bytes = bytemuck::bytes_of(&self.params);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&self.label),
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
