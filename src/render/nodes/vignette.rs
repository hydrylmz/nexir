// src/render/nodes/vignette.rs

use std::sync::Arc;
use std::sync::Mutex;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId, ResourceDescriptor, ResolutionSource, TextureAccess};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Push constants for the Vignette compute shader. 32 bytes.
/// Matches WGSL `struct VignetteParams` exactly.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VignetteParams {
    /// Intensity: 0.0 (none) to 1.0 (full dark falloff)
    pub intensity: f32,
    /// Inner radius: typically 0.5 .. 1.5
    pub radius:    f32,
    /// Softness: typically 0.2 .. 0.8
    pub softness:  f32,
    /// Roundness: 1.0 = circular (aspect-corrected), 0.0 = oval (fits rectangular frame)
    pub roundness: f32,
    /// Normalized center X (default 0.5)
    pub center_x:  f32,
    /// Normalized center Y (default 0.5)
    pub center_y:  f32,
    pub width:     u32,
    pub height:    u32,
}

const _: () = assert!(std::mem::size_of::<VignetteParams>() == 32);

impl VignetteParams {
    pub fn default_preset(width: u32, height: u32) -> Self {
        Self {
            intensity: 0.5,
            radius:    0.75,
            softness:  0.45,
            roundness: 1.0,
            center_x:  0.5,
            center_y:  0.5,
            width,
            height,
        }
    }
}

pub struct VignetteNode {
    pub in_rgba:           ResourceId,
    pub out_rgba:          ResourceId,
    pub params:            VignetteParams,
    pipeline:              Arc<wgpu::ComputePipeline>,
    bind_group_layout:     wgpu::BindGroupLayout,
    device:                Arc<wgpu::Device>,
    bg_cache:              Mutex<Option<([ViewId; 2], wgpu::BindGroup)>>,
}

impl VignetteNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        params:         VignetteParams,
    ) -> Self {
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("vignette_bgl"),
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
                label: Some("vignette_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            },
        );

        let shader_mod = shaders.get(BuiltinShader::Vignette);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::Vignette, entry_point: "cs_main" },
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
            bg_cache: Mutex::new(None),
        }
    }

    pub fn set_params(&mut self, new_params: VignetteParams) {
        self.params = new_params;
    }
}

impl RenderNode for VignetteNode {
    fn name(&self) -> &str { "Vignette" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("Vignette_{}", self.out_rgba.0)),
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
                label: Some("vignette_bg"),
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
            label: Some("vignette"),
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
