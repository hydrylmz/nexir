// src/render/nodes/chroma_key.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::context::RenderContext;
use std::sync::Mutex;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Push constants for chroma key. 32 bytes.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ChromaKeyParams {
    /// Key colour hue in degrees [0, 360). Green screen: ~120°. Blue: ~240°.
    pub key_hue:        f32,
    /// Hue tolerance in degrees. Pixels within this distance are keyed out.
    pub tolerance:      f32,
    /// Softness width in degrees. Must be < tolerance.
    pub softness:       f32,
    /// Minimum saturation for a pixel to be keyed (avoids keying grey pixels).
    pub min_saturation: f32,
    /// Minimum value (brightness) for a pixel to be keyed (avoids keying black).
    pub min_value:      f32,
    /// Spill suppression. 0.0 = none, 1.0 = full.
    pub spill_suppress: f32,
    /// Texture dimensions for bounds check.
    pub width:          u32,
    pub height:         u32,
}

// Compile-time size assertion: must be exactly 32 bytes
const _: () = assert!(std::mem::size_of::<ChromaKeyParams>() == 32);

impl ChromaKeyParams {
    /// Green-screen preset.
    pub fn green_screen(width: u32, height: u32) -> Self {
        Self {
            key_hue:        120.0,
            tolerance:      40.0,
            softness:       10.0,
            min_saturation: 0.15,
            min_value:      0.08,
            spill_suppress: 0.3,
            width,
            height,
        }
    }

    /// Blue-screen preset.
    pub fn blue_screen(width: u32, height: u32) -> Self {
        Self {
            key_hue:        240.0,
            tolerance:      40.0,
            softness:       10.0,
            min_saturation: 0.15,
            min_value:      0.08,
            spill_suppress: 0.3,
            width,
            height,
        }
    }

    /// Validate that softness < tolerance (otherwise smoothstep is degenerate).
    pub fn validate(&self) -> Result<(), ChromaKeyError> {
        if self.tolerance <= 0.0 {
            return Err(ChromaKeyError::ZeroTolerance);
        }
        if self.softness >= self.tolerance {
            return Err(ChromaKeyError::SoftnessExceedsTolerance {
                softness:  self.softness,
                tolerance: self.tolerance,
            });
        }
        Ok(())
    }
}

pub struct ChromaKeyNode {
    pub in_rgba:       ResourceId,
    pub out_rgba:      ResourceId,
    pub params:        ChromaKeyParams,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
    bg_cache:          Mutex<Option<([ViewId; 2], wgpu::BindGroup)>>,
}

impl ChromaKeyNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        params:         ChromaKeyParams,
    ) -> Self {
        // Same two-binding pattern as ColorCorrectionNode
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("chroma_key_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
                label: Some("chroma_key_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            },
        );

        let shader_mod = shaders.get(BuiltinShader::ChromaKey);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::ChromaKey, entry_point: "cs_main" },
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

    pub fn set_params(&mut self, params: ChromaKeyParams) {
        self.params = params;
    }
}

impl RenderNode for ChromaKeyNode {
    fn name(&self) -> &str { "ChromaKey" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::TextureAccess;
        builder.read(self.in_rgba, TextureAccess::Sampled);
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

        // Views are already native format from pool
        let mut cache = self.bg_cache.lock().unwrap();
        let cache_key = [in_res.view_id, out_res.view_id];
        
        if cache.is_none() || cache.as_ref().unwrap().0 != cache_key {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("chroma_key_bg"),
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
            label: Some("chroma_key"),
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

#[derive(Debug)]
pub enum ChromaKeyError {
    SoftnessExceedsTolerance { softness: f32, tolerance: f32 },
    ZeroTolerance,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_softness_exceeds_tolerance() {
        let params = ChromaKeyParams {
            key_hue: 120.0,
            tolerance: 10.0,
            softness: 15.0,
            min_saturation: 0.1,
            min_value: 0.05,
            spill_suppress: 0.0,
            width: 1,
            height: 1,
        };
        assert!(matches!(
            params.validate(),
            Err(ChromaKeyError::SoftnessExceedsTolerance { .. })
        ));
    }

    #[test]
    fn validate_rejects_zero_tolerance() {
        let params = ChromaKeyParams {
            key_hue: 120.0, tolerance: 0.0, softness: 0.0,
            min_saturation: 0.1, min_value: 0.05, spill_suppress: 0.0,
            width: 1, height: 1,
        };
        assert!(matches!(params.validate(), Err(ChromaKeyError::ZeroTolerance)));
    }

    #[test]
    fn validate_accepts_valid_green_screen() {
        assert!(ChromaKeyParams::green_screen(1, 1).validate().is_ok());
    }
}
