// src/render/nodes/yuv_to_rgb.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::context::RenderContext;
use std::sync::Mutex;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};
use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients};

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
    color_info:        ColorInfo,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
    bg_cache:          Mutex<Option<([ViewId; 3], wgpu::BindGroup)>>,
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
        color_info:     ColorInfo,
    ) -> Self {
        // Step 1 — Build bind group layout: Y(r8), UV(rg8), RGBA(rgba16f)
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("yuv_to_rgb_bgl"),
                entries: &[
                    // binding 0: Y plane
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
                    // binding 1: UV plane
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
            color_info,
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
            bg_cache: Mutex::new(None),
        }
    }
}

impl RenderNode for YuvToRgbNode {
    fn name(&self) -> &str { "YuvToRgb" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("YuvToRgb_{}", self.out_rgba.0)),
            size: ResolutionSource::Fixed(self.width, self.height),
            format: wgpu::TextureFormat::Rgba16Float,
        }));
        builder.read(self.in_y, TextureAccess::Sampled);
        builder.read(self.in_uv, TextureAccess::Sampled);
        builder.write(self.out_rgba, TextureAccess::StorageWrite);
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

        // Step 2 — Create per-frame bind group if cache missed
        let mut cache = self.bg_cache.lock().unwrap();
        let cache_key = [y_res.view_id, uv_res.view_id, rgba_res.view_id];
        
        if cache.is_none() || cache.as_ref().unwrap().0 != cache_key {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yuv_to_rgb_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(y_res.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(uv_res.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(rgba_res.view) },
                ],
            });
            *cache = Some((cache_key, bind_group));
        }

        let bind_group = &cache.as_ref().unwrap().1;

        // Step 4 — Build push constants from ColorInfo
        let color_space_u32 = match self.color_info.effective_matrix(self.width, self.height) {
            MatrixCoefficients::Bt601  => 0u32,
            MatrixCoefficients::Bt709  => 1u32,
            MatrixCoefficients::Bt2020 => 2u32,
            MatrixCoefficients::Unknown => 1u32,
        };
        let limited_range = match self.color_info.effective_range() {
            ColorRange::Full    => 0u32,
            ColorRange::Limited => 1u32,
            ColorRange::Unknown => 1u32,
        };
        let params = YuvParams {
            color_space:   color_space_u32,
            limited_range,
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
            &mut pass, &self.pipeline, bind_group,
            Some(push_bytes),
            self.width, self.height,
        );
        drop(pass);
    }
}
