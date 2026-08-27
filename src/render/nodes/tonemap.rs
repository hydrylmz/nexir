// src/render/nodes/tonemap.rs
// HDR -> SDR tone-mapping compute node.

use std::sync::{Arc, Mutex};
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Configuration for which HDR->SDR conversions to perform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMapMode {
    /// No tone mapping — clamp values that exceed 1.0.
    ClampOnly,
    /// ACES filmic curve (natural highlight roll-off).
    AcesFilmic,
    /// Reinhard luminance-based mapping.
    Reinhard,
}

/// Input transfer function of the source content.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputTransferFn {
    /// Already linear light (e.g. after YUV->RGB with a linear TRC).
    Linear,
    /// SMPTE ST 2084 Perceptual Quantizer (HDR10).
    Pq,
    /// ARIB STD-B67 Hybrid Log-Gamma.
    Hlg,
    /// sRGB gamma (no tone mapping required, pass-through).
    Srgb,
}

/// Gamut conversion applied before tone mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GamutConversion {
    /// No gamut conversion — signal is already in BT.709 space.
    None,
    /// Convert BT.2020 wide-gamut to BT.709 display-referred.
    Bt2020ToBt709,
}

/// Push constants matching struct ToneMapParams in tonemap.wgsl (32 bytes).
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ToneMapPushConstants {
    /// Transfer function selector: 0=Linear, 1=PQ, 2=HLG, 3=sRGB.
    pub transfer_fn:  u32,
    /// Gamut conversion: 0=None (pass-through), 1=BT.2020->BT.709.
    pub gamut_conv:   u32,
    /// Tone-mapping mode: 0=Clamp, 1=ACES, 2=Reinhard, 3=BT.2446a.
    pub tonemap_mode: u32,
    /// Reference peak luminance in nits (e.g. 1000.0 for HDR10, 203.0 for HLG).
    pub peak_nits:    f32,
    /// Output texture width in pixels.
    pub width:        u32,
    /// Output texture height in pixels.
    pub height:       u32,
    /// Exposure gain multiplier (default 1.0).
    pub exposure:     f32,
    pub _pad:         f32,
}

const _: () = assert!(std::mem::size_of::<ToneMapPushConstants>() == 32);

impl ToneMapPushConstants {
    pub fn for_sdr_preview(
        transfer_fn:  InputTransferFn,
        gamut:        GamutConversion,
        tonemap_mode: ToneMapMode,
        peak_nits:    f32,
        width:        u32,
        height:       u32,
    ) -> Self {
        let trc_id = match transfer_fn {
            InputTransferFn::Linear => 0,
            InputTransferFn::Pq     => 1,
            InputTransferFn::Hlg    => 2,
            InputTransferFn::Srgb   => 3,
        };
        let gamut_id = match gamut {
            GamutConversion::None          => 0,
            GamutConversion::Bt2020ToBt709 => 1,
        };
        let tonemap_id = match tonemap_mode {
            ToneMapMode::ClampOnly  => 0,
            ToneMapMode::AcesFilmic => 1,
            ToneMapMode::Reinhard   => 2,
        };
        Self {
            transfer_fn:  trc_id,
            gamut_conv:   gamut_id,
            tonemap_mode: tonemap_id,
            peak_nits,
            width,
            height,
            exposure: 1.0,
            _pad: 0.0,
        }
    }
}

pub struct ToneMapNode {
    pub in_rgba:  ResourceId,
    pub out_rgba: ResourceId,
    pub params:   ToneMapPushConstants,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
    bg_cache:          Mutex<Option<([ViewId; 2], wgpu::BindGroup)>>,
}

impl ToneMapNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        params:         ToneMapPushConstants,
    ) -> Self {
        // Build bind group layout: in (rgba16f read), out (rgba16f write)
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("tonemap_bgl"),
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
                label: Some("tonemap_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..32,
                }],
            },
        );

        let shader_mod = shaders.get(BuiltinShader::ToneMap);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::ToneMap, entry_point: "cs_main" },
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

    /// Build params for an HDR clip being shown on an SDR viewport/monitor.
    pub fn hdr_to_sdr(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_rgba:        ResourceId,
        out_rgba:       ResourceId,
        transfer_fn:    InputTransferFn,
        gamut:          GamutConversion,
        width:          u32,
        height:         u32,
    ) -> Self {
        let params = ToneMapPushConstants::for_sdr_preview(
            transfer_fn,
            gamut,
            ToneMapMode::AcesFilmic,
            1000.0,
            width,
            height,
        );
        Self::new(device, shaders, pipeline_cache, in_rgba, out_rgba, params)
    }
}

impl RenderNode for ToneMapNode {
    fn name(&self) -> &str { "ToneMap" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("ToneMap_{}", self.out_rgba.0)),
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
                label: Some("tonemap_bg"),
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
            label: Some("tonemap"),
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
