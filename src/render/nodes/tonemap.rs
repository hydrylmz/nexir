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
    /// Leave linear light untouched (no compression, no clamp above 1.0).
    ///
    /// This is the mode an HDR *output* needs: the highlights above diffuse white
    /// are the whole point of the format, and any of the three modes above would
    /// flatten them before the PQ/HLG encode.
    Passthrough,
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

impl InputTransferFn {
    /// Shader selector id.  Shared by the input and output slots — the WGSL uses
    /// one mapping for both.
    fn shader_id(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Pq     => 1,
            Self::Hlg    => 2,
            Self::Srgb   => 3,
        }
    }

    /// The transfer function matching a clip's `ColorInfo`.
    ///
    /// Everything that is not explicitly PQ or HLG is treated as display-encoded
    /// SDR, because that is what `yuv_to_rgb.wgsl` leaves in the texture: it
    /// applies the colour matrix but no EOTF, so the values are still R'G'B' in
    /// the source's own curve.
    pub fn from_color_info(color: &crate::timeline::source::ColorInfo) -> Self {
        use crate::timeline::source::TransferFunction;
        match color.transfer_fn {
            TransferFunction::Pq     => Self::Pq,
            TransferFunction::Hlg    => Self::Hlg,
            TransferFunction::Linear => Self::Linear,
            _                        => Self::Srgb,
        }
    }
}

/// Gamut conversion applied before tone mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GamutConversion {
    /// No gamut conversion — signal is already in the target's primaries.
    None,
    /// Convert BT.2020 wide-gamut to BT.709 display-referred.
    Bt2020ToBt709,
    /// Convert BT.709 to BT.2020 — for placing an SDR clip in an HDR timeline.
    Bt709ToBt2020,
}

/// Push constants matching struct ToneMapParams in tonemap.wgsl (48 bytes).
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ToneMapPushConstants {
    /// Input transfer function selector: 0=Linear, 1=PQ, 2=HLG, 3=sRGB.
    pub transfer_fn:  u32,
    /// Gamut conversion: 0=None, 1=BT.2020->BT.709, 2=BT.709->BT.2020.
    pub gamut_conv:   u32,
    /// Tone-mapping mode: 0=Clamp, 1=ACES, 2=Reinhard, 3=Passthrough.
    pub tonemap_mode: u32,
    /// Reference peak luminance in nits (e.g. 1000.0 for HDR10, 203.0 for HLG).
    pub peak_nits:    f32,
    /// Output texture width in pixels.
    pub width:        u32,
    /// Output texture height in pixels.
    pub height:       u32,
    /// Exposure gain multiplier (default 1.0).
    pub exposure:     f32,
    /// Output transfer function selector, same encoding as `transfer_fn`.
    ///
    /// P1.7 — this is what makes the node usable in both directions. It used not
    /// to exist: the shader always wrote linear light, which the rest of the chain
    /// then misread as display-encoded.
    pub output_transfer_fn: u32,
    /// Luminance that linear 1.0 stands for, in cd/m². 100.0 by convention.
    pub sdr_reference_nits: f32,
    pub _pad0:        f32,
    pub _pad1:        f32,
    pub _pad2:        f32,
}

const _: () = assert!(std::mem::size_of::<ToneMapPushConstants>() == 48);

/// Byte size of the tone-map push-constant block, shared by the pipeline layout
/// and the dispatch so the two can never disagree.
pub const TONEMAP_PUSH_CONSTANT_SIZE: u32 = 48;

/// Conventional SDR diffuse-white reference, in cd/m².
pub const SDR_REFERENCE_NITS: f32 = 100.0;

impl ToneMapPushConstants {
    /// HDR (or SDR) input -> SDR display-encoded output.
    ///
    /// Output is sRGB-encoded, NOT linear: every consumer of a tone-mapped
    /// texture in this engine (the preview blit, the RGBA16F->RGBA8 conversion in
    /// the CPU encoder, the ABGR10 repack) treats its input as display-encoded.
    pub fn for_sdr_preview(
        transfer_fn:  InputTransferFn,
        gamut:        GamutConversion,
        tonemap_mode: ToneMapMode,
        peak_nits:    f32,
        width:        u32,
        height:       u32,
    ) -> Self {
        Self {
            transfer_fn:  transfer_fn.shader_id(),
            gamut_conv:   gamut.shader_id(),
            tonemap_mode: tonemap_mode.shader_id(),
            peak_nits,
            width,
            height,
            exposure: 1.0,
            output_transfer_fn: InputTransferFn::Srgb.shader_id(),
            sdr_reference_nits: SDR_REFERENCE_NITS,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Convert a clip into an HDR **output** curve, preserving its highlights.
    ///
    /// P1.7 — this is the HDR export path. Tone mapping is `Passthrough`, so
    /// nothing is compressed into [0,1]; the pass exists to line the clip's
    /// transfer function and primaries up with the output's.
    ///
    /// Both directions matter for a mixed timeline: an HDR clip going to an HDR
    /// output is (near) identity, while an SDR Rec.709 clip is decoded, moved to
    /// BT.2020 primaries and re-encoded to PQ so it sits at its correct diffuse
    /// brightness in an HDR file instead of being stretched to peak white.
    pub fn for_hdr_output(
        input_transfer_fn:  InputTransferFn,
        gamut:              GamutConversion,
        output_transfer_fn: InputTransferFn,
        peak_nits:          f32,
        width:              u32,
        height:             u32,
    ) -> Self {
        Self {
            transfer_fn:  input_transfer_fn.shader_id(),
            gamut_conv:   gamut.shader_id(),
            tonemap_mode: ToneMapMode::Passthrough.shader_id(),
            peak_nits,
            width,
            height,
            exposure: 1.0,
            output_transfer_fn: output_transfer_fn.shader_id(),
            sdr_reference_nits: SDR_REFERENCE_NITS,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

impl GamutConversion {
    fn shader_id(self) -> u32 {
        match self {
            Self::None          => 0,
            Self::Bt2020ToBt709 => 1,
            Self::Bt709ToBt2020 => 2,
        }
    }

    /// Pick the conversion that takes `from` primaries to `to` primaries.
    pub fn between(
        from: crate::timeline::source::ColorPrimaries,
        to:   crate::timeline::source::ColorPrimaries,
    ) -> Self {
        use crate::timeline::source::ColorPrimaries;
        match (from, to) {
            (ColorPrimaries::Bt2020, ColorPrimaries::Bt709)  => Self::Bt2020ToBt709,
            (ColorPrimaries::Bt709,  ColorPrimaries::Bt2020) => Self::Bt709ToBt2020,
            // Unknown on either side means "no reliable information", and guessing
            // a matrix is worse than leaving the primaries alone.
            _ => Self::None,
        }
    }
}

impl ToneMapMode {
    fn shader_id(self) -> u32 {
        match self {
            Self::ClampOnly   => 0,
            Self::AcesFilmic  => 1,
            Self::Reinhard    => 2,
            Self::Passthrough => 3,
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
                    range: 0..TONEMAP_PUSH_CONSTANT_SIZE,
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
