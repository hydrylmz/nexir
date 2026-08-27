// src/render/effect_chain.rs

use std::collections::HashMap;
use crate::render::device::GpuDevice;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceId, ResourceDescriptor, ResolutionSource};
use crate::render::compute::ComputePipelineCache;
use crate::render::shader::registry::ShaderRegistry;
use crate::render::nodes::color_correction::{ColorCorrectionNode, ColorCorrectionParams};
use crate::render::nodes::chroma_key::{ChromaKeyNode, ChromaKeyParams};
use crate::render::nodes::gaussian_blur::{BlurPassNode, BlurParams};
use crate::render::nodes::sharpen::{SharpenNode, SharpenParams};
use crate::render::nodes::vignette::{VignetteNode, VignetteParams};
use crate::timeline::effect::{EffectKind, EffectStore};
use crate::timeline::transform::EffectParams;

pub struct EffectChainBuilder<'a> {
    device:         &'a GpuDevice,
    shaders:        &'a ShaderRegistry,
    pipeline_cache: &'a ComputePipelineCache,
    canvas_width:   u32,
    canvas_height:  u32,
}

impl<'a> EffectChainBuilder<'a> {
    pub fn new(
        device:         &'a GpuDevice,
        shaders:        &'a ShaderRegistry,
        pipeline_cache: &'a ComputePipelineCache,
        canvas_width:   u32,
        canvas_height:  u32,
    ) -> Self {
        Self { device, shaders, pipeline_cache, canvas_width, canvas_height }
    }

    /// Build a chain of nodes for a clip's effect list.
    ///
    /// Returns the node list, the final output ResourceId, and descriptors for transient textures.
    pub fn build(
        &self,
        store:        &EffectStore,
        effect_start: u32,
        effect_count: u16,
        in_rgba:      ResourceId,
        id_counter:   &mut u32,
    ) -> EffectChainOutput {
        let mut nodes: Vec<Box<dyn RenderNode>> = Vec::new();
        let mut descriptors: HashMap<ResourceId, ResourceDescriptor> = HashMap::new();
        let mut current_input = in_rgba;

        let rgba16_desc = || ResourceDescriptor {
            label: None,
            size: ResolutionSource::Fixed(self.canvas_width, self.canvas_height),
            format: wgpu::TextureFormat::Rgba16Float,
        };

        for (_effect_id, kind, params) in store.iter_clip_effects(effect_start, effect_count) {
            match kind {
                EffectKind::ColorCorrection => {
                    let transient_out = ResourceId::next(id_counter);
                    descriptors.insert(transient_out, rgba16_desc());

                    let cc_params = Self::decode_color_correction(params, self.canvas_width, self.canvas_height);
                    let node = ColorCorrectionNode::new(
                        self.device,
                        self.shaders,
                        self.pipeline_cache,
                        current_input,
                        transient_out,
                        cc_params,
                    );
                    nodes.push(Box::new(node));
                    current_input = transient_out;
                }

                EffectKind::GaussianBlur => {
                    let radius = if params.data[0].is_finite() && params.data[0] > 0.0 { params.data[0] } else { 10.0 };
                    let sigma = if params.data[1].is_finite() && params.data[1] > 0.0 { params.data[1] } else { radius * 0.5 };

                    let h_out = ResourceId::next(id_counter);
                    descriptors.insert(h_out, rgba16_desc());
                    let h_node = BlurPassNode::new(
                        self.device,
                        self.shaders,
                        self.pipeline_cache,
                        current_input,
                        h_out,
                        BlurParams::horizontal(radius, sigma, self.canvas_width, self.canvas_height),
                        "BlurH",
                    );
                    nodes.push(Box::new(h_node));

                    let v_out = ResourceId::next(id_counter);
                    descriptors.insert(v_out, rgba16_desc());
                    let v_node = BlurPassNode::new(
                        self.device,
                        self.shaders,
                        self.pipeline_cache,
                        h_out,
                        v_out,
                        BlurParams::vertical(radius, sigma, self.canvas_width, self.canvas_height),
                        "BlurV",
                    );
                    nodes.push(Box::new(v_node));
                    current_input = v_out;
                }

                EffectKind::Sharpen => {
                    let amount = if params.data[0].is_finite() { params.data[0] } else { 0.5 };
                    let transient_out = ResourceId::next(id_counter);
                    descriptors.insert(transient_out, rgba16_desc());

                    let node = SharpenNode::new(
                        self.device,
                        self.shaders,
                        self.pipeline_cache,
                        current_input,
                        transient_out,
                        SharpenParams::new(amount, self.canvas_width, self.canvas_height),
                    );
                    nodes.push(Box::new(node));
                    current_input = transient_out;
                }

                EffectKind::Vignette => {
                    let intensity = if params.data[0].is_finite() { params.data[0] } else { 0.5 };
                    let radius = if params.data[1].is_finite() && params.data[1] > 0.0 { params.data[1] } else { 0.75 };
                    let softness = if params.data[2].is_finite() && params.data[2] > 0.0 { params.data[2] } else { 0.45 };
                    let roundness = if params.data[3].is_finite() { params.data[3] } else { 1.0 };
                    let center_x = if params.data[4].is_finite() { params.data[4] } else { 0.5 };
                    let center_y = if params.data[5].is_finite() { params.data[5] } else { 0.5 };

                    let transient_out = ResourceId::next(id_counter);
                    descriptors.insert(transient_out, rgba16_desc());

                    let node = VignetteNode::new(
                        self.device,
                        self.shaders,
                        self.pipeline_cache,
                        current_input,
                        transient_out,
                        VignetteParams {
                            intensity,
                            radius,
                            softness,
                            roundness,
                            center_x,
                            center_y,
                            width: self.canvas_width,
                            height: self.canvas_height,
                        },
                    );
                    nodes.push(Box::new(node));
                    current_input = transient_out;
                }

                EffectKind::ChromaKey => {
                    let transient_out = ResourceId::next(id_counter);
                    descriptors.insert(transient_out, rgba16_desc());

                    let ck_params = Self::decode_chroma_key(params, self.canvas_width, self.canvas_height);
                    let node = ChromaKeyNode::new(
                        self.device,
                        self.shaders,
                        self.pipeline_cache,
                        current_input,
                        transient_out,
                        ck_params,
                    );
                    nodes.push(Box::new(node));
                    current_input = transient_out;
                }

                EffectKind::Custom(_) | EffectKind::Transform2D => {}
            }
        }

        EffectChainOutput {
            nodes,
            final_out: current_input,
            descriptors,
        }
    }

    /// Decode ColorCorrectionParams from the raw EffectParams float array.
    ///
    /// Layout:
    ///   data[0..3]:  lift  [R, G, B]
    ///   data[4..7]:  gamma [R, G, B]
    ///   data[8..11]: gain  [R, G, B]
    ///   data[12]:    saturation
    ///   data[13]:    brightness
    ///   data[14]:    contrast
    ///   data[15]:    hue_shift
    fn decode_color_correction(
        params: &EffectParams,
        width:  u32,
        height: u32,
    ) -> ColorCorrectionParams {
        let d = &params.data;

        let mut gamma_r = d[4];
        let mut gamma_g = d[5];
        let mut gamma_b = d[6];
        if !gamma_r.is_finite() || gamma_r <= 0.0 { gamma_r = 1.0; }
        if !gamma_g.is_finite() || gamma_g <= 0.0 { gamma_g = 1.0; }
        if !gamma_b.is_finite() || gamma_b <= 0.0 { gamma_b = 1.0; }

        let saturation = if d[12].is_finite() { d[12] } else { 1.0 };
        let brightness = if d[13].is_finite() { d[13] } else { 0.0 };
        let contrast   = if d[14].is_finite() && d[14] >= 0.0 { d[14] } else { 1.0 };
        let hue_shift  = if d[15].is_finite() { d[15] } else { 0.0 };

        ColorCorrectionParams {
            lift:       [d[0],   d[1],   d[2],   0.0],
            gamma:      [gamma_r, gamma_g, gamma_b, 1.0],
            gain:       [d[8],   d[9],   d[10],  1.0],
            saturation,
            brightness,
            contrast,
            hue_shift,
            width,
            height,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    /// Decode ChromaKeyParams from the raw EffectParams float array.
    fn decode_chroma_key(
        params: &EffectParams,
        width:  u32,
        height: u32,
    ) -> ChromaKeyParams {
        let d = &params.data;
        let ck_params = ChromaKeyParams {
            key_hue:        d[0],
            tolerance:      d[1],
            softness:       d[2],
            min_saturation: d[3],
            min_value:      d[4],
            spill_suppress: d[5],
            width,
            height,
        };

        if ck_params.validate().is_err() {
            return ChromaKeyParams::green_screen(width, height);
        }

        ck_params
    }
}

pub struct EffectChainOutput {
    /// Nodes to register with the RenderGraphCompiler, in chain order.
    pub nodes:       Vec<Box<dyn RenderNode>>,
    /// The ResourceId of the final output texture (input to the compositor).
    pub final_out:   ResourceId,
    /// Descriptors for all transient textures created by this chain.
    pub descriptors: HashMap<ResourceId, ResourceDescriptor>,
}
