// src/render/nodes/fused_grade.rs
//
// P2.3 — colour correction + 3D LUT + chroma key in ONE compute pass.
//
// WHY, AND WHAT THE UPPER BOUND IS. On benchmark 7 the three passes this replaces are
// the graph's three dearest per layer: 1.481 + 1.553 + 1.455 = 4.489 ms of a 7.366 ms
// graph, in a ~36 ms frame (AGENTS.md gotcha 27's per-instance table). So the ceiling
// on this change is ~12% of a frame and it CANNOT close the 4K60 gate — Task I's floor
// says the four-source decode alone needs 31.7 ms against a 16.67 ms budget. It is
// worth doing because it is bandwidth the graph does not have to spend, and it is
// worth doing THIS way because the arithmetic is copied rather than rewritten.
//
// WHAT IT ACTUALLY SAVES. Each of the three passes reads a full canvas-sized
// `rgba16float` texture and writes another; at 4K that is 66.4 MB in and 66.4 MB out,
// three times. Fused it is one read and one write, so the traffic drops from 6 canvas
// crossings to 2 and the two intermediates disappear from the pool entirely. Gotcha
// 27 is the direct evidence that this is what those passes spend: the same shaders on
// the same clocks cost 5.881 ms on compressible content and 11.366 ms on noisy, and
// the implied per-pass bandwidth crosses this card's 224 GB/s bus rate. **A fusion
// measured only on low-entropy fixtures would therefore be measuring the compressor**
// — the plan's two-content-set rule exists for exactly this and
// `target/.../nexir_media_bars` is kept as the control.
//
// THREE THINGS THAT ARE LOAD-BEARING, each with the failure it prevents:
//
//   * **The pool's bucket cap must be re-read afterwards** (gotcha 14). Fusing removes
//     two canvas-sized `Rgba16Float` intermediates per layer, so benchmark 7's
//     `peak bucket 16/32` becomes 8/32 — that is a REDUCTION, which is safe, but the
//     rule is that the number comes from a bench run rather than from this comment.
//   * **`params.width`/`height` are set by the constructor, not left at zero.** This
//     is `LutNode`'s gotcha 10 with three chances to happen instead of one, so the
//     dimensions are constructor arguments and `declare_resources` asserts them.
//   * **The LUT texture is node-owned, exactly as in `LutNode`.** It is not a graph
//     resource: one cube is uploaded per node and the graph never sees it, so the
//     import/pool ownership split (gotcha 18) is untouched by this file.
//
// PRECISION IS THE ONE BEHAVIOURAL DIFFERENCE, and it is a difference in the fused
// pass's FAVOUR. The chain rounds to f16 at each intermediate store; this keeps f32 in
// registers the whole way. So the outputs are not bit-identical, and
// `tests::fused_grade` asserts agreement within f16 quantisation rather than equality.

use std::sync::Arc;
use std::sync::Mutex;

use half::f16;

use crate::colour::lut_parser::Lut3D;
use crate::render::compute::{ComputePassHelper, ComputePipelineCache, PipelineKey};
use crate::render::context::RenderContext;
use crate::render::device::GpuDevice;
use crate::render::frame_state::FrameState;
use crate::render::graph::RenderNode;
use crate::render::nodes::chroma_key::ChromaKeyParams;
use crate::render::nodes::color_correction::ColorCorrectionParams;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::shader::registry::{BuiltinShader, ShaderRegistry};

/// Push constants for the fused pass. 112 bytes.
///
/// **Packed to 112 rather than laid out as the three structs concatenated, and the
/// reason is the device limit.** `ColorCorrectionParams` is 80 bytes, `LutParams` 16
/// and `ChromaKeyParams` 32 — 128 exactly, which is the whole
/// `max_push_constant_size` this crate requests (`GpuDevice::new_headless`). Three
/// copies of `width`/`height` are among them; declaring the pair once removes 16 bytes
/// and — more importantly — makes it impossible for the three stages to disagree about
/// the frame they are processing.
///
/// Keep byte-identical to `struct FusedGradeParams` in `fused_grade.wgsl`. The scalar
/// groups are `vec4` because WGSL aligns a `vec4<f32>` to 16 bytes inside a struct, so
/// a run of bare `f32`s after the three colour vectors would be read at offsets this
/// side does not agree with — the same trap `yuv_to_rgb.wgsl`'s comment records.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FusedGradeParams {
    /// Per-channel lift [R, G, B, unused].
    pub lift: [f32; 4],
    /// Per-channel gamma [R, G, B, unused]. 1.0 = identity.
    pub gamma: [f32; 4],
    /// Per-channel gain [R, G, B, unused]. 1.0 = identity.
    pub gain: [f32; 4],
    /// `[saturation, brightness, contrast, hue_shift]` — 1, 0, 1, 0 = identity.
    /// `hue_shift` is in RADIANS, as `ColorCorrectionParams` carries it.
    pub grade: [f32; 4],
    /// `[key_hue, tolerance, softness, min_saturation]`, the first three in degrees.
    pub key_a: [f32; 4],
    /// `[min_value, spill_suppress, lut_strength, unused]`.
    pub key_b: [f32; 4],
    /// Frame dimensions, declared once for all three stages.
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

// The WGSL side is 7 × 16 bytes. A mismatch here shifts `width`/`height` and the
// shader reads a zero-sized frame — the same class of failure as gotcha 6's
// `_colour_pad`, which is why this is a compile-time assertion rather than a comment.
const _: () = assert!(std::mem::size_of::<FusedGradeParams>() == 112);

/// How many bytes of push constants this node's pipeline layout declares.
pub const FUSED_GRADE_PUSH_CONSTANT_BYTES: u32 = 112;

impl FusedGradeParams {
    /// Build from the three nodes' own parameter structs.
    ///
    /// **Taking the real structs rather than loose floats is the point.** Every value
    /// below is copied from the type the unfused node would have been given, so a
    /// caller that already builds `ColorCorrectionParams::identity(w, h)` and
    /// `ChromaKeyParams::green_screen(w, h)` gets the same grade with no second
    /// interpretation of what its fields mean. `pipeline_wiring`-style drift — a fused
    /// path quietly keying on a different hue — cannot happen through this door.
    ///
    /// The dimensions come from `cc.width`/`cc.height` and the chroma key's are
    /// checked against them: they describe the same frame, and disagreeing values mean
    /// one of the two callers is wrong about the canvas.
    pub fn from_parts(cc: ColorCorrectionParams, lut_strength: f32, key: ChromaKeyParams) -> Self {
        debug_assert_eq!(
            (cc.width, cc.height),
            (key.width, key.height),
            "the colour-correction and chroma-key params describe different frames \
             ({}x{} vs {}x{}); fused, there is only one frame",
            cc.width,
            cc.height,
            key.width,
            key.height
        );
        Self {
            lift: cc.lift,
            gamma: cc.gamma,
            gain: cc.gain,
            grade: [cc.saturation, cc.brightness, cc.contrast, cc.hue_shift],
            key_a: [
                key.key_hue,
                key.tolerance,
                key.softness,
                key.min_saturation,
            ],
            key_b: [key.min_value, key.spill_suppress, lut_strength, 0.0],
            width: cc.width,
            height: cc.height,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Colour correction → 3D LUT → chroma key, in one pass.
///
/// Replaces `ColorCorrectionNode` + `LutNode` + `ChromaKeyNode` where all three run
/// back-to-back on one layer, which is the Heavy graph's shape (benchmarks 5, 6, 7, 8
/// and `EffectChainBuilder`'s common case). See the module comment for what it saves,
/// what it cannot, and how to measure it honestly.
pub struct FusedGradeNode {
    pub in_rgba: ResourceId,
    pub out_rgba: ResourceId,
    pub params: FusedGradeParams,
    /// The LUT cube, node-owned exactly as `LutNode` owns its own — never a graph
    /// resource, so the pool and the import path are untouched by this node.
    _lut_texture: wgpu::Texture,
    lut_view: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<wgpu::Device>,
    /// Keyed on the two `ViewId`s this node was handed, as the eleven other nodes are
    /// (gotcha 16): the pool is FIFO so a stable frame shape keeps this cache warm.
    bg_cache: Mutex<Option<([ViewId; 2], wgpu::BindGroup)>>,
}

impl FusedGradeNode {
    /// Build the fused node.
    ///
    /// `width`/`height` come from `cc.width`/`cc.height` via
    /// [`FusedGradeParams::from_parts`], so unlike [`super::lut::LutNode`] there is no
    /// second call a caller can forget (gotcha 10). `declare_resources` still asserts
    /// them, because a caller can pass `ColorCorrectionParams` with zeroed dimensions.
    pub fn new(
        device: &GpuDevice,
        shaders: &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        lut: &Lut3D,
        in_rgba: ResourceId,
        out_rgba: ResourceId,
        params: FusedGradeParams,
    ) -> Self {
        let n = lut.size;

        // ── The LUT cube — the same upload `LutNode::new` performs ─────────────
        let lut_texture = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fused_grade_lut_3d"),
            size: wgpu::Extent3d {
                width: n,
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

        let mut lut_bytes: Vec<u8> = Vec::with_capacity((n * n * n * 8) as usize);
        for entry in &lut.data {
            lut_bytes.extend_from_slice(&f16::from_f32(entry[0]).to_le_bytes());
            lut_bytes.extend_from_slice(&f16::from_f32(entry[1]).to_le_bytes());
            lut_bytes.extend_from_slice(&f16::from_f32(entry[2]).to_le_bytes());
            lut_bytes.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
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
                bytes_per_row: Some(n * 8),
                rows_per_image: Some(n),
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: n,
            },
        );

        let lut_sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fused_grade_lut_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
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

        // ── Bindings ──────────────────────────────────────────────────────────
        //
        // **`in` is a read-only STORAGE texture, not a sampled one, and that choice is
        // what makes this a drop-in for the chain.** `ColorCorrectionNode` reads
        // storage and `ChromaKeyNode` samples, so the fused input has to pick one — and
        // the usages the graph aggregates differ (`StorageRead` adds STORAGE_BINDING).
        // Storage-read is the stricter of the two: `TextureAccess::StorageRead` maps to
        // `TEXTURE_BINDING | STORAGE_BINDING`, so a producer written for either
        // unfused node still satisfies it, and `textureLoad` at integer coordinates is
        // exactly what both shaders did.
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fused_grade_bgl"),
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
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fused_grade_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..FUSED_GRADE_PUSH_CONSTANT_BYTES,
                    }],
                });

        let shader_mod = shaders.get(BuiltinShader::FusedGrade);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey {
                shader: BuiltinShader::FusedGrade,
                entry_point: "cs_main",
            },
            &pipeline_layout,
            &shader_mod,
        );

        Self {
            in_rgba,
            out_rgba,
            params,
            _lut_texture: lut_texture,
            lut_view,
            lut_sampler,
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
            bg_cache: Mutex::new(None),
        }
    }

    /// Update the grade between frames. No pipeline recompile — same as the three
    /// nodes' own `set_params`.
    pub fn set_params(&mut self, params: FusedGradeParams) {
        self.params = params;
    }
}

impl RenderNode for FusedGradeNode {
    /// **A distinct name, because the per-node timing report aggregates by it.**
    /// `bench`'s `aggregate_node_timings` groups brackets by name and gotcha 27's
    /// `UNEVEN` check compares instances within a group, so reusing
    /// `"ColorCorrection"` here would silently pool a fused pass with an unfused one
    /// and make the two rows incomparable — precisely the confusion that gotcha's
    /// per-instance split exists to remove.
    fn name(&self) -> &str {
        "FusedGrade"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResolutionSource, ResourceDescriptor, TextureAccess};
        // Gotcha 10's assertion, kept even though `from_parts` carries the dimensions:
        // zeroed `ColorCorrectionParams` would otherwise ask wgpu for a zero-sized
        // texture and fail as `Dimension X is zero` from inside `create_texture`,
        // naming neither this node nor the caller.
        assert!(
            self.params.width > 0 && self.params.height > 0,
            "FusedGradeNode was built with {}x{} dimensions: its output would be a \
             zero-sized texture. The frame size travels in FusedGradeParams (from \
             ColorCorrectionParams), so the caller passed zeroed params.",
            self.params.width,
            self.params.height
        );
        builder.creates.push((
            self.out_rgba,
            ResourceDescriptor {
                label: Some(format!("FusedGrade_{}", self.out_rgba.0)),
                size: ResolutionSource::Fixed(self.params.width, self.params.height),
                format: wgpu::TextureFormat::Rgba16Float,
            },
        ));
        builder.read(self.in_rgba, TextureAccess::StorageRead);
        builder.write(self.out_rgba, TextureAccess::StorageWrite);
        // The LUT cube is node-owned, not a graph resource — as in `LutNode`.
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &RenderContext, _frame: &FrameState) {
        let in_res = ctx.get(self.in_rgba);
        let out_res = ctx.get(self.out_rgba);

        let mut cache = self.bg_cache.lock().unwrap();
        let cache_key = [in_res.view_id, out_res.view_id];

        if cache.is_none() || cache.as_ref().unwrap().0 != cache_key {
            let bind_group = self
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("fused_grade_bg"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(in_res.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(out_res.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&self.lut_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                        },
                    ],
                });
            *cache = Some((cache_key, bind_group));
        }

        let bind_group = &cache.as_ref().unwrap().1;
        let push_bytes = bytemuck::bytes_of(&self.params);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_grade"),
            timestamp_writes: None,
        });
        ComputePassHelper::dispatch(
            &mut pass,
            &self.pipeline,
            bind_group,
            Some(push_bytes),
            self.params.width,
            self.params.height,
        );
        drop(pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The push-constant struct must be the size the pipeline layout declares.
    ///
    /// Restated at runtime as well as at compile time so a failure names both numbers
    /// (the same reasoning `render::resource::tests::the_bucket_cap_covers_a_whole_frames_peak`
    /// gives). The failure this guards is gotcha 6's: a struct one field short shifts
    /// `width`/`height` and the shader reads a zero-sized frame.
    #[test]
    fn the_push_constant_struct_matches_the_declared_range() {
        assert_eq!(
            std::mem::size_of::<FusedGradeParams>(),
            FUSED_GRADE_PUSH_CONSTANT_BYTES as usize,
            "the pipeline layout declares {FUSED_GRADE_PUSH_CONSTANT_BYTES} bytes"
        );
        // And it must fit the limit this crate requests, with the three unfused nodes'
        // 128 bytes as the thing it is beating. A `const` block because both sides are
        // constants and clippy is right that a runtime `assert!` over two of them is a
        // compile-time question wearing a test's clothes.
        const _: () = assert!(
            FUSED_GRADE_PUSH_CONSTANT_BYTES <= 128,
            "max_push_constant_size is 128 in GpuDevice"
        );
    }

    /// Every field of the three parameter structs must reach the fused struct.
    ///
    /// **This is the wiring test, and it uses distinct non-identity values on purpose.**
    /// A fused shader silently keying on the wrong hue, or applying identity gamma
    /// because the field never arrived, produces a plausible picture — so each value
    /// below is unique and is checked at its own offset. `from_parts` is the single
    /// door between the unfused parameter types and this one, which is what makes one
    /// test enough.
    #[test]
    fn every_parameter_reaches_the_fused_struct() {
        let cc = ColorCorrectionParams {
            lift: [0.01, 0.02, 0.03, 0.0],
            gamma: [1.1, 1.2, 1.3, 1.0],
            gain: [1.4, 1.5, 1.6, 1.0],
            saturation: 1.7,
            brightness: 0.18,
            contrast: 1.9,
            hue_shift: 0.20,
            width: 1920,
            height: 1080,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let key = ChromaKeyParams {
            key_hue: 121.0,
            tolerance: 42.0,
            softness: 11.0,
            min_saturation: 0.16,
            min_value: 0.09,
            spill_suppress: 0.31,
            width: 1920,
            height: 1080,
        };
        let p = FusedGradeParams::from_parts(cc, 0.75, key);

        assert_eq!(p.lift, cc.lift);
        assert_eq!(p.gamma, cc.gamma);
        assert_eq!(p.gain, cc.gain);
        assert_eq!(
            p.grade,
            [cc.saturation, cc.brightness, cc.contrast, cc.hue_shift],
            "the grade scalars are packed [saturation, brightness, contrast, hue]"
        );
        assert_eq!(
            p.key_a,
            [
                key.key_hue,
                key.tolerance,
                key.softness,
                key.min_saturation
            ]
        );
        assert_eq!(
            p.key_b,
            [key.min_value, key.spill_suppress, 0.75, 0.0],
            "lut_strength rides in key_b.z"
        );
        assert_eq!((p.width, p.height), (1920, 1080));
    }

    /// An identity grade must be identity in every stage.
    ///
    /// The control for the test above: with `ColorCorrectionParams::identity` and
    /// `lut_strength` 0 the fused pass must be a copy, so a shader that applied a
    /// stage unconditionally would fail `fused_grade`'s pixel test rather than pass it
    /// by accident. Checked here as a parameter property because it is cheap and needs
    /// no GPU.
    #[test]
    fn an_identity_grade_carries_identity_values() {
        let p = FusedGradeParams::from_parts(
            ColorCorrectionParams::identity(64, 64),
            0.0,
            ChromaKeyParams::green_screen(64, 64),
        );
        assert_eq!(p.lift, [0.0; 4]);
        assert_eq!(p.gamma, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(p.gain, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(p.grade, [1.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.key_b[2], 0.0, "a zero-strength LUT must be a bypass");
    }
}
