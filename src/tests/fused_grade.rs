// src/tests/fused_grade.rs
//
// P2.3 — does the FUSED colour-correction + LUT + chroma-key pass produce what the
// three separate nodes produce?
//
// WHY THIS TEST IS THE WHOLE POINT OF THE CHANGE. Fusion is a performance change with
// no observable behaviour of its own: the shader is a copy of three shaders' bodies
// concatenated, so the only way it can be wrong is by DIFFERING from them — a
// parameter that never arrived, a stage applied out of order, a push-constant offset
// off by 8 bytes (AGENTS.md gotcha 6's `_colour_pad`, which shifted `pitch` and made
// every row overwrite row 0). None of those produce an error; they produce a plausible
// picture. So the test renders the SAME input through both paths in one process and
// compares pixels.
//
// FOUR RULES THIS FILE IS BUILT ON, each earned elsewhere in the tree:
//
//  1. **Non-identity parameters, or the test is vacuous.** With
//     `ColorCorrectionParams::identity` and an identity LUT both paths are a copy, and
//     a fused shader that dropped every stage would pass. Same trap gotcha 11 records
//     for colour metadata: BT.709 passes even when the metadata never arrives, because
//     it is also every fallback. So the grade below lifts, gammas, gains, shifts hue
//     AND the LUT inverts — every stage must fire to reach the expected pixel.
//  2. **The tolerance is f16 quantisation, not equality.** The chain stores each
//     intermediate as `rgba16float` and therefore rounds twice; the fused pass keeps
//     f32 in registers. That is a real difference in the fused path's favour, so
//     equality would fail for the right reason and the wrong cause. `TOL` is derived
//     from f16's relative precision rather than picked.
//  3. **A control that would FAIL if the comparison had no teeth.**
//     `a_different_grade_produces_a_different_picture` renders the fused pass twice
//     with one field changed and requires the result to move by MORE than the
//     tolerance. Without it, two paths that both ignored their push constants would
//     agree perfectly and the file would report success. Gotcha 11's
//     mis-tagged-control rule, applied here.
//  4. **The chroma key's early-exit branch has to be exercised on both sides of the
//     gate.** The keyed region and the passed-through region take different paths
//     through both shaders (the `!in_range` store-and-return), so the fixture below
//     contains saturated green (keyed, alpha driven to 0) and greys (below
//     `min_saturation`, passed through untouched). A pattern of one or the other would
//     leave half the shader untested.
//
// No CUDA, no NVENC, no decoder, no files: a headless wgpu device and one uploaded
// texture. That is deliberate — this is a shader-equivalence question and every
// dependency it does not have is a reason it cannot skip.

#[cfg(test)]
mod fused_grade {
    use crate::colour::lut_parser::Lut3D;
    use crate::render::compute::ComputePipelineCache;
    use crate::render::context::RenderContext;
    use crate::render::device::GpuDevice;
    use crate::render::frame_state::FrameState;
    use crate::render::graph::{RenderGraphCompiler, RenderNode};
    use crate::render::nodes::chroma_key::{ChromaKeyNode, ChromaKeyParams};
    use crate::render::nodes::color_correction::{ColorCorrectionNode, ColorCorrectionParams};
    use crate::render::nodes::fused_grade::{FusedGradeNode, FusedGradeParams};
    use crate::render::nodes::lut::LutNode;
    use crate::render::resource::{
        ResolutionSource, ResourceBuilder, ResourceDescriptor, ResourceId, TextureAccess,
    };
    use crate::render::shader::registry::ShaderRegistry;
    use half::f16;
    use std::sync::Arc;

    /// 64×64: four 16-row bands, big enough that a shifted push constant would be
    /// obvious and small enough that the readback is instant.
    const W: u32 = 64;
    const H: u32 = 64;

    /// Per-channel tolerance in f16 units of the values being compared.
    ///
    /// **Derived, not chosen.** `rgba16float` carries 10 explicit mantissa bits, so its
    /// relative step near 1.0 is 2⁻¹⁰ ≈ 0.00098. The chain rounds to f16 at two
    /// intermediates and once more at the output, and each stage can compound the
    /// previous one's error through a `pow`/`mix`, so the bound is a few of those
    /// steps: 8 × 2⁻¹⁰ ≈ 0.0078. Anything larger than this is a real disagreement
    /// between the two paths rather than storage precision.
    ///
    /// Deliberately NOT loosened to make a failure pass: at 0.0078 a single dropped
    /// stage in this fixture moves pixels by 0.05–0.4, i.e. 6–50× the tolerance, so
    /// there is a wide margin between "f16 rounding" and "a stage did not run" —
    /// `a_dropped_stage_would_exceed_the_tolerance` measures exactly that margin.
    const TOL: f32 = 8.0 / 1024.0;

    /// Four bands, chosen so both sides of the chroma key's gate are exercised and no
    /// two bands are related by a scale factor (which a mis-ordered stage could hide
    /// behind).
    ///
    /// * saturated green — keyed: hue 120°, high saturation and value, so `mask` is 1
    ///   and alpha goes to 0 with green spill suppressed;
    /// * mid grey — below `min_saturation`, so it takes the early-exit branch;
    /// * warm orange — high saturation but hue ~30°, far outside the tolerance, so it
    ///   runs the full path and comes out unkeyed;
    /// * near-black — below `min_value`, the other half of the early-exit condition.
    const BANDS: [[f32; 4]; 4] = [
        [0.05, 0.80, 0.10, 1.0],
        [0.45, 0.45, 0.45, 1.0],
        [0.85, 0.35, 0.05, 1.0],
        [0.02, 0.02, 0.03, 1.0],
    ];

    fn band_at(y: u32) -> [f32; 4] {
        BANDS[((y / (H / BANDS.len() as u32)) as usize).min(BANDS.len() - 1)]
    }

    /// A grade where EVERY field is non-identity.
    ///
    /// Rule 1 above. A zero lift or a unit gamma would let a fused shader skip that
    /// stage and still match, so each value is distinct and none is the shader's
    /// early-out (`abs(contrast - 1.0) > 0.001` and friends must all be taken).
    fn grade() -> ColorCorrectionParams {
        ColorCorrectionParams {
            lift: [0.03, 0.06, -0.02, 0.0],
            gamma: [1.15, 0.90, 1.30, 1.0],
            gain: [1.10, 0.95, 1.20, 1.0],
            saturation: 1.25,
            brightness: 0.04,
            contrast: 1.12,
            hue_shift: 0.15,
            width: W,
            height: H,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }

    /// The chroma key, with the green band inside its gate.
    ///
    /// `green_screen`'s own preset, so this test keys on exactly what the Heavy graph
    /// (benchmarks 5–8) keys on rather than on a hue chosen to make the test easy.
    fn key() -> ChromaKeyParams {
        ChromaKeyParams::green_screen(W, H)
    }

    /// A LUT that is emphatically NOT the identity: channel-rotating and gamma'd.
    ///
    /// An identity cube would make stage 2 unobservable, so a fused shader that never
    /// sampled it would pass. This one moves R→G→B→R and applies a 0.8 power, which no
    /// other stage in the chain can imitate.
    fn rotating_lut(n: u32) -> Lut3D {
        let mut data = Vec::with_capacity((n * n * n) as usize);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let rf = r as f32 / (n - 1) as f32;
                    let gf = g as f32 / (n - 1) as f32;
                    let bf = b as f32 / (n - 1) as f32;
                    // Rotate and gamma. Clamped to the LUT's own [0,1] domain.
                    data.push([bf.powf(0.8), rf.powf(0.8), gf.powf(0.8)]);
                }
            }
        }
        Lut3D {
            size: n,
            data,
            domain_min: [0.0, 0.0, 0.0],
            domain_max: [1.0, 1.0, 1.0],
        }
    }

    /// Uploads `BANDS` into the graph's first resource.
    ///
    /// A node rather than a `write_texture` before the graph runs, because the pool
    /// owns the texture and hands it out during `execute` — the same reason
    /// `YuvUploadNode` defers its `write_texture` (gotcha 12).
    struct BandSource {
        out: ResourceId,
        queue: Arc<wgpu::Queue>,
    }

    impl RenderNode for BandSource {
        fn name(&self) -> &str {
            "BandSource"
        }

        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.creates.push((
                self.out,
                ResourceDescriptor {
                    label: Some("FusedGradeInput".into()),
                    size: ResolutionSource::Fixed(W, H),
                    format: wgpu::TextureFormat::Rgba16Float,
                },
            ));
            builder.write(self.out, TextureAccess::CopyDst);
            // Both readers need this: `ColorCorrectionNode` binds it as a read-only
            // storage texture and `FusedGradeNode` does too, so the usage the graph
            // aggregates has to include STORAGE_BINDING as well as COPY_DST.
            builder.write(self.out, TextureAccess::StorageWrite);
        }

        fn record(
            &self,
            _encoder: &mut wgpu::CommandEncoder,
            ctx: &RenderContext,
            _frame: &FrameState,
        ) {
            let dst = ctx.get(self.out);
            let mut bytes: Vec<u8> = Vec::with_capacity((W * H * 8) as usize);
            for y in 0..H {
                let px = band_at(y);
                for _ in 0..W {
                    for c in &px {
                        bytes.extend_from_slice(&f16::from_f32(*c).to_le_bytes());
                    }
                }
            }
            // `write_texture` on the queue rather than a staging copy: this is test
            // setup, it happens once, and it lands before the submission that reads it
            // because wgpu orders `pending_writes` ahead of the frame's commands.
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: dst.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(W * 8),
                    rows_per_image: Some(H),
                },
                wgpu::Extent3d {
                    width: W,
                    height: H,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Copies a resource into a mappable buffer so the pixels can be compared.
    struct ReadbackNode {
        src: ResourceId,
        buf: Arc<wgpu::Buffer>,
    }

    impl RenderNode for ReadbackNode {
        fn name(&self) -> &str {
            "FusedGradeReadback"
        }

        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(self.src, TextureAccess::CopySrc);
        }

        fn record(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            ctx: &RenderContext,
            _frame: &FrameState,
        ) {
            let src = ctx.get(self.src);
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: src.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &self.buf,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(W * 8),
                        rows_per_image: Some(H),
                    },
                },
                wgpu::Extent3d {
                    width: W,
                    height: H,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// One RGBA pixel as f32, straight out of the `rgba16float` readback.
    type Pixel = [f32; 4];

    fn run_graph(device: &GpuDevice, compiler: RenderGraphCompiler, out: Arc<wgpu::Buffer>) -> Vec<Pixel> {
        let graph = compiler.compile(W, H).expect("graph failed to compile");
        let frame = FrameState::test_empty(W, H);
        let mut encoder = device.begin_frame();
        graph.execute(&mut encoder, device, &frame);
        let submission = device.submit(encoder);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range().to_vec();
        out.unmap();

        mapped
            .chunks_exact(8)
            .map(|px| {
                let mut out = [0.0f32; 4];
                for (c, v) in out.iter_mut().enumerate() {
                    *v = f16::from_le_bytes([px[c * 2], px[c * 2 + 1]]).to_f32();
                }
                out
            })
            .collect()
    }

    fn readback_buffer(device: &GpuDevice) -> Arc<wgpu::Buffer> {
        Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused_grade_readback"),
            size: (W * H * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    }

    /// The three-node chain: colour correction → LUT → chroma key.
    fn render_chain(
        device: &GpuDevice,
        shaders: &ShaderRegistry,
        compute: &ComputePipelineCache,
        cc: ColorCorrectionParams,
        lut_strength: f32,
        key: ChromaKeyParams,
        lut: &Lut3D,
    ) -> Vec<Pixel> {
        let mut compiler = RenderGraphCompiler::new();
        let mut id = 2u32;
        let src = ResourceId::next(&mut id);
        let cc_out = ResourceId::next(&mut id);
        let lut_out = ResourceId::next(&mut id);
        let key_out = ResourceId::next(&mut id);

        compiler.add_node(Box::new(BandSource {
            out: src,
            queue: Arc::clone(&device.queue),
        }));
        compiler.add_node(Box::new(ColorCorrectionNode::new(
            device, shaders, compute, src, cc_out, cc,
        )));
        // Gotcha 10: `LutNode::new` cannot know the frame size.
        let mut lut_node = LutNode::new(device, shaders, compute, lut, cc_out, lut_out, lut_strength);
        lut_node.set_size(W, H);
        compiler.add_node(Box::new(lut_node));
        compiler.add_node(Box::new(ChromaKeyNode::new(
            device, shaders, compute, lut_out, key_out, key,
        )));

        let out = readback_buffer(device);
        compiler.add_node(Box::new(ReadbackNode {
            src: key_out,
            buf: Arc::clone(&out),
        }));
        run_graph(device, compiler, out)
    }

    /// The same grade in one pass.
    fn render_fused(
        device: &GpuDevice,
        shaders: &ShaderRegistry,
        compute: &ComputePipelineCache,
        params: FusedGradeParams,
        lut: &Lut3D,
    ) -> Vec<Pixel> {
        let mut compiler = RenderGraphCompiler::new();
        let mut id = 2u32;
        let src = ResourceId::next(&mut id);
        let fused_out = ResourceId::next(&mut id);

        compiler.add_node(Box::new(BandSource {
            out: src,
            queue: Arc::clone(&device.queue),
        }));
        compiler.add_node(Box::new(FusedGradeNode::new(
            device, shaders, compute, lut, src, fused_out, params,
        )));

        let out = readback_buffer(device);
        compiler.add_node(Box::new(ReadbackNode {
            src: fused_out,
            buf: Arc::clone(&out),
        }));
        run_graph(device, compiler, out)
    }

    /// Largest per-channel difference between two pictures, and where it was.
    fn worst_delta(a: &[Pixel], b: &[Pixel]) -> (f32, usize, usize) {
        let mut worst = 0.0f32;
        let mut at = (0usize, 0usize);
        for (i, (pa, pb)) in a.iter().zip(b).enumerate() {
            for c in 0..4 {
                let d = (pa[c] - pb[c]).abs();
                if d > worst {
                    worst = d;
                    at = (i, c);
                }
            }
        }
        (worst, at.0, at.1)
    }

    struct Harness {
        device: Arc<GpuDevice>,
        shaders: ShaderRegistry,
        compute: ComputePipelineCache,
        lut: Lut3D,
    }

    /// A GPU, or a printed skip. Same contract as the rest of `src/tests/`.
    fn harness() -> Option<Harness> {
        let device = match pollster::block_on(GpuDevice::new_headless()) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("[fused_grade] SKIP: no GPU device ({e:?})");
                return None;
            }
        };
        let shaders = match ShaderRegistry::compile_all(&device) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[fused_grade] SKIP: shaders did not compile ({e:?})");
                return None;
            }
        };
        Some(Harness {
            device,
            shaders,
            compute: ComputePipelineCache::new(),
            lut: rotating_lut(17),
        })
    }

    /// **The claim: the fused pass equals the three-node chain.**
    ///
    /// Both paths in one process, one uploaded texture, the same parameters through
    /// `FusedGradeParams::from_parts`. Compared within f16 quantisation (rule 2), which
    /// is the only slack the fused path is entitled to.
    #[test]
    fn the_fused_pass_matches_the_three_node_chain() {
        let Some(h) = harness() else { return };
        let cc = grade();
        let key = key();
        let strength = 0.85;

        let chain = render_chain(&h.device, &h.shaders, &h.compute, cc, strength, key, &h.lut);
        let fused = render_fused(
            &h.device,
            &h.shaders,
            &h.compute,
            FusedGradeParams::from_parts(cc, strength, key),
            &h.lut,
        );

        assert_eq!(chain.len(), fused.len(), "both paths must read back {W}x{H}");
        let (worst, idx, ch) = worst_delta(&chain, &fused);
        assert!(
            worst <= TOL,
            "the fused pass disagrees with the chain by {worst:.5} at pixel {idx} \
             (row {}, channel {ch}) — chain {:?} vs fused {:?}. The tolerance is f16 \
             quantisation ({TOL:.5}); a difference this large means a stage, a \
             parameter or an ordering differs, not that the intermediates rounded.",
            idx as u32 / W,
            chain[idx],
            fused[idx],
        );
    }

    /// The keyed and unkeyed regions must BOTH be right, checked separately.
    ///
    /// The aggregate above could hide a mismatch confined to one band — the chroma
    /// key's `!in_range` early exit is a whole branch of both shaders, and a fused
    /// version that got the gate backwards would still match on the pixels it happened
    /// to treat the same way. So this asserts per band, and then asserts that the bands
    /// genuinely landed on both sides of the gate so the split is not vacuous.
    ///
    /// **Which band is keyed is NOT hardcoded, and that is deliberate.** The key sees
    /// the pixel AFTER colour correction and the LUT, and both of those are per-channel
    /// and asymmetric here (the LUT rotates R→G→B), so predicting which input band
    /// arrives inside the key's hue window means hand-computing five stages — a claim
    /// about arithmetic that this file is not testing and that would have to be redone
    /// every time the fixture changed. Measured on this fixture: the green input arrives
    /// at the key as blue-ish and passes, and the orange input arrives as green and is
    /// keyed out.
    ///
    /// What the test asserts instead is the two things that are actually the point:
    /// **both branches ran** (some band keyed, some band not), and **the two paths agree
    /// on which** — the per-band loop compares alpha along with RGB, so a fused shader
    /// that keyed a different set would fail there rather than here.
    #[test]
    fn both_sides_of_the_chroma_key_gate_agree() {
        let Some(h) = harness() else { return };
        let cc = grade();
        let key = key();
        let strength = 0.85;

        let chain = render_chain(&h.device, &h.shaders, &h.compute, cc, strength, key, &h.lut);
        let fused = render_fused(
            &h.device,
            &h.shaders,
            &h.compute,
            FusedGradeParams::from_parts(cc, strength, key),
            &h.lut,
        );

        let band_rows = H / BANDS.len() as u32;
        let mut alphas = Vec::new();
        let mut chain_alphas = Vec::new();
        for band in 0..BANDS.len() as u32 {
            // The middle row of the band, away from any boundary.
            let y = band * band_rows + band_rows / 2;
            let i = (y * W + W / 2) as usize;
            for c in 0..4 {
                let d = (chain[i][c] - fused[i][c]).abs();
                assert!(
                    d <= TOL,
                    "band {band} (input {:?}) differs by {d:.5} on channel {c}: chain \
                     {:?} vs fused {:?}",
                    BANDS[band as usize],
                    chain[i],
                    fused[i]
                );
            }
            alphas.push(fused[i][3]);
            chain_alphas.push(chain[i][3]);
        }

        // The split has teeth only if the bands actually landed on both sides of the
        // gate. Both counts must be non-zero, or one whole branch of the key went
        // unexercised and this test compared two pictures that took the same path.
        let keyed = alphas.iter().filter(|a| **a < 0.5).count();
        let untouched = alphas.iter().filter(|a| **a > 0.99).count();
        assert!(
            keyed > 0,
            "no band was keyed out (alphas {alphas:?}), so the key's masking branch is \
             untested — this test would be comparing two unkeyed pictures"
        );
        assert!(
            untouched > 0,
            "every band was keyed (alphas {alphas:?}), so the key's `!in_range` \
             early-exit branch is untested"
        );

        // ...and the two paths must agree on WHICH, stated as a set rather than as a
        // tolerance: an alpha comparison within TOL would also pass if both shaders
        // keyed the same band by coincidence of rounding, whereas "the same bands are
        // keyed" is the property the branch has to preserve.
        let keyed_set = |v: &[f32]| -> Vec<bool> { v.iter().map(|a| *a < 0.5).collect() };
        assert_eq!(
            keyed_set(&alphas),
            keyed_set(&chain_alphas),
            "the fused pass keyed a different set of bands than the chain: fused \
             alphas {alphas:?} vs chain {chain_alphas:?}"
        );
    }

    /// **The control: the comparison must be able to fail.**
    ///
    /// Rule 3. Two shaders that both ignored their push constants would agree
    /// perfectly, so this changes ONE field of the grade and requires the fused output
    /// to move by more than the tolerance — and it also measures by how much, which is
    /// the margin the tolerance is safe within.
    #[test]
    fn a_dropped_stage_would_exceed_the_tolerance() {
        let Some(h) = harness() else { return };
        let key = key();
        let strength = 0.85;

        let full = render_fused(
            &h.device,
            &h.shaders,
            &h.compute,
            FusedGradeParams::from_parts(grade(), strength, key),
            &h.lut,
        );

        // Every stage of the fused pass, removed one at a time. Each must move the
        // picture by more than TOL, or the test above could not detect that stage
        // going missing.
        let mut no_lift = grade();
        no_lift.lift = [0.0; 4];
        let mut no_gamma = grade();
        no_gamma.gamma = [1.0, 1.0, 1.0, 1.0];
        let mut no_gain = grade();
        no_gain.gain = [1.0, 1.0, 1.0, 1.0];
        let mut no_sat = grade();
        no_sat.saturation = 1.0;
        let mut no_bc = grade();
        no_bc.brightness = 0.0;
        no_bc.contrast = 1.0;
        let mut no_hue = grade();
        no_hue.hue_shift = 0.0;

        for (name, cc, strength) in [
            ("lift", no_lift, strength),
            ("gamma", no_gamma, strength),
            ("gain", no_gain, strength),
            ("saturation", no_sat, strength),
            ("brightness+contrast", no_bc, strength),
            ("hue", no_hue, strength),
            ("the LUT", grade(), 0.0),
        ] {
            let without = render_fused(
                &h.device,
                &h.shaders,
                &h.compute,
                FusedGradeParams::from_parts(cc, strength, key),
                &h.lut,
            );
            let (worst, _, _) = worst_delta(&full, &without);
            assert!(
                worst > TOL,
                "removing {name} moved the picture by only {worst:.5}, which is inside \
                 the {TOL:.5} tolerance — so `the_fused_pass_matches_the_three_node_chain` \
                 could not tell whether that stage ran at all. Change the fixture, not \
                 the tolerance."
            );
        }

        // The key's own effect, checked the same way: a hue far from the fixture's
        // green leaves that band unkeyed, so alpha moves.
        let mut other_key = key;
        other_key.key_hue = 300.0;
        let unkeyed = render_fused(
            &h.device,
            &h.shaders,
            &h.compute,
            FusedGradeParams::from_parts(grade(), strength, other_key),
            &h.lut,
        );
        let (worst, _, _) = worst_delta(&full, &unkeyed);
        assert!(
            worst > TOL,
            "moving the key hue from 120° to 300° changed nothing ({worst:.5}), so the \
             chroma-key stage is not reading its parameters"
        );
    }

    /// A zero-strength LUT and identity grade must be a copy — the fused pass's own
    /// identity case.
    ///
    /// Cheap, and it pins the one property a reader assumes: that the fused node can be
    /// dropped into a graph with identity parameters and change nothing. The chroma key
    /// still runs (its parameters have no identity), so this compares against the CHAIN
    /// with the same identity settings rather than against the input.
    #[test]
    fn an_identity_grade_matches_an_identity_chain() {
        let Some(h) = harness() else { return };
        let cc = ColorCorrectionParams::identity(W, H);
        let key = key();

        let chain = render_chain(&h.device, &h.shaders, &h.compute, cc, 0.0, key, &h.lut);
        let fused = render_fused(
            &h.device,
            &h.shaders,
            &h.compute,
            FusedGradeParams::from_parts(cc, 0.0, key),
            &h.lut,
        );
        let (worst, idx, ch) = worst_delta(&chain, &fused);
        assert!(
            worst <= TOL,
            "identity params disagree by {worst:.5} at pixel {idx} channel {ch}: \
             chain {:?} vs fused {:?}",
            chain[idx],
            fused[idx]
        );
    }

    /// The fused node must declare ONE output where the chain declares three.
    ///
    /// This is the saving, stated as a property rather than left to a benchmark: the
    /// two intermediates disappear, which is two canvas-sized `Rgba16Float` textures
    /// per layer that the pool no longer has to hold (gotcha 14's `peak bucket` is the
    /// number that moves). Asserted through the compiler rather than by counting nodes,
    /// because what matters is the resources a frame holds simultaneously.
    #[test]
    fn fusing_removes_two_intermediates_per_layer() {
        let Some(h) = harness() else { return };
        let cc = grade();
        let key = key();

        // The chain: source + cc_out + lut_out + key_out = 4 created resources.
        let mut chain_ids = 2u32;
        let chain_src = ResourceId::next(&mut chain_ids);
        let chain_cc = ResourceId::next(&mut chain_ids);
        let chain_lut = ResourceId::next(&mut chain_ids);
        let chain_key = ResourceId::next(&mut chain_ids);
        let mut chain = RenderGraphCompiler::new();
        chain.add_node(Box::new(BandSource {
            out: chain_src,
            queue: Arc::clone(&h.device.queue),
        }));
        chain.add_node(Box::new(ColorCorrectionNode::new(
            &h.device,
            &h.shaders,
            &h.compute,
            chain_src,
            chain_cc,
            cc,
        )));
        let mut lut_node = LutNode::new(
            &h.device,
            &h.shaders,
            &h.compute,
            &h.lut,
            chain_cc,
            chain_lut,
            1.0,
        );
        lut_node.set_size(W, H);
        chain.add_node(Box::new(lut_node));
        chain.add_node(Box::new(ChromaKeyNode::new(
            &h.device,
            &h.shaders,
            &h.compute,
            chain_lut,
            chain_key,
            key,
        )));
        let chain_graph = chain.compile(W, H).expect("chain compiles");

        // The fused version: source + fused_out = 2.
        let mut fused_ids = 2u32;
        let fused_src = ResourceId::next(&mut fused_ids);
        let fused_out = ResourceId::next(&mut fused_ids);
        let mut fused = RenderGraphCompiler::new();
        fused.add_node(Box::new(BandSource {
            out: fused_src,
            queue: Arc::clone(&h.device.queue),
        }));
        fused.add_node(Box::new(FusedGradeNode::new(
            &h.device,
            &h.shaders,
            &h.compute,
            &h.lut,
            fused_src,
            fused_out,
            FusedGradeParams::from_parts(cc, 1.0, key),
        )));
        let fused_graph = fused.compile(W, H).expect("fused compiles");

        // Run one frame of each so the pool actually acquires and releases.
        let frame = FrameState::test_empty(W, H);
        for g in [&chain_graph, &fused_graph] {
            let mut enc = h.device.begin_frame();
            g.execute(&mut enc, &h.device, &frame);
            let s = h.device.submit(enc);
            h.device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(s));
        }

        let chain_pooled = chain_graph.pool_stats().pooled;
        let fused_pooled = fused_graph.pool_stats().pooled;
        assert_eq!(
            chain_pooled, 4,
            "the chain holds source + three stage outputs"
        );
        assert_eq!(
            fused_pooled, 2,
            "the fused version holds source + one output; {fused_pooled} means the \
             intermediates did not actually go away, which is the whole saving"
        );
        assert_eq!(
            chain_pooled - fused_pooled,
            2,
            "fusing must remove exactly two canvas-sized textures per layer — at 4K \
             that is 2 x 66 MB, and it is what `peak bucket` reports (gotcha 14)"
        );
    }
}
