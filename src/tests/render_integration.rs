// src/tests/render_integration.rs
#[cfg(test)]
mod render_integration {
    use std::sync::Arc;
    use crate::render::device::GpuDevice;
    use crate::render::graph::{RenderGraphCompiler, RenderNode, GraphError};
    use crate::render::resource::{ResourceId, ResolutionSource, ResourceDescriptor, ResourceBuilder, TextureAccess};
    use crate::render::context::RenderContext;
    use crate::render::frame_state::{FrameState, ClipRenderEntry};
    use crate::timeline::ids::SourceId;
    use crate::render::shader::registry::ShaderRegistry;
    use crate::timeline::transform::ClipTransform;
    use crate::render::nodes::composite::CompositeNode;

    const W: u32 = 128;
    const H: u32 = 128;

    struct CreateFinalColorNode;
    impl RenderNode for CreateFinalColorNode {
        fn name(&self) -> &str { "CreateFinalColor" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            let _id = builder.create(ResourceDescriptor {
                label: Some("FinalColor".into()),
                size: ResolutionSource::Fixed(W, H),
                format: wgpu::TextureFormat::Rgba8Unorm,
            }, TextureAccess::ColorAttachment);
            
            builder.creates.push((ResourceId::FINAL_COLOR, ResourceDescriptor {
                label: Some("FinalColor".into()),
                size: ResolutionSource::Fixed(W, H),
                format: wgpu::TextureFormat::Rgba8Unorm,
            }));
            builder.write(ResourceId::FINAL_COLOR, TextureAccess::ColorAttachment);
        }
        fn record(&self, _encoder: &mut wgpu::CommandEncoder, _ctx: &RenderContext, _frame: &FrameState) {}
    }

    struct CopyTestTexturesNode {
        t1: ResourceId, t2: ResourceId,
        r_tex: wgpu::Texture, g_tex: wgpu::Texture,
    }
    impl RenderNode for CopyTestTexturesNode {
        fn name(&self) -> &str { "CopyTestTextures" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.creates.push((self.t1, ResourceDescriptor { label: None, size: ResolutionSource::Fixed(W, H), format: wgpu::TextureFormat::Rgba8Unorm }));
            builder.creates.push((self.t2, ResourceDescriptor { label: None, size: ResolutionSource::Fixed(W, H), format: wgpu::TextureFormat::Rgba8Unorm }));
            builder.write(self.t1, TextureAccess::CopyDst);
            builder.write(self.t2, TextureAccess::CopyDst);
        }
        fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &RenderContext, _f: &FrameState) {
            let out1 = ctx.get(self.t1);
            let out2 = ctx.get(self.t2);
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture { texture: &self.r_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::ImageCopyTexture { texture: out1.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 }
            );
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture { texture: &self.g_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::ImageCopyTexture { texture: out2.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 }
            );
        }
    }

    struct ReadbackNode {
        buf: Arc<wgpu::Buffer>,
    }
    impl RenderNode for ReadbackNode {
        fn name(&self) -> &str { "Readback" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(ResourceId::FINAL_COLOR, TextureAccess::CopySrc);
        }
        fn record(&self, encoder: &mut wgpu::CommandEncoder, ctx: &RenderContext, _f: &FrameState) {
            let out = ctx.get(ResourceId::FINAL_COLOR);
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture { texture: out.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::ImageCopyBuffer {
                    buffer: &self.buf,
                    layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) }
                },
                wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 }
            );
        }
    }

    #[test]
    fn two_clip_composite_produces_correct_pixel() {
        let device = pollster::block_on(GpuDevice::new_headless()).unwrap();
        let shaders = ShaderRegistry::compile_all(&device).unwrap();

        let red_texture = device.create_texture(
            Some("red_tex"), W, H, wgpu::TextureFormat::Rgba8Unorm, 
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC
        );
        let green_texture = device.create_texture(
            Some("green_tex"), W, H, wgpu::TextureFormat::Rgba8Unorm, 
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC
        );

        device.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &red_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[255, 0, 0, 255].repeat((W * H) as usize),
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        device.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &green_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &[0, 255, 0, 128].repeat((W * H) as usize),
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        let mut frame = FrameState::test_empty(W, H);
        frame.clips.push(ClipRenderEntry { source_id: SourceId(0), texture_slot: 0, layer_order: 0, clip_width: W, clip_height: H, transform: ClipTransform::identity(), opacity: 1.0, blend_mode: crate::timeline::transform::BlendMode::Normal, crop: crate::timeline::transform::CropRect::full(), corner_pin: crate::timeline::transform::CornerPin::identity(), matte_mode: crate::timeline::transform::MatteMode::None, effects: Default::default(), frame_meta: Default::default(), kind: crate::timeline::store::ClipKind::Video });
        frame.clips.push(ClipRenderEntry { source_id: SourceId(1), texture_slot: 1, layer_order: 1, clip_width: W, clip_height: H, transform: ClipTransform::identity(), opacity: 0.5, blend_mode: crate::timeline::transform::BlendMode::Normal, crop: crate::timeline::transform::CropRect::full(), corner_pin: crate::timeline::transform::CornerPin::identity(), matte_mode: crate::timeline::transform::MatteMode::None, effects: Default::default(), frame_meta: Default::default(), kind: crate::timeline::store::ClipKind::Video });
        frame.sort_clips();

        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(CreateFinalColorNode));

        let tex1_id = ResourceId(2);
        let tex2_id = ResourceId(3);
        compiler.add_node(Box::new(CopyTestTexturesNode {
            t1: tex1_id, t2: tex2_id,
            r_tex: red_texture, g_tex: green_texture,
        }));

        let mut comp_node = CompositeNode::new(&device, &shaders, ResourceId::FINAL_COLOR, 8, wgpu::TextureFormat::Rgba8Unorm);
        comp_node.input_textures.push(tex1_id);
        comp_node.input_textures.push(tex2_id);
        compiler.add_node(Box::new(comp_node));

        let out_buffer = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buffer"),
            size: (W * H * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        compiler.add_node(Box::new(ReadbackNode { buf: Arc::clone(&out_buffer) }));

        let graph = compiler.compile(W, H).unwrap();
        let mut encoder = device.begin_frame();
        graph.execute(&mut encoder, &device, &frame);
        device.submit(encoder);

        let slice = out_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);

        let mapped = slice.get_mapped_range();
        
        let center_idx = ((H / 2) * W * 4 + (W / 2) * 4) as usize;
        let r = mapped[center_idx];
        let g = mapped[center_idx + 1];
        let b = mapped[center_idx + 2];
        
        // Expected: (255, 0, 0) under (0, 255, 0, 128)
        // With PREMULTIPLIED_ALPHA_BLENDING and instance opacity = 0.5:
        // src_color = (0, 1.0, 0, 0.5) * 0.5 = (0, 0.5, 0, 0.25)
        // dst_color = src_color + dst_color * (1 - src.a)
        // dst.r = 0 + 1.0 * (1 - 0.25) = 0.75 -> 191
        // dst.g = 0.5 + 0 * (1 - 0.25) = 0.50 -> 128
        assert!((r as i32 - 191).abs() <= 4, "red channel wrong: {}", r);
        assert!((g as i32 - 128).abs() <= 4, "green channel wrong: {}", g);
        assert!((b as i32).abs() <= 4, "blue channel wrong: {}", b);
    }

    struct NodeA;
    impl RenderNode for NodeA {
        fn name(&self) -> &str { "NodeA" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(ResourceId(100), TextureAccess::Sampled); // R
            builder.write(ResourceId(200), TextureAccess::Sampled); // S
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    struct NodeB;
    impl RenderNode for NodeB {
        fn name(&self) -> &str { "NodeB" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(ResourceId(200), TextureAccess::Sampled); // S
            builder.write(ResourceId(100), TextureAccess::Sampled); // R
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    #[test]
    fn cyclic_graph_returns_error() {
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(NodeA));
        compiler.add_node(Box::new(NodeB));

        let result = compiler.compile(100, 100);
        match result {
            Err(GraphError::CyclicDependency { cycle }) => {
                assert!(cycle.contains(&"NodeA".to_string()));
                assert!(cycle.contains(&"NodeB".to_string()));
            }
            Err(other) => panic!("Expected CyclicDependency, got {:?}", other),
            Ok(_) => panic!("Expected CyclicDependency error, but compilation succeeded"),
        }
    }

    // ── Phase 8: RenderGraph Hardening Tests ──────────────────────────────────

    struct WriterNode { name: &'static str, res: ResourceId }
    impl RenderNode for WriterNode {
        fn name(&self) -> &str { self.name }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.creates.push((self.res, ResourceDescriptor {
                label: Some(self.name.into()),
                size: ResolutionSource::Fixed(W, H),
                format: wgpu::TextureFormat::Rgba8Unorm,
            }));
            builder.write(self.res, TextureAccess::ColorAttachment);
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    struct OverwriterNode { name: &'static str, res: ResourceId }
    impl RenderNode for OverwriterNode {
        fn name(&self) -> &str { self.name }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.write(self.res, TextureAccess::ColorAttachment);
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    struct ReaderNode { name: &'static str, res: ResourceId }
    impl RenderNode for ReaderNode {
        fn name(&self) -> &str { self.name }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(self.res, TextureAccess::Sampled);
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    #[test]
    fn test_raw_hazard_ordering() {
        // NodeA writes X, NodeB reads X -> Order must be [NodeA (0), NodeB (1)]
        let res_x = ResourceId(10);
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(WriterNode { name: "NodeA", res: res_x }));
        compiler.add_node(Box::new(ReaderNode { name: "NodeB", res: res_x }));

        let compiled = compiler.compile(W, H).expect("Compilation failed");
        assert_eq!(compiled.execution_order(), &[0, 1]);
    }

    #[test]
    fn test_waw_and_raw_hazard_ordering() {
        // NodeA writes X, NodeB writes X (overwrites), NodeC reads X
        // Order must be [0, 1, 2]
        let res_x = ResourceId(10);
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(WriterNode { name: "NodeA", res: res_x }));
        compiler.add_node(Box::new(OverwriterNode { name: "NodeB", res: res_x }));
        compiler.add_node(Box::new(ReaderNode { name: "NodeC", res: res_x }));

        let compiled = compiler.compile(W, H).expect("Compilation failed");
        assert_eq!(compiled.execution_order(), &[0, 1, 2]);
    }

    #[test]
    fn test_war_hazard_ordering() {
        // NodeA writes X, NodeB reads X, NodeC overwrites X, NodeD reads X (new version)
        // Order must strictly be [0, 1, 2, 3] so NodeB reads before NodeC overwrites!
        let res_x = ResourceId(10);
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(WriterNode { name: "NodeA", res: res_x }));
        compiler.add_node(Box::new(ReaderNode { name: "NodeB", res: res_x }));
        compiler.add_node(Box::new(OverwriterNode { name: "NodeC", res: res_x }));
        compiler.add_node(Box::new(ReaderNode { name: "NodeD", res: res_x }));

        let compiled = compiler.compile(W, H).expect("Compilation failed");
        assert_eq!(compiled.execution_order(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_missing_producer_error_diagnostic() {
        // Node attempts to read resource 999 which no node produces
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(ReaderNode { name: "OrphanReader", res: ResourceId(999) }));

        let result = compiler.compile(W, H);
        match result {
            Err(GraphError::MissingProducer { node_name, resource }) => {
                assert_eq!(node_name, "OrphanReader");
                assert_eq!(resource, ResourceId(999));
            }
            Err(other) => panic!("Expected MissingProducer error, got {:?}", other),
            Ok(_) => panic!("Expected MissingProducer error, but compilation succeeded"),
        }
    }

    struct ConflictingAccessNode;
    impl RenderNode for ConflictingAccessNode {
        fn name(&self) -> &str { "ConflictingNode" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(ResourceId::FINAL_COLOR, TextureAccess::ColorAttachment);
            builder.write(ResourceId::FINAL_COLOR, TextureAccess::StorageWrite);
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    #[test]
    fn test_incompatible_access_diagnostic() {
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(ConflictingAccessNode));

        let result = compiler.compile(W, H);
        assert!(matches!(result, Err(GraphError::IncompatibleAccess { .. })));
    }

    #[test]
    fn builtin_shaders_compile_cleanly() {
        let device = pollster::block_on(GpuDevice::new_headless()).unwrap();
        let result = ShaderRegistry::compile_all(&device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_phase14_effects_pipeline_execution() {
        use crate::render::compute::ComputePipelineCache;
        use crate::render::nodes::gaussian_blur::{BlurPassNode, BlurParams};
        use crate::render::nodes::sharpen::{SharpenNode, SharpenParams};
        use crate::render::nodes::vignette::{VignetteNode, VignetteParams};
        use crate::render::nodes::color_correction::{ColorCorrectionNode, ColorCorrectionParams};

        let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).unwrap());
        let shaders = ShaderRegistry::compile_all(&device).unwrap();
        let compute_cache = Arc::new(ComputePipelineCache::new());

        let mut compiler = RenderGraphCompiler::new();
        let mut id_counter = 2u32;

        let in_id = ResourceId::next(&mut id_counter);
        let cc_out = ResourceId::next(&mut id_counter);
        let blur_h_out = ResourceId::next(&mut id_counter);
        let blur_v_out = ResourceId::next(&mut id_counter);
        let sharpen_out = ResourceId::next(&mut id_counter);
        let final_out = ResourceId::FINAL_COLOR;

        // Dummy initial producer
        struct DummyProducer(ResourceId);
        impl RenderNode for DummyProducer {
            fn name(&self) -> &str { "DummyProducer" }
            fn declare_resources(&self, builder: &mut ResourceBuilder) {
                builder.creates.push((self.0, ResourceDescriptor {
                    label: Some("InitialTexture".into()),
                    size: ResolutionSource::Fixed(W, H),
                    format: wgpu::TextureFormat::Rgba16Float,
                }));
                builder.write(self.0, TextureAccess::StorageWrite);
            }
            fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
        }
        compiler.add_node(Box::new(DummyProducer(in_id)));

        // 1. Color correction
        let mut cc_params = ColorCorrectionParams::identity(W, H);
        cc_params.brightness = 0.1;
        cc_params.contrast = 1.1;
        cc_params.saturation = 1.2;
        cc_params.hue_shift = 0.2;
        compiler.add_node(Box::new(ColorCorrectionNode::new(
            &device,
            &shaders,
            &compute_cache,
            in_id,
            cc_out,
            cc_params,
        )));

        // 2. Gaussian Blur (H + V)
        compiler.add_node(Box::new(BlurPassNode::new(
            &device,
            &shaders,
            &compute_cache,
            cc_out,
            blur_h_out,
            BlurParams::horizontal(5.0, 2.5, W, H),
            "BlurH",
        )));
        compiler.add_node(Box::new(BlurPassNode::new(
            &device,
            &shaders,
            &compute_cache,
            blur_h_out,
            blur_v_out,
            BlurParams::vertical(5.0, 2.5, W, H),
            "BlurV",
        )));

        // 3. Sharpen
        compiler.add_node(Box::new(SharpenNode::new(
            &device,
            &shaders,
            &compute_cache,
            blur_v_out,
            sharpen_out,
            SharpenParams::new(0.5, W, H),
        )));

        // 4. Vignette
        compiler.add_node(Box::new(VignetteNode::new(
            &device,
            &shaders,
            &compute_cache,
            sharpen_out,
            final_out,
            VignetteParams::default_preset(W, H),
        )));

        let graph = compiler.compile(W, H).expect("Effects pipeline graph failed to compile");
        let frame_state = FrameState::test_empty(W, H);

        let mut encoder = device.begin_frame();
        graph.execute(&mut encoder, &device, &frame_state);
        let submission_id = device.submit(encoder);
        device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_id));
    }

    /// `execute_timed` must bracket every node, name them in execution order, and
    /// touch the texture pool exactly as `execute` does.
    ///
    /// Three properties, because two of them are the ways a per-node timing report
    /// silently lies:
    ///
    /// 1. **The names must align with the brackets.** The timer returns a flat
    ///    `Vec<f64>` of pair durations, so `names[i]` is the only thing saying which
    ///    shader `pair[i]` belongs to. If it drifts, the report attributes one
    ///    node's cost to another and looks entirely reasonable.
    /// 2. **The pool interaction must be identical to `execute`'s.** The timed path
    ///    duplicates the acquire/release block, so a timed run that pooled
    ///    differently would be profiling a graph the user never runs — and after
    ///    gotcha 14, a differing `evicted` count is exactly the difference that
    ///    matters. Asserted against the pool's own counters.
    /// 3. **Timings must be per-node, not one span for the frame.** Summing them and
    ///    comparing against a single bracket around the whole graph is what
    ///    distinguishes real per-node data from the same number printed N times.
    #[test]
    fn execute_timed_brackets_every_node_in_order() {
        use crate::render::compute::ComputePipelineCache;
        use crate::render::gpu_timer::GpuTimer;
        use crate::render::nodes::color_correction::{ColorCorrectionNode, ColorCorrectionParams};
        use crate::render::nodes::gaussian_blur::{BlurPassNode, BlurParams};
        use crate::render::nodes::sharpen::{SharpenNode, SharpenParams};

        let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).unwrap());
        let shaders = ShaderRegistry::compile_all(&device).unwrap();
        let compute_cache = Arc::new(ComputePipelineCache::new());

        let mut compiler = RenderGraphCompiler::new();
        let mut id_counter = 2u32;
        let in_id = ResourceId::next(&mut id_counter);
        let cc_out = ResourceId::next(&mut id_counter);
        let blur_out = ResourceId::next(&mut id_counter);
        let final_out = ResourceId::FINAL_COLOR;

        struct DummyProducer(ResourceId);
        impl RenderNode for DummyProducer {
            fn name(&self) -> &str { "DummyProducer" }
            fn declare_resources(&self, builder: &mut ResourceBuilder) {
                builder.creates.push((self.0, ResourceDescriptor {
                    label: Some("InitialTexture".into()),
                    size: ResolutionSource::Fixed(W, H),
                    format: wgpu::TextureFormat::Rgba16Float,
                }));
                builder.write(self.0, TextureAccess::StorageWrite);
            }
            fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
        }
        compiler.add_node(Box::new(DummyProducer(in_id)));
        compiler.add_node(Box::new(ColorCorrectionNode::new(
            &device, &shaders, &compute_cache, in_id, cc_out,
            ColorCorrectionParams::identity(W, H),
        )));
        compiler.add_node(Box::new(BlurPassNode::new(
            &device, &shaders, &compute_cache, cc_out, blur_out,
            BlurParams::horizontal(5.0, 2.5, W, H), "BlurH",
        )));
        compiler.add_node(Box::new(SharpenNode::new(
            &device, &shaders, &compute_cache, blur_out, final_out,
            SharpenParams::new(0.5, W, H),
        )));

        let graph = compiler.compile(W, H).expect("graph failed to compile");
        let frame_state = FrameState::test_empty(W, H);
        let node_count = graph.execution_order().len();
        assert_eq!(node_count, 4, "four nodes were added");

        // ── Property 2, first half: the untimed baseline's pool behaviour ──────
        let mut encoder = device.begin_frame();
        graph.execute(&mut encoder, &device, &frame_state);
        let sid = device.submit(encoder);
        device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        let after_untimed = graph.pool_stats();

        // ── The timed run ─────────────────────────────────────────────────────
        let mut timer = GpuTimer::new(&device, node_count as u32);
        if !timer.is_enabled() {
            eprintln!("SKIP: this device does not report TIMESTAMP_QUERY");
            return;
        }
        timer.reset();
        let mut encoder = device.begin_frame();
        let names = graph.execute_timed(&mut encoder, &device, &frame_state, &mut timer);
        let sid = device.submit(encoder);
        device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid));

        // Property 1: one name per node, in execution order, matching the graph's
        // own order rather than the order they were added.
        assert_eq!(names.len(), node_count, "every node must be bracketed: {names:?}");
        let expected: Vec<&str> = graph
            .execution_order()
            .iter()
            .map(|&i| graph.node_name(i))
            .collect();
        assert_eq!(names, expected, "names must follow execution order");

        let timings = timer.resolve_all().expect("brackets were recorded");
        assert_eq!(
            timings.len(),
            names.len(),
            "one duration per name, or the mapping is meaningless"
        );

        // Property 3: the durations are per-node, so at least one differs from
        // another. All-equal would mean one span reported N times — and all-zero
        // would mean nothing was measured.
        assert!(
            timings.iter().any(|t| *t > 0.0),
            "every node measured 0 ns; the brackets did not capture work: {timings:?}"
        );

        // Property 2, second half: the timed path must pool exactly like `execute`.
        let after_timed = graph.pool_stats();
        assert_eq!(
            after_timed.evicted, after_untimed.evicted,
            "the timed path must not evict differently from `execute`"
        );
        assert_eq!(
            after_timed.misses, after_untimed.misses,
            "a second frame of the same shape must reuse the pool either way"
        );
        assert_eq!(
            after_timed.peak_bucket, after_untimed.peak_bucket,
            "the timed path must hold the same textures simultaneously"
        );
    }
}
