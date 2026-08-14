// src/tests/render_integration.rs
#[cfg(test)]
mod render_integration {
    use std::sync::Arc;
    use crate::render::device::GpuDevice;
    use crate::render::graph::{RenderGraphCompiler, RenderNode, GraphError};
    use crate::render::resource::{ResourceId, ResolutionSource, ResourceDescriptor, ResourceBuilder};
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING
            });
            // Just for the test, we override FINAL_COLOR's meaning if it's not 0, 
            // but actually ResourceBuilder::create returns a new ID.
            // Wait, ResourceBuilder increments a counter. FINAL_COLOR is defined as 0.
            // We can't use builder.create() and get ResourceId(0).
            // We should just write(ResourceId::FINAL_COLOR) and rely on the compiler recognizing it as an external or we just create it as a transient texture.
            // Wait, if it's transient, CompiledGraph needs its descriptor. So we need to put it into creates manually.
            // Let's just have it write(FINAL_COLOR), and the test setup will register it... wait, CompiledGraph only allocates textures for `creates`.
            // So if no node `creates` FINAL_COLOR, it will panic when we `get(FINAL_COLOR)`.
            // We can inject `FINAL_COLOR` descriptor by making a node that explicitly adds it to `creates`.
            builder.creates.push((ResourceId::FINAL_COLOR, ResourceDescriptor {
                label: Some("FinalColor".into()),
                size: ResolutionSource::Fixed(W, H),
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING
            }));
            builder.writes.push(ResourceId::FINAL_COLOR);
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
            builder.creates.push((self.t1, ResourceDescriptor { label: None, size: ResolutionSource::Fixed(W, H), format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST }));
            builder.creates.push((self.t2, ResourceDescriptor { label: None, size: ResolutionSource::Fixed(W, H), format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST }));
            builder.writes.push(self.t1);
            builder.writes.push(self.t2);
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
            builder.read(ResourceId::FINAL_COLOR);
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
            &vec![255, 0, 0, 255].repeat((W * H) as usize),
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        device.queue.write_texture(
            wgpu::ImageCopyTexture { texture: &green_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &vec![0, 255, 0, 128].repeat((W * H) as usize),
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        let mut frame = FrameState::test_empty(W, H);
        frame.clips.push(ClipRenderEntry { source_id: SourceId(0), texture_slot: 0, layer_order: 0, clip_width: W, clip_height: H, transform: ClipTransform::identity(), opacity: 1.0, is_nv12: false });
        frame.clips.push(ClipRenderEntry { source_id: SourceId(1), texture_slot: 1, layer_order: 1, clip_width: W, clip_height: H, transform: ClipTransform::identity(), opacity: 0.5, is_nv12: false });
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
        assert!((b as i32 - 0).abs() <= 4, "blue channel wrong: {}", b);
    }

    struct NodeA;
    impl RenderNode for NodeA {
        fn name(&self) -> &str { "NodeA" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(ResourceId(100)); // R
            builder.write(ResourceId(200)); // S
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    struct NodeB;
    impl RenderNode for NodeB {
        fn name(&self) -> &str { "NodeB" }
        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(ResourceId(200)); // S
            builder.write(ResourceId(100)); // R
        }
        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    #[test]
    fn cyclic_graph_returns_error() {
        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(NodeA));
        compiler.add_node(Box::new(NodeB));
        
        let result = compiler.compile(100, 100);
        assert!(matches!(result, Err(GraphError::CyclicDependency)));
    }

    #[test]
    fn builtin_shaders_compile_cleanly() {
        let device = pollster::block_on(GpuDevice::new_headless()).unwrap();
        let result = ShaderRegistry::compile_all(&device);
        assert!(result.is_ok());
    }
}
