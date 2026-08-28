// src/render/nodes/blit.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

pub struct BlitToScreenNode {
    pub in_color:          ResourceId,
    pipeline:              wgpu::RenderPipeline,
    sampler:               wgpu::Sampler,
    bind_group_layout:     wgpu::BindGroupLayout,
    pub current_surface_view: Option<wgpu::TextureView>,
    device:                Arc<wgpu::Device>,
    bg_cache:              std::sync::Mutex<Option<(crate::render::resource::ViewId, wgpu::BindGroup)>>,
}

impl BlitToScreenNode {
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        in_color:       ResourceId,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bind_group_layout = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = shaders.get(BuiltinShader::Blit);

        let pipeline = device.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            in_color,
            pipeline,
            sampler,
            bind_group_layout,
            current_surface_view: None,
            device: Arc::clone(&device.device),
            bg_cache: std::sync::Mutex::new(None),
        }
    }
}

impl RenderNode for BlitToScreenNode {
    fn name(&self) -> &str {
        "BlitToScreen"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::TextureAccess;
        builder.read(self.in_color, TextureAccess::Sampled);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        let surface_view = self.current_surface_view.as_ref()
            .expect("BlitToScreenNode::current_surface_view not set before record()");

        let rtt = ctx.get(self.in_color);
        
        let mut cache = self.bg_cache.lock().unwrap();
        if cache.is_none() || cache.as_ref().unwrap().0 != rtt.view_id {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(rtt.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            *cache = Some((rtt.view_id, bind_group));
        }
        
        let bind_group = &cache.as_ref().unwrap().1;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}
