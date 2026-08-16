// src/render/nodes/composite.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Per-instance GPU data for one clip. 64 bytes, Pod.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClipInstance {
    pub col0:      [f32; 4],
    pub col1:      [f32; 4],
    pub col2:      [f32; 4],
    pub opacity:   f32,
    pub tex_index: u32,
    pub _pad:      [f32; 2],
}

pub struct CompositeNode {
    pub out_color:       ResourceId,
    pub input_textures:  Vec<ResourceId>,
    instance_buffer:     wgpu::Buffer,
    max_instances:       u32,
    pipeline:            wgpu::RenderPipeline,
    bind_group_layout:   wgpu::BindGroupLayout,
    sampler:             wgpu::Sampler,
    device:              Arc<wgpu::Device>,
    queue:               Arc<wgpu::Queue>,
    out_format:          wgpu::TextureFormat,
    has_binding_arrays:  bool,
}

impl CompositeNode {
    pub fn new(
        device:        &GpuDevice,
        shaders:       &ShaderRegistry,
        out_color:     ResourceId,
        max_instances: u32,
        surface_format: wgpu::TextureFormat, // Note: the RTT format might be hdr_format, not surface_format, but we'll use what's passed
    ) -> Self {
        let instance_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite_instance_buffer"),
            size: (max_instances as u64 * std::mem::size_of::<ClipInstance>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: if device.has_binding_arrays {
                        std::num::NonZeroU32::new(max_instances)
                    } else {
                        None
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_id = if device.has_binding_arrays {
            BuiltinShader::Composite
        } else {
            BuiltinShader::CompositeSingle
        };
        let shader = shaders.get(shader_id);

        let pipeline = device.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffers
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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

        let sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            out_color,
            input_textures: Vec::new(),
            instance_buffer,
            max_instances,
            pipeline,
            bind_group_layout,
            sampler,
            device: Arc::clone(&device.device),
            queue: Arc::clone(&device.queue),
            out_format: surface_format,
            has_binding_arrays: device.has_binding_arrays,
        }
    }

    pub fn upload_instances(&self, frame: &FrameState) -> u32 {
        let mut instances = Vec::with_capacity(frame.clips.len());
        for (i, entry) in frame.clips.iter().enumerate() {
            let m = entry.transform.to_matrix(
                entry.clip_width as f32, 
                entry.clip_height as f32,
                frame.canvas_width as f32,
                frame.canvas_height as f32
            );
            instances.push(ClipInstance {
                col0: [m[0], m[1], m[2], 0.0],
                col1: [m[3], m[4], m[5], 0.0],
                col2: [m[6], m[7], m[8], 0.0],
                opacity: entry.opacity,
                tex_index: i as u32,
                _pad: [0.0; 2],
            });
        }

        let bytes = bytemuck::cast_slice(&instances);
        self.queue.write_buffer(&self.instance_buffer, 0, bytes);
        instances.len() as u32
    }
}

impl RenderNode for CompositeNode {
    fn name(&self) -> &str {
        "Composite"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource};
        builder.creates.push((self.out_color, ResourceDescriptor {
            label: Some("FinalColor".into()),
            size: ResolutionSource::Canvas,
            format: self.out_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        }));
        for &id in &self.input_textures {
            builder.read(id);
        }
        builder.write(self.out_color);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        frame:   &FrameState,
    ) {
        let count = self.upload_instances(frame);
        if count == 0 {
            return; // Nothing to draw
        }

        let mut views = Vec::with_capacity(self.max_instances as usize);
        for id in &self.input_textures {
            views.push(ctx.get(*id).view);
        }
        // Pad the rest with the first view if needed to satisfy the array length
        let fallback_view = *views.first().unwrap();
        while views.len() < self.max_instances as usize {
            views.push(fallback_view);
        }

        let mut single_bg = None;
        let mut multiple_bgs = Vec::new();

        if self.has_binding_arrays {
            single_bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.instance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureViewArray(&views),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }));
        } else {
            for i in 0..count {
                multiple_bgs.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite_bg_single"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.instance_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(views[i as usize]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }));
            }
        }

        let out = ctx.get(self.out_color);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: out.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);

        if self.has_binding_arrays {
            pass.set_bind_group(0, single_bg.as_ref().unwrap(), &[]);
            pass.draw(0..6, 0..count);
        } else {
            for i in 0..count {
                pass.set_bind_group(0, &multiple_bgs[i as usize], &[]);
                pass.draw(0..6, i..(i + 1));
            }
        }
    }
}
