// src/render/nodes/composite.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::context::RenderContext;
use std::sync::Mutex;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};

/// Per-instance GPU data for one clip. 80 bytes (5 × vec4), Pod.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ClipInstance {
    pub col0:         [f32; 4],
    pub col1:         [f32; 4],
    pub col2:         [f32; 4],
    pub crop:         [f32; 4],
    pub opacity:      f32,
    pub tex_index:    u32,
    pub blend_mode:   u32,
    pub crop_feather: f32,
}

#[derive(Clone)]
pub struct CompositePipelines {
    pub pipeline:            Arc<wgpu::RenderPipeline>,
    pub pipeline_add:        Arc<wgpu::RenderPipeline>,
    pub pipeline_multiply:   Arc<wgpu::RenderPipeline>,
    pub pipeline_screen:     Arc<wgpu::RenderPipeline>,
    pub pipeline_darken:     Arc<wgpu::RenderPipeline>,
    pub pipeline_lighten:    Arc<wgpu::RenderPipeline>,
    pub pipeline_difference: Arc<wgpu::RenderPipeline>,
    pub bind_group_layout:   Arc<wgpu::BindGroupLayout>,
    pub sampler:             Arc<wgpu::Sampler>,
    pub out_format:          wgpu::TextureFormat,
    pub has_binding_arrays:  bool,
}

impl CompositePipelines {
    pub fn new(
        device:        &GpuDevice,
        shaders:       &ShaderRegistry,
        max_instances: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
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

        let make_pipeline = |blend: wgpu::BlendState, label: &'static str| {
            device.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
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
                        blend: Some(blend),
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
            })
        };

        let pipeline = Arc::new(make_pipeline(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING, "composite_pipeline_normal"));

        let pipeline_add = Arc::new(make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            "composite_pipeline_add",
        ));

        let pipeline_multiply = Arc::new(make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::DstAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            "composite_pipeline_multiply",
        ));

        let pipeline_screen = Arc::new(make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            "composite_pipeline_screen",
        ));

        let pipeline_darken = Arc::new(make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Min,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Min,
                },
            },
            "composite_pipeline_darken",
        ));

        let pipeline_lighten = Arc::new(make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
            },
            "composite_pipeline_lighten",
        ));

        let pipeline_difference = Arc::new(make_pipeline(
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::ReverseSubtract,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            "composite_pipeline_difference",
        ));

        let sampler = Arc::new(device.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        }));

        Self {
            pipeline,
            pipeline_add,
            pipeline_multiply,
            pipeline_screen,
            pipeline_darken,
            pipeline_lighten,
            pipeline_difference,
            bind_group_layout: Arc::new(bind_group_layout),
            sampler,
            out_format: surface_format,
            has_binding_arrays: device.has_binding_arrays,
        }
    }
}

pub struct CompositeNode {
    pub out_color:       ResourceId,
    pub input_textures:  Vec<ResourceId>,
    instance_buffer:     wgpu::Buffer,
    max_instances:       u32,
    pub pipelines:       Arc<CompositePipelines>,
    device:              Arc<wgpu::Device>,
    queue:               Arc<wgpu::Queue>,
    bg_array_cache:      Mutex<Option<(Vec<ViewId>, wgpu::BindGroup)>>,
    bg_single_cache:     Mutex<Vec<Option<(ViewId, wgpu::BindGroup)>>>,
}

impl CompositeNode {
    pub fn new(
        device:        &GpuDevice,
        shaders:       &ShaderRegistry,
        out_color:     ResourceId,
        max_instances: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let pipelines = Arc::new(CompositePipelines::new(device, shaders, max_instances, surface_format));
        Self::with_pipelines(device, pipelines, out_color, max_instances)
    }

    pub fn with_pipelines(
        device:        &GpuDevice,
        pipelines:     Arc<CompositePipelines>,
        out_color:     ResourceId,
        max_instances: u32,
    ) -> Self {
        let instance_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite_instance_buffer"),
            size: (max_instances as u64 * std::mem::size_of::<ClipInstance>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            out_color,
            input_textures: Vec::new(),
            instance_buffer,
            max_instances,
            pipelines,
            device: Arc::clone(&device.device),
            queue: Arc::clone(&device.queue),
            bg_array_cache: Mutex::new(None),
            bg_single_cache: Mutex::new((0..max_instances).map(|_| None).collect()),
        }
    }

    pub fn pipeline_for_blend_mode(&self, mode: crate::timeline::transform::BlendMode) -> &wgpu::RenderPipeline {
        use crate::timeline::transform::BlendMode;
        match mode {
            BlendMode::Add => &self.pipelines.pipeline_add,
            BlendMode::Multiply => &self.pipelines.pipeline_multiply,
            BlendMode::Screen | BlendMode::ColorDodge => &self.pipelines.pipeline_screen,
            BlendMode::Darken | BlendMode::ColorBurn => &self.pipelines.pipeline_darken,
            BlendMode::Lighten => &self.pipelines.pipeline_lighten,
            BlendMode::Difference | BlendMode::Exclusion => &self.pipelines.pipeline_difference,
            BlendMode::Normal | BlendMode::Overlay | BlendMode::HardLight | BlendMode::SoftLight => &self.pipelines.pipeline,
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
                crop: entry.crop.to_gpu(),
                opacity: entry.opacity,
                tex_index: i as u32,
                blend_mode: entry.blend_mode.as_u32(),
                crop_feather: entry.crop.feather,
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
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((self.out_color, ResourceDescriptor {
            label: Some("FinalColor".into()),
            size: ResolutionSource::Canvas,
            format: self.pipelines.out_format,
        }));
        for &id in &self.input_textures {
            builder.read(id, TextureAccess::Sampled);
        }
        builder.write(self.out_color, crate::render::resource::TextureAccess::ColorAttachment);

        // Explicitly inject usages required by external consumers of the graph's final output:
        // - UI preview uses it as a TextureBinding (Sampled)
        // - FFmpeg CPU export copies it to a staging buffer (CopySrc)
        builder.read(self.out_color, crate::render::resource::TextureAccess::Sampled);
        builder.read(self.out_color, crate::render::resource::TextureAccess::CopySrc);
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
        let mut view_ids = Vec::with_capacity(self.max_instances as usize);
        
        for id in &self.input_textures {
            let res = ctx.get(*id);
            views.push(res.view);
            view_ids.push(res.view_id);
        }
        // Pad the rest with the first view if needed to satisfy the array length
        let fallback_view = *views.first().unwrap();
        let fallback_view_id = *view_ids.first().unwrap();
        while views.len() < self.max_instances as usize {
            views.push(fallback_view);
            view_ids.push(fallback_view_id);
        }

        if self.pipelines.has_binding_arrays {
            let mut cache = self.bg_array_cache.lock().unwrap();
            if cache.is_none() || cache.as_ref().unwrap().0 != view_ids {
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite_bg"),
                    layout: &self.pipelines.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.instance_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureViewArray(&views) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.pipelines.sampler) },
                    ],
                });
                *cache = Some((view_ids.clone(), bg));
            }
            // we will borrow it inside the pass
        } else {
            let mut cache = self.bg_single_cache.lock().unwrap();
            for i in 0..count {
                let idx = i as usize;
                let v_id = view_ids[idx];
                if cache[idx].is_none() || cache[idx].as_ref().unwrap().0 != v_id {
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("composite_bg_single"),
                        layout: &self.pipelines.bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: self.instance_buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(views[idx]) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.pipelines.sampler) },
                        ],
                    });
                    cache[idx] = Some((v_id, bg));
                }
            }
        }

        let cache_array = self.pipelines.has_binding_arrays.then(|| self.bg_array_cache.lock().unwrap());
        let cache_single = (!self.pipelines.has_binding_arrays).then(|| self.bg_single_cache.lock().unwrap());

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

        if self.pipelines.has_binding_arrays {
            // Binding-array path: group consecutive clips with the same blend mode
            // and emit one instanced draw call per group.
            let bg = &cache_array.as_ref().unwrap().as_ref().unwrap().1;
            pass.set_bind_group(0, bg, &[]);

            let mut group_start = 0u32;
            let mut current_mode = frame.clips[0].blend_mode;
            pass.set_pipeline(self.pipeline_for_blend_mode(current_mode));

            for i in 1..count {
                let mode = frame.clips[i as usize].blend_mode;
                if mode != current_mode {
                    pass.draw(0..6, group_start..i);
                    current_mode = mode;
                    group_start = i;
                    pass.set_pipeline(self.pipeline_for_blend_mode(current_mode));
                }
            }
            pass.draw(0..6, group_start..count);
        } else {
            // Single-texture path: set pipeline per clip draw call.
            let cache = cache_single.as_ref().unwrap();
            for i in 0..count {
                let mode = frame.clips[i as usize].blend_mode;
                pass.set_pipeline(self.pipeline_for_blend_mode(mode));
                pass.set_bind_group(0, &cache[i as usize].as_ref().unwrap().1, &[]);
                pass.draw(0..6, i..(i + 1));
            }
        }
    }
}
