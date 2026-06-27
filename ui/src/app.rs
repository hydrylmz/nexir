use egui::Context;
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use winit::window::Window;
use winit::event::WindowEvent;
use nexir::render::device::GpuDevice;
use nexir::project::Project;
use nexir::render::shader::registry::ShaderRegistry;
use nexir::render::compute::ComputePipelineCache;
use nexir::render::nodes::yuv_to_rgb::YuvParams;
use nexir::timeline::query::query_active;
use nexir::timeline::ids::SourceId;
use nexir::io::demuxer::Demuxer;
use nexir::render::shader::registry::BuiltinShader;
use nexir::render::compute::{ComputePassHelper, PipelineKey};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::collections::HashMap;
use crate::layout::inspector::InspectorState;
use crate::layout::media_pool::MediaPoolState;
use crate::layout::timeline::TimelineState;
use nexir::audio::audio_decoder::AudioDecoder;
use nexir::audio::output_stream::AudioOutputStream;
use nexir::audio::ring_buffer::AudioRingBuffer;
use nexir::sync::master_clock::MasterClock;
use nexir::timeline::rational::Rational;

pub struct PreviewState {
    pub texture:      Option<wgpu::Texture>,
    pub texture_id:   Option<egui::TextureId>,
    pub width:        u32,   // texture / panel size
    pub height:       u32,
    pub video_width:  u32,   // actual decoded frame dimensions
    pub video_height: u32,
}

/// A fully decoded YUV frame ready to upload to the GPU.
#[derive(Clone)]
pub struct DecodedFrame {
    pub source_id: SourceId,
    pub pts:       i64,
    pub width:     u32,
    pub height:    u32,
    pub data:      Vec<u8>,  // planar YUV420p: Y then U then V
    pub is_nv12:   bool,
}

/// Request sent from render thread → decode thread.
struct DecodeRequest {
    source_id:    SourceId,
    pts:          i64,
    path:         std::path::PathBuf,
    width:        u32,
    height:       u32,
    /// True when user scrubbed/jumped — forces a seek even if moving forward.
    force_seek:   bool,
}

/// Per-source GPU textures for YUV planes.
struct ClipTextures {
    y_tex:         wgpu::Texture,
    uv_tex:        wgpu::Texture,
    rgba_tex:      wgpu::Texture,
    tex_width:     u32,
    tex_height:    u32,
    yuv_bgl:       wgpu::BindGroupLayout,
    yuv_pipeline:  Arc<wgpu::ComputePipeline>,
    blit_bgl:      wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    sampler:       wgpu::Sampler,
}

impl ClipTextures {
    fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        compute_cache:  &ComputePipelineCache,
        width:          u32,
        height:         u32,
    ) -> Self {
        let y_tex = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv_y"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let uv_tex = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv_uv"),
            size: wgpu::Extent3d { width: width / 2, height: height / 2, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let rgba_tex = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("yuv_rgba"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let yuv_bgl = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None },
            ],
        });

        let yuv_pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yuv_pl"), bind_group_layouts: &[&yuv_bgl],
            push_constant_ranges: &[wgpu::PushConstantRange { stages: wgpu::ShaderStages::COMPUTE, range: 0..16 }],
        });
        let yuv_shader = shaders.get(BuiltinShader::YuvToRgb);
        let yuv_pipeline = compute_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::YuvToRgb, entry_point: "cs_main" },
            &yuv_pipeline_layout,
            &yuv_shader,
        );

        let blit_bgl = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None },
            ],
        });
        let blit_pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pl"), bind_group_layouts: &[&blit_bgl], push_constant_ranges: &[],
        });
        let blit_shader = shaders.get(BuiltinShader::Blit);
        let blit_pipeline = device.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"), layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState { module: &blit_shader, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader, entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None, write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None,
        });

        let sampler = device.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self { y_tex, uv_tex, rgba_tex, tex_width: width, tex_height: height,
               yuv_bgl, yuv_pipeline, blit_bgl, blit_pipeline, sampler }
    }
}

pub struct NexirApp {
    egui_ctx:         Context,
    egui_state:       State,
    pub egui_renderer: Renderer,
    pub preview:      PreviewState,
    pub inspector:    InspectorState,
    pub media_pool:   MediaPoolState,
    pub timeline:     TimelineState,
    pub project:      Project,
    pub shaders:      ShaderRegistry,
    pub compute_cache: ComputePipelineCache,

    // Video decode thread communication
    decode_tx:        std::sync::mpsc::SyncSender<DecodeRequest>,
    decode_rx:        std::sync::mpsc::Receiver<DecodedFrame>,
    last_frame:       Option<DecodedFrame>,
    last_playhead:    i64,

    clip_textures:    HashMap<SourceId, ClipTextures>,

    // Audio engine
    _audio_out:        Option<AudioOutputStream>,
    audio_ring:        Arc<AudioRingBuffer>,
    audio_clock:       Arc<MasterClock>,
    audio_shutdown:    Arc<AtomicBool>,
    audio_seek:        Arc<Mutex<Option<(i64, i64)>>>,
    audio_path:        Option<std::path::PathBuf>,  // currently playing audio file
    audio_was_playing: bool,
}

pub struct AppResponse {
    pub consumed: bool,
}

impl NexirApp {
    pub fn new(device: &GpuDevice, window: &Window) -> Self {
        let egui_ctx = Context::default();

        let mut style = (*egui_ctx.style()).clone();
        style.visuals.window_fill = egui::Color32::from_rgb(26, 26, 26);
        style.visuals.panel_fill = egui::Color32::from_rgb(26, 26, 26);
        style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(35, 35, 35);
        style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(45, 45, 45);
        style.visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(55, 55, 55);
        style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0, 153, 255);
        style.visuals.selection.bg_fill = egui::Color32::from_rgb(0, 153, 255);
        egui_ctx.set_style(style);

        let egui_state = State::new(
            egui_ctx.clone(), egui::ViewportId::ROOT, window,
            Some(window.scale_factor() as f32), None,
        );
        let egui_renderer = Renderer::new(device.device.as_ref(), device.surface_format, None, 1);

        let mut project = Project::new("Untitled Project");
        let _ = project.add_video_track("Video 1");
        let _ = project.add_video_track("Video 2");
        let _ = project.add_audio_track("Audio 1");

        let shaders = ShaderRegistry::compile_all(&device).unwrap();
        let compute_cache = ComputePipelineCache::new();

        // Spawn background decode thread
        let (req_tx, req_rx) = std::sync::mpsc::sync_channel::<DecodeRequest>(4);
        let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<DecodedFrame>(4);

        std::thread::spawn(move || {
            decode_thread(req_rx, frame_tx);
        });

        // Set up audio engine
        let project_tb = Rational { num: 1, den: 90_000 };
        let audio_ring = AudioRingBuffer::new(1 << 17); // 131072 samples
        let audio_clock = MasterClock::new(project_tb, 48_000);
        let audio_shutdown = Arc::new(AtomicBool::new(false));
        let audio_seek = Arc::new(Mutex::new(None::<(i64, i64)>));

        let audio_out = AudioOutputStream::open(
            Arc::clone(&audio_ring),
            Arc::clone(&audio_clock),
        ).map_err(|e| eprintln!("[audio] Failed to open output: {:?}", e)).ok();

        Self {
            egui_ctx,
            egui_state,
            egui_renderer,
            preview: PreviewState { texture: None, texture_id: None, width: 0, height: 0, video_width: 0, video_height: 0 },
            inspector: InspectorState::default(),
            media_pool: MediaPoolState::default(),
            timeline: TimelineState::default(),
            project,
            shaders,
            compute_cache,
            decode_tx: req_tx,
            decode_rx: frame_rx,
            last_frame: None,
            last_playhead: -1,
            clip_textures: HashMap::new(),
            _audio_out: audio_out,
            audio_ring,
            audio_clock,
            audio_shutdown,
            audio_seek,
            audio_path: None,
            audio_was_playing: false,
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> AppResponse {
        let response = self.egui_state.on_window_event(window, event);
        AppResponse { consumed: response.consumed }
    }

    pub fn update(&mut self, window: &Window) -> egui::Vec2 {
        let raw_input = self.egui_state.take_egui_input(window);
        self.egui_ctx.begin_frame(raw_input);

        egui::TopBottomPanel::top("top_bar").show(&self.egui_ctx, |ui| {
            crate::layout::top_bar::draw(ui);
        });

        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .default_height(300.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::timeline::draw(ui, &mut self.project, &mut self.timeline, &mut self.media_pool.dragging_item);
            });

        egui::SidePanel::left("media_pool")
            .resizable(true)
            .default_width(350.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::media_pool::draw(ui, &mut self.media_pool);
            });

        // Process media imports: probe file and register with video dimensions
        for entry in &self.media_pool.entries {
            let already = self.project.sources.read().unwrap().path_registered(&entry.path);
            if !already {
                let (vid_info, aud_info) = {
                    use nexir::timeline::source::{VideoStreamInfo, AudioStreamInfo, PixelFormat, ColorSpace, SampleFormat};
                    use nexir::timeline::rational::Rational;
                    if let Ok(demuxer) = Demuxer::open(&entry.path) {
                        let project_tb = Rational { num: 1, den: 90_000 };
                        
                        let vi = demuxer.video_stream.as_ref().map(|s| VideoStreamInfo {
                            width:        s.width.unwrap_or(1920),
                            height:       s.height.unwrap_or(1080),
                            frame_rate:   s.frame_rate.unwrap_or(Rational { num: 30, den: 1 }),
                            pixel_fmt:    PixelFormat::Yuv420p,
                            color_space:  ColorSpace::Bt709,
                            // convert from stream timebase to project timebase
                            duration_pts: project_tb.from_pts(s.duration, s.time_base),
                        });

                        let ai = demuxer.audio_stream.as_ref().map(|s| AudioStreamInfo {
                            sample_rate:  48000,
                            channels:     2,
                            sample_fmt:   SampleFormat::F32Interleaved,
                            duration_pts: project_tb.from_pts(s.duration, s.time_base),
                        });

                        (vi, ai)
                    } else {
                        (None, None)
                    }
                };
                self.project.register_source(entry.path.clone(), vid_info, aud_info);
            }
        }

        egui::SidePanel::right("inspector")
            .resizable(true)
            .default_width(350.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::inspector::draw(
                    ui,
                    &mut self.inspector,
                    &mut self.project,
                    self.timeline.selected_clip,
                );
            });

        let mut viewport_size = egui::Vec2::ZERO;
        egui::CentralPanel::default().show(&self.egui_ctx, |ui| {
            viewport_size = crate::layout::viewport::draw(
                ui,
                self.preview.texture_id,
                self.preview.video_width,
                self.preview.video_height,
            );
        });

        let mouse_released = self.egui_ctx.input(|i| i.pointer.any_released());
        if mouse_released {
            self.media_pool.dragging_item = None;
        }

        // Drain any completed decoded frames from the background thread
        while let Ok(frame) = self.decode_rx.try_recv() {
            self.preview.video_width  = frame.width;
            self.preview.video_height = frame.height;
            self.last_frame = Some(frame);
        }

        // Check if playback just started this frame
        let just_started_playing = self.timeline.playing && !self.audio_was_playing;

        // Sync playhead to audio clock if playing (and not just started, to avoid pulling it back to old clock)
        if self.timeline.playing && self.audio_path.is_some() && !just_started_playing {
            let clock_pts = self.audio_clock.pts();
            self.timeline.playhead_frame = self.project.pts_to_frame(clock_pts);
        }
        
        if self.timeline.playing != self.audio_was_playing {
            if let Some(audio_out) = &self._audio_out {
                if self.timeline.playing {
                    let _ = audio_out.play();
                } else {
                    let _ = audio_out.pause();
                }
            }
            self.audio_was_playing = self.timeline.playing;
        }

        // Check if playhead moved OR if playback just started
        let playhead_pts = self.project.frame_to_pts(self.timeline.playhead_frame);
        if playhead_pts != self.last_playhead || just_started_playing {
            // Detect a scrub/jump:
            // 1. We just started playing (need to sync decoder to playhead)
            // 2. We moved backward (always a jump)
            // 3. We jumped more than 2s forward (definitely not normal playback)
            // 4. We moved while PAUSED (always a scrub, even if it's a small forward move)
            let pts_delta = playhead_pts - self.last_playhead;
            let mut force_seek = just_started_playing 
                              || pts_delta < 0 
                              || pts_delta > 180_000 
                              || !self.timeline.playing; // Moved while paused
                              
            self.last_playhead = playhead_pts;

            let mut active_clips = Vec::new();
            query_active(&self.project.clips, playhead_pts, &mut active_clips);

            // Separate active clips by track kind
            let mut top_video_clip = None;
            let mut top_audio_clip = None;
            let mut fallback_video_audio = None;
            for clip in &active_clips {
                let track_id = self.project.clips.track_id_at(clip.store_index);
                if let Some(track) = self.project.tracks.get(track_id) {
                    use nexir::timeline::track::TrackKind;
                    match track.kind {
                        TrackKind::Video => {
                            if top_video_clip.is_none() { top_video_clip = Some(clip.clone()); }
                            if fallback_video_audio.is_none() {
                                let source_id = self.project.clips.source_id_at(clip.store_index);
                                if self.project.sources.read().unwrap().audio_info(source_id).is_ok() {
                                    fallback_video_audio = Some(clip.clone());
                                }
                            }
                        }
                        TrackKind::Audio { .. } => {
                            if top_audio_clip.is_none() { top_audio_clip = Some(clip.clone()); }
                        }
                        _ => {}
                    }
                }
            }
            let top_audio_clip = top_audio_clip.or(fallback_video_audio);

            // ── HANDLE VIDEO CLIP ──
            if let Some(clip) = top_video_clip {
                let source_id = self.project.clips.source_id_at(clip.store_index);
                let source_pts = clip.source_pts;

                let path_and_size = {
                    let sources = self.project.sources.read().unwrap();
                    if let Ok(vi) = sources.video_info(source_id) {
                        let path = sources.path(source_id);
                        path.map(|p| (p.as_ref().clone(), vi.width, vi.height))
                    } else {
                        None
                    }
                };

                if let Some((path, w, h)) = path_and_size {
                    let _ = self.decode_tx.try_send(DecodeRequest {
                        source_id, pts: source_pts, path, width: w, height: h, force_seek,
                    });
                }
            }

            // ── HANDLE AUDIO CLIP ──
            if let Some(clip) = top_audio_clip {
                let source_id = self.project.clips.source_id_at(clip.store_index);
                let source_pts = clip.source_pts;
                let path = {
                    let sources = self.project.sources.read().unwrap();
                    sources.path(source_id).map(|p| p.as_ref().clone())
                };

                if let Some(path) = path {
                    // Check if we need to switch to a new audio file
                    let need_new_decoder = self.audio_path.as_ref() != Some(&path);
                    if need_new_decoder {
                        // Stop current decoder if running
                        self.audio_shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
                        // Give it a brief moment to exit
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        
                        self.audio_ring.clear();
                        self.audio_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                        
                        let clock = std::sync::Arc::clone(&self.audio_clock);
                        let ring = std::sync::Arc::clone(&self.audio_ring);
                        let seek = std::sync::Arc::clone(&self.audio_seek);
                        let shutdown = std::sync::Arc::clone(&self.audio_shutdown);
                        let project_tb = nexir::timeline::rational::Rational { num: 1, den: 90_000 };
                        
                        let path_clone = path.clone();
                        eprintln!("[app] Starting audio decoder for {:?}", path_clone);
                        std::thread::spawn(move || {
                            match AudioDecoder::new(
                                &path_clone, ring, clock, project_tb, shutdown, seek
                            ) {
                                Ok(mut decoder) => decoder.run(),
                                Err(e) => eprintln!("Failed to open audio decoder for {:?}: {:?}", path_clone, e),
                            }
                        });
                        
                        self.audio_path = Some(path);
                        force_seek = true; // force a seek when opening a new file
                    }
                    
                    if force_seek {
                        *self.audio_seek.lock().unwrap() = Some((source_pts, playhead_pts));
                        self.audio_clock.seek(playhead_pts);
                    }
                }
            }
            // NOTE: We do NOT kill the decoder when top_audio_clip is None.
            // The decoder keeps running for the current file — it will naturally
            // hit EOF and stop producing samples. Explicit stop happens on pause.
        }

        // Stop audio decoder when user pauses / stops playback
        if !self.timeline.playing && self.audio_path.is_some() {
            self.audio_shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
            self.audio_path = None;
            self.audio_ring.clear();
            // Reset clock so next play starts fresh
            self.audio_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        }

        viewport_size
    }

    pub fn render(&mut self, device: &GpuDevice, surface: &wgpu::Surface, window: &Window, viewport_size: egui::Vec2) {
        let width = viewport_size.x as u32;
        let height = viewport_size.y as u32;

        if width > 0 && height > 0 {
            if self.preview.width != width || self.preview.height != height {
                let texture = device.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Preview Texture"),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let tex_id = self.egui_renderer.register_native_texture(
                    device.device.as_ref(), &view, wgpu::FilterMode::Linear,
                );
                if let Some(old_id) = self.preview.texture_id {
                    self.egui_renderer.free_texture(&old_id);
                }
                self.preview.texture = Some(texture);
                self.preview.texture_id = Some(tex_id);
                self.preview.width = width;
                self.preview.height = height;
            }
        }

        let output = self.egui_ctx.end_frame();
        let clipped_primitives = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);

        let surface_texture = match surface.get_current_texture() {
            Ok(t)  => t,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => { log::error!("Dropped frame: {:?}", e); return; }
        };
        let surface_view = surface_texture.texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.begin_frame();

        // If we have a decoded frame, upload it and blit to preview
        if let (Some(frame), Some(preview_texture)) = (&self.last_frame.clone(), &self.preview.texture) {
            let vid_w = frame.width;
            let vid_h = frame.height;
            let source_id = frame.source_id;

            // Ensure GPU textures exist for this source
            if !self.clip_textures.contains_key(&source_id) {
                let ct = ClipTextures::new(device, &self.shaders, &self.compute_cache, vid_w, vid_h);
                self.clip_textures.insert(source_id, ct);
            }

            if let Some(ct) = self.clip_textures.get(&source_id) {
                // Upload Y plane
                let y_size = (vid_w * vid_h) as usize;
                let y_bpr  = (vid_w + 255) & !255;
                let mut y_staging = vec![0u8; (y_bpr * vid_h) as usize];
                for row in 0..vid_h as usize {
                    let src = &frame.data[row * vid_w as usize .. (row+1) * vid_w as usize];
                    let dst_off = row * y_bpr as usize;
                    y_staging[dst_off .. dst_off + vid_w as usize].copy_from_slice(src);
                }
                device.queue.write_texture(
                    wgpu::ImageCopyTexture { texture: &ct.y_tex, mip_level: 0,
                        origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &y_staging,
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(y_bpr), rows_per_image: Some(vid_h) },
                    wgpu::Extent3d { width: vid_w, height: vid_h, depth_or_array_layers: 1 },
                );

                // Upload UV plane (interleaved Rg8Unorm, half res)
                let uv_half_w = vid_w / 2;
                let uv_half_h = vid_h / 2;
                let uv_row_bytes = vid_w; // width/2 * 2 bytes = width
                let uv_bpr = (uv_row_bytes + 255) & !255;
                let mut uv_staging = vec![0u8; (uv_bpr * uv_half_h) as usize];

                let uv_plane_size = (uv_half_w * uv_half_h) as usize;
                if !frame.is_nv12 && frame.data.len() >= y_size + uv_plane_size * 2 {
                    // YUV420p: separate U and V planes → interleave into RG
                    let u_plane = &frame.data[y_size .. y_size + uv_plane_size];
                    let v_plane = &frame.data[y_size + uv_plane_size ..
                                              (y_size + uv_plane_size * 2).min(frame.data.len())];
                    for row in 0..uv_half_h as usize {
                        for col in 0..uv_half_w as usize {
                            let src_i = row * uv_half_w as usize + col;
                            let dst_i = row * uv_bpr as usize + col * 2;
                            uv_staging[dst_i]     = u_plane[src_i];
                            uv_staging[dst_i + 1] = if src_i < v_plane.len() { v_plane[src_i] } else { 128 };
                        }
                    }
                } else if frame.data.len() > y_size {
                    // NV12: already interleaved UV
                    let src_uv = &frame.data[y_size..];
                    for row in 0..uv_half_h as usize {
                        let src_row_start = row * vid_w as usize;
                        let src_row_end   = (src_row_start + vid_w as usize).min(src_uv.len());
                        let dst_off = row * uv_bpr as usize;
                        let len = (src_row_end - src_row_start).min(uv_row_bytes as usize);
                        uv_staging[dst_off .. dst_off + len]
                            .copy_from_slice(&src_uv[src_row_start .. src_row_start + len]);
                    }
                }
                device.queue.write_texture(
                    wgpu::ImageCopyTexture { texture: &ct.uv_tex, mip_level: 0,
                        origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &uv_staging,
                    wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(uv_bpr), rows_per_image: Some(uv_half_h) },
                    wgpu::Extent3d { width: uv_half_w, height: uv_half_h, depth_or_array_layers: 1 },
                );

                // YUV → RGB compute pass
                {
                    let y_view    = ct.y_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let uv_view   = ct.uv_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let rgba_view = ct.rgba_tex.create_view(&wgpu::TextureViewDescriptor::default());

                    let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("yuv_bg"), layout: &ct.yuv_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&y_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&uv_view) },
                            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&rgba_view) },
                        ],
                    });

                    let params = YuvParams { color_space: 1, limited_range: 1, width: vid_w, height: vid_h };
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("yuv_to_rgb"), timestamp_writes: None,
                    });
                    ComputePassHelper::dispatch(&mut pass, &ct.yuv_pipeline, &bg,
                        Some(bytemuck::bytes_of(&params)), vid_w, vid_h);
                }

                // Blit RGBA16Float → Rgba8UnormSrgb preview
                {
                    let rgba_view    = ct.rgba_tex.create_view(&wgpu::TextureViewDescriptor::default());
                    let preview_view = preview_texture.create_view(&wgpu::TextureViewDescriptor::default());

                    let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("blit_bg"), layout: &ct.blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&rgba_view) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ct.sampler) },
                        ],
                    });

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("blit_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &preview_view, resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                    });
                    pass.set_pipeline(&ct.blit_pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        } else if let Some(ref preview_texture) = self.preview.texture {
            // Clear preview when no frame
            let preview_view = preview_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &preview_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
        }

        // egui textures
        for (id, image_delta) in &output.textures_delta.set {
            self.egui_renderer.update_texture(device.device.as_ref(), device.queue.as_ref(), *id, image_delta);
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [surface_texture.texture.width(), surface_texture.texture.height()],
            pixels_per_point: window.scale_factor() as f32,
        };
        self.egui_renderer.update_buffers(
            device.device.as_ref(), device.queue.as_ref(), &mut encoder, &clipped_primitives, &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            self.egui_renderer.render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        device.submit(encoder);
        surface_texture.present();

        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

// ── Background decode thread ──────────────────────────────────────────────────

fn decode_thread(
    rx: std::sync::mpsc::Receiver<DecodeRequest>,
    tx: std::sync::mpsc::SyncSender<DecodedFrame>,
) {
    // Keep one demuxer+decoder open per source path to avoid reopening on every request.
    let mut demuxers: HashMap<std::path::PathBuf, Demuxer> = HashMap::new();
    let mut decoders: HashMap<std::path::PathBuf, nexir::io::decoder::Decoder> = HashMap::new();

    // Track the last PTS we decoded per path so we can skip seeking for sequential playback.
    let mut last_decoded_pts: HashMap<std::path::PathBuf, i64> = HashMap::new();

    while let Ok(req) = rx.recv() {
        // Detect still images — handled specially: no seeking, software decode only.
        let is_still_image = req.path.extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "tif" | "webp" | "gif"))
            .unwrap_or(false);

        // ── STILL IMAGE PATH ────────────────────────────────────────────────────
        if is_still_image {
            // Reopen demuxer fresh each time (seeking images is unreliable)
            let demuxer = match Demuxer::open(&req.path) {
                Ok(d)  => d,
                Err(e) => { log::warn!("Image demux open failed: {:?}", e); continue; }
            };
            demuxers.insert(req.path.clone(), demuxer);

            let demuxer = demuxers.get_mut(&req.path).unwrap();
            if let Some(stream_info) = &demuxer.video_stream {
                let si = stream_info.clone();
                // Software decode only — CUDA cannot handle MJPEG
                if let Ok(dec) = nexir::io::decoder::Decoder::open_sw(&si, si.codecpar) {
                    decoders.insert(req.path.clone(), dec);
                }
            }

            let demuxer = match demuxers.get_mut(&req.path) { Some(d) => d, None => continue };
            let decoder = match decoders.get_mut(&req.path) { Some(d) => d, None => continue };

            let y_size        = (req.width * req.height) as usize;
            let uv_plane_size = ((req.width / 2) * (req.height / 2)) as usize;
            let mut buf       = vec![0u8; y_size + uv_plane_size * 2];

            while let Ok(Some(pkt)) = demuxer.next_video_packet() {
                if let Ok(Some((_, is_nv12, w, h))) = decoder.decode_into(&pkt, &mut buf, None) {
                    let _ = tx.try_send(DecodedFrame {
                        source_id: req.source_id,
                        pts: 0,
                        width: w,
                        height: h,
                        data: buf.clone(),
                        is_nv12,
                    });
                    break;
                }
            }
            continue;
        }

        // ── VIDEO PATH ──────────────────────────────────────────────────────────

        // Open demuxer+decoder if not already open.
        if !demuxers.contains_key(&req.path) {
            match Demuxer::open(&req.path) {
                Ok(d)  => { demuxers.insert(req.path.clone(), d); }
                Err(e) => { log::warn!("Demux open failed: {:?}", e); continue; }
            }
        }
        if !decoders.contains_key(&req.path) {
            if let Some(stream_info) = demuxers.get(&req.path).and_then(|d| d.video_stream.as_ref()) {
                let si = stream_info.clone();
                match nexir::io::decoder::Decoder::open(&si, si.codecpar, true) {
                    Ok(dec) => { decoders.insert(req.path.clone(), dec); }
                    Err(e)  => { log::warn!("Decoder open failed: {:?}", e); continue; }
                }
            }
        }

        // Decide whether to seek.
        // We seek only when: user forced a seek (scrub/jump) OR we have no prior position.
        // For normal sequential playback we just keep reading forward — this is the key
        // optimization that makes smooth playback possible.
        let prev_pts = last_decoded_pts.get(&req.path).copied().unwrap_or(i64::MIN);
        let need_seek = req.force_seek || prev_pts == i64::MIN;

        let project_tb = nexir::timeline::rational::Rational { num: 1, den: 90_000 };
        let stream_tb = match demuxers.get(&req.path).and_then(|d| d.video_stream.as_ref()) {
            Some(s) => s.time_base,
            None => continue,
        };
        let target_stream_pts = project_tb.rescale_pts(req.pts, stream_tb);

        let stream_pts = if need_seek {
            let demuxer = match demuxers.get_mut(&req.path) { Some(d) => d, None => continue };
            let decoder = match decoders.get_mut(&req.path) { Some(d) => d, None => continue };
            match demuxer.seek(req.pts, project_tb) {
                Ok(p) => {
                    decoder.flush();
                    p
                }
                Err(e) => { log::warn!("Seek failed: {:?}", e); continue; }
            }
        } else {
            // Sequential: target PTS is the request PTS in stream time.
            // We'll read forward until we meet or pass it.
            target_stream_pts
        };

        let demuxer = match demuxers.get_mut(&req.path) { Some(d) => d, None => continue };
        let decoder = match decoders.get_mut(&req.path) { Some(d) => d, None => continue };

        // Allocate output buffer — large enough for YUV420p or NV12.
        let y_size        = (req.width * req.height) as usize;
        let uv_plane_size = ((req.width / 2) * (req.height / 2)) as usize;
        let mut buf       = vec![0u8; y_size + uv_plane_size * 2];

        // Decode packets until we reach or pass the target PTS.
        let mut last_pkt_pts = prev_pts;
        'decode: for _ in 0..600 {
            let pkt = match demuxer.next_video_packet() {
                Ok(Some(p)) => p,
                _           => break,  // EOF
            };
            let pkt_pts = pkt.pts;
            match decoder.decode_into(&pkt, &mut buf, None) {
                Ok(Some((frame_pts, is_nv12, actual_w, actual_h))) => {
                    let effective_pts = if frame_pts == 0 { pkt_pts } else { frame_pts };
                    last_pkt_pts = effective_pts;
                    if effective_pts >= stream_pts {
                        last_decoded_pts.insert(req.path.clone(), effective_pts);
                        let _ = tx.try_send(DecodedFrame {
                            source_id: req.source_id,
                            pts:       effective_pts,
                            width:     actual_w,
                            height:    actual_h,
                            data:      buf,
                            is_nv12,
                        });
                        break 'decode;
                    }
                    // Not at target yet — keep reading forward
                }
                Ok(None) => continue,
                Err(e)   => { log::warn!("Decode error: {:?}", e); break; }
            }
            let _ = last_pkt_pts; // suppress unused warning
        }
    }
}
