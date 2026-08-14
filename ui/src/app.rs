use crate::history::HistoryState;
use crate::layout::inspector::InspectorState;
use crate::layout::media_pool::MediaPoolState;
use crate::layout::timeline::TimelineState;
use egui::Context;
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use nexir::audio::audio_decoder::AudioDecoder;
use nexir::audio::output_stream::AudioOutputStream;
use nexir::audio::ring_buffer::AudioRingBuffer;
use nexir::io::demuxer::Demuxer;
use nexir::project::Project;
use nexir::project_file::ProjectFile;
use nexir::render::compute::ComputePipelineCache;
use nexir::render::device::GpuDevice;
use nexir::render::shader::registry::BuiltinShader;
use nexir::render::shader::registry::ShaderRegistry;
use nexir::sync::master_clock::MasterClock;
use nexir::timeline::query::query_active;
use nexir::timeline::rational::Rational;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::layout::export_settings::ExportSettings;
use nexir::export::progress::ProgressReceiver;
use nexir::interop::capability::InteropCapability;
use nexir::interop::cuda_context::CudaContext;
use nexir::io::frame_cache::FrameCache;
use nexir::io::io_layer::IoLayer;
use nexir::io::prefetch::{PrefetchRequest, PrefetchWorker, spawn_prefetch_worker};
use nexir::io::slot_pool::FrameSlotPool;
use nexir::render::graph::RenderGraphCompiler;
use nexir::render::nodes::composite::CompositeNode;
use nexir::render::nodes::yuv_upload::YuvUploadNode;
use nexir::render::resource::ResourceId;
use nexir::scheduler::frame_scheduler::FrameScheduler;

use crate::image_still::{StillImageCache, StillImageUploadNode};

pub struct PreviewState {
    pub texture: Option<wgpu::Texture>,
    pub texture_id: Option<egui::TextureId>,
    pub width: u32, // texture / panel size
    pub height: u32,
    pub video_width: u32, // actual decoded frame dimensions
    pub video_height: u32,
}

#[derive(Clone, Debug)]
pub struct ActiveClipAudioInfo {
    pub clip_id: nexir::timeline::ids::ClipId,
    pub path: PathBuf,
    pub volume: f32,
    pub pan: f32,
    pub muted: bool,
    pub speed: f32,
    pub pitch: f32,
    pub source_pts: i64,
    pub timeline_pts: i64,
    pub source_in_pts: i64,
    pub source_out_pts: i64,
}

pub struct NexirApp {
    egui_ctx: Context,
    egui_state: State,
    pub egui_renderer: Renderer,
    pub preview: PreviewState,
    pub inspector: InspectorState,
    pub media_pool: MediaPoolState,
    pub timeline: TimelineState,
    pub history: HistoryState,
    pub project: Project,
    pub shaders: Arc<ShaderRegistry>,
    pub compute_cache: Arc<ComputePipelineCache>,

    // Backend rendering state
    io_layer: Arc<IoLayer>,
    frame_scheduler: FrameScheduler,
    last_playhead: i64,

    // UI blit resources
    blit_bgl: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,

    // Audio engine
    _audio_out: Option<AudioOutputStream>,
    audio_ring: Arc<AudioRingBuffer>,
    audio_clock: Arc<MasterClock>,
    audio_shutdown: Arc<AtomicBool>,
    audio_seek: Arc<Mutex<Option<(i64, i64)>>>,
    audio_path: Option<std::path::PathBuf>, // currently playing audio file
    audio_volume: f32,
    audio_pan: f32,
    audio_muted: bool,
    audio_speed: f32,
    audio_pitch: f32,
    audio_was_playing: bool,

    // Waveform display cache
    waveform_cache: crate::waveform::WaveformCache,

    // Project file management
    current_project_path: Option<PathBuf>,

    // GPU device (shared for export)
    device: Arc<GpuDevice>,

    // Export state
    export_progress: Option<ProgressReceiver>,
    export_status: Option<String>,
    export_progress_pct: f32,
    export_settings_open: bool,
    export_pending_path: Option<PathBuf>,
    export_settings: ExportSettings,
    interop_capability: InteropCapability,
    cuda_ctx: Option<Arc<CudaContext>>,
    still_cache: Mutex<StillImageCache>,
}

pub struct AppResponse {
    pub consumed: bool,
}

impl NexirApp {
    pub fn new(device: Arc<GpuDevice>, window: &Window) -> Self {
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
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
        );
        log::info!("NexirApp::new: creating egui_renderer...");
        let egui_renderer = Renderer::new(
            device.device.as_ref(),
            *device.surface_format.lock().unwrap(),
            None,
            1,
        );
        log::info!("NexirApp::new: egui_renderer created");

        let mut project = Project::new("Untitled Project");
        let _ = project.add_video_track("Video 1");
        let _ = project.add_video_track("Video 2");
        let _ = project.add_audio_track("Audio 1");

        log::info!("NexirApp::new: compiling shaders...");
        let shaders = Arc::new(ShaderRegistry::compile_all(&device).unwrap());
        log::info!("NexirApp::new: shaders compiled successfully");

        let compute_cache = Arc::new(ComputePipelineCache::new());

        // Initialize Backend Systems
        let project_tb = Rational {
            num: 1,
            den: 90_000,
        };

        let pool = Arc::new(FrameSlotPool::new(&device));
        let cache = Arc::new(FrameCache::new(pool.clone(), 32));
        log::info!("NexirApp::new: pool and cache created");

        let (prefetch_tx, prefetch_rx) = std::sync::mpsc::sync_channel::<PrefetchRequest>(16);

        let io_layer = Arc::new(IoLayer::new(
            device.device.clone(),
            pool,
            cache.clone(),
            project.sources.clone(),
            prefetch_tx,
            project_tb,
        ));
        log::info!("NexirApp::new: io_layer created");

        let prefetch_shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker = PrefetchWorker::new(prefetch_rx, io_layer.clone(), cache, prefetch_shutdown);
        spawn_prefetch_worker(worker);

        // Setup FrameScheduler — use project canvas resolution
        let canvas_w = project.settings.width;
        let canvas_h = project.settings.height;
        let frame_scheduler = FrameScheduler::new(io_layer.clone(), canvas_w, canvas_h);
        log::info!("NexirApp::new: frame_scheduler created");

        // Blit pipeline for preview
        let blit_bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("app_blit_bgl"),
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
        let blit_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("app_blit_pl"),
                    bind_group_layouts: &[&blit_bgl],
                    push_constant_ranges: &[],
                });
        let blit_shader = shaders.get(BuiltinShader::Blit);
        let blit_pipeline = device
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("app_blit_pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
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
        log::info!("NexirApp::new: blit resources created");

        // Set up audio engine
        let project_tb = Rational {
            num: 1,
            den: 90_000,
        };
        let audio_ring = AudioRingBuffer::new(1 << 17); // 131072 samples
        let audio_clock = MasterClock::new(project_tb, 48_000);
        let audio_shutdown = Arc::new(AtomicBool::new(false));
        let audio_seek = Arc::new(Mutex::new(None::<(i64, i64)>));

        log::info!("NexirApp::new: CPAL audio stream opening...");
        let audio_out = AudioOutputStream::open(Arc::clone(&audio_ring), Arc::clone(&audio_clock))
            .map_err(|e| eprintln!("[audio] Failed to open output: {:?}", e))
            .ok();
        log::info!(
            "NexirApp::new: CPAL audio stream status: {:?}",
            audio_out.is_some()
        );

        log::info!("NexirApp::new: InteropCapability probing...");
        let interop_capability = InteropCapability::probe(&device);
        log::info!(
            "NexirApp::new: InteropCapability probed: {:?}",
            interop_capability
        );

        log::info!("NexirApp::new: completed successfully!");
        Self {
            egui_ctx,
            egui_state,
            egui_renderer,
            preview: PreviewState {
                texture: None,
                texture_id: None,
                width: 0,
                height: 0,
                video_width: 0,
                video_height: 0,
            },
            inspector: InspectorState::default(),
            media_pool: MediaPoolState::default(),
            timeline: TimelineState::default(),
            history: HistoryState::default(),
            project,
            shaders,
            compute_cache,
            io_layer,
            frame_scheduler,
            last_playhead: -1,
            blit_bgl,
            blit_pipeline,
            sampler,
            _audio_out: audio_out,
            audio_ring,
            audio_clock,
            audio_shutdown,
            audio_seek,
            audio_path: None,
            current_project_path: None,
            audio_volume: 1.0,
            audio_pan: 0.0,
            audio_muted: false,
            audio_speed: 1.0,
            audio_pitch: 0.0,
            audio_was_playing: false,
            waveform_cache: crate::waveform::WaveformCache::new(),
            device,
            export_progress: None,
            export_status: None,
            export_progress_pct: 0.0,
            export_settings_open: false,
            export_pending_path: None,
            export_settings: ExportSettings::default(),
            interop_capability,
            cuda_ctx: None,
            still_cache: Mutex::new(StillImageCache::default()),
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> AppResponse {
        let response = self.egui_state.on_window_event(window, event);
        AppResponse {
            consumed: response.consumed,
        }
    }

    pub fn update(&mut self, window: &Window) -> egui::Vec2 {
        let raw_input = self.egui_state.take_egui_input(window);
        self.egui_ctx.begin_frame(raw_input);

        let (undo_pressed, redo_pressed) = self.egui_ctx.input(|i| {
            let cmd = i.modifiers.command || i.modifiers.ctrl;
            (
                cmd && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                (cmd && i.key_pressed(egui::Key::Y))
                    || (cmd && i.modifiers.shift && i.key_pressed(egui::Key::Z)),
            )
        });
        if undo_pressed && self.history.undo(&mut self.project) {
            self.timeline.clear_interaction();
            self.timeline.selected_clip = None;
        } else if redo_pressed && self.history.redo(&mut self.project) {
            self.timeline.clear_interaction();
            self.timeline.selected_clip = None;
        }

        let top_bar_action = {
            let can_undo = self.history.can_undo();
            let can_redo = self.history.can_redo();
            let mut action: Option<crate::layout::top_bar::TopBarAction> = None;
            egui::TopBottomPanel::top("top_bar").show(&self.egui_ctx, |ui| {
                action = crate::layout::top_bar::draw(
                    ui,
                    can_undo,
                    can_redo,
                    &mut self.project.settings,
                );
            });
            action
        };
        if let Some(action) = top_bar_action {
            match action {
                crate::layout::top_bar::TopBarAction::Undo => {
                    if self.history.undo(&mut self.project) {
                        self.timeline.clear_interaction();
                        self.timeline.selected_clip = None;
                    }
                }
                crate::layout::top_bar::TopBarAction::Redo => {
                    if self.history.redo(&mut self.project) {
                        self.timeline.clear_interaction();
                        self.timeline.selected_clip = None;
                    }
                }
                crate::layout::top_bar::TopBarAction::NewProject => {
                    self.new_project();
                }
                crate::layout::top_bar::TopBarAction::OpenProject => {
                    self.open_project();
                }
                crate::layout::top_bar::TopBarAction::Save => {
                    self.save_project();
                }
                crate::layout::top_bar::TopBarAction::SaveAs => {
                    self.save_project_as();
                }
                crate::layout::top_bar::TopBarAction::Export => {
                    self.open_export_settings();
                }
            }
        }

        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .default_height(300.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::timeline::draw(
                    ui,
                    &mut self.project,
                    &mut self.timeline,
                    &mut self.media_pool.dragging_item,
                    &mut self.history,
                    &self.waveform_cache,
                    &self.still_cache,
                );
            });

        let just_started_playing = self.timeline.playing && !self.audio_was_playing;

        egui::SidePanel::left("media_pool")
            .resizable(true)
            .default_width(350.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::media_pool::draw(ui, &mut self.media_pool);
            });

        // Process media imports: probe file and register with video dimensions
        for entry in &self.media_pool.entries {
            let already = self
                .project
                .sources
                .read()
                .unwrap()
                .path_registered(&entry.path);
            if !already {
                let (vid_info, aud_info) = {
                    use nexir::timeline::rational::Rational;
                    use nexir::timeline::source::{
                        AudioStreamInfo, ColorSpace, PixelFormat, SampleFormat, VideoStreamInfo,
                    };
                    if let Ok(demuxer) = Demuxer::open(&entry.path) {
                        let project_tb = Rational {
                            num: 1,
                            den: 90_000,
                        };
                        let is_still_image =
                            nexir::timeline::source::is_still_image_path(&entry.path);

                        let vi = demuxer.video_stream.as_ref().map(|s| VideoStreamInfo {
                            width: s.width.unwrap_or(1920),
                            height: s.height.unwrap_or(1080),
                            frame_rate: if is_still_image {
                                Rational { num: 0, den: 1 }
                            } else {
                                s.frame_rate.unwrap_or(Rational { num: 30, den: 1 })
                            },
                            pixel_fmt: PixelFormat::Yuv420p,
                            color_space: ColorSpace::Bt709,
                            // convert from stream timebase to project timebase
                            duration_pts: if is_still_image {
                                0
                            } else {
                                project_tb.from_pts(s.duration, s.time_base)
                            },
                            is_vfr: !is_still_image && s.is_vfr,
                            time_base: s.time_base,
                        });

                        let ai = demuxer.audio_stream.as_ref().map(|s| AudioStreamInfo {
                            sample_rate: 48000,
                            channels: 2,
                            sample_fmt: SampleFormat::F32Interleaved,
                            duration_pts: project_tb.from_pts(s.duration, s.time_base),
                        });

                        (vi, ai)
                    } else {
                        (None, None)
                    }
                };
                self.project
                    .register_source(entry.path.clone(), vid_info, aud_info);
                // Evict this specific path from still-image cache
                self.still_cache.lock().unwrap().evict(&entry.path);
            }
        }

        self.advance_playhead(just_started_playing);

        egui::SidePanel::right("inspector")
            .resizable(true)
            .default_width(350.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::inspector::draw(
                    ui,
                    &mut self.inspector,
                    &mut self.project,
                    self.timeline.selected_clip,
                    &mut self.history,
                );
            });

        // Query active clips to pass to viewport for interaction (hit testing, drawing handles)
        let playhead_pts = self.project.frame_to_pts(self.timeline.playhead_frame);
        let mut active_clips = Vec::new();
        query_active(&self.project.clips, playhead_pts, &mut active_clips);

        let mut viewport_size = egui::Vec2::ZERO;
        egui::CentralPanel::default().show(&self.egui_ctx, |ui| {
            viewport_size = crate::layout::viewport::draw(
                ui,
                self.preview.texture_id,
                self.preview.video_width,
                self.preview.video_height,
                &mut self.timeline,
                &mut self.project,
                &active_clips,
                &mut self.history,
            );
        });

        let mouse_released = self.egui_ctx.input(|i| i.pointer.any_released());
        if mouse_released {
            self.media_pool.dragging_item = None;
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

            // Respect mute / solo: pre-compute whether any track is soloed.
            let any_soloed = self.project.tracks.any_soloed();

            // Separate active clips by track kind.
            let mut top_audio_clip = None;
            let mut fallback_video_audio = None;
            for clip in &active_clips {
                let track_id = self.project.clips.track_id_at(clip.store_index);
                if let Some(track) = self.project.tracks.get(track_id) {
                    let track_active = track.is_active(any_soloed);
                    use nexir::timeline::track::TrackKind;
                    match track.kind {
                        TrackKind::Video => {
                            // Only use embedded audio if the track is not muted/solo'd out.
                            if track_active && fallback_video_audio.is_none() {
                                let source_id = self.project.clips.source_id_at(clip.store_index);
                                if self
                                    .project
                                    .sources
                                    .read()
                                    .unwrap()
                                    .audio_info(source_id)
                                    .is_ok()
                                {
                                    fallback_video_audio = Some(clip.clone());
                                }
                            }
                        }
                        TrackKind::Audio { .. } => {
                            // Dedicated audio tracks respect mute/solo.
                            if track_active && top_audio_clip.is_none() {
                                top_audio_clip = Some(clip.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }
            let top_audio_clip = top_audio_clip.or(fallback_video_audio);

            // ── HANDLE AUDIO CLIP ──
            if let Some(ref clip) = top_audio_clip {
                let source_id = self.project.clips.source_id_at(clip.store_index);
                let source_pts = clip.source_pts;
                let volume = self.project.clips.volume_at(clip.store_index);
                let pan = self.project.clips.pan_at(clip.store_index);
                let audio_muted = self.project.clips.audio_muted_at(clip.store_index);
                let speed = self.project.clips.speed_at(clip.store_index);
                let pitch = self.project.clips.pitch_at(clip.store_index);
                let source_in_pts = self.project.clips.source_in_at(clip.store_index);
                let pts_out = self.project.clips.pts_out_at(clip.store_index);
                let pts_in = self.project.clips.pts_in_at(clip.store_index);
                let duration_pts = pts_out - pts_in;
                let source_out_pts = source_in_pts + (duration_pts as f32 * speed).round() as i64;
                let path = {
                    let sources = self.project.sources.read().unwrap();
                    sources.path(source_id).map(|p| p.as_ref().clone())
                };

                if let Some(path) = path {
                    // Check if we need to switch to a new audio file or if audio properties changed.
                    let need_new_decoder = self.audio_path.as_ref() != Some(&path)
                        || self.audio_volume != volume
                        || self.audio_pan != pan
                        || self.audio_muted != audio_muted
                        || self.audio_speed != speed
                        || self.audio_pitch != pitch;
                    if need_new_decoder {
                        // Stop current decoder if running
                        self.audio_shutdown
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        // Give it a brief moment to exit
                        std::thread::sleep(std::time::Duration::from_millis(5));

                        self.audio_ring.clear();
                        self.audio_shutdown =
                            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

                        let clock = std::sync::Arc::clone(&self.audio_clock);
                        let ring = std::sync::Arc::clone(&self.audio_ring);
                        let seek = std::sync::Arc::clone(&self.audio_seek);
                        let shutdown = std::sync::Arc::clone(&self.audio_shutdown);
                        let project_tb = nexir::timeline::rational::Rational {
                            num: 1,
                            den: 90_000,
                        };

                        let path_clone = path.clone();
                        eprintln!(
                            "[app] Starting audio decoder for {:?} (volume: {}, pan: {}, muted: {}, speed: {}, pitch: {})",
                            path_clone, volume, pan, audio_muted, speed, pitch
                        );
                        std::thread::spawn(move || {
                            match AudioDecoder::new(
                                &path_clone,
                                ring,
                                clock,
                                project_tb,
                                shutdown,
                                seek,
                                volume,
                                pan,
                                audio_muted,
                                speed,
                                pitch,
                                source_in_pts,
                                source_out_pts,
                            ) {
                                Ok(decoder) => decoder.run(),
                                Err(e) => eprintln!(
                                    "Failed to open audio decoder for {:?}: {:?}",
                                    path_clone, e
                                ),
                            }
                        });

                        self.audio_path = Some(path);
                        self.audio_volume = volume;
                        self.audio_pan = pan;
                        self.audio_muted = audio_muted;
                        self.audio_speed = speed;
                        self.audio_pitch = pitch;
                        force_seek = true; // force a seek when opening a new file
                    }

                    if force_seek {
                        *self.audio_seek.lock().unwrap() = Some((source_pts, playhead_pts));
                        self.audio_clock.seek(playhead_pts);
                    }
                }
            }

            // ── AUDIO MUTE: kill the decoder immediately when the audio track is
            //    muted (or unsolo'd), so the ring drains to silence.
            if top_audio_clip.is_none() && self.audio_path.is_some() {
                self.audio_shutdown
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                self.audio_path = None;
                self.audio_ring.clear();
                self.audio_shutdown =
                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            }
        }

        // Stop audio decoder when user pauses / stops playback
        if !self.timeline.playing && self.audio_path.is_some() {
            self.audio_shutdown
                .store(true, std::sync::atomic::Ordering::Relaxed);
            self.audio_path = None;
            self.audio_ring.clear();
            // Reset clock so next play starts fresh
            self.audio_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        }

        // ── Export progress polling ──────────────────────────────────────────
        if let Some(ref progress_rx) = self.export_progress {
            if let Some(update) = progress_rx.try_recv() {
                let pct = if update.total_frames > 0 {
                    update.frames_done as f64 / update.total_frames as f64 * 100.0
                } else {
                    0.0
                };
                self.export_progress_pct = (pct / 100.0) as f32;
                self.export_status = Some(format!(
                    "Export: {:.0}% ({} / {} frames, {:.1} fps)",
                    pct, update.frames_done, update.total_frames, update.fps
                ));
                if update.phase == nexir::export::progress::ExportPhase::Done {
                    self.export_progress = None;
                    self.export_progress_pct = 1.0;
                    self.export_status = Some("Export complete!".to_string());
                } else if let nexir::export::progress::ExportPhase::Failed(ref err) = update.phase {
                    self.export_progress = None;
                    self.export_status = Some(format!("Export failed: {}", err));
                }
            }
        }

        // ── Export progress UI ───────────────────────────────────────────────
        if let Some(ref status) = self.export_status.clone() {
            let is_active = self.export_progress.is_some();
            egui::Window::new("Export")
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .collapsible(false)
                .resizable(false)
                .show(&self.egui_ctx, |ui| {
                    ui.label(status);
                    if is_active {
                        ui.add(egui::ProgressBar::new(self.export_progress_pct).show_percentage());
                    } else {
                        if ui.button("Close").clicked() {
                            self.export_status = None;
                        }
                    }
                });
        }
        // ── Export settings panel ────────────────────────────────────────────
        if self.export_settings_open {
            if let Some(ref path) = self.export_pending_path.clone() {
                let do_export = crate::layout::export_settings::draw(
                    &self.egui_ctx,
                    &mut self.export_settings_open,
                    &mut self.export_settings,
                    &self.interop_capability,
                    path,
                );
                if do_export {
                    self.start_export();
                }
            }
        }

        viewport_size
    }

    fn advance_playhead(&mut self, just_started_playing: bool) {
        if !self.timeline.playing {
            return;
        }

        let max_frame = self.project.duration_frames();
        if max_frame <= 0 {
            self.timeline.playhead_frame = 0;
            self.timeline.last_tick = Some(std::time::Instant::now());
            return;
        }

        let now = std::time::Instant::now();

        if self.audio_path.is_some() && !just_started_playing {
            let clock_pts = self.audio_clock.pts();
            self.timeline.playhead_frame = self.project.pts_to_frame(clock_pts);
        } else if let Some(last) = self.timeline.last_tick {
            let fps = self.project.settings.frame_rate.num.max(1);
            let elapsed_secs = now.duration_since(last).as_secs_f64();
            let frames_to_advance = (elapsed_secs * fps as f64) as i64;
            if frames_to_advance > 0 {
                self.timeline.playhead_frame += frames_to_advance;
                self.timeline.last_tick = Some(now);
            }
        } else {
            self.timeline.last_tick = Some(now);
        }

        if self.timeline.playhead_frame >= max_frame {
            self.timeline.playhead_frame = 0;
            self.timeline.last_tick = Some(now);
            self.audio_clock.seek(0);
        }

        self.egui_ctx.request_repaint();
    }

    // ─────────────────────────────────────────────
    // Project file management
    // ─────────────────────────────────────────────

    /// Stop any running audio decoder and reset audio state.
    fn stop_audio(&mut self) {
        self.audio_shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.audio_path = None;
        self.audio_ring.clear();
        self.audio_shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.audio_was_playing = false;
        self.timeline.playing = false;
    }

    /// Reset the project to a blank slate, clearing history.
    fn new_project(&mut self) {
        self.stop_audio();
        self.project = Project::new("Untitled Project");
        let _ = self.project.add_video_track("Video 1");
        let _ = self.project.add_video_track("Video 2");
        let _ = self.project.add_audio_track("Audio 1");
        // Sync IoLayer: copy new (empty) registry into IoLayer's shared Arc,
        // then point the project to use that same Arc so future registrations are visible.
        self.io_layer.reset_for_new_project(&self.project.sources);
        self.project.sources = self.io_layer.source_reg.clone();
        // Clear still-image cache because source registry changed.
        self.still_cache.lock().unwrap().clear();
        self.current_project_path = None;
        self.history = crate::history::HistoryState::default();
        self.timeline.clear_interaction();
        self.timeline.selected_clip = None;
        self.media_pool = crate::layout::media_pool::MediaPoolState::default();
    }

    /// Show a file-open dialog and load a .nexp project.
    fn open_project(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter("Nexir Project", &["nexp"])
            .set_title("Open Project")
            .pick_file();

        if let Some(path) = file {
            match ProjectFile::load(&path) {
                Ok(mut project) => {
                    self.stop_audio();
                    // Sync IoLayer: copy loaded registry into IoLayer's shared Arc,
                    // then point the project to use that same Arc.
                    self.io_layer.reset_for_new_project(&project.sources);
                    project.sources = self.io_layer.source_reg.clone();
                    // Clear still-image cache because source registry changed.
                    self.still_cache.lock().unwrap().clear();
                    self.project = project;
                    self.current_project_path = Some(path);
                    self.history = crate::history::HistoryState::default();
                    self.timeline.clear_interaction();
                    self.timeline.selected_clip = None;
                    self.media_pool = crate::layout::media_pool::MediaPoolState::default();
                }
                Err(e) => {
                    log::error!("Failed to load project: {:?}", e);
                }
            }
        }
    }

    /// Save to the current path, or prompt if none set.
    fn save_project(&mut self) {
        if let Some(ref path) = self.current_project_path {
            if let Err(e) = ProjectFile::save(path, &self.project) {
                log::error!("Failed to save project: {:?}", e);
            }
        } else {
            self.save_project_as();
        }
    }

    /// Always show a file-save dialog, then save.
    fn save_project_as(&mut self) {
        let file = rfd::FileDialog::new()
            .add_filter("Nexir Project", &["nexp"])
            .set_title("Save Project As")
            .save_file();

        if let Some(path) = file {
            // Ensure the path ends with .nexp
            let path = if path.extension().map_or(true, |ext| ext != "nexp") {
                path.with_extension("nexp")
            } else {
                path
            };
            if let Err(e) = ProjectFile::save(&path, &self.project) {
                log::error!("Failed to save project: {:?}", e);
            } else {
                self.current_project_path = Some(path);
            }
        }
    }

    // ─────────────────────────────────────────────
    // Export
    // ─────────────────────────────────────────────

    /// Compile a render graph from the current scheduled frame.
    /// Returns (CompiledGraph, nodes, rtt_id) for use with ExportEngine.
    fn compile_export_graph(
        &self,
        device: &GpuDevice,
        frame: &nexir::render::frame_state::FrameState,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Result<
        (
            nexir::render::graph::CompiledGraph,
            Vec<Box<dyn nexir::render::graph::RenderNode>>,
            ResourceId,
        ),
        nexir::render::graph::GraphError,
    > {
        let mut compiler = RenderGraphCompiler::new();
        let mut id_counter = 2; // 0=FINAL_COLOR, 1=SCREEN

        let mut comp_node = CompositeNode::new(
            device,
            &self.shaders,
            ResourceId::FINAL_COLOR,
            8, // max clips
            wgpu::TextureFormat::Rgba16Float,
        );

        for clip in &frame.clips {
            // If this clip's source is a still image, use the StillImageUploadNode
            let is_still = self
                .project
                .sources
                .read()
                .unwrap()
                .path(clip.source_id)
                .map(|p| nexir::timeline::source::is_still_image_path(p.as_ref()))
                .unwrap_or(false);

            if is_still {
                if let Some(path) = self.project.sources.read().unwrap().path(clip.source_id) {
                    log::info!("compile_export_graph: clip source_id={:?} path={:?} detected as still", clip.source_id, path);
                    if let Some(cached) = self.still_cache.lock().unwrap().get_or_load(device, path.as_ref()) {
                        let rgba_id = ResourceId::next(&mut id_counter);
                        compiler.add_node(Box::new(StillImageUploadNode::new(cached, rgba_id)));
                        comp_node.input_textures.push(rgba_id);
                        continue;
                    } else {
                        log::warn!("Still image load failed for {:?}", path);
                    }
                } else {
                    log::warn!("compile_export_graph: clip source_id={:?} has no registered path", clip.source_id);
                }
                // Fallthrough to YUV path if still image failed to load.
            }

            let tier = (clip.texture_slot >> 16) as u8;
            let index = (clip.texture_slot & 0xFFFF) as u16;
            let slot_id = nexir::io::slot_pool::FrameSlotId { tier, index };

            let y_id = ResourceId::next(&mut id_counter);
            let uv_id = ResourceId::next(&mut id_counter);

            let mut upload_node =
                YuvUploadNode::new(device, 0, clip.clip_width, clip.clip_height, y_id, uv_id);

            // Upload YUV data from the slot pool into staging buffers
            self.io_layer.pool.with_buffer_read(slot_id, |data| {
                upload_node.upload_frame(data, clip.is_nv12, clip.clip_width, clip.clip_height);
            });

            compiler.add_node(Box::new(upload_node));

            // Add YuvToRgb node
            let rgba_id = ResourceId::next(&mut id_counter);
            compiler.add_node(Box::new(
                nexir::render::nodes::yuv_to_rgb::YuvToRgbNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    y_id,
                    uv_id,
                    rgba_id,
                    clip.clip_width,
                    clip.clip_height,
                    nexir::timeline::source::ColorSpace::Bt709,
                    true, // limited range
                ),
            ));

            // Provide RGBA texture to compositor
            comp_node.input_textures.push(rgba_id);
        }

        compiler.add_node(Box::new(comp_node));

        let graph = compiler.compile(canvas_width, canvas_height)?;
        // NOTE: nodes are stored unused in ExportRenderer (dead_code), so pass empty vec
        Ok((graph, Vec::new(), ResourceId::FINAL_COLOR))
    }

    fn open_export_settings(&mut self) {
        self.stop_audio();

        let file = rfd::FileDialog::new()
            .add_filter("Video Files", &["mp4", "mkv", "mov"])
            .set_title("Export Video")
            .save_file();

        if let Some(path) = file {
            self.export_pending_path = Some(path);
            self.export_settings_open = true;
        }
    }

    /// Open a save dialog and start an export job.
    fn start_export(&mut self) {
        use nexir::export::engine::ExportEngine;
        use nexir::export::job::ExportJob;

        let path = match self.export_pending_path.take() {
            Some(p) => p,
            None => return,
        };

        // Ensure the path extension matches the container format from export_settings.
        let ext = self.export_settings.container.format_name();
        let path = if path.extension().map_or(true, |e| e != ext) {
            path.with_extension(ext)
        } else {
            path
        };

        // Build export job from current project state
        let project_tb = Rational {
            num: 1,
            den: 90_000,
        };
        let fps = self.project.settings.frame_rate;
        let total_duration = self.project.frame_to_pts(self.project.duration_frames());
        let width = self.project.settings.width;
        let height = self.project.settings.height;

        let job = ExportJob {
            output_path: path,
            container: self.export_settings.container,
            video_codec: self.export_settings.video_codec,
            audio_codec: self.export_settings.audio_codec,
            quality: self.export_settings.video_quality(),
            audio_bitrate: 192_000,
            pts_in: 0,
            pts_out: total_duration,
            width,
            height,
            frame_rate: fps,
            project_tb,
            render_threads: num_cpus::get().max(2) / 2,
            cpu_preset: self.export_settings.cpu_preset,
        };

        // NOTE (Encode Interop): Prepare CudaContext if hardware interop is available and not forced to CPU.
        // This enables the zero-copy GPU path (wgpu -> CUDA -> NVENC), drastically improving export speed
        // by avoiding CPU memory readbacks and providing hardware HEVC/H.264 encode capabilities.
        let cuda_ctx = if self.interop_capability.is_available() && !self.export_settings.force_cpu
        {
            if self.cuda_ctx.is_none() {
                self.cuda_ctx = CudaContext::new(&self.interop_capability)
                    .ok()
                    .map(Arc::new);
            }
            self.cuda_ctx.clone()
        } else {
            None
        };

        // Build a completely fresh IoLayer for export — isolated from live playback.
        //
        // We intentionally drop the prefetch channel receiver immediately so that
        // prime_export_prefetch's try_send calls do nothing. Export is sequential
        // (frame 0, 1, 2...) and decode_blocking already tracks last_decoded_pts,
        // so each call just reads the next packet without seeking — no prefetch
        // benefit. A live prefetch worker causes a deadlock: it fills all 32 pool
        // slots, evicts one, then tries to lock that slot's buffer mutex while the
        // render thread holds it in upload_frame_data.
        let export_pool = Arc::new(FrameSlotPool::new(&self.device));
        let export_cache = Arc::new(FrameCache::new(export_pool.clone(), 16));
        let (export_prefetch_tx, _export_prefetch_rx) =
            std::sync::mpsc::sync_channel::<PrefetchRequest>(1);
        let export_io = Arc::new(IoLayer::new(
            self.device.device.clone(),
            export_pool,
            export_cache,
            self.project.sources.clone(),
            export_prefetch_tx,
            project_tb,
        ));
        let export_scheduler = Arc::new(FrameScheduler::new(export_io, width, height));
        let engine = ExportEngine::new(
            Arc::clone(&self.device),
            job,
            export_scheduler,
            Arc::new(std::sync::RwLock::new(self.project.clips.clone())),
            Arc::new(std::sync::RwLock::new(self.project.tracks.clone())),
            Arc::new(std::sync::RwLock::new(
                self.project.sources.read().unwrap().clone(),
            )),
            self.interop_capability.clone(),
            cuda_ctx,
            self.export_settings.force_cpu,
        );

        match engine.start(Arc::clone(&self.shaders), Arc::clone(&self.compute_cache)) {
            Ok(rx) => {
                self.export_progress = Some(rx);
                self.export_status = Some("Export started...".to_string());
            }
            Err(e) => {
                self.export_status = Some(format!("Export failed: {:?}", e));
            }
        }
    }

    pub fn render(
        &mut self,
        device: &GpuDevice,
        surface: &wgpu::Surface,
        window: &Window,
        viewport_size: egui::Vec2,
    ) {
        let width = viewport_size.x as u32;
        let height = viewport_size.y as u32;

        if width > 0 && height > 0 {
            if self.preview.width != width || self.preview.height != height {
                let texture = device.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Preview Texture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let tex_id = self.egui_renderer.register_native_texture(
                    device.device.as_ref(),
                    &view,
                    wgpu::FilterMode::Linear,
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
        let clipped_primitives = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);

        let surface_texture = match surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => {
                log::error!("Dropped frame: {:?}", e);
                return;
            }
        };
        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.begin_frame();

        // 1. Get playhead and schedule frame
        let playhead_pts = self.project.frame_to_pts(self.timeline.playhead_frame);
        let frame = self.frame_scheduler.schedule_frame(
            playhead_pts,
            &self.project.clips,
            &self.project.tracks,
            &self.project.sources.read().unwrap(),
        );

        if !frame.clips.is_empty() {
            if let Some(ref preview_texture) = self.preview.texture {
                if let Ok((graph, _nodes, rtt_id)) = self.compile_export_graph(
                    device,
                    &frame,
                    self.preview.width,
                    self.preview.height,
                ) {
                    graph.execute_with_callback(&mut encoder, device, &frame, |enc, ctx| {
                        let final_res = ctx.get(rtt_id);

                        let preview_view =
                            preview_texture.create_view(&wgpu::TextureViewDescriptor::default());

                        let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("blit_bg"),
                            layout: &self.blit_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(final_res.view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        });

                        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("blit_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &preview_view,
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
                        pass.set_pipeline(&self.blit_pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        pass.draw(0..3, 0..1);
                    });
                }
            }
        } else if let Some(ref preview_texture) = self.preview.texture {
            // Clear preview when no frame
            let preview_view = preview_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview_clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &preview_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        // egui textures
        for (id, image_delta) in &output.textures_delta.set {
            self.egui_renderer.update_texture(
                device.device.as_ref(),
                device.queue.as_ref(),
                *id,
                image_delta,
            );
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [
                surface_texture.texture.width(),
                surface_texture.texture.height(),
            ],
            pixels_per_point: window.scale_factor() as f32,
        };
        self.egui_renderer.update_buffers(
            device.device.as_ref(),
            device.queue.as_ref(),
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut render_pass, &clipped_primitives, &screen_descriptor);
        }

        device.submit(encoder);
        surface_texture.present();

        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}
