use crate::autosave::{AutosaveState, RecoveryInfo};
use crate::history::HistoryState;
use crate::layout::inspector::InspectorState;
use crate::layout::media_pool::MediaPoolState;
use crate::layout::recovery_dialog::{self, RecoveryAction};
use crate::layout::relink_dialog::{self, RelinkDialogState};
use crate::layout::timeline::TimelineState;
use crate::self_update::UpdaterState;
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
    /// CPU-side copy of the last rendered preview frame (width, height, RGBA8 bytes).
    /// Updated each frame for use by the eyedropper magnifier.
    pub preview_pixels: Option<(u32, u32, Vec<u8>)>,
}

pub struct ActiveAudioDecoder {
    shutdown: Arc<AtomicBool>,
    seek: Arc<Mutex<Option<(i64, i64)>>>,
    ring: Arc<AudioRingBuffer>,
    /// cached params — used to detect if we need to restart the decoder
    volume: f32,
    pan: f32,
    muted: bool,
    fade_in_pts: i64,
    fade_out_pts: i64,
    speed: f32,
    source_in_pts: i64,
    source_out_pts: i64,
}

#[derive(Clone, Debug)]
pub struct ActiveClipAudioInfo {
    pub clip_id: nexir::timeline::ids::ClipId,
    pub path: PathBuf,
    pub volume: f32,
    pub pan: f32,
    pub muted: bool,
    pub fade_in_pts: i64,
    pub fade_out_pts: i64,
    pub speed: f32,
    pub pitch: f32,
    pub source_pts: i64,
    /// Timeline PTS this clip's playback was armed at.
    ///
    /// Written when the mixer is handed a clip (see `app.rs` where
    /// `ActiveClipAudioInfo` is built) and carried for parity with
    /// `audio_mixer::ClipAudio::timeline_pts`, which the decoder's seek pair
    /// consumes.  Nothing in the UI reads it back, hence the allow.
    #[allow(dead_code)]
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
    audio_clock: Arc<MasterClock>,
    audio_mixer_bufs: nexir::audio::output_stream::MixerBusList,
    /// Map from ClipId -> (shutdown_flag, seek_channel, ring_buffer, cached params)
    active_audio_decoders: std::collections::HashMap<nexir::timeline::ids::ClipId, ActiveAudioDecoder>,
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
    text_cache: Mutex<nexir::render::text_cache::TextCache>,
    composite_pipelines: Arc<nexir::render::nodes::composite::CompositePipelines>,

    // Autosave & Crash Recovery
    pub autosave: AutosaveState,
    pub pending_recovery: Option<RecoveryInfo>,
    pub relink_dialog: RelinkDialogState,
    updater: UpdaterState,
}

pub struct AppResponse {
    pub consumed: bool,
}

/// What [`NexirApp::compile_export_graph`] hands back: the compiled graph, the
/// node objects it borrows (which must outlive it), and the id of the texture
/// holding the final colour output.
type CompiledExportGraph = (
    nexir::render::graph::CompiledGraph,
    Vec<Box<dyn nexir::render::graph::RenderNode>>,
    ResourceId,
);

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

        // G2c — NVDEC decodes straight into per-source textures the render graph
        // binds, which is what removes the ~10 ms `GPU transfer` row from a 4K
        // frame. Every source this cannot serve (no CUDA, no NVDEC, >8-bit) falls
        // back to the slot-pool upload path automatically.
        let interop_targets = Arc::new(nexir::io::interop_decode::InteropDecodeTargets::new(
            Arc::clone(&device),
            nexir::interop::capability::InteropCapability::probe(&device),
        ));

        let io_layer = Arc::new(IoLayer::new(
            device.device.clone(),
            pool,
            cache.clone(),
            project.sources.clone(),
            prefetch_tx,
            project_tb,
            Arc::clone(&interop_targets),
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
        let audio_mixer_bufs: nexir::audio::output_stream::MixerBusList = Arc::new(Mutex::new(Vec::new()));
        let audio_clock = MasterClock::new(project_tb, 48_000);

        log::info!("NexirApp::new: CPAL audio stream opening...");
        let audio_out = AudioOutputStream::open(Arc::clone(&audio_mixer_bufs), Arc::clone(&audio_clock))
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

        let composite_pipelines = Arc::new(nexir::render::nodes::composite::CompositePipelines::new(
            &device,
            &shaders,
            8,
            wgpu::TextureFormat::Rgba16Float,
        ));

        let pending_recovery = AutosaveState::check_for_recovery(None);
        let autosave = AutosaveState::new();
        let relink_dialog = RelinkDialogState::default();

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
                preview_pixels: None,
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
            audio_clock,
            audio_mixer_bufs,
            active_audio_decoders: std::collections::HashMap::new(),
            current_project_path: None,
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
            text_cache: Mutex::new(nexir::render::text_cache::TextCache::default()),
            composite_pipelines,
            autosave,
            pending_recovery,
            relink_dialog,
            updater: UpdaterState::new(),
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> AppResponse {
        let response = self.egui_state.on_window_event(window, event);
        AppResponse {
            consumed: response.consumed,
        }
    }

    pub fn update(&mut self, window: &Window) -> egui::Vec2 {
        self.updater.poll();
        if self.updater.ready_to_restart() {
            let token = self.history.change_token();
            let project_path = self.current_project_path.as_deref();
            self.autosave.flush(&self.project, token, project_path);
            match self.updater.restart() {
                Ok(()) => self
                    .egui_ctx
                    .send_viewport_cmd(egui::ViewportCommand::Close),
                Err(message) => self.updater.set_restart_error(message),
            }
        }
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
        // Undo and redo clear the same interaction state, so one branch covers
        // both; `||` short-circuits, which is what keeps redo from running when
        // undo already consumed the keypress.
        if (undo_pressed && self.history.undo(&mut self.project))
            || (redo_pressed && self.history.redo(&mut self.project))
        {
            self.timeline.clear_interaction();
            self.timeline.selected_clip = None;
        }

        let prev_settings = self.project.settings.clone();

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
                    self.updater.view(),
                );
            });
            action
        };
        
        if prev_settings.width != self.project.settings.width || prev_settings.height != self.project.settings.height {
            self.frame_scheduler = FrameScheduler::new(
                self.io_layer.clone(),
                self.project.settings.width,
                self.project.settings.height,
            );
        }
        // P2.7 — a settings change is an undoable edit.
        //
        // The top bar's combo boxes mutate `project.settings` in place during
        // `draw`, so there is no point at which a caller can `record()` first;
        // the change is detected afterwards by comparing against the value from
        // before the panel ran.  Without this the change was never recorded, and
        // the user's next Ctrl+Z reverted an EARLIER clip edit while leaving the
        // new frame rate in place — which reads as undo corrupting the timeline.
        if prev_settings.width != self.project.settings.width
            || prev_settings.height != self.project.settings.height
            || prev_settings.frame_rate != self.project.settings.frame_rate
            || prev_settings.timebase != self.project.settings.timebase
        {
            self.history
                .record_settings_change(&self.project, prev_settings.clone());
        }
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
                crate::layout::top_bar::TopBarAction::InstallUpdate => {
                    let token = self.history.change_token();
                    let project_path = self.current_project_path.as_deref();
                    self.autosave.flush(&self.project, token, project_path);
                    self.updater.install();
                }
                crate::layout::top_bar::TopBarAction::RetryUpdate => {
                    self.updater.retry_check();
                }
                crate::layout::top_bar::TopBarAction::DismissUpdateError => {
                    self.updater.dismiss_error();
                }
                crate::layout::top_bar::TopBarAction::RestartUpdatedApp => {
                    match self.updater.restart() {
                        Ok(()) => self
                            .egui_ctx
                            .send_viewport_cmd(egui::ViewportCommand::Close),
                        Err(message) => self.updater.set_restart_error(message),
                    }
                }
            }
        }

        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .default_height(300.0)
            .height_range(100.0..=400.0)
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
            .default_width(320.0)
            .width_range(180.0..=800.0)
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
                        AudioStreamInfo, PixelFormat, SampleFormat, VideoStreamInfo,
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
                            pixel_fmt: if s.color_info.bit_depth >= 10 {
                                PixelFormat::P010
                            } else {
                                PixelFormat::Yuv420p
                            },
                            color_info: s.color_info,
                            // convert from stream timebase to project timebase
                            duration_pts: if is_still_image {
                                0
                            } else {
                                project_tb.from_pts(s.duration, s.time_base)
                            },
                            is_vfr: !is_still_image && s.is_vfr,
                            time_base: s.time_base,
                            rotation: nexir::timeline::source::VideoRotation::None,
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

        let playhead_pts = self.project.frame_to_pts(self.timeline.playhead_frame);

        egui::SidePanel::right("inspector")
            .resizable(true)
            .default_width(320.0)
            .width_range(180.0..=800.0)
            .show(&self.egui_ctx, |ui| {
                crate::layout::inspector::draw(
                    ui,
                    &mut self.inspector,
                    &mut self.project,
                    self.timeline.selected_clip,
                    &mut self.history,
                    playhead_pts,
                );
            });

        // Query active clips to pass to viewport for interaction (hit testing, drawing handles)
        let mut active_clips = Vec::new();
        query_active(&self.project.clips, playhead_pts, &mut active_clips);

        let mut vp_result = crate::layout::viewport::ViewportDrawResult {
            size: egui::Vec2::ZERO,
            eyedropper_pick: None,
            eyedropper_hover_uv: None,
        };
        egui::CentralPanel::default().show(&self.egui_ctx, |ui| {
            vp_result = crate::layout::viewport::draw(
                ui,
                self.preview.texture_id,
                self.preview.video_width,
                self.preview.video_height,
                &mut self.timeline,
                &mut self.project,
                &active_clips,
                &mut self.history,
                self.inspector.eyedropper_active,
                self.preview.preview_pixels.as_ref(),
            );
        });
        let viewport_size = vp_result.size;

        // While eyedropper is hovering, request continuous repaints for the live magnifier
        if self.inspector.eyedropper_active && vp_result.eyedropper_hover_uv.is_some() {
            self.egui_ctx.request_repaint();
        }

        // Eyedropper: if a pick was requested and we have a preview texture, do a GPU readback.
        if let Some(uv) = vp_result.eyedropper_pick
            && let Some(ref tex) = self.preview.texture {
                let w = tex.width();
                let h = tex.height();
                let px = (uv.x * w as f32) as u32;
                let py = (uv.y * h as f32) as u32;
                let px = px.min(w.saturating_sub(1));
                let py = py.min(h.saturating_sub(1));

                // Each texel is 4 bytes (RGBA8). We read a 4-byte block.
                let bytes_per_row = (w * 4 + 255) & !255u32; // align to 256
                let buf_size = (bytes_per_row * h) as u64;

                let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("eyedropper_staging"),
                    size: buf_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut enc = self.device.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("eyedropper_enc") }
                );
                enc.copy_texture_to_buffer(
                    tex.as_image_copy(),
                    wgpu::ImageCopyBuffer {
                        buffer: &staging,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(bytes_per_row),
                            rows_per_image: Some(h),
                        },
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                self.device.queue.submit([enc.finish()]);
                self.device.device.poll(wgpu::Maintain::Wait);

                {
                    let buf_slice = staging.slice(..);
                    buf_slice.map_async(wgpu::MapMode::Read, |_| {});
                    self.device.device.poll(wgpu::Maintain::Wait);
                    let data = buf_slice.get_mapped_range();
                    let byte_offset = (py * bytes_per_row + px * 4) as usize;
                    if byte_offset + 3 < data.len() {
                        let r = data[byte_offset]     as f32 / 255.0;
                        let g = data[byte_offset + 1] as f32 / 255.0;
                        let b = data[byte_offset + 2] as f32 / 255.0;
                        self.inspector.chroma_key_color = [r, g, b];
                        if let Some(idx) = self.timeline.selected_clip {
                            self.history.record(&self.project);
                            let mut eff = self.project.clips.effects_at(idx);
                            eff.chroma_key_color = [r, g, b];
                            self.project.clips.set_effects_at(idx, eff);
                        }
                        log::info!("Eyedropper picked color [{:.3}, {:.3}, {:.3}] at ({}, {})", r, g, b, px, py);
                    }
                }
                staging.unmap();
                self.inspector.eyedropper_active = false;
                self.egui_ctx.request_repaint();
            }

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
            let pts_delta = playhead_pts - self.last_playhead;
            let force_seek = just_started_playing
                || !(0..=180_000).contains(&pts_delta)
                || !self.timeline.playing;
            self.last_playhead = playhead_pts;

            if !self.timeline.playing {
                // Paused: kill all decoders immediately
                self.stop_all_audio_decoders();
            } else {
                // Build the set of currently-active audio clips
                let any_soloed = self.project.tracks.any_soloed();
                let mut desired: Vec<ActiveClipAudioInfo> = Vec::new();

                for clip in &active_clips {
                    let track_id = self.project.clips.track_id_at(clip.store_index);
                    if let Some(track) = self.project.tracks.get(track_id) {
                        if !track.is_active(any_soloed) { continue; }
                        use nexir::timeline::track::TrackKind;
                        let source_id = self.project.clips.source_id_at(clip.store_index);
                        let has_audio = self.project.sources.read().unwrap().audio_info(source_id).is_ok();
                        if !has_audio { continue; }
                        match track.kind {
                            TrackKind::Video | TrackKind::Audio { .. } => {
                                let path = {
                                    let sources = self.project.sources.read().unwrap();
                                    sources.path(source_id).map(|p| p.as_ref().clone())
                                };
                                if let Some(path) = path {
                                    let volume = self.project.clips.volume_at(clip.store_index) * track.gain;
                                    let pan = (self.project.clips.pan_at(clip.store_index) + track.pan).clamp(-1.0, 1.0);
                                    let muted = self.project.clips.audio_muted_at(clip.store_index);
                                    let fade_in_pts = self.project.clips.fade_in_pts_at(clip.store_index);
                                    let fade_out_pts = self.project.clips.fade_out_pts_at(clip.store_index);
                                    let speed = self.project.clips.speed_at(clip.store_index);
                                    let pitch = self.project.clips.pitch_at(clip.store_index);
                                    let source_in_pts = self.project.clips.source_in_at(clip.store_index);
                                    let pts_out = self.project.clips.pts_out_at(clip.store_index);
                                    let pts_in = self.project.clips.pts_in_at(clip.store_index);
                                    let duration_pts = pts_out - pts_in;
                                    let source_out_pts = source_in_pts + (duration_pts as f32 * speed).round() as i64;
                                    let clip_id = self.project.clips.clip_id_at(clip.store_index);
                                    desired.push(ActiveClipAudioInfo {
                                        clip_id,
                                        path,
                                        volume,
                                        pan,
                                        muted,
                                        fade_in_pts,
                                        fade_out_pts,
                                        speed,
                                        pitch,
                                        source_pts: clip.source_pts,
                                        timeline_pts: playhead_pts,
                                        source_in_pts,
                                        source_out_pts,
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Shut down decoders no longer needed
                let desired_ids: std::collections::HashSet<_> = desired.iter().map(|d| d.clip_id).collect();
                let to_remove: Vec<_> = self.active_audio_decoders.keys()
                    .filter(|id| !desired_ids.contains(id))
                    .cloned().collect();
                for id in to_remove {
                    self.stop_audio_decoder(id);
                }

                // Start or update decoders
                let project_tb = nexir::timeline::rational::Rational { num: 1, den: 90_000 };
                for info in desired {
                    let needs_restart = if let Some(existing) = self.active_audio_decoders.get(&info.clip_id) {
                        force_seek
                            || existing.volume != info.volume
                            || existing.pan != info.pan
                            || existing.muted != info.muted
                            || existing.fade_in_pts != info.fade_in_pts
                            || existing.fade_out_pts != info.fade_out_pts
                            || existing.speed != info.speed
                            || existing.source_in_pts != info.source_in_pts
                            || existing.source_out_pts != info.source_out_pts
                    } else {
                        true // new clip
                    };

                    if needs_restart {
                        // Kill any existing decoder for this clip
                        if self.active_audio_decoders.contains_key(&info.clip_id) {
                            self.stop_audio_decoder(info.clip_id);
                        }

                        // Spawn a fresh decoder with its own ring buffer
                        let ring = AudioRingBuffer::new(32768); // ~680ms at 48kHz stereo
                        let shutdown = Arc::new(AtomicBool::new(false));
                        let seek = Arc::new(Mutex::new(Some((info.source_pts, playhead_pts))));
                        let mixer_bufs = Arc::clone(&self.audio_mixer_bufs);

                        // Register ring in the mixer list
                        {
                            let mut bufs = mixer_bufs.lock().unwrap();
                            bufs.push(Arc::clone(&ring));
                        }

                        let ring_dec = Arc::clone(&ring);
                        let shutdown_dec = Arc::clone(&shutdown);
                        let seek_dec = Arc::clone(&seek);
                        let info_clone = info.clone();

                        std::thread::spawn(move || {
                            match AudioDecoder::new(
                                &info_clone.path,
                                ring_dec,
                                project_tb,
                                shutdown_dec,
                                seek_dec,
                                info_clone.volume,
                                info_clone.pan,
                                info_clone.muted,
                                info_clone.fade_in_pts,
                                info_clone.fade_out_pts,
                                info_clone.speed,
                                info_clone.pitch,
                                info_clone.source_in_pts,
                                info_clone.source_out_pts,
                            ) {
                                Ok(decoder) => decoder.run(),
                                Err(e) => eprintln!("[audio] Failed to open decoder for {:?}: {:?}", info_clone.path, e),
                            }
                        });

                        self.active_audio_decoders.insert(info.clip_id, ActiveAudioDecoder {
                            shutdown,
                            seek,
                            ring,
                            volume: info.volume,
                            pan: info.pan,
                            muted: info.muted,
                            fade_in_pts: info.fade_in_pts,
                            fade_out_pts: info.fade_out_pts,
                            speed: info.speed,
                            source_in_pts: info.source_in_pts,
                            source_out_pts: info.source_out_pts,
                        });
                    } else if force_seek {
                        // Seek existing decoder
                        if let Some(dec) = self.active_audio_decoders.get(&info.clip_id) {
                            *dec.seek.lock().unwrap() = Some((info.source_pts, playhead_pts));
                            dec.ring.clear();
                        }
                    }
                }

                if force_seek {
                    self.audio_clock.seek(playhead_pts);
                }
            }
        }

        // When paused — also kill decoders
        if !self.timeline.playing && !self.active_audio_decoders.is_empty() {
            self.stop_all_audio_decoders();
        }

        // ── Export progress polling ──────────────────────────────────────────
        if let Some(ref progress_rx) = self.export_progress
            && let Some(update) = progress_rx.try_recv() {
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
                    } else if ui.button("Close").clicked() {
                        self.export_status = None;
                    }
                });
        }
        // ── Export settings panel ────────────────────────────────────────────
        if self.export_settings_open
            && let Some(ref path) = self.export_pending_path.clone() {
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

        // ── Crash recovery modal ─────────────────────────────────────────────
        // Shown on the first frame after startup (or project open) when a stale
        // autosave was detected.  Blocks all other interaction via modal window.
        if let Some(info) = self.pending_recovery.clone()
            && let Some(action) = recovery_dialog::draw(&self.egui_ctx, &info) {
                match action {
                    RecoveryAction::Restore => {
                        match ProjectFile::load(&info.autosave_path) {
                            Ok(mut project) => {
                                self.stop_playback();
                                self.io_layer.reset_for_new_project(&project.sources);
                                project.sources = self.io_layer.source_reg.clone();
                                self.still_cache.lock().unwrap().clear();
                                self.project = project;
                                self.history = crate::history::HistoryState::default();
                                self.timeline.clear_interaction();
                                self.timeline.selected_clip = None;
                                self.media_pool = crate::layout::media_pool::MediaPoolState::default();
                                // Remove autosave now that it's been restored.
                                ProjectFile::delete_autosave(&info.autosave_path);
                                log::info!("[recovery] project restored from autosave");
                                // Check for missing media in the recovered project.
                                self.relink_dialog.refresh(&self.project);
                            }
                            Err(e) => {
                                log::error!("[recovery] failed to load autosave: {:?}", e);
                            }
                        }
                    }
                    RecoveryAction::Discard => {
                        ProjectFile::delete_autosave(&info.autosave_path);
                        log::info!("[recovery] autosave discarded by user");
                    }
                }
                self.pending_recovery = None;
            }

        // ── Missing media / relink dialog ────────────────────────────────────
        relink_dialog::draw(
            &self.egui_ctx,
            &mut self.relink_dialog,
            &mut self.project,
            &self.still_cache,
        );

        // ── Autosave tick ────────────────────────────────────────────────────
        // Don't tick autosave while the recovery dialog is open — we only want
        // to start accumulating saves once the user is actually editing.
        if self.pending_recovery.is_none() {
            let token = self.history.change_token();
            let proj_path = self.current_project_path.as_deref();
            self.autosave.tick(&self.project, token, proj_path);
        }

        viewport_size
    }

    fn stop_audio_decoder(&mut self, clip_id: nexir::timeline::ids::ClipId) {
        if let Some(dec) = self.active_audio_decoders.remove(&clip_id) {
            dec.shutdown.store(true, std::sync::atomic::Ordering::Relaxed);
            // Remove its ring buffer from the CPAL mixer list
            let ring_ptr = Arc::as_ptr(&dec.ring);
            let mut bufs = self.audio_mixer_bufs.lock().unwrap();
            bufs.retain(|b| Arc::as_ptr(b) != ring_ptr);
        }
    }

    fn stop_all_audio_decoders(&mut self) {
        let ids: Vec<_> = self.active_audio_decoders.keys().cloned().collect();
        for id in ids {
            self.stop_audio_decoder(id);
        }
        self.audio_clock.seek(self.project.frame_to_pts(self.timeline.playhead_frame));
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

        if !self.active_audio_decoders.is_empty() && !just_started_playing {
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
    fn stop_playback(&mut self) {
        self.stop_all_audio_decoders();
        self.audio_was_playing = false;
        self.timeline.playing = false;
    }

    /// Reset the project to a blank slate, clearing history.
    fn new_project(&mut self) {
        self.stop_playback();
        // Flush any unsaved changes to the autosave file before wiping the project.
        let token = self.history.change_token();
        let proj_path = self.current_project_path.as_deref();
        self.autosave.flush(&self.project, token, proj_path);
        // Delete the old autosave now that it has been flushed and we are clearing the project.
        self.autosave.delete_last_autosave();
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
        self.pending_recovery = None;
        self.relink_dialog = RelinkDialogState::default();
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
                    self.stop_playback();
                    // Flush & delete the previous project's autosave before switching.
                    let old_token = self.history.change_token();
                    let old_path = self.current_project_path.as_deref();
                    self.autosave.flush(&self.project, old_token, old_path);
                    self.autosave.delete_last_autosave();
                    // Sync IoLayer: copy loaded registry into IoLayer's shared Arc,
                    // then point the project to use that same Arc.
                    self.io_layer.reset_for_new_project(&project.sources);
                    project.sources = self.io_layer.source_reg.clone();
                    // Clear still-image cache because source registry changed.
                    self.still_cache.lock().unwrap().clear();
                    self.project = project;
                    self.current_project_path = Some(path.clone());
                    self.history = crate::history::HistoryState::default();
                    self.timeline.clear_interaction();
                    self.timeline.selected_clip = None;
                    self.media_pool = crate::layout::media_pool::MediaPoolState::default();
                    self.relink_dialog = RelinkDialogState::default();

                    // Check if there is an autosave for the freshly-opened project
                    // that is *newer* than the saved .nexp file (e.g. the user had
                    // unsaved changes from a previous session).
                    let saved_mtime = std::fs::metadata(&path)
                        .ok()
                        .and_then(|m| m.modified().ok());
                    let autosave_path = ProjectFile::autosave_path_for(Some(&path));
                    if let Some(recovery) = ProjectFile::find_autosave_newer_than(&autosave_path, saved_mtime) {
                        use crate::autosave::peek_project_name_from;
                        let project_name = peek_project_name_from(&recovery)
                            .unwrap_or_else(|| "Untitled".into());
                        let modified = std::fs::metadata(&recovery).ok().and_then(|m| m.modified().ok());
                        self.pending_recovery = Some(RecoveryInfo {
                            autosave_path: recovery,
                            project_name,
                            modified,
                        });
                    } else {
                        self.pending_recovery = None;
                        // Delete any stale autosave for this project path (older than saved file).
                        ProjectFile::delete_autosave(&autosave_path);
                        // Check for offline/missing media in the loaded project.
                        self.relink_dialog.refresh(&self.project);
                    }
                }
                Err(e) => {
                    log::error!("Failed to load project: {:?}", e);
                }
            }
        }
    }

    /// Save to the current path, or prompt if none set.
    fn save_project(&mut self) {
        if let Some(ref path) = self.current_project_path.clone() {
            if let Err(e) = ProjectFile::save(path, &self.project) {
                log::error!("Failed to save project: {:?}", e);
            } else {
                // Clean up the autosave file — the project is now explicitly saved.
                let autosave_path = ProjectFile::autosave_path_for(Some(path));
                ProjectFile::delete_autosave(&autosave_path);
                self.autosave.delete_last_autosave();
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
            let path = if path.extension().is_none_or(|ext| ext != "nexp") {
                path.with_extension("nexp")
            } else {
                path
            };
            if let Err(e) = ProjectFile::save(&path, &self.project) {
                log::error!("Failed to save project: {:?}", e);
            } else {
                // Delete the autosave for both the old and new path.
                let old_autosave = ProjectFile::autosave_path_for(self.current_project_path.as_deref());
                ProjectFile::delete_autosave(&old_autosave);
                self.autosave.delete_last_autosave();
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
    ) -> Result<CompiledExportGraph, nexir::render::graph::GraphError> {
        let mut compiler = RenderGraphCompiler::new();
        // G2c/G2d — two ids per clip are reserved below this for interop planes,
        // whether or not a clip uses them. `FrameScheduler::interop_y_id` picks the
        // same numbers when it binds the frame's imports, and a counter that skipped
        // the CPU-path clips would shift every later clip's ids as soon as one
        // source fell back.
        let mut id_counter = FrameScheduler::interop_id_counter_start(frame.clips.len());

        let mut comp_node = CompositeNode::with_pipelines(
            device,
            Arc::clone(&self.composite_pipelines),
            ResourceId::FINAL_COLOR,
            8, // max clips
        );

        // Enumerated because an interop clip's plane ids are derived from its index
        // in `frame.clips` — the same index `FrameScheduler::interop_bindings` used
        // when it bound them (G2d).
        for (clip_index, clip) in frame.clips.iter().enumerate() {
            // Determine input source type: Text, Still Image, or Video
            let (initial_rgba_id, clip_w, clip_h) = if let nexir::timeline::store::ClipKind::Text {
                text, font_size, color, stroke_color, stroke_width, background_color, bg_padding
            } = &clip.kind {
                let cached = self.text_cache.lock().unwrap().get_or_create(
                    device, text, *font_size, *color,
                    *stroke_color, *stroke_width, *background_color, *bg_padding,
                );
                let rgba_id = ResourceId::next(&mut id_counter);
                let (w, h) = (cached.width, cached.height);
                compiler.add_node(Box::new(nexir::render::text_cache::TextUploadNode::new(cached, rgba_id)));
                (rgba_id, w, h)
            } else {
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
                        if let Some(cached) = self.still_cache.lock().unwrap().get_or_load(device, path.as_ref()) {
                            let rgba_id = ResourceId::next(&mut id_counter);
                            let (w, h) = (cached.width, cached.height);
                            compiler.add_node(Box::new(StillImageUploadNode::new(cached, rgba_id)));
                            (rgba_id, w, h)
                        } else {
                            log::warn!("Still image load failed for {:?}", path);
                            continue;
                        }
                    } else {
                        log::warn!("compile_export_graph: clip source_id={:?} has no registered path", clip.source_id);
                        continue;
                    }
                } else {
                    // P1.6 — layout and colour come from the DECODED frame in this
                    // clip's slot, not from the source registry: the decoder may
                    // have converted the frame, and its metadata is what describes
                    // the bytes actually sitting in the pool buffer.
                    let layout     = clip.frame_meta.layout;
                    let color_info = clip.frame_meta.color;

                    // G2d — an interop clip gets NO `YuvUploadNode`. Leaving one in
                    // and skipping its upload would allocate two pooled textures
                    // nobody writes and render a black clip, with no error and
                    // nothing in the pool's counters (gotcha 18); the plane ids come
                    // from `FrameScheduler` because that is what the frame bound.
                    let (y_id, uv_id) = if clip.is_interop() {
                        (
                            FrameScheduler::interop_y_id(clip_index),
                            FrameScheduler::interop_uv_id(clip_index),
                        )
                    } else {
                        (
                            ResourceId::next(&mut id_counter),
                            ResourceId::next(&mut id_counter),
                        )
                    };

                    if !clip.is_interop() {
                        let tier = (clip.texture_slot >> 16) as u8;
                        let index = (clip.texture_slot & 0xFFFF) as u16;
                        let slot_id = nexir::io::slot_pool::FrameSlotId { tier, index };

                        let upload_node = YuvUploadNode::new_with_layout(
                            device, 0, clip.clip_width, clip.clip_height, y_id, uv_id, layout,
                        );

                        // Upload YUV data from the slot pool into staging buffers
                        self.io_layer.pool.with_buffer_read(slot_id, |data| {
                            upload_node.upload_frame(
                                data,
                                layout.semi_planar,
                                clip.clip_width,
                                clip.clip_height,
                            );
                        });

                        compiler.add_node(Box::new(upload_node));
                    }

                    // Add YuvToRgb node
                    let rgba_id = ResourceId::next(&mut id_counter);

                    compiler.add_node(Box::new(
                        nexir::render::nodes::yuv_to_rgb::YuvToRgbNode::new_with_layout(
                            device,
                            &self.shaders,
                            &self.compute_cache,
                            y_id,
                            uv_id,
                            rgba_id,
                            clip.clip_width,
                            clip.clip_height,
                            color_info,
                            layout.semi_planar,
                        )
                        .with_imported_planes(clip.is_interop()),
                    ));

                    // Insert tone-mapping for HDR clips targeting the SDR viewport.
                    //
                    // The preview is always SDR — there is no HDR swapchain — so
                    // this stays a one-way HDR→SDR map, unlike the export path
                    // which now converts in whichever direction the job needs.
                    let final_rgba_id = if color_info.is_hdr() {
                        use nexir::render::nodes::tonemap::{
                            ToneMapNode, InputTransferFn, GamutConversion, ToneMapPushConstants,
                            ToneMapMode,
                        };

                        let tonemapped_id = ResourceId::next(&mut id_counter);
                        // `from_color_info` rather than a local match: a BT.2020
                        // 10-bit clip with a plain BT.2020 (non-PQ) curve is
                        // `is_hdr()` too, and treating its display-encoded samples
                        // as linear light made it come out washed out.
                        let trc = InputTransferFn::from_color_info(&color_info);
                        let gamut = GamutConversion::between(
                            color_info.effective_primaries(clip.clip_width, clip.clip_height),
                            nexir::timeline::source::ColorPrimaries::Bt709,
                        );
                        let tm_params = ToneMapPushConstants::for_sdr_preview(
                            trc,
                            gamut,
                            ToneMapMode::AcesFilmic,
                            1000.0,
                            clip.clip_width,
                            clip.clip_height,
                        );
                        compiler.add_node(Box::new(ToneMapNode::new(
                            device,
                            &self.shaders,
                            &self.compute_cache,
                            rgba_id,
                            tonemapped_id,
                            tm_params,
                        )));
                        tonemapped_id
                    } else {
                        rgba_id
                    };

                    (final_rgba_id, clip.clip_width, clip.clip_height)
                }
            };

            let mut cur_id = initial_rgba_id;

            let eff = &clip.effects;

            // 1. Color & Light Adjustment (Brightness, Contrast, Saturation, Hue)
            if eff.color_enabled
                && (eff.brightness.abs() > 0.001
                    || (eff.contrast - 1.0).abs() > 0.001
                    || (eff.saturation - 1.0).abs() > 0.001
                    || eff.hue.abs() > 0.001)
            {
                let cc_out = ResourceId::next(&mut id_counter);
                let mut params = nexir::render::nodes::color_correction::ColorCorrectionParams::identity(
                    clip_w,
                    clip_h,
                );
                params.brightness = eff.brightness;
                params.contrast = eff.contrast.max(0.0);
                params.saturation = eff.saturation.max(0.0);
                params.hue_shift = eff.hue.to_radians();

                compiler.add_node(Box::new(nexir::render::nodes::color_correction::ColorCorrectionNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    cur_id,
                    cc_out,
                    params,
                )));
                cur_id = cc_out;
            }

            // 2. Gaussian Blur
            if eff.blur_enabled && eff.blur_radius > 0.1 {
                use nexir::render::nodes::gaussian_blur::{BlurPassNode, BlurParams};
                let h_out = ResourceId::next(&mut id_counter);
                compiler.add_node(Box::new(BlurPassNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    cur_id,
                    h_out,
                    BlurParams::horizontal(eff.blur_radius, eff.blur_sigma, clip_w, clip_h),
                    "BlurH",
                )));
                let v_out = ResourceId::next(&mut id_counter);
                compiler.add_node(Box::new(BlurPassNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    h_out,
                    v_out,
                    BlurParams::vertical(eff.blur_radius, eff.blur_sigma, clip_w, clip_h),
                    "BlurV",
                )));
                cur_id = v_out;
            }

            // 3. Sharpen
            if eff.sharpen_enabled && eff.sharpen_amount > 0.001 {
                use nexir::render::nodes::sharpen::{SharpenNode, SharpenParams};
                let sharp_out = ResourceId::next(&mut id_counter);
                compiler.add_node(Box::new(SharpenNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    cur_id,
                    sharp_out,
                    SharpenParams::new(eff.sharpen_amount, clip_w, clip_h),
                )));
                cur_id = sharp_out;
            }

            // 4. Vignette
            if eff.vignette_enabled && eff.vignette_intensity > 0.001 {
                use nexir::render::nodes::vignette::{VignetteNode, VignetteParams};
                let vig_out = ResourceId::next(&mut id_counter);
                compiler.add_node(Box::new(VignetteNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    cur_id,
                    vig_out,
                    VignetteParams {
                        intensity: eff.vignette_intensity,
                        radius: eff.vignette_radius,
                        softness: eff.vignette_softness,
                        roundness: eff.vignette_roundness,
                        center_x: 0.5,
                        center_y: 0.5,
                        width: clip_w,
                        height: clip_h,
                    },
                )));
                cur_id = vig_out;
            }

            // 5. Chroma Key
            if eff.chroma_key_enabled {
                let r = eff.chroma_key_color[0];
                let g = eff.chroma_key_color[1];
                let b = eff.chroma_key_color[2];
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                let delta = max - min;
                let hue = if delta < 1e-5 {
                    0.0
                } else if (max - r).abs() < 1e-5 {
                    ((g - b) / delta).rem_euclid(6.0) * 60.0
                } else if (max - g).abs() < 1e-5 {
                    ((b - r) / delta + 2.0) * 60.0
                } else {
                    ((r - g) / delta + 4.0) * 60.0
                };
                let tol  = (eff.chroma_key_tolerance * 180.0).max(1.0);
                let soft = (eff.chroma_key_softness  * 180.0).min(tol - 0.1).max(0.01);
                let ck_params = nexir::render::nodes::chroma_key::ChromaKeyParams {
                    key_hue: hue,
                    tolerance: tol,
                    softness: soft,
                    min_saturation: eff.chroma_key_min_saturation,
                    min_value: 0.05,
                    spill_suppress: eff.chroma_key_spill_suppress,
                    width: clip_w,
                    height: clip_h,
                };
                let ck_out_id = ResourceId::next(&mut id_counter);
                compiler.add_node(Box::new(nexir::render::nodes::chroma_key::ChromaKeyNode::new(
                    device,
                    &self.shaders,
                    &self.compute_cache,
                    cur_id,
                    ck_out_id,
                    ck_params,
                )));
                cur_id = ck_out_id;
            }

            let post_process_id = cur_id;

            // Provide RGBA texture to compositor
            comp_node.input_textures.push(post_process_id);
        }

        compiler.add_node(Box::new(comp_node));

        let graph = compiler.compile(canvas_width, canvas_height)?;
        // NOTE: nodes are stored unused in ExportRenderer (dead_code), so pass empty vec
        Ok((graph, Vec::new(), ResourceId::FINAL_COLOR))
    }

    fn open_export_settings(&mut self) {
        self.stop_playback();

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
        let path = if path.extension().is_none_or(|e| e != ext) {
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

        let mut job = ExportJob {
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
            // SDR default: the export renderer tone-maps every HDR clip down to
            // Rec.709 before the encoder sees it, so that is what the output is
            // tagged as.  `set_hdr10` below overrides both halves when the user
            // asked for an HDR export.
            output_color: nexir::timeline::source::ColorInfo::bt709(),
            hdr10: None,
        };

        // P1.7 — HDR10 export.  `set_hdr10` sets the colour description AND the
        // mastering metadata together, and refuses a codec that cannot carry
        // 10-bit, so a mis-set panel can never produce a file tagged HDR over SDR
        // pixels.  The panel already hides the toggle for those codecs; this is the
        // backstop.
        if let Some(hdr) = self.export_settings.hdr10()
            && let Err(e) = job.set_hdr10(hdr)
        {
            log::error!(
                "[export] HDR10 was requested but cannot be used for this job \
                 ({e:?}) — exporting SDR instead"
            );
            self.export_status = Some(
                "HDR10 is not supported for this codec — exported as SDR".to_string(),
            );
        }

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
        // Export keeps the CPU upload path. The interop path's target holds ONE
        // frame per source, so it cannot serve `ExportDecodeWorker`'s 16-frame
        // lookahead — the whole point of which is to decode ahead of the render
        // cursor into the CPU `FrameCache`. Giving export a disabled registry states
        // that rather than leaving it to a runtime coincidence.
        let export_io = Arc::new(IoLayer::new(
            self.device.device.clone(),
            export_pool,
            export_cache,
            self.project.sources.clone(),
            export_prefetch_tx,
            project_tb,
            Arc::new(nexir::io::interop_decode::InteropDecodeTargets::disabled(
                Arc::clone(&self.device),
            )),
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

        if width > 0 && height > 0
            && (self.preview.width != width || self.preview.height != height) {
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
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
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
            if let Some(ref preview_texture) = self.preview.texture
                && let Ok((graph, _nodes, rtt_id)) = self.compile_export_graph(
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

        let submission = device.submit(encoder);
        surface_texture.present();

        // G2c — stamp every interop source this frame sampled with the submission
        // that reads its textures. The next decode for that source waits on this
        // before overwriting them; without it NVDEC writes into a texture the
        // submitted graph is still reading, which is a torn frame with no error and
        // nothing in the pool's counters. Cheap: the stamp is a clone of a
        // submission index per interop clip, and it is a no-op on the CPU path.
        self.io_layer
            .interop()
            .mark_submitted(&frame.clips, &submission);

        // Eyedropper magnifier: keep preview_pixels current after the frame has been committed.
        // We do this after device.submit() so the blit into preview_texture is guaranteed complete.
        // Only runs when eyedropper is active to avoid per-frame overhead.
        if self.inspector.eyedropper_active {
            if let Some(ref preview_texture) = self.preview.texture {
                let pw = preview_texture.width();
                let ph = preview_texture.height();
                if pw > 0 && ph > 0 {
                    let bytes_per_row = (pw * 4 + 255) & !255u32;
                    let buf_size = (bytes_per_row * ph) as u64;
                    let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("eyedropper_preview_staging"),
                        size: buf_size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    let mut copy_enc = device.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: Some("preview_pixel_copy") }
                    );
                    copy_enc.copy_texture_to_buffer(
                        preview_texture.as_image_copy(),
                        wgpu::ImageCopyBuffer {
                            buffer: &staging,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(bytes_per_row),
                                rows_per_image: Some(ph),
                            },
                        },
                        wgpu::Extent3d { width: pw, height: ph, depth_or_array_layers: 1 },
                    );
                    device.queue.submit([copy_enc.finish()]);
                    device.device.poll(wgpu::Maintain::Wait);

                    let buf_slice = staging.slice(..);
                    buf_slice.map_async(wgpu::MapMode::Read, |_| {});
                    device.device.poll(wgpu::Maintain::Wait);
                    let data = buf_slice.get_mapped_range();
                    // Strip row padding: copy only the pixel data (pw * 4 bytes per row)
                    let mut pixels = Vec::with_capacity((pw * ph * 4) as usize);
                    for row in 0..ph {
                        let row_start = (row * bytes_per_row) as usize;
                        let row_end = row_start + (pw * 4) as usize;
                        if row_end <= data.len() {
                            pixels.extend_from_slice(&data[row_start..row_end]);
                        }
                    }
                    drop(data);
                    staging.unmap();
                    self.preview.preview_pixels = Some((pw, ph, pixels));
                }
            }
        } else {
            // Clear cached pixels when eyedropper is not active
            self.preview.preview_pixels = None;
        }

        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}
