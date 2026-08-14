// src/engine/playback.rs  (Phase 5 update)

use crate::audio::output_stream::AudioOutputStream;
use crate::audio::ring_buffer::AudioRingBuffer;
use crate::render::device::GpuDevice;
use crate::render::graph::CompiledGraph;
use crate::render::nodes::blit::BlitToScreenNode;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::sync::drift_corrector::DriftCorrector;
use crate::sync::master_clock::MasterClock;
use crate::sync::presentation::{PresentAction, PresentationDecider};
#[cfg(any(test, debug_assertions))]
use crate::sync::sync_probe::SyncProbe;
use crate::timeline::rational::Rational;
use crate::timeline::source::SourceRegistry;
use crate::timeline::store::TimelineStore;
use crate::timeline::track::TrackList;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

pub struct PlaybackEngine {
    device: Arc<GpuDevice>,
    surface: wgpu::Surface<'static>,
    graph: CompiledGraph,
    scheduler: Arc<FrameScheduler>,
    timeline: Arc<std::sync::RwLock<TimelineStore>>,
    tracks: Arc<std::sync::RwLock<TrackList>>,
    sources: Arc<std::sync::RwLock<SourceRegistry>>,
    clock: Arc<MasterClock>,
    decider: PresentationDecider,
    corrector: DriftCorrector,
    #[cfg(any(test, debug_assertions))]
    probe: SyncProbe,
    _audio_stream: AudioOutputStream, // kept alive to prevent stream close
    project_tb: Rational,
    shutdown: Arc<AtomicBool>,
    seek_request: Arc<Mutex<Option<i64>>>,
    ring: Arc<AudioRingBuffer>,
}

impl PlaybackEngine {
    pub fn new(
        device: Arc<GpuDevice>,
        surface: wgpu::Surface<'static>,
        graph: CompiledGraph,
        scheduler: Arc<FrameScheduler>,
        timeline: Arc<std::sync::RwLock<TimelineStore>>,
        tracks: Arc<std::sync::RwLock<TrackList>>,
        sources: Arc<std::sync::RwLock<SourceRegistry>>,
        ring: Arc<AudioRingBuffer>,
        project_tb: Rational,
        frame_rate: Rational,
        vsync_period_ns: i64,
        shutdown: Arc<AtomicBool>,
        seek_request: Arc<Mutex<Option<i64>>>,
    ) -> Result<Self, crate::audio::output_stream::AudioStreamError> {
        let clock = MasterClock::new(project_tb, crate::audio::audio_decoder::OUT_SAMPLE_RATE);

        let audio_stream = AudioOutputStream::open(Arc::clone(&ring), Arc::clone(&clock))?;

        let corrector = DriftCorrector::new(Arc::clone(&clock));
        let decider =
            PresentationDecider::new(Arc::clone(&clock), project_tb, frame_rate, vsync_period_ns);

        #[cfg(any(test, debug_assertions))]
        let probe = SyncProbe::new(project_tb);

        Ok(Self {
            device,
            surface,
            graph,
            scheduler,
            timeline,
            tracks,
            sources,
            clock,
            decider,
            corrector,
            #[cfg(any(test, debug_assertions))]
            probe,
            _audio_stream: audio_stream,
            project_tb,
            shutdown,
            seek_request,
            ring,
        })
    }

    /// Run the playback loop until shutdown is set.
    pub fn run(mut self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            let pts = self.clock.pts();
            let next_pts = self.decider.next_frame_pts();
            let action = self.decider.decide(next_pts);

            match action {
                PresentAction::Drop => {
                    // Schedule a frame to keep cache warm, but do NOT render or present
                    let _frame_state = self.scheduler.schedule_frame(
                        next_pts,
                        &self.timeline.read().unwrap(),
                        &self.tracks.read().unwrap(),
                        &self.sources.read().unwrap(),
                    );
                    continue;
                }
                PresentAction::Hold => {
                    // Sleep until next vsync boundary approx
                    let drift_ns = self.project_tb.pts_to_ns(next_pts - pts);
                    let mut sleep_ns = drift_ns.abs();
                    // Just in case, clamp to [0, vsync_period] (assume 16.6ms)
                    let vsync_ns = 16_666_667;
                    if sleep_ns > vsync_ns {
                        sleep_ns = vsync_ns;
                    }
                    std::thread::sleep(std::time::Duration::from_nanos(sleep_ns as u64));
                    continue;
                }
                PresentAction::Present => {
                    let frame_state = self.scheduler.schedule_frame(
                        next_pts,
                        &self.timeline.read().unwrap(),
                        &self.tracks.read().unwrap(),
                        &self.sources.read().unwrap(),
                    );

                    let output = match self.surface.get_current_texture() {
                        Ok(o) => o,
                        Err(wgpu::SurfaceError::Outdated) | Err(wgpu::SurfaceError::Lost) => {
                            continue;
                        }
                        Err(e) => {
                            eprintln!("Surface error: {:?}", e);
                            continue;
                        }
                    };

                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    for node in self.graph.nodes_mut() {
                        if let Some(any) = node.as_any_mut() {
                            if let Some(blit_node) = any.downcast_mut::<BlitToScreenNode>() {
                                blit_node.current_surface_view = Some(view);
                                break;
                            }
                        }
                    }

                    let mut encoder = self.device.begin_frame();
                    self.graph.execute(&mut encoder, &self.device, &frame_state);
                    self.device.submit(encoder);

                    output.present();

                    #[cfg(any(test, debug_assertions))]
                    self.probe.record(next_pts, self.clock.pts());

                    let _correction = self.corrector.update();
                }
            }
        }
    }

    /// Seek the entire engine to a new PTS.
    pub fn seek(&mut self, new_pts: i64) {
        self.clock.seek(new_pts);
        self.corrector.reset();
        *self.seek_request.lock().unwrap() = Some(new_pts);
        self.ring.clear();
    }

    #[cfg(any(test, debug_assertions))]
    pub fn rms_sync_error_ns(&self) -> f64 {
        self.probe.rms_error_ns()
    }
}
