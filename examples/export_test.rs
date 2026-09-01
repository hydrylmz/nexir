use nexir::export::engine::ExportEngine;
use nexir::export::job::{AudioCodec, Container, CpuPreset, ExportJob, VideoCodec, VideoQuality};
use nexir::io::frame_cache::FrameCache;
use nexir::io::slot_pool::FrameSlotPool;
use nexir::io::io_layer::IoLayer;
use nexir::render::device::GpuDevice;
use nexir::timeline::rational::Rational;
use nexir::timeline::store::TimelineStore;
use nexir::timeline::track::TrackList;
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).unwrap());

    let job = ExportJob {
        output_path: PathBuf::from("test_export.mp4"),
        container: Container::Mp4,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::Aac,
        width: 1920,
        height: 1080,
        frame_rate: Rational::new(30, 1),
        quality: VideoQuality::Crf(23),
        audio_bitrate: 192000,
        render_threads: 4,
        project_tb: Rational::new(1, 90000),
        pts_in: 0,
        pts_out: 3000,
        cpu_preset: CpuPreset::Medium,
        output_color: nexir::timeline::source::ColorInfo::bt709(),
        hdr10: None,
    };

    println!("Starting export...");

    let capability = nexir::interop::capability::InteropCapability::none();
    let sources = Arc::new(std::sync::RwLock::new(
        nexir::timeline::source::SourceRegistry::new(),
    ));

    // Build the IoLayer with the current 6-argument API.
    let pool  = Arc::new(FrameSlotPool::new(&device));
    let cache = Arc::new(FrameCache::new(Arc::clone(&pool), 128));
    let (prefetch_tx, prefetch_rx) = std::sync::mpsc::sync_channel(64);
    let io_layer = Arc::new(IoLayer::new(
        Arc::clone(&device.device),
        Arc::clone(&pool),
        Arc::clone(&cache),
        Arc::clone(&sources),
        prefetch_tx,
        Rational::new(1, 90000),
        // This example forces the CPU path (`capability` above is `none()`), so a
        // disabled registry is the honest description rather than a probe whose
        // answer would be ignored.
        Arc::new(nexir::io::interop_decode::InteropDecodeTargets::disabled(
            Arc::clone(&device),
        )),
    ));
    // Spawn the prefetch worker so the channel doesn't block.
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let worker = nexir::io::prefetch::PrefetchWorker::new(
        prefetch_rx,
        Arc::clone(&io_layer),
        Arc::clone(&cache),
        Arc::clone(&shutdown),
    );
    let _prefetch_thread = nexir::io::prefetch::spawn_prefetch_worker(worker);

    let scheduler = Arc::new(nexir::scheduler::frame_scheduler::FrameScheduler::new(
        io_layer, 1920, 1080,
    ));

    let store_arc  = Arc::new(std::sync::RwLock::new(TimelineStore::new()));
    let tracks_arc = Arc::new(std::sync::RwLock::new(TrackList::new()));

    let engine = ExportEngine::new(
        device.clone(),
        job,
        scheduler,
        store_arc,
        tracks_arc,
        sources,
        capability,
        None,
        false,
    );

    let shaders =
        Arc::new(nexir::render::shader::registry::ShaderRegistry::compile_all(&device).unwrap());
    let compute = Arc::new(nexir::render::compute::ComputePipelineCache::new());
    match engine.start(shaders, compute) {
        Ok(_)  => println!("Export started"),
        Err(e) => println!("Export failed: {:?}", e),
    }
}
