use nexir::export::engine::ExportEngine;
use nexir::export::job::{AudioCodec, Container, ExportJob, VideoCodec, VideoQuality};
use nexir::render::device::GpuDevice;
use nexir::timeline::rational::Rational;
use nexir::timeline::store::TimelineStore;
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).unwrap());
    let mut store = TimelineStore::new();

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
    };

    println!("Starting export...");

    let capability = nexir::interop::capability::InteropCapability::none();
    let io_layer = Arc::new(nexir::io::io_layer::IoLayer::new(&device, 1024, 8));
    let scheduler = Arc::new(nexir::scheduler::frame_scheduler::FrameScheduler::new(
        io_layer, 1920, 1080,
    ));
    let sources = Arc::new(std::sync::RwLock::new(
        nexir::timeline::source::SourceRegistry::new(),
    ));
    let store_arc = Arc::new(std::sync::RwLock::new(store));

    let engine = ExportEngine::new(
        device.clone(),
        job,
        scheduler,
        store_arc,
        sources,
        capability,
        None,
        false,
    );

    let shaders =
        Arc::new(nexir::render::shader::registry::ShaderRegistry::compile_all(&device).unwrap());
    let compute = Arc::new(nexir::render::compute::ComputePipelineCache::new());
    match engine.start(shaders, compute) {
        Ok(_) => println!("Export started"),
        Err(e) => println!("Export failed: {:?}", e),
    }
}
