use std::sync::Arc;
use crate::export::job::ExportJob;
use crate::export::partitioner::SegmentPartitioner;
use crate::export::renderer::ExportRenderer;
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::video_encoder::VideoEncoder;
use crate::export::audio_encoder::AudioMuxEncoder;
use crate::export::muxer::Muxer;
use crate::export::progress::{progress_channel, ProgressReceiver, ProgressSender};
use crate::render::device::GpuDevice;
use crate::render::graph::{CompiledGraph, RenderNode};
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::io::ffi::avutil::AVRational;

pub struct ExportEngine {
    device:     Arc<GpuDevice>,
    job:        Arc<ExportJob>,
    scheduler:  Arc<FrameScheduler>,
}

impl ExportEngine {
    pub fn new(
        device:    Arc<GpuDevice>,
        job:       ExportJob,
        scheduler: Arc<FrameScheduler>,
    ) -> Self {
        Self { device, job: Arc::new(job), scheduler }
    }

    pub fn start(
        self,
        graph: CompiledGraph,
        nodes: Vec<Box<dyn RenderNode>>,
        rtt_id: crate::render::resource::ResourceId,
    ) -> Result<ProgressReceiver, ExportError> {
        self.job.validate().map_err(ExportError::JobInvalid)?;

        let (prog_tx, prog_rx) = progress_channel(self.job.total_frames());

        let video_enc = VideoEncoder::open(&self.job).map_err(ExportError::EncoderOpen)?;
        let mut audio_enc = AudioMuxEncoder::open(&self.job).map_err(ExportError::EncoderOpen)?;

        let enc_video_tb = AVRational { num: self.job.frame_rate.den as i32, den: self.job.frame_rate.num as i32 };
        let enc_audio_tb = AVRational { num: 1, den: 48000 };

        let muxer = Arc::new(Muxer::open(&self.job, &video_enc, &audio_enc, enc_video_tb, enc_audio_tb)
            .map_err(ExportError::MuxerOpen)?);

        let queue = Arc::new(EncoderQueue::new());
        let segments = SegmentPartitioner::partition(&self.job);

        let job_clone     = Arc::clone(&self.job);
        let queue_clone   = Arc::clone(&queue);
        let muxer_clone   = Arc::clone(&muxer);
        let device_clone  = Arc::clone(&self.device);
        let scheduler_clone = Arc::clone(&self.scheduler);

        let prog_tx_enc = prog_tx.clone();
        std::thread::Builder::new().name("ve-encoder".into()).spawn(move || {
            Self::encoder_thread(queue_clone, video_enc, muxer_clone, prog_tx_enc);
        }).map_err(ExportError::ThreadSpawn)?;

        let job_clone2 = Arc::clone(&self.job);
        let muxer_clone2 = Arc::clone(&muxer);
        std::thread::Builder::new().name("ve-audio-enc".into()).spawn(move || {
            let _ = audio_enc.encode_all(&job_clone2, &mut |pkt| {
                muxer_clone2.write_packet(pkt, false).unwrap();
            });
        }).map_err(ExportError::ThreadSpawn)?;

        std::thread::Builder::new().name("ve-export-dispatch".into()).spawn(move || {
            let mut renderer = ExportRenderer::new(
                Arc::clone(&device_clone), graph, nodes,
                Arc::clone(&scheduler_clone), rtt_id, Arc::clone(&job_clone)
            ).unwrap();

            for seg in &segments {
                renderer.render_segment(seg, &queue, &prog_tx).unwrap();
            }
            queue.push(QueueItem::AllDone);
        }).map_err(ExportError::ThreadSpawn)?;

        Ok(prog_rx)
    }

    fn encoder_thread(
        queue:     Arc<EncoderQueue>,
        mut video_enc: VideoEncoder,
        muxer:     Arc<Muxer>,
        _prog_tx:   ProgressSender,
    ) {
        loop {
            match queue.pop() {
                Some(QueueItem::Frame(raw)) => {
                    let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                        muxer.write_packet(pkt, true).unwrap();
                    };
                    video_enc.encode_frame(&raw, &mut sink).unwrap();
                }
                Some(QueueItem::SegmentDone { .. }) => {}
                Some(QueueItem::AllDone) | None => {
                    let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                        muxer.write_packet(pkt, true).unwrap();
                    };
                    video_enc.flush(&mut sink).unwrap();
                    break;
                }
            }
        }
        
        if let Ok(m) = Arc::try_unwrap(muxer) {
            m.finalise().unwrap();
        }
    }
}

#[derive(Debug)]
pub enum ExportError {
    JobInvalid(crate::export::job::JobError),
    EncoderOpen(crate::export::video_encoder::EncodeError),
    MuxerOpen(crate::export::muxer::MuxError),
    ThreadSpawn(std::io::Error),
}
