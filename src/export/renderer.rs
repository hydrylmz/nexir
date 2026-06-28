use std::sync::Arc;
use crate::render::device::GpuDevice;
use crate::render::graph::{CompiledGraph, RenderNode};
use crate::render::resource::ResourceId;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::export::job::ExportJob;
use crate::timeline::store::TimelineStore;
use crate::timeline::source::SourceRegistry;
use crate::export::partitioner::ExportSegment;
use crate::export::readback::FrameReadback;
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::progress::{ProgressSender, ExportPhase};

pub struct ExportRenderer {
    device:    Arc<GpuDevice>,
    graph:     CompiledGraph,
    #[allow(dead_code)]
    nodes:     Vec<Box<dyn RenderNode>>,
    scheduler: Arc<FrameScheduler>,
    readback:  FrameReadback,
    rtt_id:    ResourceId,
    job:       Arc<ExportJob>,
    timeline:  Arc<std::sync::RwLock<TimelineStore>>,
    sources:   Arc<std::sync::RwLock<SourceRegistry>>,
}

impl ExportRenderer {
    pub fn new(
        device:    Arc<GpuDevice>,
        graph:     CompiledGraph,
        nodes:     Vec<Box<dyn RenderNode>>,
        scheduler: Arc<FrameScheduler>,
        rtt_id:    ResourceId,
        job:       Arc<ExportJob>,
        timeline:  Arc<std::sync::RwLock<TimelineStore>>,
        sources:   Arc<std::sync::RwLock<SourceRegistry>>,
    ) -> Result<Self, RenderError> {
        let readback = FrameReadback::new(&device, job.width, job.height)
            .map_err(RenderError::ReadbackInit)?;

        Ok(Self {
            device,
            graph,
            nodes,
            scheduler,
            readback,
            rtt_id,
            job,
            timeline,
            sources,
        })
    }

    pub fn render_segment(
        &mut self,
        segment:  &ExportSegment,
        queue:    &EncoderQueue,
        progress: &ProgressSender,
    ) -> Result<(), RenderError> {
        let mut active_slot = 0usize;
        let mut pending_frame: Option<usize> = None;

        for frame_idx in segment.frame_start..=segment.frame_end {
            if frame_idx < segment.frame_end {
                let pts = self.job.frame_pts(frame_idx);
                let frame_state = self.scheduler.schedule_frame(
                    pts,
                    &self.timeline.read().unwrap(),
                    &self.sources.read().unwrap()
                );
                
                let mut encoder = self.device.begin_frame();
                
                self.graph.execute_with_callback(&mut encoder, &self.device, &frame_state, |enc, ctx| {
                    let rtt_id = self.rtt_id;
                    let ctx = std::panic::AssertUnwindSafe(ctx);
                    if let Ok(res) = std::panic::catch_unwind(|| ctx.get(rtt_id)) {
                        self.readback.record_copy(enc, res.texture, active_slot);
                    }
                });
                
                self.device.submit(encoder);
            }

            if let Some(prev_idx) = pending_frame {
                let prev_slot = 1 - active_slot;
                
                let view = self.readback.map_read(prev_slot, &self.device)
                    .map_err(|_| RenderError::GpuTimeout)?;
                
                let bytes = self.readback.strip_padding(&view);
                
                drop(view);
                self.readback.unmap(prev_slot);
                
                queue.push(QueueItem::Frame(RawFrame {
                    frame_index: prev_idx,
                    pts: self.job.frame_pts(prev_idx),
                    data: bytes,
                }));
                
                progress.report(1, ExportPhase::Rendering);
            }

            active_slot = 1 - active_slot;
            pending_frame = if frame_idx < segment.frame_end { Some(frame_idx) } else { None };
        }
        
        queue.push(QueueItem::SegmentDone { segment_index: segment.index });
        
        Ok(())
    }
}

pub struct RawFrame {
    pub frame_index: usize,
    pub pts:         i64,
    pub data:        Vec<u8>,
}

#[derive(Debug)]
pub enum RenderError {
    GraphOutput,
    ReadbackInit(String),
    ScheduleFailure,
    GpuTimeout,
}
