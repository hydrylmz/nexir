// src/export/renderer.rs

use std::sync::Arc;
use crate::render::device::GpuDevice;
use crate::render::graph::{CompiledGraph, RenderGraphCompiler};
use crate::render::resource::ResourceId;
use crate::render::nodes::yuv_upload::YuvUploadNode;
use crate::render::nodes::yuv_to_rgb::YuvToRgbNode;
use crate::render::nodes::composite::CompositeNode;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::export::job::ExportJob;
use crate::timeline::store::TimelineStore;
use crate::timeline::source::{SourceRegistry, ColorSpace};
use crate::export::partitioner::ExportSegment;
use crate::export::readback::FrameReadback;
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::progress::{ProgressSender, ExportPhase};
use crate::render::shader::registry::ShaderRegistry;
use crate::render::compute::ComputePipelineCache;
use crate::render::frame_state::FrameState;

/// Maximum number of clips composited in a single export frame.
const MAX_CLIPS: usize = 8;

/// Describes the set of active clips for a frame.  When this changes between
/// frames the render graph must be recompiled (different clip dimensions →
/// different texture sizes → different pipeline bind groups).
#[derive(Clone, PartialEq, Eq, Debug)]
struct ClipSignature {
    clip_width:  u32,
    clip_height: u32,
    is_nv12:     bool,
}

pub struct ExportRenderer {
    device:         Arc<GpuDevice>,
    scheduler:      Arc<FrameScheduler>,
    readback:       FrameReadback,
    job:            Arc<ExportJob>,
    timeline:       Arc<std::sync::RwLock<TimelineStore>>,
    sources:        Arc<std::sync::RwLock<SourceRegistry>>,
    shaders:        Arc<ShaderRegistry>,
    compute_cache:  Arc<ComputePipelineCache>,

    /// Current compiled render graph (None before first frame).
    cached_graph:     Option<CompiledGraph>,
    /// Clip signatures that the current cached graph was built for.
    cached_sig:       Vec<ClipSignature>,
    /// Indices of the YuvUploadNodes inside the cached graph (one per clip).
    upload_indices:   Vec<usize>,

    pub frames_done: usize,
}

impl ExportRenderer {
    pub fn new(
        device:        Arc<GpuDevice>,
        scheduler:     Arc<FrameScheduler>,
        job:           Arc<ExportJob>,
        timeline:      Arc<std::sync::RwLock<TimelineStore>>,
        sources:       Arc<std::sync::RwLock<SourceRegistry>>,
        shaders:       Arc<ShaderRegistry>,
        compute_cache: Arc<ComputePipelineCache>,
    ) -> Result<Self, RenderError> {
        log::info!("[export] ExportRenderer::new — canvas {}x{}, {} total frames",
            job.width, job.height, job.total_frames());

        let readback = FrameReadback::new(&device, job.width, job.height)
            .map_err(RenderError::ReadbackInit)?;

        Ok(Self {
            device,
            scheduler,
            readback,
            job,
            timeline,
            sources,
            shaders,
            compute_cache,
            cached_graph:   None,
            cached_sig:     Vec::new(),
            upload_indices: Vec::new(),
            frames_done:    0,
        })
    }

    /// Compute the clip signature for `frame`.
    fn signature(frame: &FrameState) -> Vec<ClipSignature> {
        frame.clips.iter().map(|c| ClipSignature {
            clip_width:  c.clip_width,
            clip_height: c.clip_height,
            is_nv12:     c.is_nv12,
        }).collect()
    }

    /// Compile (or reuse the cached) render graph for the active clips in `frame`.
    ///
    /// Recompilation happens only when the number of active clips or their
    /// dimensions change — typically at segment boundaries or on the very first
    /// frame. Within a single clip's span the graph is reused every frame.
    fn ensure_graph(&mut self, frame: &FrameState) -> Result<(), RenderError> {
        let sig = Self::signature(frame);

        if self.cached_graph.is_some() && self.cached_sig == sig {
            return Ok(()); // cache hit — nothing to do
        }

        log::info!(
            "[export] (re)compiling graph for {} clip(s) (was {})",
            sig.len(), self.cached_sig.len()
        );

        let mut compiler   = RenderGraphCompiler::new();
        let mut id_counter = 2u32; // 0 = FINAL_COLOR, 1 = SCREEN (reserved)

        let mut comp_node = CompositeNode::new(
            &self.device,
            &self.shaders,
            ResourceId::FINAL_COLOR,
            MAX_CLIPS as u32,
            wgpu::TextureFormat::Rgba16Float,
        );

        let mut upload_indices = Vec::with_capacity(sig.len());

        for (slot, clip) in frame.clips.iter().enumerate() {
            let y_id    = ResourceId::next(&mut id_counter);
            let uv_id   = ResourceId::next(&mut id_counter);
            let rgba_id = ResourceId::next(&mut id_counter);

            // Create YuvUpload with the clip's ACTUAL dimensions so the GPU
            // textures are exactly the right size — no wasted rows, no green fill.
            let upload_node = YuvUploadNode::new(
                &self.device,
                slot as u32,
                clip.clip_width,
                clip.clip_height,
                y_id,
                uv_id,
            );

            let node_idx = compiler.add_node(Box::new(upload_node));
            upload_indices.push(node_idx);

            // YuvToRgb operates on clip dimensions and outputs an RGBA texture
            // at clip resolution.  The Composite node handles letterboxing via
            // the ClipTransform (which maps clip space → canvas space).
            compiler.add_node(Box::new(YuvToRgbNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                y_id,
                uv_id,
                rgba_id,
                clip.clip_width,
                clip.clip_height,
                ColorSpace::Bt709,
                true, // limited range
            )));

            comp_node.input_textures.push(rgba_id);
        }

        compiler.add_node(Box::new(comp_node));

        let graph = compiler
            .compile(self.job.width, self.job.height)
            .map_err(|_| RenderError::GraphOutput)?;

        log::info!("[export] render graph compiled successfully ({} clip(s))", sig.len());

        self.cached_graph   = Some(graph);
        self.cached_sig     = sig;
        self.upload_indices = upload_indices;

        Ok(())
    }

    /// Upload YUV data for every active clip into the staging buffers of the
    /// corresponding `YuvUploadNode`s in the compiled graph.
    ///
    /// The graph must have been compiled by `ensure_graph` before calling this.
    fn upload_frame_data(&mut self, frame: &FrameState) {
        let io    = self.scheduler.io_layer();
        let graph = self.cached_graph.as_mut().expect("graph not compiled");
        let nodes = graph.nodes_mut();

        for (slot_idx, clip) in frame.clips.iter().enumerate() {
            if slot_idx >= self.upload_indices.len() {
                log::warn!("[export] slot_idx={slot_idx} out of upload_indices range");
                break;
            }

            let node_idx = self.upload_indices[slot_idx];
            let node     = &mut nodes[node_idx];

            if let Some(upload) = node
                .as_any_mut()
                .and_then(|n| n.downcast_mut::<YuvUploadNode>())
            {
                let tier    = (clip.texture_slot >> 16) as u8;
                let index   = (clip.texture_slot & 0xFFFF) as u16;
                let slot_id = crate::io::slot_pool::FrameSlotId { tier, index };



                io.pool.with_buffer_read(slot_id, |data| {
                    upload.upload_frame(data, clip.is_nv12, clip.clip_width, clip.clip_height);
                });
            } else {
                log::warn!("[export] slot {slot_idx}: failed to downcast YuvUploadNode");
            }
        }
    }

    pub fn render_segment(
        &mut self,
        segment:  &ExportSegment,
        queue:    &EncoderQueue,
        progress: &ProgressSender,
    ) -> Result<(), RenderError> {
        log::info!("[export] render_segment {} — frames {}..{}",
            segment.index, segment.frame_start, segment.frame_end);

        let mut active_slot   = 0usize;
        let mut pending_frame: Option<usize> = None;

        for frame_idx in segment.frame_start..=segment.frame_end {


            // ── Render the current frame ──────────────────────────────────────
            if frame_idx < segment.frame_end {
                let pts = self.job.frame_pts(frame_idx);


                let frame_state = self.scheduler.schedule_frame(
                    pts,
                    &self.timeline.read().unwrap(),
                    &self.sources.read().unwrap(),
                );


                // Recompile graph if clip dimensions changed (or on first frame).
                self.ensure_graph(&frame_state)?;

                // Copy YUV data from slot-pool buffers into wgpu staging buffers.
                // This is non-blocking (queue.write_buffer internally) — no GPU stall.
                self.upload_frame_data(&frame_state);


                let mut encoder = self.device.begin_frame();
                let rtt_id      = ResourceId::FINAL_COLOR;

                let graph = self.cached_graph.as_mut().unwrap();
                graph.execute_with_callback(
                    &mut encoder,
                    &self.device,
                    &frame_state,
                    |enc, ctx| {
                        let ctx = std::panic::AssertUnwindSafe(ctx);
                        if let Ok(res) = std::panic::catch_unwind(|| ctx.get(rtt_id)) {
                            self.readback.record_copy(enc, res.texture, active_slot);
                        } else {
                            log::error!("[export] frame {frame_idx}: FINAL_COLOR missing from RenderContext!");
                        }
                    },
                );

                self.device.submit(encoder);


                // Non-blocking poll — lets the GPU start without blocking this thread.
                self.device.device.poll(wgpu::Maintain::Poll);
            }

            // ── Read back the previous frame ──────────────────────────────────
            if let Some(prev_idx) = pending_frame {
                let prev_slot = 1 - active_slot;


                let view = self.readback.map_read(prev_slot, &self.device)
                    .map_err(|e| {
                        log::error!("[export] map_read failed for frame {prev_idx}: {e}");
                        RenderError::GpuTimeout
                    })?;

                let bytes = self.readback.strip_padding(&view);
                drop(view);
                self.readback.unmap(prev_slot);



                queue.push(QueueItem::Frame(RawFrame {
                    frame_index: prev_idx,
                    pts:  self.job.frame_pts(prev_idx),
                    data: bytes,
                }));

                self.frames_done += 1;
                progress.report(self.frames_done, ExportPhase::Rendering);
            }

            active_slot   = 1 - active_slot;
            pending_frame = if frame_idx < segment.frame_end { Some(frame_idx) } else { None };
        }

        log::info!("[export] render_segment {} done", segment.index);
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
