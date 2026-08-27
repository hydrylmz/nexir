// src/export/renderer.rs

use crate::export::job::ExportJob;
use crate::export::muxer::Muxer;
use crate::export::partitioner::ExportSegment;
use crate::export::progress::{ExportPhase, ProgressSender};
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::readback::FrameReadback;
use crate::export::video_encoder::VideoEncoderBackend;
use crate::interop::encode_interop::Abgr10RepackNode;
use crate::render::compute::ComputePipelineCache;
use crate::render::device::GpuDevice;
use crate::render::frame_state::FrameState;
use crate::render::graph::{CompiledGraph, RenderGraphCompiler};
use crate::render::nodes::composite::CompositeNode;
use crate::render::nodes::yuv_to_rgb::YuvToRgbNode;
use crate::render::nodes::yuv_upload::YuvUploadNode;
use crate::render::nodes::tonemap::{ToneMapNode, ToneMapPushConstants, InputTransferFn, GamutConversion, ToneMapMode};
use crate::render::resource::ResourceId;
use crate::render::shader::registry::ShaderRegistry;
use crate::render::still_image::StillImageCache;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients, TransferFunction, ColorPrimaries, SourceRegistry, is_still_image_path};
use crate::timeline::store::TimelineStore;
use crate::timeline::track::TrackList;
use std::sync::Arc;

pub enum ExportBackend {
    Cpu {
        readback: FrameReadback,
    },
    GpuNvenc {
        video_enc: VideoEncoderBackend,
        muxer: Arc<Muxer>,
        repack: Abgr10RepackNode,
    },
}

/// Minimum binding-array size passed to CompositeNode — avoids creating
/// a zero-slot layout when the frame has no clips.
const MIN_COMPOSITE_SLOTS: u32 = 1;

/// Describes the set of active clips for a frame.  When this changes between
/// frames the render graph must be recompiled (different clip dimensions →
/// different texture sizes → different pipeline bind groups).
#[derive(Clone, PartialEq, Eq, Debug)]
struct ClipSignature {
    clip_width: u32,
    clip_height: u32,
    is_nv12: bool,
    /// True when this clip is a still image (PNG/JPEG/etc).  Still images use
    /// `StillImageUploadNode` instead of `YuvUploadNode → YuvToRgbNode`, so a
    /// change in this flag must trigger graph recompilation.
    is_still_image: bool,
}

pub struct ExportRenderer {
    device: Arc<GpuDevice>,
    scheduler: Arc<FrameScheduler>,
    pub backend: ExportBackend,
    job: Arc<ExportJob>,
    timeline: Arc<std::sync::RwLock<TimelineStore>>,
    tracks: Arc<std::sync::RwLock<TrackList>>,
    sources: Arc<std::sync::RwLock<SourceRegistry>>,
    shaders: Arc<ShaderRegistry>,
    compute_cache: Arc<ComputePipelineCache>,

    /// Cache for still-image CPU decoding + GPU staging buffers.
    still_image_cache: StillImageCache,

    /// Current compiled render graph (None before first frame).
    cached_graph: Option<CompiledGraph>,
    /// Clip signatures that the current cached graph was built for.
    cached_sig: Vec<ClipSignature>,
    /// Indices of the YuvUploadNodes inside the cached graph (one per YUV clip).
    /// Still-image clips do not have an entry here — their upload node writes its
    /// staging buffer at load time and is a no-op during `upload_frame_data`.
    upload_indices: Vec<Option<usize>>,

    pub frames_done: usize,
}

impl ExportRenderer {
    pub fn new(
        device: Arc<GpuDevice>,
        scheduler: Arc<FrameScheduler>,
        job: Arc<ExportJob>,
        timeline: Arc<std::sync::RwLock<TimelineStore>>,
        tracks: Arc<std::sync::RwLock<TrackList>>,
        sources: Arc<std::sync::RwLock<SourceRegistry>>,
        shaders: Arc<ShaderRegistry>,
        compute_cache: Arc<ComputePipelineCache>,
        backend: ExportBackend,
    ) -> Self {
        log::info!(
            "[export] ExportRenderer::new — canvas {}x{}, {} total frames",
            job.width,
            job.height,
            job.total_frames()
        );

        Self {
            device,
            scheduler,
            backend,
            job,
            timeline,
            tracks,
            sources,
            shaders,
            compute_cache,
            still_image_cache: StillImageCache::default(),
            cached_graph: None,
            cached_sig: Vec::new(),
            upload_indices: Vec::new(),
            frames_done: 0,
        }
    }

    /// Compute the clip signature for `frame`, consulting the source registry
    /// to detect still images.
    fn signature(frame: &FrameState, sources: &SourceRegistry) -> Vec<ClipSignature> {
        frame
            .clips
            .iter()
            .map(|c| {
                let is_still = sources
                    .path(c.source_id)
                    .map(|p| is_still_image_path(p.as_ref()))
                    .unwrap_or(false);
                ClipSignature {
                    clip_width: c.clip_width,
                    clip_height: c.clip_height,
                    is_nv12: c.is_nv12,
                    is_still_image: is_still,
                }
            })
            .collect()
    }

    /// Compile (or reuse the cached) render graph for the active clips in `frame`.
    ///
    /// Recompilation happens only when the number of active clips, their
    /// dimensions, or their still-image flag changes — typically at segment
    /// boundaries or on the very first frame.  Within a single clip's span the
    /// graph is reused every frame.
    ///
    /// Still-image clips (PNG, JPEG, etc.) bypass the YUV pipeline entirely:
    /// a `StillImageUploadNode` copies the pre-decoded Rgba16Float staging
    /// buffer directly into the composite input texture, avoiding the green-box
    /// artefact that the YUV path produces for non-YUV pixel data.
    fn ensure_graph(&mut self, frame: &FrameState) -> Result<(), RenderError> {
        let sig = {
            let sources = self.sources.read().unwrap();
            Self::signature(frame, &sources)
        };

        if self.cached_graph.is_some() && self.cached_sig == sig {
            return Ok(()); // cache hit — nothing to do
        }

        log::info!(
            "[export] (re)compiling graph for {} clip(s) (was {})",
            sig.len(),
            self.cached_sig.len()
        );

        let mut compiler = RenderGraphCompiler::new();
        let mut id_counter = 2u32; // 0 = FINAL_COLOR, 1 = SCREEN (reserved)

        // Use the exact clip count so the binding-array layout is tight.
        // ensure_graph recompiles whenever sig changes, so this is always correct.
        let clip_count = (sig.len() as u32).max(MIN_COMPOSITE_SLOTS);
        let mut comp_node = CompositeNode::new(
            &self.device,
            &self.shaders,
            ResourceId::FINAL_COLOR,
            clip_count,
            wgpu::TextureFormat::Rgba16Float,
        );

        // `upload_indices` maps clip slot → YuvUploadNode index in the graph.
        // Still-image clips are stored as `None` — they have no YuvUploadNode.
        let mut upload_indices: Vec<Option<usize>> = Vec::with_capacity(sig.len());

        for (slot, (clip, clip_sig)) in frame.clips.iter().zip(sig.iter()).enumerate() {
            let rgba_id = ResourceId::next(&mut id_counter);

            if clip_sig.is_still_image {
                // ── Still-image path ──────────────────────────────────────────
                // Load the PNG/JPEG once; subsequent frames reuse the cache.
                let path = {
                    let sources = self.sources.read().unwrap();
                    sources.path(clip.source_id).map(|p| p.to_path_buf())
                };

                if let Some(path) = path {
                    if let Some(cached) = self.still_image_cache.get_or_load(&self.device, &path) {
                        compiler.add_node(Box::new(
                            crate::render::still_image::StillImageUploadNode::new(cached, rgba_id),
                        ));
                        comp_node.input_textures.push(rgba_id);
                        upload_indices.push(None);
                        continue;
                    } else {
                        log::warn!(
                            "[export] slot {slot}: failed to load still image {:?}, skipping clip",
                            path
                        );
                    }
                } else {
                    log::warn!("[export] slot {slot}: still image has no registered path, skipping");
                }

                // If load failed, push a sentinel so the slot count stays in sync.
                upload_indices.push(None);
                continue;
            }

            // ── YUV video path ────────────────────────────────────────────────
            let y_id  = ResourceId::next(&mut id_counter);
            let uv_id = ResourceId::next(&mut id_counter);

            // Look up the real ColorInfo from the source. Fall back to BT.709/Limited
            // for any source that doesn't have registered video info.
            let color_info = {
                let sources = self.sources.read().unwrap();
                sources
                    .video_info(clip.source_id)
                    .map(|vi| vi.color_info)
                    .unwrap_or(ColorInfo {
                        matrix:      MatrixCoefficients::Bt709,
                        range:       ColorRange::Limited,
                        transfer_fn: TransferFunction::Bt709,
                        primaries:   ColorPrimaries::Bt709,
                        bit_depth:   8,
                    })
            };

            // Create YuvUpload with the clip's ACTUAL dimensions so the GPU
            // textures are exactly the right size — no wasted rows, no green fill.
            let upload_node = YuvUploadNode::new_with_depth(
                &self.device,
                slot as u32,
                clip.clip_width,
                clip.clip_height,
                y_id,
                uv_id,
                color_info.bit_depth,
            );

            let node_idx = compiler.add_node(Box::new(upload_node));
            upload_indices.push(Some(node_idx));


            compiler.add_node(Box::new(YuvToRgbNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                y_id,
                uv_id,
                rgba_id,
                clip.clip_width,
                clip.clip_height,
                color_info,
            )));

            // Tone-map HDR clips to SDR for CPU/GPU export pipelines
            let final_rgba_id = if color_info.is_hdr() {
                use crate::timeline::source::{TransferFunction, ColorPrimaries};
                let tonemapped_id = ResourceId::next(&mut id_counter);
                let trc = match color_info.transfer_fn {
                    TransferFunction::Pq  => InputTransferFn::Pq,
                    TransferFunction::Hlg => InputTransferFn::Hlg,
                    _                     => InputTransferFn::Linear,
                };
                let gamut = if color_info.effective_primaries(
                    clip.clip_width, clip.clip_height
                ) == ColorPrimaries::Bt2020 {
                    GamutConversion::Bt2020ToBt709
                } else {
                    GamutConversion::None
                };
                let tm_params = ToneMapPushConstants::for_sdr_preview(
                    trc,
                    gamut,
                    ToneMapMode::AcesFilmic,
                    1000.0,
                    clip.clip_width,
                    clip.clip_height,
                );
                compiler.add_node(Box::new(ToneMapNode::new(
                    &self.device,
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

            comp_node.input_textures.push(final_rgba_id);
        }


        compiler.add_node(Box::new(comp_node));

        let graph = compiler
            .compile(self.job.width, self.job.height)
            .map_err(|_| RenderError::GraphOutput)?;

        log::info!(
            "[export] render graph compiled successfully ({} clip(s))",
            sig.len()
        );

        self.cached_graph = Some(graph);
        self.cached_sig = sig;
        self.upload_indices = upload_indices;

        Ok(())
    }

    /// Upload YUV data for every active *video* clip into the staging buffers
    /// of the corresponding `YuvUploadNode`s in the compiled graph.
    ///
    /// Still-image clips are skipped — their staging buffer is written once at
    /// load time by `StillImageCache::get_or_load` and never needs refreshing.
    ///
    /// The graph must have been compiled by `ensure_graph` before calling this.
    fn upload_frame_data(&mut self, frame: &FrameState) {
        let io = self.scheduler.io_layer();
        let graph = self.cached_graph.as_mut().expect("graph not compiled");
        let nodes = graph.nodes_mut();

        for (slot_idx, clip) in frame.clips.iter().enumerate() {
            if slot_idx >= self.upload_indices.len() {
                log::warn!("[export] slot_idx={slot_idx} out of upload_indices range");
                break;
            }

            // None means this slot is a still image — no YUV upload needed.
            let node_idx = match self.upload_indices[slot_idx] {
                Some(idx) => idx,
                None => continue,
            };

            let node = &mut nodes[node_idx];

            if let Some(upload) = node
                .as_any_mut()
                .and_then(|n| n.downcast_mut::<YuvUploadNode>())
            {
                let tier = (clip.texture_slot >> 16) as u8;
                let index = (clip.texture_slot & 0xFFFF) as u16;
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
        segment: &ExportSegment,
        queue: &EncoderQueue,
        progress: &ProgressSender,
    ) -> Result<(), RenderError> {
        log::info!(
            "[export] render_segment {} — frames {}..{}",
            segment.index,
            segment.frame_start,
            segment.frame_end
        );

        let is_cpu = matches!(self.backend, ExportBackend::Cpu { .. });

        if is_cpu {
            let mut active_slot = 0usize;
            // Pair the pending frame index with the SubmissionIndex of the command
            // buffer that wrote its readback slot.  A single Option keeps them in sync
            // and eliminates the SID-is-None panic on the final drain iteration.
            let mut pending: Option<(usize, wgpu::SubmissionIndex)> = None;

            for frame_idx in segment.frame_start..=segment.frame_end {
                // ── Check cancellation & pause ───────────────────────────────
                if progress.control().is_cancelled() {
                    log::info!("[export] cancelled by user during CPU render at frame {frame_idx}");
                    progress.report(self.frames_done, ExportPhase::Cancelled);
                    return Ok(());
                }
                while progress.control().is_paused() {
                    if progress.control().is_cancelled() {
                        progress.report(self.frames_done, ExportPhase::Cancelled);
                        return Ok(());
                    }
                    progress.report(self.frames_done, ExportPhase::Paused);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                // ── Render the current frame ──────────────────────────────────────
                if frame_idx < segment.frame_end {
                    let pts = self.job.frame_pts(frame_idx);

                    let frame_state = self.scheduler.schedule_frame(
                        pts,
                        &self.timeline.read().unwrap(),
                        &self.tracks.read().unwrap(),
                        &self.sources.read().unwrap(),
                    );

                    // Recompile graph if clip dimensions changed (or on first frame).
                    self.ensure_graph(&frame_state)?;

                    // Copy YUV data from slot-pool buffers into wgpu staging buffers.
                    // This is non-blocking (queue.write_buffer internally) — no GPU stall.
                    self.upload_frame_data(&frame_state);

                    let mut encoder = self.device.begin_frame();
                    let rtt_id = ResourceId::FINAL_COLOR;

                    let graph = self.cached_graph.as_mut().unwrap();

                    if let ExportBackend::Cpu { readback } = &mut self.backend {
                        graph.execute_with_callback(
                            &mut encoder,
                            &self.device,
                            &frame_state,
                            |enc, ctx| {
                                if ctx.contains(rtt_id) {
                                    readback.record_copy(enc, ctx.get(rtt_id).texture, active_slot);
                                } else {
                                    log::error!("[export] frame {frame_idx}: FINAL_COLOR missing from RenderContext!");
                                }
                            },
                        );
                    }

                    let sid = self.device.submit(encoder);

                    // Non-blocking poll — lets the GPU start without blocking this thread.
                    self.device.device.poll(wgpu::Maintain::Poll);

                    // ── Read back the frame rendered two iterations ago ───────────
                    if let Some((prev_idx, prev_sid)) = pending.take() {
                        let prev_slot = 1 - active_slot;

                        if let ExportBackend::Cpu { readback } = &mut self.backend {
                            let bytes = readback
                                .map_strip_unmap(prev_slot, &self.device, prev_sid)
                                .map_err(|e| {
                                    log::error!(
                                        "[export] map_strip_unmap failed for frame {prev_idx}: {e}"
                                    );
                                    RenderError::GpuTimeout
                                })?;

                            queue.push(QueueItem::Frame(RawFrame {
                                frame_index: prev_idx,
                                pts: self.job.frame_pts(prev_idx),
                                data: bytes,
                            }));
                        }

                        self.frames_done += 1;
                        progress.report(self.frames_done, ExportPhase::Rendering);
                    }

                    // Store this frame as the next pending readback, paired with its SID.
                    pending = Some((frame_idx, sid));
                    active_slot = 1 - active_slot;
                } else {
                    // ── Final drain: read back the last rendered frame ────────────
                    if let Some((prev_idx, prev_sid)) = pending.take() {
                        let prev_slot = 1 - active_slot;

                        if let ExportBackend::Cpu { readback } = &mut self.backend {
                            let bytes = readback.map_strip_unmap(prev_slot, &self.device, prev_sid)
                                .map_err(|e| {
                                    log::error!("[export] map_strip_unmap failed for frame {prev_idx} (drain): {e}");
                                    RenderError::GpuTimeout
                                })?;

                            queue.push(QueueItem::Frame(RawFrame {
                                frame_index: prev_idx,
                                pts: self.job.frame_pts(prev_idx),
                                data: bytes,
                            }));
                        }

                        self.frames_done += 1;
                        progress.report(self.frames_done, ExportPhase::Rendering);
                    }
                }
            }

            log::info!("[export] render_segment {} done", segment.index);
            queue.push(QueueItem::SegmentDone {
                segment_index: segment.index,
            });
        } else {
            // ── GPU / NVENC pipelined path ─────────────────────────────────────
            //
            // Classic double-buffered pipeline: while NVENC encodes frame N-1 from
            // slot (N-1)%2, the GPU renders frame N into slot N%2.  We only stall on
            // `WaitForSubmissionIndex(prev_sid)` — the exact submission that wrote the
            // *previous* slot — so the current GPU render can proceed concurrently.
            //
            //  Frame timeline (ideal, both units fully pipelined):
            //  ┌──────────┬──────────┬──────────┬──────────┐
            //  │ GPU: F0  │ GPU: F1  │ GPU: F2  │ GPU: F3  │  (slot 0, 1, 0, 1 …)
            //  └──────────┴──────────┴──────────┴──────────┘
            //             ┌──────────┬──────────┬──────────┐
            //             │ ENC: F0  │ ENC: F1  │ ENC: F2  │
            //             └──────────┴──────────┴──────────┘
            //
            // `pending` holds (frame_idx, SubmissionIndex, abgr10_slot) for the most
            // recently submitted — but not yet encoded — frame.

            // `pending` = (frame_idx, submission_index, abgr10_slot) of the last
            // submitted frame that has not been encoded yet.
            let mut pending: Option<(usize, wgpu::SubmissionIndex, usize)> = None;
            let mut active_slot = 0usize;

            // Iterate frame_start..=frame_end: the extra iteration at frame_end
            // skips rendering and only drains the final pending frame (same
            // pattern as the CPU readback path).
            for frame_idx in segment.frame_start..=segment.frame_end {
                // ── Check cancellation & pause ───────────────────────────────
                if progress.control().is_cancelled() {
                    log::info!("[export] cancelled by user during CPU render at frame {frame_idx}");
                    progress.report(self.frames_done, ExportPhase::Cancelled);
                    return Ok(());
                }
                while progress.control().is_paused() {
                    if progress.control().is_cancelled() {
                        progress.report(self.frames_done, ExportPhase::Cancelled);
                        return Ok(());
                    }
                    progress.report(self.frames_done, ExportPhase::Paused);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                // ── Render: submit GPU work for this frame ────────────────────
                if frame_idx < segment.frame_end {
                    let pts = self.job.frame_pts(frame_idx);

                    let frame_state = self.scheduler.schedule_frame(
                        pts,
                        &self.timeline.read().unwrap(),
                        &self.tracks.read().unwrap(),
                        &self.sources.read().unwrap(),
                    );

                    self.ensure_graph(&frame_state)?;
                    self.upload_frame_data(&frame_state);

                    let mut encoder = self.device.begin_frame();
                    let rtt_id    = ResourceId::FINAL_COLOR;
                    let width     = self.job.width;
                    let height    = self.job.height;
                    let device_ref = &self.device;
                    let slot      = active_slot; // captured for the closure below

                    let graph = self.cached_graph.as_mut().unwrap();

                    if let ExportBackend::GpuNvenc { video_enc, repack, .. } = &mut self.backend {
                        let interop = video_enc.nvenc_interop()
                            .expect("nvenc_interop: invariant — GpuNvenc arm");
                        let abgr10_texture = interop.abgr10_texture_for_slot(slot);
                        let abgr10_view =
                            abgr10_texture.create_view(&wgpu::TextureViewDescriptor::default());

                        graph.execute_with_callback(
                            &mut encoder,
                            device_ref,
                            &frame_state,
                            |enc, ctx| {
                                if ctx.contains(rtt_id) {
                                    let in_view = ctx.get(rtt_id).texture.create_view(
                                        &wgpu::TextureViewDescriptor::default(),
                                    );
                                    repack.record(enc, device_ref, &in_view, &abgr10_view, width, height);
                                } else {
                                    log::error!(
                                        "[export] frame {frame_idx}: FINAL_COLOR missing from RenderContext!"
                                    );
                                }
                            },
                        );
                    }

                    // Submit — non-blocking. GPU starts executing immediately.
                    // We do NOT stall here; we overlap with encoding the prior frame below.
                    let sid = self.device.submit(encoder);

                    // Non-blocking poll: lets the driver queue the work without a CPU stall.
                    self.device.device.poll(wgpu::Maintain::Poll);

                    // ── Encode: process the previously submitted frame ─────────
                    if let Some((prev_idx, prev_sid, prev_slot)) = pending.take() {
                        // Wait only for the specific submission that wrote prev_slot.
                        // The current frame's GPU work (sid) can still run concurrently.
                        self.device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(prev_sid));

                        let encode_result = if let ExportBackend::GpuNvenc { video_enc, muxer, .. } =
                            &mut self.backend
                        {
                            let prev_pts = self.job.frame_pts(prev_idx);
                            let interop = video_enc.nvenc_interop_mut()
                                .expect("nvenc_interop_mut: invariant — GpuNvenc arm");
                            let muxer_ref = &**muxer;
                            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                                if let Err(e) = muxer_ref.write_packet(pkt, true) {
                                    log::error!("[export] NVENC mux write_packet failed for frame {prev_idx}: {e:?}");
                                }
                            };
                            interop.encode_frame(prev_pts, prev_slot)
                                .map_err(|e| {
                                    log::error!("[export] encode_frame failed for frame {prev_idx}: {e:?}");
                                    RenderError::GpuTimeout
                                })
                                .map(|(bytes, frame_pts)| {
                                    if !bytes.is_empty() {
                                        // Wrap compressed bytes in a shim AVPacket for the muxer.
                                        unsafe {
                                            use crate::io::ffi::avutil::{av_packet_alloc, av_packet_free};
                                            let pkt = av_packet_alloc();
                                            if !pkt.is_null() {
                                                (*pkt).data     = bytes.as_ptr() as *mut u8;
                                                (*pkt).size     = bytes.len() as i32;
                                                (*pkt).pts      = frame_pts;
                                                (*pkt).dts      = frame_pts;
                                                (*pkt).duration = 0;
                                                sink(pkt);
                                                (*pkt).data = std::ptr::null_mut();
                                                (*pkt).size = 0;
                                                av_packet_free(&mut (pkt as *mut _));
                                            }
                                        }
                                    }
                                })
                        } else {
                            Ok(())
                        };

                        encode_result?;

                        self.frames_done += 1;
                        progress.report(self.frames_done, ExportPhase::Rendering);
                    }

                    // Store this frame as the next pending encode, then advance slot.
                    pending = Some((frame_idx, sid, active_slot));
                    active_slot = 1 - active_slot;

                } else {
                    // ── Drain: encode the last pending frame ──────────────────
                    if let Some((prev_idx, prev_sid, prev_slot)) = pending.take() {
                        // Wait for the final GPU submission to complete before encoding.
                        self.device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(prev_sid));

                        if let ExportBackend::GpuNvenc { video_enc, muxer, .. } = &mut self.backend {
                            let prev_pts = self.job.frame_pts(prev_idx);
                            let interop  = video_enc.nvenc_interop_mut()
                                .expect("nvenc_interop_mut: invariant — GpuNvenc arm");
                            let muxer_ref = &**muxer;
                            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                                if let Err(e) = muxer_ref.write_packet(pkt, true) {
                                    log::error!("[export] NVENC mux write_packet (drain) failed for frame {prev_idx}: {e:?}");
                                }
                            };
                            match interop.encode_frame(prev_pts, prev_slot) {
                                Ok((bytes, frame_pts)) if !bytes.is_empty() => {
                                    unsafe {
                                        use crate::io::ffi::avutil::{av_packet_alloc, av_packet_free};
                                        let pkt = av_packet_alloc();
                                        if !pkt.is_null() {
                                            (*pkt).data     = bytes.as_ptr() as *mut u8;
                                            (*pkt).size     = bytes.len() as i32;
                                            (*pkt).pts      = frame_pts;
                                            (*pkt).dts      = frame_pts;
                                            (*pkt).duration = 0;
                                            sink(pkt);
                                            (*pkt).data = std::ptr::null_mut();
                                            (*pkt).size = 0;
                                            av_packet_free(&mut (pkt as *mut _));
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("[export] encode_frame failed for frame {prev_idx} (drain): {e:?}");
                                    return Err(RenderError::GpuTimeout);
                                }
                                Ok(_) => {} // empty bitstream (e.g. B-frame delay) — not an error
                            }
                        }

                        self.frames_done += 1;
                        progress.report(self.frames_done, ExportPhase::Rendering);
                    }
                }
            }

            log::info!(
                "[export] render_segment {} done (NVENC pipelined)",
                segment.index
            );
        }

        Ok(())
    }
}

pub struct RawFrame {
    pub frame_index: usize,
    pub pts: i64,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub enum RenderError {
    GraphOutput,
    ReadbackInit(String),
    ScheduleFailure,
    GpuTimeout,
}
