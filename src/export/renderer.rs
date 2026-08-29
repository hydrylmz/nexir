// src/export/renderer.rs

use crate::export::job::ExportJob;
use crate::export::muxer::Muxer;
use crate::export::partitioner::ExportSegment;
use crate::export::progress::{ExportPhase, ProgressSender};
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::readback::FrameReadback;
use crate::export::video_encoder::VideoEncoderBackend;
use crate::interop::nv12_encode::Nv12EncodeNode;
use crate::render::compute::ComputePipelineCache;
use crate::render::device::GpuDevice;
use crate::render::frame_state::FrameState;
use crate::render::graph::{CompiledGraph, RenderGraphCompiler};
use crate::render::nodes::composite::CompositeNode;
use crate::render::nodes::color_correction::{ColorCorrectionNode, ColorCorrectionParams};
use crate::render::nodes::chroma_key::{ChromaKeyNode, ChromaKeyParams};
use crate::render::nodes::gaussian_blur::{BlurPassNode, BlurParams};
use crate::render::nodes::sharpen::{SharpenNode, SharpenParams};
use crate::render::nodes::vignette::{VignetteNode, VignetteParams};
use crate::render::nodes::yuv_to_rgb::YuvToRgbNode;
use crate::render::nodes::yuv_upload::YuvUploadNode;
use crate::render::nodes::tonemap::{ToneMapNode, ToneMapPushConstants, InputTransferFn, GamutConversion, ToneMapMode};
use crate::render::resource::ResourceId;
use crate::render::shader::registry::ShaderRegistry;
use crate::render::still_image::StillImageCache;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::timeline::source::{DecodedFrameMeta, SourceRegistry, is_still_image_path};
use crate::timeline::ids::SourceId;
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
        /// RGB → NV12 conversion, writing straight into the NVENC input buffer.
        ///
        /// Replaced `Abgr10RepackNode` in P1.9 step 3: converting here rather than
        /// handing NVENC packed RGB is what removes the driver's BT.601-only
        /// RGB→YUV step, and therefore what lets a BT.709 job use zero-copy at all
        /// (see `ExportJob::nvenc_zero_copy_is_colour_safe`).
        nv12: Nv12EncodeNode,
    },
}

/// Minimum binding-array size passed to CompositeNode — avoids creating
/// a zero-slot layout when the frame has no clips.
const MIN_COMPOSITE_SLOTS: u32 = 1;

/// Describes the set of active clips for a frame.  When this changes between
/// frames the render graph must be recompiled (different clip dimensions →
/// different texture sizes → different pipeline bind groups).
#[derive(Clone, PartialEq, Debug)]
struct ClipSignature {
    source_id: SourceId,
    clip_width: u32,
    clip_height: u32,
    /// Pixel layout and colour of the frame currently in this clip's slot.
    ///
    /// P1.6 — the signature keys off the DECODED metadata, not the container's,
    /// and includes it in full.  Both halves matter: a clip whose colour or bit
    /// depth changes mid-timeline (a source with per-frame metadata, or a decoder
    /// that switched to a swscale fallback) needs a recompile, because
    /// `YuvToRgbNode` bakes the conversion in at construction time.
    frame_meta: DecodedFrameMeta,
    /// True when this clip is a still image (PNG/JPEG/etc).  Still images use
    /// `StillImageUploadNode` instead of `YuvUploadNode → YuvToRgbNode`, so a
    /// change in this flag must trigger graph recompilation.
    is_still_image: bool,
    effects: crate::timeline::transform::ClipEffects,
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
                    source_id: c.source_id,
                    clip_width: c.clip_width,
                    clip_height: c.clip_height,
                    // Straight from the decoder — see the field's comment for why
                    // this is not read from `sources.video_info()`.
                    frame_meta: c.frame_meta,
                    is_still_image: is_still,
                    effects: c.effects,
                }
            })
            .collect()
    }

    /// Compile (or reuse the cached) render graph for the active clips in `frame`.
    ///
    /// Recompilation happens when the active clips, their dimensions, their
    /// still-image flags, or their effect settings change. Within a stable
    /// clip span the graph is reused every frame.
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
                        // A still image is sRGB full-range Rec.709.  For an SDR job
                        // that needs no transform at all; for an HDR job it needs
                        // the same 709→2020 + PQ encode any SDR clip gets, or it
                        // would be read as PQ code values and come out near-black.
                        let transformed_id = self.add_color_transform(
                            &mut compiler,
                            rgba_id,
                            &crate::timeline::source::ColorInfo::srgb(),
                            clip.clip_width,
                            clip.clip_height,
                            &mut id_counter,
                        );
                        let final_id = self.add_effect_chain(&mut compiler, transformed_id, clip, &mut id_counter);
                        comp_node.input_textures.push(final_id);
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

            // P1.6 — layout and colour both come from the frame the decoder
            // actually produced.  Reading them off the source registry instead was
            // wrong whenever the decoder converted the frame (swscale fallback) or
            // the stream header disagreed with the frame's own metadata.
            let layout     = clip_sig.frame_meta.layout;
            let color_info = clip_sig.frame_meta.color;

            // Create YuvUpload with the clip's ACTUAL dimensions so the GPU
            // textures are exactly the right size — no wasted rows, no green fill.
            let upload_node = YuvUploadNode::new_with_layout(
                &self.device,
                slot as u32,
                clip.clip_width,
                clip.clip_height,
                y_id,
                uv_id,
                layout,
            );

            let node_idx = compiler.add_node(Box::new(upload_node));
            upload_indices.push(Some(node_idx));


            compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                y_id,
                uv_id,
                rgba_id,
                clip.clip_width,
                clip.clip_height,
                color_info,
                layout.semi_planar,
            )));

            // ── Colour transform into the output's space ──────────────────────
            //
            // P1.7 — this used to be an unconditional HDR->SDR tone-map, applied
            // only to HDR clips.  It is now driven by the JOB's target colour
            // space, which is what makes a real HDR export possible:
            //
            //   * SDR job, HDR clip  → decode PQ/HLG, BT.2020→709, ACES tone-map,
            //                          re-encode sRGB.  (What it always did, minus
            //                          the bug of writing linear light out.)
            //   * HDR job, HDR clip  → line the transfer function and primaries up
            //                          with the output's, highlights intact.
            //   * HDR job, SDR clip  → decode sRGB, BT.709→2020, re-encode PQ, so
            //                          the clip sits at its correct diffuse
            //                          brightness in the HDR file rather than
            //                          being stretched to peak white.
            //   * SDR job, SDR clip  → no node at all.
            let final_rgba_id = self.add_color_transform(
                &mut compiler,
                rgba_id,
                &color_info,
                clip.clip_width,
                clip.clip_height,
                &mut id_counter,
            );

            let final_rgba_id = self.add_effect_chain(&mut compiler, final_rgba_id, clip, &mut id_counter);
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

    /// Insert whatever colour-space conversion this clip needs to land in the
    /// job's output colour space, returning the resource holding the result.
    ///
    /// Returns `input` unchanged when nothing is needed — an SDR clip in an SDR
    /// job, which is the overwhelmingly common case, adds no node and costs no
    /// pass.
    ///
    /// P1.7 — the previous logic here only ever went one direction (HDR clip →
    /// SDR output) and was keyed off the clip alone, so the job's own target
    /// colour space had no influence on the pixels at all. That is what made an
    /// `output_color = bt2020(true, 10)` export a file with HDR tags over SDR
    /// pixels. Both the source and the destination are now consulted.
    fn add_color_transform(
        &self,
        compiler:    &mut RenderGraphCompiler,
        input:       ResourceId,
        clip_color:  &crate::timeline::source::ColorInfo,
        width:       u32,
        height:      u32,
        id_counter:  &mut u32,
    ) -> ResourceId {
        use crate::timeline::source::TransferFunction;

        let job_is_hdr  = self.job.is_hdr();
        let clip_is_hdr = matches!(
            clip_color.transfer_fn,
            TransferFunction::Pq | TransferFunction::Hlg
        );

        let src_primaries = clip_color.effective_primaries(width, height);
        let dst_primaries = self.job.output_color.effective_primaries(
            self.job.width, self.job.height,
        );
        let gamut = GamutConversion::between(src_primaries, dst_primaries);
        let src_trc = InputTransferFn::from_color_info(clip_color);

        // Nothing to do: SDR in, SDR out, same primaries.  Skip the pass entirely
        // rather than running an identity decode/encode round trip, which would
        // cost a full-frame dispatch and lose a little precision to the two
        // transfer-function conversions.
        if !job_is_hdr && !clip_is_hdr && gamut == GamutConversion::None {
            return input;
        }

        let output_id = ResourceId::next(id_counter);

        let params = if job_is_hdr {
            // HDR target: match the output's curve and keep the highlights.
            let dst_trc = match self.job.output_color.transfer_fn {
                TransferFunction::Hlg => InputTransferFn::Hlg,
                // Everything else on an HDR job is PQ — `ExportJob::is_hdr` only
                // returns true for PQ or HLG.
                _                     => InputTransferFn::Pq,
            };
            let peak = self
                .job
                .hdr10
                .as_ref()
                .map(|h| h.peak_nits())
                .unwrap_or(1000.0);
            log::info!(
                "[export] colour transform: {src_trc:?}/{src_primaries:?} → \
                 {dst_trc:?}/{dst_primaries:?} (HDR passthrough, peak {peak} nits)"
            );
            ToneMapPushConstants::for_hdr_output(
                src_trc, gamut, dst_trc, peak, width, height,
            )
        } else {
            // SDR target: tone-map down to display-referred sRGB.
            log::info!(
                "[export] colour transform: {src_trc:?}/{src_primaries:?} → \
                 SDR sRGB/{dst_primaries:?} (ACES tone-map)"
            );
            ToneMapPushConstants::for_sdr_preview(
                src_trc,
                gamut,
                ToneMapMode::AcesFilmic,
                1000.0,
                width,
                height,
            )
        };

        compiler.add_node(Box::new(ToneMapNode::new(
            &self.device,
            &self.shaders,
            &self.compute_cache,
            input,
            output_id,
            params,
        )));
        output_id
    }

    fn add_effect_chain(
        &self,
        compiler: &mut RenderGraphCompiler,
        input: ResourceId,
        clip: &crate::render::frame_state::ClipRenderEntry,
        id_counter: &mut u32,
    ) -> ResourceId {
        let effects = clip.effects;
        let width = clip.clip_width;
        let height = clip.clip_height;
        let mut current = input;

        if effects.color_enabled {
            let output = ResourceId::next(id_counter);
            compiler.add_node(Box::new(ColorCorrectionNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                current,
                output,
                ColorCorrectionParams {
                    lift: [0.0; 4],
                    gamma: [1.0; 4],
                    gain: [1.0; 4],
                    saturation: effects.saturation,
                    brightness: effects.brightness,
                    contrast: effects.contrast,
                    hue_shift: effects.hue.to_radians(),
                    width,
                    height,
                    _pad0: 0.0,
                    _pad1: 0.0,
                },
            )));
            current = output;
        }

        if effects.blur_enabled {
            let horizontal = ResourceId::next(id_counter);
            compiler.add_node(Box::new(BlurPassNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                current,
                horizontal,
                BlurParams::horizontal(effects.blur_radius, effects.blur_sigma, width, height),
                "ExportBlurH",
            )));
            let vertical = ResourceId::next(id_counter);
            compiler.add_node(Box::new(BlurPassNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                horizontal,
                vertical,
                BlurParams::vertical(effects.blur_radius, effects.blur_sigma, width, height),
                "ExportBlurV",
            )));
            current = vertical;
        }

        if effects.sharpen_enabled {
            let output = ResourceId::next(id_counter);
            compiler.add_node(Box::new(SharpenNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                current,
                output,
                SharpenParams::new(effects.sharpen_amount, width, height),
            )));
            current = output;
        }

        if effects.vignette_enabled {
            let output = ResourceId::next(id_counter);
            compiler.add_node(Box::new(VignetteNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                current,
                output,
                VignetteParams {
                    intensity: effects.vignette_intensity,
                    radius: effects.vignette_radius,
                    softness: effects.vignette_softness,
                    roundness: effects.vignette_roundness,
                    center_x: 0.5,
                    center_y: 0.5,
                    width,
                    height,
                },
            )));
            current = output;
        }

        if effects.chroma_key_enabled {
            let output = ResourceId::next(id_counter);
            let [red, green, blue] = effects.chroma_key_color;
            let max = red.max(green).max(blue);
            let min = red.min(green).min(blue);
            let delta = max - min;
            let hue = if delta <= f32::EPSILON {
                0.0
            } else if (max - red).abs() <= f32::EPSILON {
                60.0 * ((green - blue) / delta).rem_euclid(6.0)
            } else if (max - green).abs() <= f32::EPSILON {
                60.0 * ((blue - red) / delta + 2.0)
            } else {
                60.0 * ((red - green) / delta + 4.0)
            };
            compiler.add_node(Box::new(ChromaKeyNode::new(
                &self.device,
                &self.shaders,
                &self.compute_cache,
                current,
                output,
                ChromaKeyParams {
                    key_hue: hue,
                    tolerance: effects.chroma_key_tolerance * 360.0,
                    softness: effects.chroma_key_softness * 360.0,
                    min_saturation: 0.15,
                    min_value: 0.08,
                    spill_suppress: 0.3,
                    width,
                    height,
                },
            )));
            current = output;
        }

        current
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
                    upload.upload_frame(
                        data,
                        clip.frame_meta.layout.semi_planar,
                        clip.clip_width,
                        clip.clip_height,
                    );
                });
            } else {
                log::warn!("[export] slot {slot_idx}: failed to downcast YuvUploadNode");
            }
        }
    }

    /// Hand a batch of NVENC bitstreams to the muxer, in the order NVENC produced
    /// them (decode order).  Called for every packet the encoder returns, wherever
    /// it came from — submission, slot reclaim, or the final drain.
    fn mux_nvenc_packets(
        &self,
        packets: Vec<crate::interop::encode_interop::EncodedPacket>,
        context: &str,
    ) -> Result<(), RenderError> {
        if packets.is_empty() {
            return Ok(());
        }
        let ExportBackend::GpuNvenc { muxer, .. } = &self.backend else {
            return Ok(());
        };
        let muxer_ref = &**muxer;
        for packet in &packets {
            if packet.bytes.is_empty() {
                continue;
            }
            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                if let Err(e) = muxer_ref.write_packet(pkt, true) {
                    log::error!("[export] NVENC mux write_packet failed ({context}): {e:?}");
                }
            };
            crate::export::video_encoder::write_nvenc_packet(packet, &mut sink).map_err(|e| {
                log::error!("[export] packet hand-off failed ({context}): {e:?}");
                RenderError::GpuTimeout
            })?;
        }
        Ok(())
    }

    /// Free one pipeline slot: block until NVENC has finished with every in-flight
    /// picture that still owns it, muxing whatever comes out.
    ///
    /// Must be called before recording GPU work that overwrites the slot's ABGR10
    /// texture — that is the resource-lifetime guarantee of the async pipeline.
    fn nvenc_reclaim_slot(&mut self, slot: usize) -> Result<(), RenderError> {
        let packets = match &mut self.backend {
            ExportBackend::GpuNvenc { video_enc, .. } => {
                let interop = video_enc
                    .nvenc_interop_mut()
                    .expect("nvenc_interop_mut: invariant — GpuNvenc arm");
                interop.reclaim_slot(slot).map_err(|e| {
                    log::error!("[export] reclaim_slot({slot}) failed: {e:?}");
                    RenderError::GpuTimeout
                })?
            }
            _ => Vec::new(),
        };
        self.mux_nvenc_packets(packets, "slot reclaim")
    }

    /// Submit one rendered frame to NVENC and mux anything that came back.
    ///
    /// Non-blocking with respect to *this* frame: it returns as soon as the driver
    /// has accepted the picture.  The packets it yields belong to earlier frames.
    fn nvenc_submit_frame(&mut self, frame_idx: usize, slot: usize) -> Result<(), RenderError> {
        let pts = self.job.frame_pts(frame_idx);
        let packets = match &mut self.backend {
            ExportBackend::GpuNvenc { video_enc, .. } => {
                let interop = video_enc
                    .nvenc_interop_mut()
                    .expect("nvenc_interop_mut: invariant — GpuNvenc arm");
                interop.encode_frame(pts, slot).map_err(|e| {
                    log::error!("[export] encode_frame failed for frame {frame_idx}: {e:?}");
                    RenderError::GpuTimeout
                })?
            }
            _ => Vec::new(),
        };
        self.mux_nvenc_packets(packets, "encode submit")
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
            // ── GPU / NVENC async pipelined path (P1.1) ─────────────────────────
            //
            // Three stages run concurrently, each on its own resources:
            //
            //   GPU:    F0 ── F1 ── F2 ── F3 ── F4 …      (renders into slot N % S)
            //   NVENC:       F0 ── F1 ── F2 ── F3 …       (encodes from slot N % S)
            //   Muxer:            P0 ── P1 ── P2 …        (writes packets, decode order)
            //
            // What used to serialise this was `encode_frame` waiting on the frame's
            // own completion event before returning, so the CPU could never get
            // ahead of the encoder.  Now submission is non-blocking and the only
            // waits are the two that correctness actually requires:
            //
            //   * `WaitForSubmissionIndex(sid)` before handing a slot to NVENC —
            //     the GPU must have finished writing that ABGR10 texture.
            //   * `reclaim_slot(slot)` before rendering into a slot again — NVENC
            //     must have finished reading it.  This is the backpressure: it
            //     blocks only when the encoder has fallen `S` frames behind.
            //
            // `S` = `EncodeInterop::slot_count()`.  Each slot owns its own interop
            // texture, registered resource, bitstream buffer and completion event,
            // so nothing is shared between frames in flight.
            let slot_count = match &self.backend {
                ExportBackend::GpuNvenc { video_enc, .. } => video_enc
                    .nvenc_interop()
                    .expect("nvenc_interop: invariant — GpuNvenc arm")
                    .slot_count(),
                _ => 1,
            };

            // How many GPU submissions may be outstanding without having been
            // handed to NVENC yet.  Keeping this strictly below `slot_count`
            // guarantees the slot about to be rendered into is never one still
            // waiting in `inflight` — those are always the 1..=GPU_LOOKAHEAD most
            // recent frames, i.e. different slots modulo `slot_count`.
            let gpu_lookahead = slot_count.saturating_sub(2).max(1);

            // (frame_idx, submission index, slot) for frames the GPU is rendering
            // or has rendered but that have not been submitted to NVENC yet.
            // Oldest first.
            let mut inflight: std::collections::VecDeque<(usize, wgpu::SubmissionIndex, usize)> =
                std::collections::VecDeque::with_capacity(gpu_lookahead + 1);

            for frame_idx in segment.frame_start..segment.frame_end {
                // ── Check cancellation & pause ───────────────────────────────
                if progress.control().is_cancelled() {
                    log::info!("[export] cancelled by user during GPU render at frame {frame_idx}");
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

                let slot = frame_idx % slot_count;

                // ── Backpressure: make sure NVENC is done with this slot ───────
                // No-op until the pipeline is full; after that it blocks on the
                // OLDEST in-flight picture, which is the correct thing to wait for.
                self.nvenc_reclaim_slot(slot)?;

                // ── Render: record and submit GPU work for this frame ──────────
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
                let device_ref = &self.device;

                let graph = self.cached_graph.as_mut().unwrap();

                if let ExportBackend::GpuNvenc { video_enc, nv12, .. } = &mut self.backend {
                    let interop = video_enc.nvenc_interop()
                        .expect("nvenc_interop: invariant — GpuNvenc arm");
                    // The pitch comes from the encoder, not from a second
                    // computation here: the shader must write at exactly the
                    // stride the NVENC registration declared, or the chroma plane
                    // lands where the driver does not read it.
                    let pitch = interop.pitch();
                    let nv12_buffer = interop.nv12_buffer_for_slot(slot);

                    graph.execute_with_callback(
                        &mut encoder,
                        device_ref,
                        &frame_state,
                        |enc, ctx| {
                            if ctx.contains(rtt_id) {
                                let in_view = ctx.get(rtt_id).texture.create_view(
                                    &wgpu::TextureViewDescriptor::default(),
                                );
                                nv12.record(enc, device_ref, &in_view, nv12_buffer, pitch);
                            } else {
                                log::error!(
                                    "[export] frame {frame_idx}: FINAL_COLOR missing from RenderContext!"
                                );
                            }
                        },
                    );
                }

                // Submit — non-blocking.  The GPU starts executing immediately and
                // this thread moves on.
                let sid = self.device.submit(encoder);

                // Non-blocking poll: lets the driver queue the work without a CPU stall.
                self.device.device.poll(wgpu::Maintain::Poll);

                inflight.push_back((frame_idx, sid, slot));

                // ── Submit finished renders to NVENC ──────────────────────────
                // Only once more than `gpu_lookahead` are outstanding, so the GPU
                // stays ahead of the encoder rather than being paced by it.
                while inflight.len() > gpu_lookahead {
                    let (done_idx, done_sid, done_slot) = inflight
                        .pop_front()
                        .expect("inflight is non-empty: len > gpu_lookahead >= 1");
                    // The one unavoidable GPU wait: NVENC must not read a texture
                    // the GPU is still writing.  It is the OLDEST submission, so by
                    // now it has usually completed already and this returns at once.
                    self.device
                        .device
                        .poll(wgpu::Maintain::WaitForSubmissionIndex(done_sid));
                    self.nvenc_submit_frame(done_idx, done_slot)?;

                    self.frames_done += 1;
                    progress.report(self.frames_done, ExportPhase::Rendering);
                }
            }

            // ── Drain: submit every remaining rendered frame ───────────────────
            // Their bitstreams are collected here or, for whatever NVENC is still
            // holding, by the EOS flush in `ExportEngine` after the last segment.
            while let Some((done_idx, done_sid, done_slot)) = inflight.pop_front() {
                self.device
                    .device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(done_sid));
                self.nvenc_submit_frame(done_idx, done_slot)?;

                self.frames_done += 1;
                progress.report(self.frames_done, ExportPhase::Rendering);
            }

            log::info!(
                "[export] render_segment {} done (NVENC async pipeline, {} slot(s), \
                 {} GPU frame(s) of lookahead)",
                segment.index, slot_count, gpu_lookahead
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
