use crate::io::io_layer::IoLayer;
use crate::render::frame_state::{ClipRenderEntry, FrameState};
use crate::scheduler::island::{build_islands, Island};
use crate::timeline::query::{query_active, ActiveClip};
use crate::timeline::source::SourceRegistry;
use crate::timeline::store::TimelineStore;
use crate::timeline::track::{TrackKind, TrackList};
use std::sync::Arc;

#[derive(Clone)]
pub struct FrameScheduler {
    io_layer: Arc<IoLayer>,
    canvas_w: u32,
    canvas_h: u32,
}

impl FrameScheduler {
    pub fn new(io_layer: Arc<IoLayer>, canvas_w: u32, canvas_h: u32) -> Self {
        Self {
            io_layer,
            canvas_w,
            canvas_h,
        }
    }

    pub fn io_layer(&self) -> &Arc<IoLayer> {
        &self.io_layer
    }

    pub fn schedule_frame(
        &self,
        pts: i64,
        store: &TimelineStore,
        tracks: &TrackList,
        source_reg: &SourceRegistry,
    ) -> FrameState {
        // Step 1 — Query active clips
        let mut active: Vec<ActiveClip> = Vec::with_capacity(32);
        query_active(store, pts, &mut active);

        // Step 2 — Filter out Audio-track clips — they must never enter the
        // GPU render pipeline (they have no visual content to composite).
        active.retain(|ac| {
            let track_id = store.track_id_at(ac.store_index);
            tracks
                .get(track_id)
                .is_none_or(|t| !matches!(t.kind, TrackKind::Audio { .. }))
        });

        // Step 3 — Build islands
        let islands = build_islands(store, source_reg, &active, pts);

        // Step 4 — Sequential island processing
        let entries: Vec<Vec<ClipRenderEntry>> = islands
            .iter()
            .map(|island| self.process_island(island))
            .collect();

        // Step 5 — Flatten and sort by layer_order
        let mut all: Vec<ClipRenderEntry> = entries.into_iter().flatten().collect();
        all.sort_by_key(|e| e.layer_order);

        // Step 6 — Build FrameState
        let imported = Self::interop_bindings(&all);
        FrameState {
            pts,
            canvas_width: self.canvas_w,
            canvas_height: self.canvas_h,
            clips: all,
            test_textures: vec![],
            imported,
        }
    }

    /// Which `ResourceId` a clip's interop luma plane is bound to.
    ///
    /// **The graph builder and the frame binding must agree on this number, and
    /// nothing else enforces it** — `ResourceBuilder::import` takes the id from the
    /// caller precisely so the reading node and the binder can share it. Deriving
    /// it from the clip's index in one place and choosing it in another is how they
    /// would drift, so both sides call these two functions.
    ///
    /// The ids start after `FINAL_COLOR` (0) and `SCREEN` (1) and are packed two
    /// per clip, which keeps them out of the range a builder allocates with
    /// `ResourceId::next` from a counter it starts at `2 + 2 * clips.len()`.
    pub fn interop_y_id(clip_index: usize) -> crate::render::resource::ResourceId {
        crate::render::resource::ResourceId(2 + 2 * clip_index as u32)
    }

    /// Which `ResourceId` a clip's interop chroma plane is bound to. See
    /// [`Self::interop_y_id`].
    pub fn interop_uv_id(clip_index: usize) -> crate::render::resource::ResourceId {
        crate::render::resource::ResourceId(3 + 2 * clip_index as u32)
    }

    /// The first `ResourceId` a graph builder may allocate for its own
    /// intermediates, given this frame's clips.
    ///
    /// Reserves two ids per clip whether or not that clip is on the interop path:
    /// a counter that skipped the CPU-path clips would shift every later clip's
    /// ids when one source fell back mid-timeline, and the ids a node holds must
    /// match the ones the frame bound.
    pub fn interop_id_counter_start(clip_count: usize) -> u32 {
        2 + 2 * clip_count as u32
    }

    /// Bind every interop clip's planes into an [`ImportedResources`] for this
    /// frame.
    ///
    /// Empty when no clip is on the interop path, which is exactly what the CPU
    /// upload path means (gotcha 18) — so a frame with no GPU decode is
    /// byte-for-byte what it was before G2c.
    fn interop_bindings(clips: &[ClipRenderEntry]) -> crate::render::resource::ImportedResources {
        let mut imported = crate::render::resource::ImportedResources::new();
        for (i, clip) in clips.iter().enumerate() {
            if let Some(planes) = &clip.interop_planes {
                imported.bind(Self::interop_y_id(i), planes.y.clone());
                imported.bind(Self::interop_uv_id(i), planes.uv.clone());
            }
        }
        imported
    }

    /// Cache-only variant of `schedule_frame` for use during export rendering.
    ///
    /// Instead of calling `decode_blocking` (which blocks for ~30–60ms per
    /// 1080p frame), this variant spins with exponential backoff waiting for
    /// the dedicated `ExportDecodeWorker` threads to populate the cache.
    ///
    /// If the cache remains empty after `max_wait_ms`, falls back to blocking
    /// decode as a last resort (same behaviour as `schedule_frame`).
    pub fn schedule_frame_cached(
        &self,
        pts: i64,
        store: &TimelineStore,
        tracks: &TrackList,
        source_reg: &SourceRegistry,
        max_wait_ms: u64,
    ) -> FrameState {
        let mut active: Vec<ActiveClip> = Vec::with_capacity(32);
        query_active(store, pts, &mut active);

        // Filter out Audio-track clips (same as schedule_frame)
        active.retain(|ac| {
            let track_id = store.track_id_at(ac.store_index);
            tracks
                .get(track_id)
                .is_none_or(|t| !matches!(t.kind, TrackKind::Audio { .. }))
        });

        let islands = build_islands(store, source_reg, &active, pts);

        let entries: Vec<Vec<ClipRenderEntry>> = islands
            .iter()
            .map(|island| self.process_island_cached(island, max_wait_ms))
            .collect();

        let mut all: Vec<ClipRenderEntry> = entries.into_iter().flatten().collect();
        all.sort_by_key(|e| e.layer_order);

        let imported = Self::interop_bindings(&all);
        FrameState {
            pts,
            canvas_width: self.canvas_w,
            canvas_height: self.canvas_h,
            clips: all,
            test_textures: vec![],
            imported,
        }
    }

    fn process_island_cached(&self, island: &Island, max_wait_ms: u64) -> Vec<ClipRenderEntry> {
        let mut entries = Vec::with_capacity(island.clips.len());

        for clip in &island.clips {
            let (fps, is_vfr, time_base, is_still_image) = {
                let source_reg = self.io_layer.source_reg.read().unwrap();
                let is_still_image = source_reg
                    .path(clip.source_id)
                    .map(|path| crate::timeline::source::is_still_image_path(path.as_ref()))
                    .unwrap_or(false);
                let (fps, is_vfr, time_base) = source_reg
                    .video_info(clip.source_id)
                    .map(|info| (info.frame_rate, info.is_vfr, info.time_base))
                    .unwrap_or((
                        crate::timeline::rational::Rational { num: 30, den: 1 },
                        false,
                        crate::timeline::rational::Rational {
                            num: 1,
                            den: 90_000,
                        },
                    ));
                (fps, is_vfr, time_base, is_still_image)
            };

            let project_tb = crate::timeline::rational::Rational {
                num: 1,
                den: 90_000,
            };

            let quantized_pts = if is_still_image || fps.num == 0 {
                0
            } else if is_vfr {
                let stream_pts = project_tb.rescale_pts(clip.source_pts, time_base);
                let stream_frame_duration =
                    time_base.den * fps.den / (time_base.num * fps.num);
                let quantized_stream_pts = if stream_frame_duration > 0 {
                    (stream_pts / stream_frame_duration) * stream_frame_duration
                } else {
                    stream_pts
                };
                time_base.rescale_pts(quantized_stream_pts, project_tb)
            } else {
                let frame_duration = 90_000 * fps.den / fps.num;
                (clip.source_pts / frame_duration) * frame_duration
            };

            // Spin-wait for the decode worker with exponential backoff.
            //
            // G2c — the interop path is tried FIRST, before the wait: the decode
            // workers populate the CPU `FrameCache`, and a source decoding into its
            // own textures has no cache to wait for. `decode_interop` returns `None`
            // for every source that cannot use it, so the wait below is unchanged
            // for those.
            if let Some(interop) =
                self.io_layer.decode_interop(clip.source_id, quantized_pts)
            {
                entries.push(ClipRenderEntry {
                    source_id: clip.source_id,
                    texture_slot: 0,
                    layer_order: clip.layer_order,
                    clip_width: interop.width,
                    clip_height: interop.height,
                    transform: clip.transform,
                    opacity: clip.opacity,
                    blend_mode: clip.blend_mode,
                    crop: clip.crop,
                    corner_pin: clip.corner_pin,
                    matte_mode: clip.matte_mode,
                    effects: clip.effects,
                    frame_meta: interop.meta,
                    kind: clip.kind.clone(),
                    interop_planes: Some(interop.planes),
                });
                continue;
            }

            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(max_wait_ms);
            let mut sleep_us = 100u64;
            let slot_info = loop {
                if let Some(s) = self.io_layer.cache.touch(clip.source_id, quantized_pts) {
                    break Some(s);
                }
                if std::time::Instant::now() >= deadline {
                    // Timeout — fall back to blocking decode as last resort.
                    break self.io_layer.decode_blocking(clip.source_id, quantized_pts);
                }
                std::thread::sleep(std::time::Duration::from_micros(sleep_us));
                sleep_us = (sleep_us * 2).min(4_000);
            };
            // If this is a still image and decode worker couldn't provide a
            // slot, still include a placeholder entry — UI will handle the
            // still-image upload path. Otherwise, require a valid slot.
            let is_text = matches!(clip.kind, crate::timeline::store::ClipKind::Text { .. });
            if (is_still_image || is_text) && slot_info.is_none() {
                entries.push(ClipRenderEntry {
                    source_id: clip.source_id,
                    texture_slot: 0,
                    layer_order: clip.layer_order,
                    clip_width: clip.clip_width,
                    clip_height: clip.clip_height,
                    transform: clip.transform,
                    opacity: clip.opacity,
                    blend_mode: clip.blend_mode,
                    crop: clip.crop,
                    corner_pin: clip.corner_pin,
                    matte_mode: clip.matte_mode,
                    effects: clip.effects,
                    // Still images and text bypass the YUV path entirely (their
                    // upload node writes RGBA directly), so this metadata is never
                    // consulted; sRGB full-range is the honest description of what
                    // those pixels are.
                    frame_meta: still_image_frame_meta(),
                    kind: clip.kind.clone(),
                    interop_planes: None,
                });
                continue;
            }

            let (slot, frame_meta) = match slot_info {
                Some(s) => s,
                None => continue,
            };

            let packed_slot = ((slot.tier as u32) << 16) | (slot.index as u32);
            entries.push(ClipRenderEntry {
                source_id: clip.source_id,
                texture_slot: packed_slot,
                layer_order: clip.layer_order,
                clip_width: clip.clip_width,
                clip_height: clip.clip_height,
                transform: clip.transform,
                opacity: clip.opacity,
                blend_mode: clip.blend_mode,
                crop: clip.crop,
                corner_pin: clip.corner_pin,
                matte_mode: clip.matte_mode,
                effects: clip.effects,
                frame_meta,
                kind: clip.kind.clone(),
                interop_planes: None,
            });
        }

        entries
    }

    fn process_island(&self, island: &Island) -> Vec<ClipRenderEntry> {
        let mut entries = Vec::with_capacity(island.clips.len());

        for clip in &island.clips {
            // Get framerate to quantize source_pts
            let (fps, is_vfr, time_base, is_still_image) = {
                let source_reg = self.io_layer.source_reg.read().unwrap();
                let is_still_image = source_reg
                    .path(clip.source_id)
                    .map(|path| crate::timeline::source::is_still_image_path(path.as_ref()))
                    .unwrap_or(false);
                let (fps, is_vfr, time_base) = source_reg
                    .video_info(clip.source_id)
                    .map(|info| (info.frame_rate, info.is_vfr, info.time_base))
                    .unwrap_or((
                        crate::timeline::rational::Rational { num: 30, den: 1 },
                        false,
                        crate::timeline::rational::Rational {
                            num: 1,
                            den: 90_000,
                        },
                    ));
                (fps, is_vfr, time_base, is_still_image)
            };

            let project_tb = crate::timeline::rational::Rational {
                num: 1,
                den: 90_000,
            };

            let quantized_pts = if is_still_image || fps.num == 0 {
                0 // For images or unknown, always ask for frame 0
            } else if is_vfr {
                let stream_pts = project_tb.rescale_pts(clip.source_pts, time_base);
                let stream_frame_duration =
                    time_base.den * fps.den / (time_base.num * fps.num);
                let quantized_stream_pts = if stream_frame_duration > 0 {
                    (stream_pts / stream_frame_duration) * stream_frame_duration
                } else {
                    stream_pts
                };
                time_base.rescale_pts(quantized_stream_pts, project_tb)
            } else {
                let frame_duration = 90_000 * fps.den / fps.num;
                (clip.source_pts / frame_duration) * frame_duration
            };

            let is_text = matches!(clip.kind, crate::timeline::store::ClipKind::Text { .. });
            if is_still_image || is_text {
                // For still images, don't attempt to decode via the video slot pool.
                // Still-image upload is handled on the UI side (StillImageUploadNode),
                // so include a placeholder entry with texture_slot=0. This prevents
                // the scheduler from dropping image clips when the slot pool has
                // no decode entry for them.
                entries.push(ClipRenderEntry {
                    source_id: clip.source_id,
                    texture_slot: 0,
                    layer_order: clip.layer_order,
                    clip_width: clip.clip_width,
                    clip_height: clip.clip_height,
                    transform: clip.transform,
                    opacity: clip.opacity,
                    blend_mode: clip.blend_mode,
                    crop: clip.crop,
                    corner_pin: clip.corner_pin,
                    matte_mode: clip.matte_mode,
                    effects: clip.effects,
                    frame_meta: still_image_frame_meta(),
                    kind: clip.kind.clone(),
                    // Still images and text never touch the YUV path at all.
                    interop_planes: None,
                });
                continue;
            }

            // G2c — GPU decode first: NVDEC writes into this source's own Y/UV
            // textures and the frame never crosses PCIe. `None` covers every
            // reason it cannot (no CUDA, no NVDEC, >8-bit, a failed copy), and
            // falls through to the CPU path below unchanged.
            if let Some(interop) = self.io_layer.decode_interop(clip.source_id, quantized_pts) {
                entries.push(ClipRenderEntry {
                    source_id: clip.source_id,
                    // Meaningless on this path and deliberately left at 0: the
                    // pixels are in `interop_planes`, and a plausible-looking slot
                    // index would invite an upload that reads an unrelated buffer.
                    texture_slot: 0,
                    layer_order: clip.layer_order,
                    // The frame's own dimensions, not the container's — the same
                    // reason `frame_meta` comes from the decoder (gotcha 11).
                    clip_width: interop.width,
                    clip_height: interop.height,
                    transform: clip.transform,
                    opacity: clip.opacity,
                    blend_mode: clip.blend_mode,
                    crop: clip.crop,
                    corner_pin: clip.corner_pin,
                    matte_mode: clip.matte_mode,
                    effects: clip.effects,
                    frame_meta: interop.meta,
                    kind: clip.kind.clone(),
                    interop_planes: Some(interop.planes),
                });
                continue;
            }

            // Try cache first (fast path); fall back to blocking decode.
            let slot_info = self
                .io_layer
                .cache
                .touch(clip.source_id, quantized_pts)
                .or_else(|| self.io_layer.decode_blocking(clip.source_id, quantized_pts));

            let (slot, frame_meta) = match slot_info {
                Some(s) => s,
                None => continue, // source not importable / pool full
            };

            let packed_slot = ((slot.tier as u32) << 16) | (slot.index as u32);
            entries.push(ClipRenderEntry {
                source_id: clip.source_id,
                texture_slot: packed_slot,
                layer_order: clip.layer_order,
                clip_width: clip.clip_width,
                clip_height: clip.clip_height,
                transform: clip.transform,
                opacity: clip.opacity,
                blend_mode: clip.blend_mode,
                crop: clip.crop,
                corner_pin: clip.corner_pin,
                matte_mode: clip.matte_mode,
                effects: clip.effects,
                frame_meta,
                kind: clip.kind.clone(),
                interop_planes: None,
            });
        }

        entries
    }
}

/// Frame metadata for clips that never go through the YUV path: still images and
/// text, whose upload nodes write RGBA(16F) straight into the composite input.
///
/// Describing them as sRGB full-range is accurate and keeps `frame_meta` a
/// meaningful field everywhere rather than a placeholder some entries lie about.
fn still_image_frame_meta() -> crate::timeline::source::DecodedFrameMeta {
    crate::timeline::source::DecodedFrameMeta {
        layout: crate::timeline::source::FrameLayout::YUV420P8,
        color:  crate::timeline::source::ColorInfo::srgb(),
    }
}
