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
        FrameState {
            pts,
            canvas_width: self.canvas_w,
            canvas_height: self.canvas_h,
            clips: all,
            test_textures: vec![],
        }
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

        FrameState {
            pts,
            canvas_width: self.canvas_w,
            canvas_height: self.canvas_h,
            clips: all,
            test_textures: vec![],
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
