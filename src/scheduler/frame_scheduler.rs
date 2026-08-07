use std::sync::Arc;
use crate::timeline::store::TimelineStore;
use crate::timeline::source::SourceRegistry;
use crate::timeline::query::{query_active, ActiveClip};
use crate::scheduler::island::{build_islands, Island};
use crate::io::io_layer::IoLayer;
use crate::render::frame_state::{FrameState, ClipRenderEntry};

#[derive(Clone)]
pub struct FrameScheduler {
    io_layer:    Arc<IoLayer>,
    canvas_w:    u32,
    canvas_h:    u32,
}

impl FrameScheduler {
    pub fn new(
        io_layer:   Arc<IoLayer>,
        canvas_w:   u32,
        canvas_h:   u32,
    ) -> Self {
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
        source_reg: &SourceRegistry
    ) -> FrameState {
        // Step 1 — Query active clips
        let mut active: Vec<ActiveClip> = Vec::with_capacity(32);
        query_active(store, pts, &mut active);

        // Step 2 — Build islands
        let islands = build_islands(store, source_reg, &active, pts);

        // Step 3 — Sequential island processing (decode_blocking is I/O and
        // would stall Rayon workers if run in parallel)
        let entries: Vec<Vec<ClipRenderEntry>> = islands
            .iter()
            .map(|island| self.process_island(island))
            .collect();

        // Step 4 — Flatten and sort by layer_order
        let mut all: Vec<ClipRenderEntry> = entries.into_iter().flatten().collect();
        all.sort_by_key(|e| e.layer_order);

        // Step 5 — Build FrameState
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
        pts:         i64,
        store:       &TimelineStore,
        source_reg:  &SourceRegistry,
        max_wait_ms: u64,
    ) -> FrameState {
        let mut active: Vec<ActiveClip> = Vec::with_capacity(32);
        query_active(store, pts, &mut active);

        let islands = build_islands(store, source_reg, &active, pts);

        let entries: Vec<Vec<ClipRenderEntry>> = islands
            .iter()
            .map(|island| self.process_island_cached(island, max_wait_ms))
            .collect();

        let mut all: Vec<ClipRenderEntry> = entries.into_iter().flatten().collect();
        all.sort_by_key(|e| e.layer_order);

        FrameState {
            pts,
            canvas_width:  self.canvas_w,
            canvas_height: self.canvas_h,
            clips: all,
            test_textures: vec![],
        }
    }

    fn process_island_cached(&self, island: &Island, max_wait_ms: u64) -> Vec<ClipRenderEntry> {
        let mut entries = Vec::with_capacity(island.clips.len());

        for clip in &island.clips {
            let fps = self.io_layer.source_reg.read().unwrap().video_info(clip.source_id)
                .map(|info| info.frame_rate)
                .unwrap_or(crate::timeline::rational::Rational { num: 30, den: 1 });

            let quantized_pts = if fps.num == 0 {
                0
            } else {
                let frame_duration = 90_000 * (fps.den as i64) / (fps.num as i64);
                (clip.source_pts / frame_duration) * frame_duration
            };

            // Spin-wait for the decode worker with exponential backoff.
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(max_wait_ms);
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

            let (slot, is_nv12) = match slot_info {
                Some(s) => s,
                None    => continue,
            };

            let packed_slot = ((slot.tier as u32) << 16) | (slot.index as u32);
            entries.push(ClipRenderEntry {
                texture_slot: packed_slot,
                layer_order:  clip.layer_order,
                clip_width:   clip.clip_width,
                clip_height:  clip.clip_height,
                transform:    clip.transform,
                opacity:      clip.opacity,
                is_nv12,
            });
        }

        entries
    }

    fn process_island(&self, island: &Island) -> Vec<ClipRenderEntry> {
        let mut entries = Vec::with_capacity(island.clips.len());

        for clip in &island.clips {
            // Get framerate to quantize source_pts
            let fps = self.io_layer.source_reg.read().unwrap().video_info(clip.source_id)
                .map(|info| info.frame_rate)
                .unwrap_or(crate::timeline::rational::Rational { num: 30, den: 1 });
            
            let quantized_pts = if fps.num == 0 {
                0 // For images or unknown, always ask for frame 0
            } else {
                let frame_duration = 90_000 * (fps.den as i64) / (fps.num as i64);
                (clip.source_pts / frame_duration) * frame_duration
            };

            // Try cache first (fast path); fall back to blocking decode.
            let slot_info = self.io_layer.cache.touch(clip.source_id, quantized_pts)
                .or_else(|| self.io_layer.decode_blocking(clip.source_id, quantized_pts));

            let (slot, is_nv12) = match slot_info {
                Some(s) => s,
                None    => continue,  // source not importable / pool full
            };

            let packed_slot = ((slot.tier as u32) << 16) | (slot.index as u32);
            entries.push(ClipRenderEntry {
                texture_slot: packed_slot,
                layer_order:  clip.layer_order,
                clip_width:   clip.clip_width,
                clip_height:  clip.clip_height,
                transform:    clip.transform,
                opacity:      clip.opacity,
                is_nv12,
            });
        }

        entries
    }
}
