// src/scheduler/frame_scheduler.rs

use std::sync::Arc;
use rayon::prelude::*;
use crate::timeline::store::TimelineStore;
use crate::timeline::source::SourceRegistry;
use crate::timeline::query::{query_active, ActiveClip};
use crate::scheduler::island::{build_islands, Island};
use crate::io::io_layer::IoLayer;
use crate::render::frame_state::{FrameState, ClipRenderEntry};

pub struct FrameScheduler {
    timeline:    Arc<std::sync::RwLock<TimelineStore>>,
    source_reg:  Arc<SourceRegistry>,
    io_layer:    Arc<IoLayer>,
    canvas_w:    u32,
    canvas_h:    u32,
}

impl FrameScheduler {
    pub fn new(
        timeline:   Arc<std::sync::RwLock<TimelineStore>>,
        source_reg: Arc<SourceRegistry>,
        io_layer:   Arc<IoLayer>,
        canvas_w:   u32,
        canvas_h:   u32,
    ) -> Self {
        Self {
            timeline,
            source_reg,
            io_layer,
            canvas_w,
            canvas_h,
        }
    }

    pub fn schedule_frame(&self, pts: i64) -> FrameState {
        // Step 1 — Query active clips
        let store = self.timeline.read().unwrap();
        let mut active: Vec<ActiveClip> = Vec::with_capacity(32);
        query_active(&store, pts, &mut active);

        // Step 2 — Build islands
        let islands = build_islands(&store, &self.source_reg, &active, pts);
        
        // Explicitly drop store read-lock before Rayon dispatch
        drop(store);

        // Step 3 — Parallel island processing with Rayon
        let entries: Vec<Vec<ClipRenderEntry>> = islands
            .par_iter()
            .map(|island| self.process_island(island))
            .collect();

        // Step 4 — Flatten and sort by layer_order
        let mut all: Vec<ClipRenderEntry> = entries.into_iter().flatten().collect();
        all.sort_unstable_by_key(|e| e.layer_order);

        // Step 5 — Build FrameState
        FrameState {
            pts,
            canvas_width: self.canvas_w,
            canvas_height: self.canvas_h,
            clips: all,
            test_textures: vec![],
        }
    }

    fn process_island(&self, island: &Island) -> Vec<ClipRenderEntry> {
        let mut entries = Vec::with_capacity(island.clips.len());

        for clip in &island.clips {
            let slot = self.io_layer.get_or_decode(clip.source_id, clip.source_pts);
            
            // Cache miss (latency policy: skip this clip for this frame)
            if slot.is_none() {
                continue;
            }

            // Cache hit
            entries.push(ClipRenderEntry {
                texture_slot: slot.unwrap().index() as u32,
                layer_order:  clip.layer_order,
                clip_width:   clip.clip_width,
                clip_height:  clip.clip_height,
                transform:    clip.transform,
                opacity:      clip.opacity,
            });
        }

        entries
    }
}
