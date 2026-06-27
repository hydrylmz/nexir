// src/scheduler/island.rs

use std::collections::HashMap;
use crate::timeline::ids::{TrackId, SourceId};
use crate::timeline::transform::ClipTransform;
use crate::timeline::store::TimelineStore;
use crate::timeline::source::SourceRegistry;
use crate::timeline::query::ActiveClip;

/// One clip's contribution within an island.
#[derive(Clone, Debug)]
pub struct IslandClip {
    pub store_index:  usize,
    pub source_id:    SourceId,
    pub source_pts:   i64,
    pub layer_order:  u16,
    pub opacity:      f32,
    pub transform:    ClipTransform,
    pub clip_width:   u32,
    pub clip_height:  u32,
    pub effect_start: u32,
    pub effect_count: u16,
}

/// A group of clips that can be decoded in parallel (one island per track).
#[derive(Clone)]
pub struct Island {
    pub track_id: TrackId,
    pub clips:    Vec<IslandClip>,
}

pub fn build_islands(
    store:          &TimelineStore,
    source_reg:     &SourceRegistry,
    active_indices: &[ActiveClip],
    query_pts:      i64,
) -> Vec<Island> {
    let mut island_map: HashMap<TrackId, Island> = HashMap::new();

    for active in active_indices {
        let idx = active.store_index;
        
        let track_id   = store.track_id_at(idx);
        let source_id  = store.source_id_at(idx);
        let pts_in     = store.pts_in_at(idx);
        let source_in  = store.source_in_at(idx);
        let source_pts = query_pts - pts_in + source_in;
        let transform  = store.transform_at(idx).clone();
        let opacity    = store.opacity_at(idx);
        let layer      = store.layer_order_at(idx);
        let (effect_start, effect_count) = store.effect_range_at(idx);
        
        let (clip_width, clip_height) = source_reg.video_info(source_id)
            .map(|i| (i.width, i.height))
            .unwrap_or((1920, 1080));

        let clip = IslandClip {
            store_index: idx,
            source_id,
            source_pts,
            layer_order: layer,
            opacity,
            transform,
            clip_width,
            clip_height,
            effect_start,
            effect_count,
        };

        island_map.entry(track_id)
            .or_insert_with(|| Island { track_id, clips: Vec::new() })
            .clips.push(clip);
    }

    island_map.into_values().collect()
}
