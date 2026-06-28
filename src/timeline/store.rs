#![allow(dead_code)]
// src/timeline/store.rs

use crate::timeline::ids::{ClipId, TrackId, SourceId};
use crate::timeline::transform::ClipTransform;

/// The central SoA clip metadata store.
/// INVARIANT: all Vec fields have identical length at all times.
/// INVARIANT: pts_in[i] <= pts_in[i+1] for all i  (sorted ascending).
/// INVARIANT: pts_in[i] < pts_out[i] for all i    (clip duration > 0).
pub struct TimelineStore {
    // --- identity (rarely touched after insert) ---
    pub(crate) ids:          Vec<ClipId>,
    pub(crate) track_ids:    Vec<TrackId>,
    pub(crate) source_ids:   Vec<SourceId>,

    // --- hot path: time bounds (read every frame during query_active) ---
    pub(crate) pts_in:       Vec<i64>,
    pub(crate) pts_out:      Vec<i64>,
    pub(crate) source_in:    Vec<i64>,

    // --- compositing (read every frame for active clips, but less hot than time) ---
    pub(crate) layer_order:  Vec<u16>,
    pub(crate) opacity:      Vec<f32>,
    pub(crate) transform:    Vec<ClipTransform>,

    // --- speed / pitch (cold path — set by inspector) ---
    /// Playback speed multiplier (1.0 = normal, 2.0 = double speed, 0.5 = half speed).
    pub(crate) speed:        Vec<f32>,
    /// Pitch shift in semitones (0.0 = no shift). Independent of speed.
    pub(crate) pitch:        Vec<f32>,

    // --- effect linkage (cold path) ---
    pub(crate) effect_start: Vec<u32>,
    pub(crate) effect_count: Vec<u16>,

    pub(crate) next_id:      u32,
}

impl Default for TimelineStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineStore {
    pub fn new() -> Self {
        TimelineStore {
            ids: Vec::new(),
            track_ids: Vec::new(),
            source_ids: Vec::new(),
            pts_in: Vec::new(),
            pts_out: Vec::new(),
            source_in: Vec::new(),
            layer_order: Vec::new(),
            opacity: Vec::new(),
            transform: Vec::new(),
            speed: Vec::new(),
            pitch: Vec::new(),
            effect_start: Vec::new(),
            effect_count: Vec::new(),
            next_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn assert_sorted(&self) {
        for window in self.pts_in.windows(2) {
            assert!(window[0] <= window[1], "TimelineStore not sorted by pts_in");
        }
    }

    pub fn assert_coherent(&self) {
        let n = self.ids.len();
        assert_eq!(self.track_ids.len(), n);
        assert_eq!(self.source_ids.len(), n);
        assert_eq!(self.pts_in.len(), n);
        assert_eq!(self.pts_out.len(), n);
        assert_eq!(self.source_in.len(), n);
        assert_eq!(self.layer_order.len(), n);
        assert_eq!(self.opacity.len(), n);
        assert_eq!(self.transform.len(), n);
        assert_eq!(self.speed.len(), n);
        assert_eq!(self.pitch.len(), n);
        assert_eq!(self.effect_start.len(), n);
        assert_eq!(self.effect_count.len(), n);
    }

    pub fn index_of(&self, id: ClipId) -> Option<usize> {
        self.ids.iter().position(|&x| x == id)
    }

    pub(crate) fn pts_in_slice(&self) -> &[i64] {
        &self.pts_in
    }

    pub(crate) fn pts_in_slice_mut(&mut self) -> &mut [i64] {
        &mut self.pts_in
    }

    pub fn pts_in_at(&self, idx: usize) -> i64 {
        self.pts_in[idx]
    }

    pub fn pts_out_at(&self, idx: usize) -> i64 {
        self.pts_out[idx]
    }

    pub fn track_id_at(&self, idx: usize) -> TrackId {
        self.track_ids[idx]
    }

    pub fn source_id_at(&self, idx: usize) -> SourceId {
        self.source_ids[idx]
    }

    pub fn source_in_at(&self, idx: usize) -> i64 {
        self.source_in[idx]
    }

    pub fn layer_order_at(&self, idx: usize) -> u16 {
        self.layer_order[idx]
    }

    pub fn opacity_at(&self, idx: usize) -> f32 {
        self.opacity[idx]
    }

    pub fn transform_at(&self, idx: usize) -> &ClipTransform {
        &self.transform[idx]
    }

    pub(crate) fn effect_range_at(&self, idx: usize) -> (u32, u16) {
        (self.effect_start[idx], self.effect_count[idx])
    }

    pub fn clip_id_at(&self, idx: usize) -> ClipId {
        self.ids[idx]
    }

    pub fn set_transform_at(&mut self, idx: usize, t: ClipTransform) {
        self.transform[idx] = t;
    }

    pub fn set_opacity_at(&mut self, idx: usize, opacity: f32) {
        self.opacity[idx] = opacity;
    }

    pub fn speed_at(&self, idx: usize) -> f32 {
        self.speed[idx]
    }

    pub fn set_speed_at(&mut self, idx: usize, speed: f32) {
        self.speed[idx] = speed;
    }

    pub fn pitch_at(&self, idx: usize) -> f32 {
        self.pitch[idx]
    }

    pub fn set_pitch_at(&mut self, idx: usize, pitch: f32) {
        self.pitch[idx] = pitch;
    }
}
