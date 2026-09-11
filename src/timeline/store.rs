#![allow(dead_code)]
// src/timeline/store.rs

use crate::timeline::ids::{ClipId, TrackId, SourceId};
use crate::timeline::transform::{ClipTransform, BlendMode, CropRect, CornerPin, MatteMode, ClipEffects};
use crate::timeline::keyframe::KeyframeStore;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClipKind {
    Video,
    Audio,
    Image,
    Text {
        text: String,
        font_size: f32,
        color: [f32; 4],
        /// Stroke (outline) around each glyph. None = disabled.
        #[serde(default)]
        stroke_color: Option<[f32; 4]>,
        /// Stroke thickness in pixels.
        #[serde(default)]
        stroke_width: f32,
        /// Solid background fill behind the text. None = transparent.
        #[serde(default)]
        background_color: Option<[f32; 4]>,
        /// Extra padding around text for the background box, in pixels (per side).
        /// Default 0 = rasteriser uses 10% of font_size automatically.
        #[serde(default)]
        bg_padding: f32,
    },
}

/// The central SoA clip metadata store.
/// INVARIANT: all Vec fields have identical length at all times.
/// INVARIANT: pts_in[i] <= pts_in[i+1] for all i  (sorted ascending).
/// INVARIANT: pts_in[i] < pts_out[i] for all i    (clip duration > 0).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TimelineStore {
    // --- identity (rarely touched after insert) ---
    pub(crate) ids:          Vec<ClipId>,
    pub(crate) track_ids:    Vec<TrackId>,
    pub(crate) source_ids:   Vec<SourceId>,

    // --- hot path: time bounds (read every frame during query_active) ---
    pub(crate) pts_in:       Vec<i64>,
    pub(crate) pts_out:      Vec<i64>,
    pub(crate) source_in:    Vec<i64>,

    // --- identity / kind ---
    #[serde(default)] // for backward compatibility
    pub(crate) kind:         Vec<ClipKind>,

    // --- compositing (read every frame for active clips, but less hot than time) ---
    pub(crate) layer_order:  Vec<u16>,
    pub(crate) opacity:      Vec<f32>,
    pub(crate) transform:    Vec<ClipTransform>,
    #[serde(default)]
    pub(crate) blend_mode:   Vec<BlendMode>,
    #[serde(default)]
    pub(crate) crop:         Vec<CropRect>,
    #[serde(default)]
    pub(crate) corner_pin:   Vec<CornerPin>,
    #[serde(default)]
    pub(crate) matte_mode:   Vec<MatteMode>,
    #[serde(default)]
    pub(crate) effects:      Vec<ClipEffects>,
    pub(crate) volume:       Vec<f32>,
    pub(crate) pan:          Vec<f32>,
    pub(crate) audio_muted:  Vec<bool>,
    /// Audio fade-in duration in timeline PTS ticks (0 = no fade).
    #[serde(default)]
    pub(crate) fade_in_pts:  Vec<i64>,
    /// Audio fade-out duration in timeline PTS ticks (0 = no fade).
    #[serde(default)]
    pub(crate) fade_out_pts: Vec<i64>,

    // --- speed / pitch (cold path — set by inspector) ---
    /// Playback speed multiplier (1.0 = normal, 2.0 = double speed, 0.5 = half speed).
    pub(crate) speed:        Vec<f32>,
    /// Pitch shift in semitones (0.0 = no shift). Independent of speed.
    pub(crate) pitch:        Vec<f32>,

    // --- effect linkage (cold path) ---
    pub(crate) effect_start: Vec<u32>,
    pub(crate) effect_count: Vec<u16>,

    // --- keyframes ---
    #[serde(default)]
    pub(crate) keyframes: KeyframeStore,

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
            kind: Vec::new(),
            layer_order: Vec::new(),
            opacity: Vec::new(),
            transform: Vec::new(),
            blend_mode: Vec::new(),
            crop: Vec::new(),
            corner_pin: Vec::new(),
            matte_mode: Vec::new(),
            effects: Vec::new(),
            volume: Vec::new(),
            pan: Vec::new(),
            audio_muted: Vec::new(),
            fade_in_pts: Vec::new(),
            fade_out_pts: Vec::new(),
            speed: Vec::new(),
            pitch: Vec::new(),
            effect_start: Vec::new(),
            effect_count: Vec::new(),
            keyframes: KeyframeStore::new(),
            next_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn sort_by_pts_in(&mut self) {
        let mut permutation: Vec<usize> = (0..self.ids.len()).collect();
        permutation.sort_by_key(|&i| self.pts_in[i]);

        fn apply_perm<T: Clone>(v: &mut [T], perm: &[usize]) {
            let old = v.to_owned();
            for (new_idx, &old_idx) in perm.iter().enumerate() {
                v[new_idx] = old[old_idx].clone();
            }
        }

        apply_perm(&mut self.ids, &permutation);
        apply_perm(&mut self.track_ids, &permutation);
        apply_perm(&mut self.source_ids, &permutation);
        apply_perm(&mut self.pts_in, &permutation);
        apply_perm(&mut self.pts_out, &permutation);
        apply_perm(&mut self.source_in, &permutation);
        apply_perm(&mut self.kind, &permutation);
        apply_perm(&mut self.layer_order, &permutation);
        apply_perm(&mut self.opacity, &permutation);
        apply_perm(&mut self.transform, &permutation);
        apply_perm(&mut self.blend_mode, &permutation);
        apply_perm(&mut self.crop, &permutation);
        apply_perm(&mut self.corner_pin, &permutation);
        apply_perm(&mut self.matte_mode, &permutation);
        apply_perm(&mut self.effects, &permutation);
        apply_perm(&mut self.volume, &permutation);
        apply_perm(&mut self.pan, &permutation);
        apply_perm(&mut self.audio_muted, &permutation);
        apply_perm(&mut self.fade_in_pts, &permutation);
        apply_perm(&mut self.fade_out_pts, &permutation);
        apply_perm(&mut self.speed, &permutation);
        apply_perm(&mut self.pitch, &permutation);
        apply_perm(&mut self.effect_start, &permutation);
        apply_perm(&mut self.effect_count, &permutation);
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
        assert_eq!(self.kind.len(), n);
        assert_eq!(self.layer_order.len(), n);
        assert_eq!(self.opacity.len(), n);
        assert_eq!(self.transform.len(), n);
        assert_eq!(self.blend_mode.len(), n);
        assert_eq!(self.crop.len(), n);
        assert_eq!(self.corner_pin.len(), n);
        assert_eq!(self.matte_mode.len(), n);
        assert_eq!(self.effects.len(), n);
        assert_eq!(self.volume.len(), n);
        assert_eq!(self.pan.len(), n);
        assert_eq!(self.audio_muted.len(), n);
        assert_eq!(self.fade_in_pts.len(), n);
        assert_eq!(self.fade_out_pts.len(), n);
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

    pub fn kind_at(&self, idx: usize) -> &ClipKind {
        &self.kind[idx]
    }

    pub fn set_kind_at(&mut self, idx: usize, kind: ClipKind) {
        self.kind[idx] = kind;
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

    pub fn volume_at(&self, idx: usize) -> f32 {
        self.volume[idx]
    }

    pub fn set_volume_at(&mut self, idx: usize, volume: f32) {
        self.volume[idx] = volume;
    }

    pub fn pan_at(&self, idx: usize) -> f32 {
        self.pan[idx]
    }

    pub fn set_pan_at(&mut self, idx: usize, pan: f32) {
        self.pan[idx] = pan;
    }

    pub fn audio_muted_at(&self, idx: usize) -> bool {
        self.audio_muted[idx]
    }

    pub fn set_audio_muted_at(&mut self, idx: usize, muted: bool) {
        self.audio_muted[idx] = muted;
    }

    pub fn fade_in_pts_at(&self, idx: usize) -> i64 {
        self.fade_in_pts[idx]
    }

    pub fn set_fade_in_pts_at(&mut self, idx: usize, pts: i64) {
        self.fade_in_pts[idx] = pts.max(0);
    }

    pub fn fade_out_pts_at(&self, idx: usize) -> i64 {
        self.fade_out_pts[idx]
    }

    pub fn set_fade_out_pts_at(&mut self, idx: usize, pts: i64) {
        self.fade_out_pts[idx] = pts.max(0);
    }

    pub fn speed_at(&self, idx: usize) -> f32 {
        self.speed[idx]
    }

    pub fn set_speed_at(&mut self, idx: usize, new_speed: f32) {
        let old_speed = self.speed[idx];
        if (old_speed - new_speed).abs() > 1e-4 {
            let duration = self.pts_out[idx] - self.pts_in[idx];
            let old_speed_scaled = (old_speed * 10_000.0).round() as i64;
            let new_speed_scaled = (new_speed * 10_000.0).round() as i64;
            let new_duration = if new_speed_scaled != 0 {
                let numer = duration as i128 * old_speed_scaled as i128;
                let den = new_speed_scaled as i128;
                let scaled = numer;
                if scaled >= 0 { ((scaled + den / 2) / den) as i64 }
                else           { ((scaled - den / 2) / den) as i64 }
            } else {
                duration
            };
            self.pts_out[idx] = self.pts_in[idx] + new_duration;
            self.speed[idx] = new_speed;
        }
    }

    pub fn pitch_at(&self, idx: usize) -> f32 {
        self.pitch[idx]
    }

    pub fn set_pitch_at(&mut self, idx: usize, pitch: f32) {
        self.pitch[idx] = pitch;
    }

    pub fn blend_mode_at(&self, idx: usize) -> BlendMode {
        self.blend_mode.get(idx).copied().unwrap_or_default()
    }

    pub fn set_blend_mode_at(&mut self, idx: usize, mode: BlendMode) {
        if idx < self.blend_mode.len() {
            self.blend_mode[idx] = mode;
        }
    }

    pub fn crop_at(&self, idx: usize) -> &CropRect {
        static DEFAULT_CROP: CropRect = CropRect { left: 0.0, top: 0.0, right: 1.0, bottom: 1.0, feather: 0.0 };
        self.crop.get(idx).unwrap_or(&DEFAULT_CROP)
    }

    pub fn set_crop_at(&mut self, idx: usize, crop: CropRect) {
        if idx < self.crop.len() {
            self.crop[idx] = crop;
        }
    }

    pub fn corner_pin_at(&self, idx: usize) -> &CornerPin {
        static DEFAULT_PIN: CornerPin = CornerPin {
            top_left: [0.0, 0.0],
            top_right: [1.0, 0.0],
            bottom_left: [0.0, 1.0],
            bottom_right: [1.0, 1.0],
        };
        self.corner_pin.get(idx).unwrap_or(&DEFAULT_PIN)
    }

    pub fn set_corner_pin_at(&mut self, idx: usize, pin: CornerPin) {
        if idx < self.corner_pin.len() {
            self.corner_pin[idx] = pin;
        }
    }

    pub fn matte_mode_at(&self, idx: usize) -> MatteMode {
        self.matte_mode.get(idx).copied().unwrap_or_default()
    }

    pub fn set_matte_mode_at(&mut self, idx: usize, mode: MatteMode) {
        if idx < self.matte_mode.len() {
            self.matte_mode[idx] = mode;
        }
    }

    pub fn effects_at(&self, idx: usize) -> ClipEffects {
        self.effects.get(idx).copied().unwrap_or_default()
    }

    pub fn set_effects_at(&mut self, idx: usize, effects: ClipEffects) {
        if idx < self.effects.len() {
            self.effects[idx] = effects;
        }
    }

    pub fn keyframes(&self) -> &KeyframeStore {
        &self.keyframes
    }

    pub fn keyframes_mut(&mut self) -> &mut KeyframeStore {
        &mut self.keyframes
    }
}