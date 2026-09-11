// src/timeline/keyframe.rs

use serde::{Deserialize, Serialize};
use crate::timeline::ids::ClipId;

/// Identifies one animatable numeric parameter on a clip.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimParam {
    // Transform & Compositing
    Opacity,
    PositionX,
    PositionY,
    Scale,
    Rotation,
    // Crop
    CropLeft,
    CropTop,
    CropRight,
    CropBottom,
    CropFeather,
    // Color & Light
    Brightness,
    Contrast,
    Saturation,
    HueShift,
    // Blur
    BlurRadius,
    BlurSigma,
    // Sharpen
    SharpenAmount,
    // Vignette
    VignetteIntensity,
    VignetteRadius,
    VignetteSoftness,
    VignetteRoundness,
    // Chroma Key
    ChromaKeyTolerance,
    ChromaKeySoftness,
    // Audio
    Volume,
    Pan,
}

impl AnimParam {
    pub fn label(&self) -> &'static str {
        match self {
            AnimParam::Opacity => "Opacity",
            AnimParam::PositionX => "Position X",
            AnimParam::PositionY => "Position Y",
            AnimParam::Scale => "Scale",
            AnimParam::Rotation => "Rotation",
            AnimParam::CropLeft => "Crop Left",
            AnimParam::CropTop => "Crop Top",
            AnimParam::CropRight => "Crop Right",
            AnimParam::CropBottom => "Crop Bottom",
            AnimParam::CropFeather => "Crop Feather",
            AnimParam::Brightness => "Brightness",
            AnimParam::Contrast => "Contrast",
            AnimParam::Saturation => "Saturation",
            AnimParam::HueShift => "Hue Shift",
            AnimParam::BlurRadius => "Blur Radius",
            AnimParam::BlurSigma => "Blur Sigma",
            AnimParam::SharpenAmount => "Sharpen Amount",
            AnimParam::VignetteIntensity => "Vignette Intensity",
            AnimParam::VignetteRadius => "Vignette Radius",
            AnimParam::VignetteSoftness => "Vignette Softness",
            AnimParam::VignetteRoundness => "Vignette Roundness",
            AnimParam::ChromaKeyTolerance => "Chroma Key Tolerance",
            AnimParam::ChromaKeySoftness => "Chroma Key Softness",
            AnimParam::Volume => "Volume",
            AnimParam::Pan => "Pan",
        }
    }
}

/// Interpolation mode between keyframes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InterpMode {
    #[default]
    Linear,
    Hold,
}

impl InterpMode {
    pub fn label(&self) -> &'static str {
        match self {
            InterpMode::Linear => "Linear",
            InterpMode::Hold => "Hold",
        }
    }

    pub fn all() -> &'static [InterpMode] {
        &[InterpMode::Linear, InterpMode::Hold]
    }
}

/// A single keyframe on a timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    /// Timeline PTS (90 kHz timebase).
    pub pts: i64,
    /// Parameter value at this keyframe.
    pub value: f32,
    /// Interpolation mode leaving this keyframe toward the next keyframe.
    pub interp: InterpMode,
}

/// An animation track of keyframes for a specific parameter on a clip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyframeTrack {
    pub clip_id: ClipId,
    pub param: AnimParam,
    /// Sorted ascending by `pts`.
    pub keys: Vec<Keyframe>,
}

impl KeyframeTrack {
    pub fn new(clip_id: ClipId, param: AnimParam) -> Self {
        Self {
            clip_id,
            param,
            keys: Vec::new(),
        }
    }

    /// Evaluate the parameter value at the given `pts`.
    ///
    /// Rules:
    /// - If no keyframes: returns None.
    /// - If query pts <= first keyframe: returns first keyframe value.
    /// - If query pts >= last keyframe: returns last keyframe value.
    /// - Between keyframes:
    ///   - `Hold`: retains the left keyframe's value until the next keyframe.
    ///   - `Linear`: linearly interpolates between left and right keyframes.
    pub fn eval(&self, pts: i64) -> Option<f32> {
        if self.keys.is_empty() {
            return None;
        }

        if self.keys.len() == 1 {
            return Some(self.keys[0].value);
        }

        if pts <= self.keys[0].pts {
            return Some(self.keys[0].value);
        }

        let last_idx = self.keys.len() - 1;
        if pts >= self.keys[last_idx].pts {
            return Some(self.keys[last_idx].value);
        }

        // Binary search to find the bounding interval: keys[i].pts <= pts < keys[i+1].pts
        let idx = match self.keys.binary_search_by_key(&pts, |k| k.pts) {
            Ok(exact) => return Some(self.keys[exact].value),
            Err(insertion_point) => insertion_point - 1,
        };

        let k0 = &self.keys[idx];
        let k1 = &self.keys[idx + 1];

        match k0.interp {
            InterpMode::Hold => Some(k0.value),
            InterpMode::Linear => {
                let dt = (k1.pts - k0.pts) as f64;
                if dt <= 0.0 {
                    return Some(k0.value);
                }
                let t = ((pts - k0.pts) as f64 / dt).clamp(0.0, 1.0) as f32;
                Some(k0.value + (k1.value - k0.value) * t)
            }
        }
    }

    /// Insert or update a keyframe at `pts`. Maintains sorted order.
    pub fn insert(&mut self, pts: i64, value: f32, interp: InterpMode) {
        match self.keys.binary_search_by_key(&pts, |k| k.pts) {
            Ok(idx) => {
                self.keys[idx].value = value;
                self.keys[idx].interp = interp;
            }
            Err(idx) => {
                self.keys.insert(
                    idx,
                    Keyframe {
                        pts,
                        value,
                        interp,
                    },
                );
            }
        }
    }

    /// Remove keyframe at `pts`. Returns true if removed.
    pub fn remove(&mut self, pts: i64) -> bool {
        if let Ok(idx) = self.keys.binary_search_by_key(&pts, |k| k.pts) {
            self.keys.remove(idx);
            true
        } else {
            false
        }
    }

    /// Move a keyframe from `old_pts` to `new_pts`.
    pub fn move_keyframe(&mut self, old_pts: i64, new_pts: i64) -> bool {
        if let Ok(idx) = self.keys.binary_search_by_key(&old_pts, |k| k.pts) {
            let key = self.keys.remove(idx);
            self.insert(new_pts, key.value, key.interp);
            true
        } else {
            false
        }
    }

    /// Shift all keyframes by `delta_pts`.
    pub fn shift_all(&mut self, delta_pts: i64) {
        for k in &mut self.keys {
            k.pts += delta_pts;
        }
    }
}

/// Central keyframe store. Stored in `TimelineStore`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KeyframeStore {
    tracks: Vec<KeyframeTrack>,
}

impl KeyframeStore {
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty() || self.tracks.iter().all(|t| t.keys.is_empty())
    }

    pub fn tracks(&self) -> &[KeyframeTrack] {
        &self.tracks
    }

    /// Find an existing track for `(clip_id, param)`.
    pub fn get_track(&self, clip_id: ClipId, param: AnimParam) -> Option<&KeyframeTrack> {
        self.tracks
            .iter()
            .find(|t| t.clip_id == clip_id && t.param == param)
    }

    /// Find or create a mutable track for `(clip_id, param)`.
    pub fn get_track_mut(&mut self, clip_id: ClipId, param: AnimParam) -> Option<&mut KeyframeTrack> {
        self.tracks
            .iter_mut()
            .find(|t| t.clip_id == clip_id && t.param == param)
    }

    pub fn get_or_create_track_mut(
        &mut self,
        clip_id: ClipId,
        param: AnimParam,
    ) -> &mut KeyframeTrack {
        if let Some(pos) = self
            .tracks
            .iter()
            .position(|t| t.clip_id == clip_id && t.param == param)
        {
            &mut self.tracks[pos]
        } else {
            self.tracks.push(KeyframeTrack::new(clip_id, param));
            self.tracks.last_mut().unwrap()
        }
    }

    /// Evaluate parameter value at `pts`. Returns `None` if no keyframes exist for this track.
    pub fn eval(&self, clip_id: ClipId, param: AnimParam, pts: i64) -> Option<f32> {
        self.get_track(clip_id, param).and_then(|t| t.eval(pts))
    }

    /// Add or update a keyframe.
    pub fn set_keyframe(
        &mut self,
        clip_id: ClipId,
        param: AnimParam,
        pts: i64,
        value: f32,
        interp: InterpMode,
    ) {
        let track = self.get_or_create_track_mut(clip_id, param);
        track.insert(pts, value, interp);
    }

    /// Remove a keyframe at exact `pts`. Cleans up empty tracks.
    pub fn remove_keyframe(&mut self, clip_id: ClipId, param: AnimParam, pts: i64) -> bool {
        let mut removed = false;
        if let Some(track) = self.get_track_mut(clip_id, param) {
            removed = track.remove(pts);
        }
        self.clean_empty_tracks();
        removed
    }

    /// Move a keyframe from `old_pts` to `new_pts`.
    pub fn move_keyframe(
        &mut self,
        clip_id: ClipId,
        param: AnimParam,
        old_pts: i64,
        new_pts: i64,
    ) -> bool {
        if let Some(track) = self.get_track_mut(clip_id, param) {
            track.move_keyframe(old_pts, new_pts)
        } else {
            false
        }
    }

    /// Check if a keyframe exists at exact `pts`.
    pub fn has_keyframe_at(&self, clip_id: ClipId, param: AnimParam, pts: i64) -> bool {
        self.get_track(clip_id, param)
            .is_some_and(|t| t.keys.binary_search_by_key(&pts, |k| k.pts).is_ok())
    }

    /// Check if any keyframes exist for this parameter on this clip.
    pub fn has_keyframes(&self, clip_id: ClipId, param: AnimParam) -> bool {
        self.get_track(clip_id, param)
            .is_some_and(|t| !t.keys.is_empty())
    }

    /// Check if any keyframes exist for this clip across all parameters.
    pub fn has_any_keyframes_for_clip(&self, clip_id: ClipId) -> bool {
        self.tracks
            .iter()
            .any(|t| t.clip_id == clip_id && !t.keys.is_empty())
    }

    /// Iterate all tracks belonging to `clip_id`.
    pub fn tracks_for_clip(&self, clip_id: ClipId) -> impl Iterator<Item = &KeyframeTrack> {
        self.tracks.iter().filter(move |t| t.clip_id == clip_id)
    }

    /// Remove all tracks and keyframes belonging to `clip_id`.
    pub fn remove_clip(&mut self, clip_id: ClipId) {
        self.tracks.retain(|t| t.clip_id != clip_id);
    }

    /// Shift all keyframes of a clip by `delta_pts` (e.g. when moving or ripple moving a clip).
    pub fn shift_clip_keyframes(&mut self, clip_id: ClipId, delta_pts: i64) {
        if delta_pts == 0 {
            return;
        }
        for track in self.tracks.iter_mut().filter(|t| t.clip_id == clip_id) {
            track.shift_all(delta_pts);
        }
    }

    /// Duplicate / copy keyframes from `source_clip_id` to `dest_clip_id`, optionally filtered by time bounds `[pts_in, pts_out)`.
    pub fn copy_clip_keyframes(
        &mut self,
        source_clip_id: ClipId,
        dest_clip_id: ClipId,
        time_range: Option<(i64, i64)>,
    ) {
        let mut new_tracks = Vec::new();
        for track in self.tracks.iter().filter(|t| t.clip_id == source_clip_id) {
            let mut new_track = KeyframeTrack::new(dest_clip_id, track.param);
            for k in &track.keys {
                if let Some((start, end)) = time_range {
                    if k.pts >= start && k.pts <= end {
                        new_track.keys.push(k.clone());
                    }
                } else {
                    new_track.keys.push(k.clone());
                }
            }
            if !new_track.keys.is_empty() {
                new_tracks.push(new_track);
            }
        }
        self.tracks.extend(new_tracks);
    }

    /// Retain only keyframes within `[pts_in, pts_out)` for `clip_id`.
    pub fn trim_clip_keyframes(&mut self, clip_id: ClipId, pts_in: i64, pts_out: i64) {
        for track in self.tracks.iter_mut().filter(|t| t.clip_id == clip_id) {
            track.keys.retain(|k| k.pts >= pts_in && k.pts <= pts_out);
        }
        self.clean_empty_tracks();
    }

    fn clean_empty_tracks(&mut self) {
        self.tracks.retain(|t| !t.keys.is_empty());
    }
}
