// src/scheduler/island.rs

use std::collections::HashMap;
use crate::timeline::ids::{TrackId, SourceId};
use crate::timeline::keyframe::AnimParam;
use crate::timeline::transform::{ClipTransform, BlendMode, CropRect, CornerPin, MatteMode, ClipEffects};
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
    pub blend_mode:   BlendMode,
    pub crop:         CropRect,
    pub corner_pin:   CornerPin,
    pub matte_mode:   MatteMode,
    pub effects:      ClipEffects,
    pub clip_width:   u32,
    pub clip_height:  u32,
    pub effect_start: u32,
    pub effect_count: u16,
    pub kind:         crate::timeline::store::ClipKind,
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
    let kf_store = store.keyframes();

    for active in active_indices {
        let idx = active.store_index;
        let clip_id = store.clip_id_at(idx);
        
        let track_id   = store.track_id_at(idx);
        let source_id  = store.source_id_at(idx);
        let source_pts = active.source_pts;
        let mut transform  = *store.transform_at(idx);
        let mut opacity    = store.opacity_at(idx);
        let blend_mode = store.blend_mode_at(idx);
        let mut crop       = *store.crop_at(idx);
        let corner_pin = *store.corner_pin_at(idx);
        let matte_mode = store.matte_mode_at(idx);
        let mut effects    = store.effects_at(idx);
        let layer      = store.layer_order_at(idx);
        let (effect_start, effect_count) = store.effect_range_at(idx);

        // Evaluate animated parameters at query_pts
        if let Some(op) = kf_store.eval(clip_id, AnimParam::Opacity, query_pts) {
            opacity = op.clamp(0.0, 1.0);
        }
        if let Some(px) = kf_store.eval(clip_id, AnimParam::PositionX, query_pts) {
            transform.position[0] = px;
        }
        if let Some(py) = kf_store.eval(clip_id, AnimParam::PositionY, query_pts) {
            transform.position[1] = py;
        }
        if let Some(scale) = kf_store.eval(clip_id, AnimParam::Scale, query_pts) {
            transform.scale = [scale, scale];
        }
        if let Some(rot) = kf_store.eval(clip_id, AnimParam::Rotation, query_pts) {
            transform.rotation = rot;
        }

        if let Some(cl) = kf_store.eval(clip_id, AnimParam::CropLeft, query_pts) {
            crop.left = cl;
        }
        if let Some(ct) = kf_store.eval(clip_id, AnimParam::CropTop, query_pts) {
            crop.top = ct;
        }
        if let Some(cr) = kf_store.eval(clip_id, AnimParam::CropRight, query_pts) {
            crop.right = cr;
        }
        if let Some(cb) = kf_store.eval(clip_id, AnimParam::CropBottom, query_pts) {
            crop.bottom = cb;
        }
        if let Some(cf) = kf_store.eval(clip_id, AnimParam::CropFeather, query_pts) {
            crop.feather = cf;
        }

        if let Some(b) = kf_store.eval(clip_id, AnimParam::Brightness, query_pts) {
            effects.brightness = b;
        }
        if let Some(c) = kf_store.eval(clip_id, AnimParam::Contrast, query_pts) {
            effects.contrast = c;
        }
        if let Some(s) = kf_store.eval(clip_id, AnimParam::Saturation, query_pts) {
            effects.saturation = s;
        }
        if let Some(h) = kf_store.eval(clip_id, AnimParam::HueShift, query_pts) {
            effects.hue = h;
        }
        if let Some(r) = kf_store.eval(clip_id, AnimParam::BlurRadius, query_pts) {
            effects.blur_radius = r;
        }
        if let Some(sig) = kf_store.eval(clip_id, AnimParam::BlurSigma, query_pts) {
            effects.blur_sigma = sig;
        }
        if let Some(sh) = kf_store.eval(clip_id, AnimParam::SharpenAmount, query_pts) {
            effects.sharpen_amount = sh;
        }
        if let Some(vi) = kf_store.eval(clip_id, AnimParam::VignetteIntensity, query_pts) {
            effects.vignette_intensity = vi;
        }
        if let Some(vr) = kf_store.eval(clip_id, AnimParam::VignetteRadius, query_pts) {
            effects.vignette_radius = vr;
        }
        if let Some(vs) = kf_store.eval(clip_id, AnimParam::VignetteSoftness, query_pts) {
            effects.vignette_softness = vs;
        }
        if let Some(vround) = kf_store.eval(clip_id, AnimParam::VignetteRoundness, query_pts) {
            effects.vignette_roundness = vround;
        }
        if let Some(ct) = kf_store.eval(clip_id, AnimParam::ChromaKeyTolerance, query_pts) {
            effects.chroma_key_tolerance = ct;
        }
        if let Some(cs) = kf_store.eval(clip_id, AnimParam::ChromaKeySoftness, query_pts) {
            effects.chroma_key_softness = cs;
        }
        
        let (clip_width, clip_height) = {
            let kind = store.kind_at(idx);
            if let crate::timeline::store::ClipKind::Text {
                text, font_size, stroke_color, stroke_width, background_color, bg_padding, ..
            } = kind {
                // Measure the rasterised text size so the GPU transform matrix
                // places it at natural size rather than stretching it to 1920×1080.
                crate::render::text_renderer::measure_text(
                    text,
                    *font_size,
                    *stroke_width,
                    stroke_color.is_some(),
                    background_color.is_some(),
                    *bg_padding,
                )
            } else {
                source_reg.video_info(source_id)
                    .map(|i| (i.width, i.height))
                    .unwrap_or((1920, 1080))
            }
        };

        let kind = store.kind_at(idx).clone();

        let clip = IslandClip {
            store_index: idx,
            source_id,
            source_pts,
            layer_order: layer,
            opacity,
            transform,
            blend_mode,
            crop,
            corner_pin,
            matte_mode,
            effects,
            clip_width,
            clip_height,
            effect_start,
            effect_count,
            kind,
        };

        island_map.entry(track_id)
            .or_insert_with(|| Island { track_id, clips: Vec::new() })
            .clips.push(clip);
    }

    island_map.into_values().collect()
}
