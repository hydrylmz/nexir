// src/timeline/mutation.rs

use crate::timeline::ids::{ClipId, SourceId, TrackId};
use crate::timeline::store::TimelineStore;
use crate::timeline::transform::{ClipTransform, BlendMode, CropRect, CornerPin, MatteMode};

/// Parameters for inserting a new clip.
#[derive(Clone, Debug)]
pub struct ClipInsertParams {
    pub track_id: TrackId,
    pub source_id: SourceId,
    pub kind: crate::timeline::store::ClipKind,
    pub pts_in: i64,
    pub pts_out: i64,
    pub source_in: i64,
    pub layer_order: u16,
    pub opacity: f32,
    pub transform: ClipTransform,
    pub blend_mode: BlendMode,
    pub crop: CropRect,
    pub corner_pin: CornerPin,
    pub matte_mode: MatteMode,
    /// Linear clip gain (1.0 = unity).
    pub volume: f32,
    /// Stereo pan (-1.0 = left, 0.0 = center, 1.0 = right).
    pub pan: f32,
    /// Per-clip audio mute.
    pub audio_muted: bool,
    /// Audio fade-in duration in timeline PTS ticks (0 = no fade).
    pub fade_in_pts: i64,
    /// Audio fade-out duration in timeline PTS ticks (0 = no fade).
    pub fade_out_pts: i64,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
    /// Pitch shift in semitones (0.0 = no shift).
    pub pitch: f32,
}

impl Default for ClipInsertParams {
    fn default() -> Self {
        Self {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 90_000,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            blend_mode: BlendMode::Normal,
            crop: CropRect::full(),
            corner_pin: CornerPin::identity(),
            matte_mode: MatteMode::None,
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
        }
    }
}

pub fn insert_clip(
    store: &mut TimelineStore,
    params: ClipInsertParams,
) -> Result<ClipId, MutationError> {
    if params.pts_in >= params.pts_out {
        return Err(MutationError::InvalidDuration {
            pts_in: params.pts_in,
            pts_out: params.pts_out,
        });
    }

    let pos = store
        .pts_in_slice()
        .partition_point(|&x| x <= params.pts_in);

    store.track_ids.insert(pos, params.track_id);
    store.source_ids.insert(pos, params.source_id);
    store.kind.insert(pos, params.kind);
    store.pts_in.insert(pos, params.pts_in);
    store.pts_out.insert(pos, params.pts_out);
    store.source_in.insert(pos, params.source_in);
    store.layer_order.insert(pos, params.layer_order);
    store.opacity.insert(pos, params.opacity);
    store.transform.insert(pos, params.transform);
    store.blend_mode.insert(pos, params.blend_mode);
    store.crop.insert(pos, params.crop);
    store.corner_pin.insert(pos, params.corner_pin);
    store.matte_mode.insert(pos, params.matte_mode);
    store.volume.insert(pos, params.volume);
    store.pan.insert(pos, params.pan);
    store.audio_muted.insert(pos, params.audio_muted);
    store.fade_in_pts.insert(pos, params.fade_in_pts);
    store.fade_out_pts.insert(pos, params.fade_out_pts);
    store.speed.insert(pos, params.speed);
    store.pitch.insert(pos, params.pitch);
    store.effect_start.insert(pos, 0);
    store.effect_count.insert(pos, 0);

    let clip_id = ClipId(store.next_id);
    store.next_id += 1;
    store.ids.insert(pos, clip_id);

    Ok(clip_id)
}

/// Insert a clip in **overwrite** mode (like every real NLE).
///
/// Any existing clip on the same track whose time range overlaps [`pts_in`, `pts_out`) is either:
/// - **Fully removed** if it lies entirely within the new clip's range.
/// - **Trimmed at the head** if it starts before the new clip but its tail overlaps.
/// - **Trimmed at the tail** if it starts inside the new clip but its tail extends past.
/// - **Split** if the new clip lands fully inside an existing clip (head stays, tail becomes a new clip).
pub fn insert_clip_overwrite(
    store: &mut TimelineStore,
    params: ClipInsertParams,
) -> Result<ClipId, MutationError> {
    if params.pts_in >= params.pts_out {
        return Err(MutationError::InvalidDuration {
            pts_in: params.pts_in,
            pts_out: params.pts_out,
        });
    }

    let new_in = params.pts_in;
    let new_out = params.pts_out;
    let track = params.track_id;

    // Collect all clip IDs on the same track that overlap the new clip range.
    // We need to handle them after collecting to avoid borrowing issues.
    let overlapping: Vec<(
        ClipId,
        i64,
        i64,
        i64,
        SourceId,
        TrackId,
        crate::timeline::store::ClipKind,
        u16,
        f32,
        ClipTransform,
        BlendMode,
        CropRect,
        CornerPin,
        MatteMode,
        f32,
        f32,
        bool,
        i64,
        i64,
        f32,
        f32,
    )> = {
        (0..store.len())
            .filter(|&i| {
                store.track_ids[i] == track
                    && store.pts_in[i] < new_out
                    && store.pts_out[i] > new_in
            })
            .map(|i| {
                (
                    store.ids[i],
                    store.pts_in[i],
                    store.pts_out[i],
                    store.source_in[i],
                    store.source_ids[i],
                    store.track_ids[i],
                    store.kind[i].clone(),
                    store.layer_order[i],
                    store.opacity[i],
                    store.transform[i],
                    store.blend_mode_at(i),
                    *store.crop_at(i),
                    *store.corner_pin_at(i),
                    store.matte_mode_at(i),
                    store.volume[i],
                    store.pan[i],
                    store.audio_muted[i],
                    store.fade_in_pts[i],
                    store.fade_out_pts[i],
                    store.speed[i],
                    store.pitch[i],
                )
            })
            .collect()
    };

    for (
        id,
        ex_in,
        ex_out,
        ex_src_in,
        ex_src_id,
        ex_track,
        ex_kind,
        ex_layer,
        ex_opacity,
        ex_transform,
        ex_blend,
        ex_crop,
        ex_corner_pin,
        ex_matte,
        ex_volume,
        ex_pan,
        ex_audio_muted,
        ex_fade_in_pts,
        ex_fade_out_pts,
        ex_speed,
        ex_pitch,
    ) in overlapping
    {
        if ex_in >= new_in && ex_out <= new_out {
            // Fully covered — delete it.
            remove_clip(store, id)?;
        } else if ex_in < new_in && ex_out > new_out {
            // New clip lands entirely inside existing clip — split it.
            // Keep the head portion: [ex_in .. new_in]
            // Create a tail portion: [new_out .. ex_out]
            let tail_src_in = ex_src_in + (new_out - ex_in);
            remove_clip(store, id)?;
            // Head
            insert_clip(
                store,
                ClipInsertParams {
                    track_id: ex_track,
                    source_id: ex_src_id,
                    kind: ex_kind.clone(),
                    pts_in: ex_in,
                    pts_out: new_in,
                    source_in: ex_src_in,
                    layer_order: ex_layer,
                    opacity: ex_opacity,
                    transform: ex_transform,
                    blend_mode: ex_blend,
                    crop: ex_crop,
                    corner_pin: ex_corner_pin,
                    matte_mode: ex_matte,
                    volume: ex_volume,
                    pan: ex_pan,
                    audio_muted: ex_audio_muted,
                    fade_in_pts: ex_fade_in_pts,
                    fade_out_pts: 0,
                    speed: ex_speed,
                    pitch: ex_pitch,
                },
            )?;
            // Tail
            insert_clip(
                store,
                ClipInsertParams {
                    track_id: ex_track,
                    source_id: ex_src_id,
                    kind: ex_kind.clone(),
                    pts_in: new_out,
                    pts_out: ex_out,
                    source_in: tail_src_in,
                    layer_order: ex_layer,
                    opacity: ex_opacity,
                    transform: ex_transform,
                    blend_mode: ex_blend,
                    crop: ex_crop,
                    corner_pin: ex_corner_pin,
                    matte_mode: ex_matte,
                    volume: ex_volume,
                    pan: ex_pan,
                    audio_muted: ex_audio_muted,
                    fade_in_pts: 0,
                    fade_out_pts: ex_fade_out_pts,
                    speed: ex_speed,
                    pitch: ex_pitch,
                },
            )?;
        } else if ex_in < new_in {
            // Existing clip starts before new clip — trim its tail to new_in.
            remove_clip(store, id)?;
            insert_clip(
                store,
                ClipInsertParams {
                    track_id: ex_track,
                    source_id: ex_src_id,
                    kind: ex_kind.clone(),
                    pts_in: ex_in,
                    pts_out: new_in,
                    source_in: ex_src_in,
                    layer_order: ex_layer,
                    opacity: ex_opacity,
                    transform: ex_transform,
                    blend_mode: ex_blend,
                    crop: ex_crop,
                    corner_pin: ex_corner_pin,
                    matte_mode: ex_matte,
                    volume: ex_volume,
                    pan: ex_pan,
                    audio_muted: ex_audio_muted,
                    fade_in_pts: ex_fade_in_pts,
                    fade_out_pts: 0,
                    speed: ex_speed,
                    pitch: ex_pitch,
                },
            )?;
        } else {
            // Existing clip starts inside new clip — trim its head to new_out.
            let new_src_in = ex_src_in + (new_out - ex_in);
            remove_clip(store, id)?;
            insert_clip(
                store,
                ClipInsertParams {
                    track_id: ex_track,
                    source_id: ex_src_id,
                    kind: ex_kind.clone(),
                    pts_in: new_out,
                    pts_out: ex_out,
                    source_in: new_src_in,
                    layer_order: ex_layer,
                    opacity: ex_opacity,
                    transform: ex_transform,
                    blend_mode: ex_blend,
                    crop: ex_crop,
                    corner_pin: ex_corner_pin,
                    matte_mode: ex_matte,
                    volume: ex_volume,
                    pan: ex_pan,
                    audio_muted: ex_audio_muted,
                    fade_in_pts: 0,
                    fade_out_pts: ex_fade_out_pts,
                    speed: ex_speed,
                    pitch: ex_pitch,
                },
            )?;
        }
    }

    // Now insert the new clip.
    insert_clip(store, params)
}

pub fn remove_clip(store: &mut TimelineStore, id: ClipId) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;

    store.ids.remove(idx);
    store.track_ids.remove(idx);
    store.source_ids.remove(idx);
    store.kind.remove(idx);
    store.pts_in.remove(idx);
    store.pts_out.remove(idx);
    store.source_in.remove(idx);
    store.layer_order.remove(idx);
    store.opacity.remove(idx);
    store.transform.remove(idx);
    store.blend_mode.remove(idx);
    store.crop.remove(idx);
    store.corner_pin.remove(idx);
    store.matte_mode.remove(idx);
    store.volume.remove(idx);
    store.pan.remove(idx);
    store.audio_muted.remove(idx);
    store.fade_in_pts.remove(idx);
    store.fade_out_pts.remove(idx);
    store.speed.remove(idx);
    store.pitch.remove(idx);
    store.effect_start.remove(idx);
    store.effect_count.remove(idx);

    Ok(())
}

pub fn move_clip(
    store: &mut TimelineStore,
    id: ClipId,
    new_pts_in: i64,
) -> Result<ClipId, MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;

    let duration = store.pts_out[idx] - store.pts_in[idx];
    let new_pts_out = new_pts_in + duration;

    let params = ClipInsertParams {
        track_id: store.track_ids[idx],
        source_id: store.source_ids[idx],
        kind: store.kind[idx].clone(),
        pts_in: new_pts_in,
        pts_out: new_pts_out,
        source_in: store.source_in[idx],
        layer_order: store.layer_order[idx],
        opacity: store.opacity[idx],
        transform: store.transform[idx],
        blend_mode: store.blend_mode_at(idx),
        crop: *store.crop_at(idx),
        corner_pin: *store.corner_pin_at(idx),
        matte_mode: store.matte_mode_at(idx),
        volume: store.volume[idx],
        pan: store.pan[idx],
        audio_muted: store.audio_muted[idx],
        fade_in_pts: store.fade_in_pts[idx],
        fade_out_pts: store.fade_out_pts[idx],
        speed: store.speed[idx],
        pitch: store.pitch[idx],
    };

    remove_clip(store, id)?;
    insert_clip(store, params)
}

pub fn trim_clip_in(
    store: &mut TimelineStore,
    id: ClipId,
    new_pts_in: i64,
) -> Result<ClipId, MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;

    if new_pts_in >= store.pts_out[idx] {
        return Err(MutationError::TrimPastOppositeEnd);
    }

    let delta = new_pts_in - store.pts_in[idx];
    let new_source_in = store.source_in[idx] + delta;

    if new_source_in < 0 {
        return Err(MutationError::SourceInBeforeStart {
            computed: new_source_in,
        });
    }

    let params = ClipInsertParams {
        track_id: store.track_ids[idx],
        source_id: store.source_ids[idx],
        kind: store.kind[idx].clone(),
        pts_in: new_pts_in,
        pts_out: store.pts_out[idx],
        source_in: new_source_in,
        layer_order: store.layer_order[idx],
        opacity: store.opacity[idx],
        transform: store.transform[idx],
        blend_mode: store.blend_mode_at(idx),
        crop: *store.crop_at(idx),
        corner_pin: *store.corner_pin_at(idx),
        matte_mode: store.matte_mode_at(idx),
        volume: store.volume[idx],
        pan: store.pan[idx],
        audio_muted: store.audio_muted[idx],
        fade_in_pts: store.fade_in_pts[idx],
        fade_out_pts: store.fade_out_pts[idx],
        speed: store.speed[idx],
        pitch: store.pitch[idx],
    };

    remove_clip(store, id)?;
    insert_clip(store, params)
}

pub fn trim_clip_out(
    store: &mut TimelineStore,
    id: ClipId,
    new_pts_out: i64,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;

    if new_pts_out <= store.pts_in[idx] {
        return Err(MutationError::TrimPastOppositeEnd);
    }

    store.pts_out[idx] = new_pts_out;
    Ok(())
}

pub fn split_clip(
    store: &mut TimelineStore,
    id: ClipId,
    split_pts: i64,
) -> Result<ClipId, MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    let pts_in = store.pts_in[idx];
    let pts_out = store.pts_out[idx];

    if split_pts <= pts_in || split_pts >= pts_out {
        return Err(MutationError::TrimPastOppositeEnd);
    }

    let delta = split_pts - pts_in;
    let old_fade_out = store.fade_out_pts[idx];

    let params = ClipInsertParams {
        track_id: store.track_ids[idx],
        source_id: store.source_ids[idx],
        kind: store.kind[idx].clone(),
        pts_in: split_pts,
        pts_out: pts_out,
        source_in: store.source_in[idx] + delta,
        layer_order: store.layer_order[idx],
        opacity: store.opacity[idx],
        transform: store.transform[idx],
        blend_mode: store.blend_mode_at(idx),
        crop: *store.crop_at(idx),
        corner_pin: *store.corner_pin_at(idx),
        matte_mode: store.matte_mode_at(idx),
        volume: store.volume[idx],
        pan: store.pan[idx],
        audio_muted: store.audio_muted[idx],
        fade_in_pts: 0,
        fade_out_pts: old_fade_out,
        speed: store.speed[idx],
        pitch: store.pitch[idx],
    };

    store.pts_out[idx] = split_pts;
    store.fade_out_pts[idx] = 0;

    insert_clip(store, params)
}

pub fn duplicate_clip(store: &mut TimelineStore, id: ClipId) -> Result<ClipId, MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;

    let duration = store.pts_out[idx] - store.pts_in[idx];

    let params = ClipInsertParams {
        track_id: store.track_ids[idx],
        source_id: store.source_ids[idx],
        kind: store.kind[idx].clone(),
        pts_in: store.pts_out[idx],
        pts_out: store.pts_out[idx] + duration,
        source_in: store.source_in[idx],
        layer_order: store.layer_order[idx],
        opacity: store.opacity[idx],
        transform: store.transform[idx],
        blend_mode: store.blend_mode_at(idx),
        crop: *store.crop_at(idx),
        corner_pin: *store.corner_pin_at(idx),
        matte_mode: store.matte_mode_at(idx),
        volume: store.volume[idx],
        pan: store.pan[idx],
        audio_muted: store.audio_muted[idx],
        fade_in_pts: store.fade_in_pts[idx],
        fade_out_pts: store.fade_out_pts[idx],
        speed: store.speed[idx],
        pitch: store.pitch[idx],
    };

    insert_clip_overwrite(store, params)
}

pub fn ripple_remove_clip(store: &mut TimelineStore, id: ClipId) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    let track_id = store.track_ids[idx];
    let pts_in = store.pts_in[idx];
    let pts_out = store.pts_out[idx];
    let duration = pts_out - pts_in;

    // Collect all other clips on the same track starting at or after pts_out
    let mut to_shift = Vec::new();
    for i in 0..store.len() {
        if store.track_ids[i] == track_id && store.pts_in[i] >= pts_out {
            to_shift.push((
                store.ids[i],
                ClipInsertParams {
                    track_id: store.track_ids[i],
                    source_id: store.source_ids[i],
                    kind: store.kind[i].clone(),
                    pts_in: store.pts_in[i] - duration,
                    pts_out: store.pts_out[i] - duration,
                    source_in: store.source_in[i],
                    layer_order: store.layer_order[i],
                    opacity: store.opacity[i],
                    transform: store.transform[i],
                    blend_mode: store.blend_mode_at(i),
                    crop: *store.crop_at(i),
                    corner_pin: *store.corner_pin_at(i),
                    matte_mode: store.matte_mode_at(i),
                    volume: store.volume[i],
                    pan: store.pan[i],
                    audio_muted: store.audio_muted[i],
                    fade_in_pts: store.fade_in_pts[i],
                    fade_out_pts: store.fade_out_pts[i],
                    speed: store.speed[i],
                    pitch: store.pitch[i],
                },
            ));
        }
    }

    // Sort to_shift by original pts_in to maintain sorted order
    to_shift.sort_by_key(|(_, params)| params.pts_in);

    // Remove the target clip
    remove_clip(store, id)?;

    // Remove all the clips that need shifting
    for (clip_id, _) in &to_shift {
        remove_clip(store, *clip_id)?;
    }

    // Re-insert the shifted clips
    for (_, params) in to_shift {
        insert_clip(store, params)?;
    }

    Ok(())
}

pub fn insert_clip_ripple(
    store: &mut TimelineStore,
    params: ClipInsertParams,
) -> Result<ClipId, MutationError> {
    if params.pts_in >= params.pts_out {
        return Err(MutationError::InvalidDuration {
            pts_in: params.pts_in,
            pts_out: params.pts_out,
        });
    }

    let new_in = params.pts_in;
    let new_out = params.pts_out;
    let duration = new_out - new_in;
    let track = params.track_id;

    // Identify primary clips on the target track that need shifting or splitting.
    let mut primary_indices = Vec::new();
    for i in 0..store.len() {
        if store.track_ids[i] == track {
            primary_indices.push(i);
        }
    }
    primary_indices.sort_by_key(|&i| store.pts_in[i]);

    let mut to_shift = Vec::new();
    let mut to_split = Vec::new();

    let mut boundary = new_in + duration;

    for i in primary_indices {
        let ex_in = store.pts_in[i];
        let ex_out = store.pts_out[i];

        if ex_out <= new_in {
            // Completely before the insertion point; unaffected.
            continue;
        }

        if ex_in < new_in && ex_out > new_in {
            // Overlaps the insertion point; split it.
            let head_params = ClipInsertParams {
                track_id: store.track_ids[i],
                source_id: store.source_ids[i],
                kind: store.kind[i].clone(),
                pts_in: ex_in,
                pts_out: new_in,
                source_in: store.source_in[i],
                layer_order: store.layer_order[i],
                opacity: store.opacity[i],
                transform: store.transform[i],
                blend_mode: store.blend_mode_at(i),
                crop: *store.crop_at(i),
                corner_pin: *store.corner_pin_at(i),
                matte_mode: store.matte_mode_at(i),
                volume: store.volume[i],
                pan: store.pan[i],
                audio_muted: store.audio_muted[i],
                fade_in_pts: store.fade_in_pts[i],
                fade_out_pts: 0,
                speed: store.speed[i],
                pitch: store.pitch[i],
            };
            let tail_params = ClipInsertParams {
                track_id: store.track_ids[i],
                source_id: store.source_ids[i],
                kind: store.kind[i].clone(),
                pts_in: boundary,
                pts_out: boundary + (ex_out - new_in),
                source_in: store.source_in[i] + (new_in - ex_in),
                layer_order: store.layer_order[i],
                opacity: store.opacity[i],
                transform: store.transform[i],
                blend_mode: store.blend_mode_at(i),
                crop: *store.crop_at(i),
                corner_pin: *store.corner_pin_at(i),
                matte_mode: store.matte_mode_at(i),
                volume: store.volume[i],
                pan: store.pan[i],
                audio_muted: store.audio_muted[i],
                fade_in_pts: 0,
                fade_out_pts: store.fade_out_pts[i],
                speed: store.speed[i],
                pitch: store.pitch[i],
            };

            to_split.push((store.ids[i], head_params, tail_params));
            boundary = boundary + (ex_out - new_in);
        } else if ex_in >= new_in {
            // Starts at or after new_in.
            if ex_in < boundary {
                // Overlaps with the current push boundary; must be shifted.
                let shift_delta = boundary - ex_in;
                to_shift.push((
                    store.ids[i],
                    ClipInsertParams {
                        track_id: store.track_ids[i],
                        source_id: store.source_ids[i],
                        kind: store.kind[i].clone(),
                        pts_in: boundary,
                        pts_out: ex_out + shift_delta,
                        source_in: store.source_in[i],
                        layer_order: store.layer_order[i],
                        opacity: store.opacity[i],
                        transform: store.transform[i],
                        blend_mode: store.blend_mode_at(i),
                        crop: *store.crop_at(i),
                        corner_pin: *store.corner_pin_at(i),
                        matte_mode: store.matte_mode_at(i),
                        volume: store.volume[i],
                        pan: store.pan[i],
                        audio_muted: store.audio_muted[i],
                        fade_in_pts: store.fade_in_pts[i],
                        fade_out_pts: store.fade_out_pts[i],
                        speed: store.speed[i],
                        pitch: store.pitch[i],
                    },
                ));
                boundary = ex_out + shift_delta;
            } else {
                // Already at or after the boundary; no shift needed.
                boundary = ex_out;
            }
        }
    }

    // Now, for every clip we decided to split or shift, we must ALSO split or shift
    // its linked clips on other tracks in exactly the same way.
    // A clip is linked if it has the same source_id and the same original pts_in.
    let mut linked_to_split = Vec::new();
    let mut linked_to_shift = Vec::new();

    for &(split_id, ref _head, ref tail) in &to_split {
        if let Some(idx) = store.index_of(split_id) {
            let pri_source_id = store.source_ids[idx];
            let pri_pts_in = store.pts_in[idx];
            let pri_track = store.track_ids[idx];

            for j in 0..store.len() {
                if store.source_ids[j] == pri_source_id
                    && store.pts_in[j] == pri_pts_in
                    && store.track_ids[j] != pri_track
                {
                    let ex_in = store.pts_in[j];
                    let ex_out = store.pts_out[j];

                    let l_head = ClipInsertParams {
                        track_id: store.track_ids[j],
                        source_id: store.source_ids[j],
                        kind: store.kind[j].clone(),
                        pts_in: ex_in,
                        pts_out: new_in,
                        source_in: store.source_in[j],
                        layer_order: store.layer_order[j],
                        opacity: store.opacity[j],
                        transform: store.transform[j],
                        blend_mode: store.blend_mode_at(j),
                        crop: *store.crop_at(j),
                        corner_pin: *store.corner_pin_at(j),
                        matte_mode: store.matte_mode_at(j),
                        volume: store.volume[j],
                        pan: store.pan[j],
                        audio_muted: store.audio_muted[j],
                        fade_in_pts: store.fade_in_pts[j],
                        fade_out_pts: 0,
                        speed: store.speed[j],
                        pitch: store.pitch[j],
                    };
                    let l_tail = ClipInsertParams {
                        track_id: store.track_ids[j],
                        source_id: store.source_ids[j],
                        kind: store.kind[j].clone(),
                        pts_in: tail.pts_in,
                        pts_out: tail.pts_in + (ex_out - new_in),
                        source_in: store.source_in[j] + (new_in - ex_in),
                        layer_order: store.layer_order[j],
                        opacity: store.opacity[j],
                        transform: store.transform[j],
                        blend_mode: store.blend_mode_at(j),
                        crop: *store.crop_at(j),
                        corner_pin: *store.corner_pin_at(j),
                        matte_mode: store.matte_mode_at(j),
                        volume: store.volume[j],
                        pan: store.pan[j],
                        audio_muted: store.audio_muted[j],
                        fade_in_pts: 0,
                        fade_out_pts: store.fade_out_pts[j],
                        speed: store.speed[j],
                        pitch: store.pitch[j],
                    };
                    linked_to_split.push((store.ids[j], l_head, l_tail));
                }
            }
        }
    }

    for &(shift_id, ref shifted_params) in &to_shift {
        if let Some(idx) = store.index_of(shift_id) {
            let pri_source_id = store.source_ids[idx];
            let pri_pts_in = store.pts_in[idx];
            let pri_track = store.track_ids[idx];
            let shift_delta = shifted_params.pts_in - pri_pts_in;

            for j in 0..store.len() {
                if store.source_ids[j] == pri_source_id
                    && store.pts_in[j] == pri_pts_in
                    && store.track_ids[j] != pri_track
                {
                    linked_to_shift.push((
                        store.ids[j],
                        ClipInsertParams {
                            track_id: store.track_ids[j],
                            source_id: store.source_ids[j],
                            kind: store.kind[j].clone(),
                            pts_in: store.pts_in[j] + shift_delta,
                            pts_out: store.pts_out[j] + shift_delta,
                            source_in: store.source_in[j],
                            layer_order: store.layer_order[j],
                            opacity: store.opacity[j],
                            transform: store.transform[j],
                            blend_mode: store.blend_mode_at(j),
                            crop: *store.crop_at(j),
                            corner_pin: *store.corner_pin_at(j),
                            matte_mode: store.matte_mode_at(j),
                            volume: store.volume[j],
                            pan: store.pan[j],
                            audio_muted: store.audio_muted[j],
                            fade_in_pts: store.fade_in_pts[j],
                            fade_out_pts: store.fade_out_pts[j],
                            speed: store.speed[j],
                            pitch: store.pitch[j],
                        },
                    ));
                }
            }
        }
    }

    to_split.extend(linked_to_split);
    to_shift.extend(linked_to_shift);

    // Remove all affected clips.
    for (split_id, _, _) in &to_split {
        remove_clip(store, *split_id)?;
    }
    for (shift_id, _) in &to_shift {
        remove_clip(store, *shift_id)?;
    }

    // Re-insert split portions.
    for (_, head, tail) in to_split {
        insert_clip(store, head)?;
        insert_clip(store, tail)?;
    }

    // Re-insert shifted clips.
    to_shift.sort_by_key(|(_, params)| params.pts_in);
    for (_, params) in to_shift {
        insert_clip(store, params)?;
    }

    // Finally insert the new clip.
    insert_clip(store, params)
}


pub fn move_clip_ripple(
    store: &mut TimelineStore,
    id: ClipId,
    new_pts_in: i64,
) -> Result<ClipId, MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;

    let duration = store.pts_out[idx] - store.pts_in[idx];
    let new_pts_out = new_pts_in + duration;

    let params = ClipInsertParams {
        track_id: store.track_ids[idx],
        source_id: store.source_ids[idx],
        kind: store.kind[idx].clone(),
        pts_in: new_pts_in,
        pts_out: new_pts_out,
        source_in: store.source_in[idx],
        layer_order: store.layer_order[idx],
        opacity: store.opacity[idx],
        transform: store.transform[idx],
        blend_mode: store.blend_mode_at(idx),
        crop: *store.crop_at(idx),
        corner_pin: *store.corner_pin_at(idx),
        matte_mode: store.matte_mode_at(idx),
        volume: store.volume[idx],
        pan: store.pan[idx],
        audio_muted: store.audio_muted[idx],
        fade_in_pts: store.fade_in_pts[idx],
        fade_out_pts: store.fade_out_pts[idx],
        speed: store.speed[idx],
        pitch: store.pitch[idx],
    };

    // First do a ripple-remove to close the gap at the original position
    ripple_remove_clip(store, id)?;

    // Then ripple-insert the clip at the new position
    insert_clip_ripple(store, params)
}

pub fn set_opacity(
    store: &mut TimelineStore,
    id: ClipId,
    opacity: f32,
) -> Result<(), MutationError> {
    if !(0.0..=1.0).contains(&opacity) {
        return Err(MutationError::InvalidOpacity(opacity));
    }
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    store.opacity[idx] = opacity;
    Ok(())
}

pub fn set_transform(
    store: &mut TimelineStore,
    id: ClipId,
    transform: ClipTransform,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    store.transform[idx] = transform;
    Ok(())
}

pub fn set_blend_mode(
    store: &mut TimelineStore,
    id: ClipId,
    mode: BlendMode,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    store.set_blend_mode_at(idx, mode);
    Ok(())
}

pub fn set_crop(
    store: &mut TimelineStore,
    id: ClipId,
    crop: CropRect,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    store.set_crop_at(idx, crop);
    Ok(())
}

pub fn set_corner_pin(
    store: &mut TimelineStore,
    id: ClipId,
    pin: CornerPin,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    store.set_corner_pin_at(idx, pin);
    Ok(())
}

pub fn set_matte_mode(
    store: &mut TimelineStore,
    id: ClipId,
    mode: MatteMode,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    store.set_matte_mode_at(idx, mode);
    Ok(())
}

/// Adjusts the media source in-point without changing the clip's timeline start, end, or duration.
pub fn slip_clip(
    store: &mut TimelineStore,
    id: ClipId,
    delta_source_pts: i64,
) -> Result<(), MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    let new_source_in = store.source_in[idx] + delta_source_pts;
    if new_source_in < 0 {
        return Err(MutationError::SourceInBeforeStart {
            computed: new_source_in,
        });
    }
    store.source_in[idx] = new_source_in;
    Ok(())
}

/// Slides a clip along the timeline, adjusting the out-point of any adjacent preceding clip
/// and the in-point of any adjacent following clip on the same track.
pub fn slide_clip(
    store: &mut TimelineStore,
    id: ClipId,
    delta_pts: i64,
) -> Result<(), MutationError> {
    if delta_pts == 0 {
        return Ok(());
    }
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    let track = store.track_ids[idx];
    let old_in = store.pts_in[idx];
    let old_out = store.pts_out[idx];
    let new_in = old_in + delta_pts;
    let new_out = old_out + delta_pts;
    if new_in < 0 {
        return Err(MutationError::InvalidDuration {
            pts_in: new_in,
            pts_out: new_out,
        });
    }

    // Check preceding and following adjacent clips on same track
    for i in 0..store.ids.len() {
        if store.track_ids[i] == track && store.ids[i] != id {
            if store.pts_out[i] == old_in {
                store.pts_out[i] = new_in;
            }
            if store.pts_in[i] == old_out {
                store.pts_in[i] = new_out;
                store.source_in[i] += delta_pts;
            }
        }
    }

    store.pts_in[idx] = new_in;
    store.pts_out[idx] = new_out;
    store.sort_by_pts_in();
    Ok(())
}

/// Adjusts the cut point between two adjacent clips simultaneously without altering the timeline length.
pub fn roll_edit(
    store: &mut TimelineStore,
    left_id: ClipId,
    right_id: ClipId,
    delta_pts: i64,
) -> Result<(), MutationError> {
    let l_idx = store.index_of(left_id).ok_or(MutationError::ClipNotFound(left_id))?;
    let r_idx = store.index_of(right_id).ok_or(MutationError::ClipNotFound(right_id))?;

    let new_cut = store.pts_out[l_idx] + delta_pts;
    if new_cut <= store.pts_in[l_idx] {
        return Err(MutationError::TrimPastOppositeEnd);
    }
    if new_cut >= store.pts_out[r_idx] {
        return Err(MutationError::TrimPastOppositeEnd);
    }

    let r_source_delta = delta_pts;
    if store.source_in[r_idx] + r_source_delta < 0 {
        return Err(MutationError::SourceInBeforeStart {
            computed: store.source_in[r_idx] + r_source_delta,
        });
    }

    store.pts_out[l_idx] = new_cut;
    store.pts_in[r_idx] = new_cut;
    store.source_in[r_idx] += r_source_delta;
    store.sort_by_pts_in();
    Ok(())
}

/// Duplicates an existing clip at a target track and timeline in-point.
pub fn duplicate_clip_to(
    store: &mut TimelineStore,
    id: ClipId,
    target_track: TrackId,
    target_pts_in: i64,
) -> Result<ClipId, MutationError> {
    let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
    let duration = store.pts_out[idx] - store.pts_in[idx];
    let params = ClipInsertParams {
        track_id: target_track,
        source_id: store.source_ids[idx],
        kind: store.kind[idx].clone(),
        pts_in: target_pts_in,
        pts_out: target_pts_in + duration,
        source_in: store.source_in[idx],
        layer_order: store.layer_order[idx],
        opacity: store.opacity[idx],
        transform: store.transform[idx],
        blend_mode: store.blend_mode_at(idx),
        crop: *store.crop_at(idx),
        corner_pin: *store.corner_pin_at(idx),
        matte_mode: store.matte_mode_at(idx),
        volume: store.volume[idx],
        pan: store.pan[idx],
        audio_muted: store.audio_muted[idx],
        fade_in_pts: store.fade_in_pts[idx],
        fade_out_pts: store.fade_out_pts[idx],
        speed: store.speed[idx],
        pitch: store.pitch[idx],
    };
    insert_clip(store, params)
}

/// Moves multiple clips concurrently by delta_pts.
pub fn batch_move_clips(
    store: &mut TimelineStore,
    clip_ids: &[ClipId],
    delta_pts: i64,
) -> Result<(), MutationError> {
    if delta_pts == 0 {
        return Ok(());
    }
    for &id in clip_ids {
        let idx = store.index_of(id).ok_or(MutationError::ClipNotFound(id))?;
        let new_in = store.pts_in[idx] + delta_pts;
        let new_out = store.pts_out[idx] + delta_pts;
        if new_in < 0 {
            return Err(MutationError::InvalidDuration {
                pts_in: new_in,
                pts_out: new_out,
            });
        }
        store.pts_in[idx] = new_in;
        store.pts_out[idx] = new_out;
    }
    store.sort_by_pts_in();
    Ok(())
}

#[derive(Debug)]
pub enum MutationError {
    ClipNotFound(ClipId),
    InvalidDuration { pts_in: i64, pts_out: i64 },
    InvalidOpacity(f32),
    TrimPastOppositeEnd,
    SourceInBeforeStart { computed: i64 },
}
