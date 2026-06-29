use crate::timeline::{
    store::TimelineStore,
    track::{Track, TrackList, TrackError},
    source::{SourceRegistry, VideoStreamInfo, AudioStreamInfo},
    ids::{TrackId, ClipId, SourceId},
    mutation::{insert_clip, ClipInsertParams, MutationError},
    rational::Rational,
};
use std::path::PathBuf;

/// Project-level settings (resolution, framerate, timebase).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProjectSettings {
    pub width: u32,
    pub height: u32,
    /// Frames per second as a rational number, e.g. `Rational::new(30, 1)`.
    pub frame_rate: Rational,
    /// All `pts_in`/`pts_out` values in `TimelineStore` are expressed in
    /// units of this timebase.
    pub timebase: Rational,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: Rational::new(30, 1),
            timebase: Rational::TIMEBASE_90K,
        }
    }
}

/// The root object that owns an entire editing session.
///
/// Bundles together:
/// - `settings`: project-wide resolution, framerate, and timebase.
/// - `tracks`: ordered list of video/audio/text tracks.
/// - `clips`: structure-of-arrays store for all clips on the timeline.
/// - `sources`: registry of imported media files.
pub struct Project {
    pub name: String,
    pub settings: ProjectSettings,
    pub tracks: TrackList,
    pub clips: TimelineStore,
    pub sources: std::sync::Arc<std::sync::RwLock<SourceRegistry>>,
}

impl Project {
    /// Create a blank project with the given name and default 1080p/30fps settings.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            settings: ProjectSettings::default(),
            tracks: TrackList::new(),
            clips: TimelineStore::new(),
            sources: std::sync::Arc::new(std::sync::RwLock::new(SourceRegistry::default())),
        }
    }

    /// Create a blank project with explicit settings.
    pub fn with_settings(name: impl Into<String>, settings: ProjectSettings) -> Self {
        Self {
            name: name.into(),
            settings,
            tracks: TrackList::new(),
            clips: TimelineStore::new(),
            sources: std::sync::Arc::new(std::sync::RwLock::new(SourceRegistry::default())),
        }
    }

    // ─────────────────────────────────────────────
    // Track helpers
    // ─────────────────────────────────────────────

    /// Add a new video track and return its id.
    pub fn add_video_track(&mut self, name: impl Into<String>) -> Result<TrackId, TrackError> {
        // We allocate a placeholder id — TrackList::push() will use its own
        // internal next_id counter and return the canonical assigned id.
        let placeholder = TrackId(self.tracks.len() as u8);
        let track = Track::new_video(placeholder, name.into());
        self.tracks.push(track)
    }

    /// Add a new audio track and return its id.
    pub fn add_audio_track(&mut self, name: impl Into<String>) -> Result<TrackId, TrackError> {
        let placeholder = TrackId(self.tracks.len() as u8);
        let track = Track::new_audio(placeholder, name.into());
        self.tracks.push(track)
    }

    // ─────────────────────────────────────────────
    // Source helpers
    // ─────────────────────────────────────────────

    /// Register a media file and return its `SourceId`.
    pub fn register_source(
        &mut self,
        path: PathBuf,
        video_info: Option<VideoStreamInfo>,
        audio_info: Option<AudioStreamInfo>,
    ) -> SourceId {
        self.sources.write().unwrap().register(path, video_info, audio_info)
    }

    // ─────────────────────────────────────────────
    // Clip helpers
    // ─────────────────────────────────────────────

    /// Insert a clip. `pts_in` and `pts_out` are in project timebase ticks.
    pub fn insert_clip(&mut self, mut params: ClipInsertParams) -> Result<ClipId, MutationError> {
        params.layer_order = self.tracks.index_of(params.track_id).unwrap_or(0) as u16;
        insert_clip(&mut self.clips, params)
    }

    /// Insert a clip using overwrite mode (trims/removes any overlapping clips on the same track).
    pub fn insert_clip_overwrite(&mut self, mut params: ClipInsertParams) -> Result<ClipId, MutationError> {
        params.layer_order = self.tracks.index_of(params.track_id).unwrap_or(0) as u16;
        use crate::timeline::mutation::insert_clip_overwrite;
        insert_clip_overwrite(&mut self.clips, params)
    }

    /// Move a clip to a new start position (keeping its duration and source-in).
    /// Returns the new ClipId (the clip is removed and re-inserted, so the id changes).
    pub fn move_clip(&mut self, id: ClipId, new_pts_in: i64) -> Result<ClipId, MutationError> {
        use crate::timeline::mutation::move_clip;
        move_clip(&mut self.clips, id, new_pts_in)
    }

    /// Move a clip to a different track AND new start position.
    pub fn move_clip_to_track(
        &mut self,
        id:         ClipId,
        new_track:  TrackId,
        new_pts_in: i64,
    ) -> Result<ClipId, MutationError> {
        use crate::timeline::mutation::{remove_clip, insert_clip};
        let idx = self.clips.index_of(id).ok_or(crate::timeline::mutation::MutationError::ClipNotFound(id))?;
        let duration   = self.clips.pts_out_at(idx) - self.clips.pts_in_at(idx);
        let source_id  = self.clips.source_id_at(idx);
        let source_in  = self.clips.source_in_at(idx);
        let layer      = self.tracks.index_of(new_track).unwrap_or(0) as u16;
        let opacity    = self.clips.opacity_at(idx);
        let transform  = *self.clips.transform_at(idx);
        let volume     = self.clips.volume_at(idx);
        let pan        = self.clips.pan_at(idx);
        let audio_muted = self.clips.audio_muted_at(idx);
        let speed      = self.clips.speed_at(idx);
        let pitch      = self.clips.pitch_at(idx);
        remove_clip(&mut self.clips, id)?;
        insert_clip(&mut self.clips, ClipInsertParams {
            track_id:    new_track,
            source_id,
            pts_in:      new_pts_in,
            pts_out:     new_pts_in + duration,
            source_in,
            layer_order: layer,
            opacity,
            transform,
            volume,
            pan,
            audio_muted,
            speed,
            pitch,
        })
    }

    // ─────────────────────────────────────────────
    // Time helpers
    // ─────────────────────────────────────────────

    /// Convert a frame number to project timebase ticks (pts).
    ///
    /// For the default 30 fps / 90 kHz timebase: 1 frame = 3 000 ticks.
    pub fn frame_to_pts(&self, frame: i64) -> i64 {
        let fr = &self.settings.frame_rate;
        let tb = &self.settings.timebase;
        // pts = frame * (tb.den / fr.num)  [when tb.num == 1]
        // General: pts = frame * fr.den * tb.den / (fr.num * tb.num)
        frame * fr.den * tb.den / (fr.num * tb.num)
    }

    /// Convert project timebase ticks (pts) to the nearest frame number.
    pub fn pts_to_frame(&self, pts: i64) -> i64 {
        let fr = &self.settings.frame_rate;
        let tb = &self.settings.timebase;
        pts * fr.num * tb.num / (fr.den * tb.den)
    }

    /// Total project duration in frames (largest `pts_out` across all clips).
    pub fn duration_frames(&self) -> i64 {
        if self.clips.is_empty() {
            return 0;
        }
        // pts_out is stored in parallel with pts_in; iterate to find the max.
        let n = self.clips.len();
        let max_pts_out = (0..n)
            .map(|i| self.clips.pts_out_at(i))
            .max()
            .unwrap_or(0);
        self.pts_to_frame(max_pts_out)
    }
}
