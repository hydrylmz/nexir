// src/timeline/track.rs

use crate::timeline::ids::TrackId;

#[derive(Debug, Clone)]
pub struct Track {
    pub id:        TrackId,
    pub kind:      TrackKind,
    pub name:      String,    
    pub mute:      bool,
    pub solo:      bool,
    pub locked:    bool,      
    pub height_px: u16,       
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackKind {
    Video,
    Audio {
        sample_rate: u32,  
        channels:    u8,   
    },
    Text,
    Effect, 
}

impl Track {
    /// Construct a named video track.
    pub fn new_video(id: TrackId, name: impl Into<String>) -> Self {
        Track { id, kind: TrackKind::Video, name: name.into(), mute: false, solo: false, locked: false, height_px: 80 }
    }

    /// Construct a stereo audio track at 48 kHz.
    pub fn new_audio(id: TrackId, name: impl Into<String>) -> Self {
        Track { id, kind: TrackKind::Audio { sample_rate: 48_000, channels: 2 }, name: name.into(), mute: false, solo: false, locked: false, height_px: 60 }
    }

    /// Returns true if this track should contribute to rendering.
    pub fn is_active(&self, any_track_is_soloed: bool) -> bool {
        if self.mute {
            false
        } else if any_track_is_soloed {
            self.solo
        } else {
            true
        }
    }
}

/// Container for all tracks, maintaining insertion order for layer compositing.
/// The z-order of video tracks is: tracks[0] is bottom, tracks[last] is top.
pub struct TrackList {
    tracks: Vec<Track>,
    next_id: u8,  
}

impl Default for TrackList {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackList {
    pub fn new() -> Self {
        TrackList { tracks: (Vec::new()), next_id: (0) }
    }

    /// Add a track to the top of the stack and return its assigned ID.
    pub fn push(&mut self, track: Track) -> Result<TrackId, TrackError> {
        if self.next_id < 255 {
            let id = TrackId(self.next_id);
            self.next_id += 1;
            self.tracks.push(track);
            Ok(id)
        } else {
            Err(TrackError::TrackLimitReached)
        }
    }

    /// Look up a track by ID. O(N) — tracks are few, this is fine.
    pub fn get(&self, id: TrackId) -> Option<&Track> {
        self.tracks.iter().find(|t| t.id == id)
    }

    /// Mutable lookup.
    pub fn get_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    /// Returns true if any track in the list has solo=true.
    pub fn any_soloed(&self) -> bool {
        self.tracks.iter().any(|t| t.solo)
    }

    /// Reorder a track: move it from its current position to `new_index`.
    pub fn reorder(&mut self, id: TrackId, new_index: usize) -> Result<(), TrackError> {
        let current_index = self.tracks.iter().position(|t| t.id == id);
        if let Some(current_index) = current_index {
            if new_index <= self.tracks.len() {
                let track = self.tracks.remove(current_index);
                self.tracks.insert(new_index, track);
                Ok(())
            } else {
                Err(TrackError::TrackLimitReached)
            }
        } else {
            Err(TrackError::TrackNotFound(id))
        }

    }

    pub fn iter(&self) -> impl Iterator<Item = &Track> {
        self.tracks.iter()
    }

    pub fn len(&self) -> usize {

        self.tracks.len()
    }
}

#[derive(Debug)]
pub enum TrackError {
    TrackLimitReached,  
    TrackNotFound(TrackId),
}