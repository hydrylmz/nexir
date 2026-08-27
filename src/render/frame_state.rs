// src/render/frame_state.rs

use crate::timeline::transform::{ClipTransform, BlendMode, CropRect, CornerPin, MatteMode};
use crate::timeline::ids::SourceId;

/// One clip's contribution to the current frame.
#[derive(Clone, Debug)]
pub struct ClipRenderEntry {
    pub source_id:     SourceId,
    /// Which GPU texture slot holds this clip's decoded RGBA data.
    pub texture_slot:  u32,
    /// Z-order: 0 = bottom, higher = top.
    pub layer_order:   u16,
    /// Source pixel dimensions
    pub clip_width:    u32,
    pub clip_height:   u32,
    pub transform:     ClipTransform,
    pub opacity:       f32,
    pub blend_mode:    BlendMode,
    pub crop:          CropRect,
    pub corner_pin:    CornerPin,
    pub matte_mode:    MatteMode,
    pub is_nv12:       bool,
    pub kind:          crate::timeline::store::ClipKind,
}

/// Everything the graph needs for one frame.
pub struct FrameState {
    pub pts:           i64,
    pub canvas_width:  u32,
    pub canvas_height: u32,
    /// All clips to composite this frame, sorted by layer_order ascending.
    pub clips:         Vec<ClipRenderEntry>,
    /// Pre-uploaded RGBA textures for test patterns (Phase 2 only).
    pub test_textures: Vec<wgpu::Texture>,
}

impl FrameState {
    /// Construct a minimal FrameState for testing.
    pub fn test_empty(canvas_width: u32, canvas_height: u32) -> Self {
        Self {
            pts: 0,
            canvas_width,
            canvas_height,
            clips: Vec::new(),
            test_textures: Vec::new(),
        }
    }

    /// Sort clips by layer_order ascending (painter's algorithm).
    pub fn sort_clips(&mut self) {
        self.clips.sort_unstable_by_key(|c| c.layer_order);
    }
}
