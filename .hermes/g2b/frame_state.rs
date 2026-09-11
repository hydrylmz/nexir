// src/render/frame_state.rs

use crate::timeline::transform::{ClipTransform, BlendMode, CropRect, CornerPin, MatteMode, ClipEffects};
use crate::timeline::ids::SourceId;
use crate::timeline::source::DecodedFrameMeta;
use crate::render::resource::ImportedResources;

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
    pub effects:       ClipEffects,
    /// Pixel layout and colour metadata of the frame sitting in `texture_slot`,
    /// as reported by the decoder for those exact pixels.
    ///
    /// P1.6 — this replaces a bare `is_nv12: bool`.  The render graph reads bit
    /// depth, chroma layout, code alignment and every colour property from here
    /// instead of from the source registry, because the decoder may have converted
    /// the frame and the container's metadata then describes something else.
    pub frame_meta:    DecodedFrameMeta,
    pub kind:          crate::timeline::store::ClipKind,
}

impl ClipRenderEntry {
    /// Whether the slot holds semi-planar chroma (NV12 / P010).
    pub fn is_nv12(&self) -> bool {
        self.frame_meta.layout.semi_planar
    }
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
    /// Textures supplied from outside the graph for this frame — the GPU decode
    /// path's Y/UV planes, which the decoder owns and the graph only reads.
    ///
    /// G2b. Per frame rather than per compiled graph because the graph shape is
    /// stable while the target holding a given source's planes is not. Empty is the
    /// CPU upload path and means the graph allocates every resource from its pool,
    /// exactly as before.
    pub imported:      ImportedResources,
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
            imported: ImportedResources::new(),
        }
    }

    /// Sort clips by layer_order ascending (painter's algorithm).
    pub fn sort_clips(&mut self) {
        self.clips.sort_unstable_by_key(|c| c.layer_order);
    }
}
