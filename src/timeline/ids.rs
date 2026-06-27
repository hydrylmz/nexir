// ID types for timeline


/// Indexes into TimelineStore parallel arrays.
/// Invariant: always < TimelineStore::len()
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClipId(pub(crate) u32);

/// Indexes into the Track list. Max 256 tracks (fits u8, cheap SIMD compare).
/// u8 is intentional — a timeline with > 256 tracks is a UI problem, not mine :3
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TrackId(pub(crate) u8);

/// Indexes into EffectStore parallel arrays.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EffectId(pub(crate) u32);

/// Indexes into SourceRegistry parallel arrays.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SourceId(pub(crate) u32);

/// Opaque handle to a compiled WGSL shader. Not used in Phase 1
/// but referenced by EffectKind so we define it here.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ShaderId(pub(crate) u32);

impl ClipId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl TrackId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl EffectId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl SourceId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}