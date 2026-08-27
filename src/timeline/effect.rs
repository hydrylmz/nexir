// src/timeline/effect.rs

use crate::timeline::ids::{EffectId, ShaderId};
use crate::timeline::transform::EffectParams;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum EffectKind {
    ColorCorrection,
    GaussianBlur,
    Sharpen,
    Vignette,
    ChromaKey,
    Transform2D,
    /// A user-supplied WGSL shader. The ShaderId is looked up in the ShaderRegistry (Phase 3).
    Custom(ShaderId),
}

/// SoA effect store. All slices are always the same length.
pub struct EffectStore {
    ids:     Vec<EffectId>,

    enabled: Vec<bool>,
    kind:    Vec<EffectKind>,

    params:  Vec<EffectParams>,

    next_id: u32,
}

impl Default for EffectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectStore {
    pub fn new() -> Self {
        EffectStore {
            ids: Vec::new(),
            enabled: Vec::new(),
            kind: Vec::new(),
            params: Vec::new(),
            next_id: 0,
        }
    }

    /// Append a new effect and return its ID.
    pub fn push(&mut self, kind: EffectKind) -> EffectId {

        let id = EffectId(self.next_id);
        self.next_id += 1;
        self.ids.push(id);
        self.enabled.push(true);
        self.kind.push(kind);
        self.params.push(EffectParams::zero());
        id
    }

    /// Disable an effect (soft-delete). Does NOT remove from arrays.
    pub fn disable(&mut self, id: EffectId) -> Result<(), EffectError> {
        self.ids.iter().position(|&e_id| e_id == id)
            .map(|idx| self.enabled[idx] = false)
            .ok_or(EffectError::NotFound(id))
    }

    /// Re-enable a previously disabled effect.
    pub fn enable(&mut self, id: EffectId) -> Result<(), EffectError> {
        self.ids.iter().position(|&e_id| e_id == id)
            .map(|idx| self.enabled[idx] = true)
            .ok_or(EffectError::NotFound(id))
    }

    /// Update parameters for an effect.
    pub fn set_params(&mut self, id: EffectId, params: EffectParams) -> Result<(), EffectError> {
        self.ids.iter().position(|&e_id| e_id == id)
            .map(|idx| self.params[idx] = params)
            .ok_or(EffectError::NotFound(id))

    }

    /// Get params (read-only).
    pub fn get_params(&self, id: EffectId) -> Result<&EffectParams, EffectError> {
        self.ids.iter().position(|&e_id| e_id == id)
            .map(|idx| &self.params[idx])
            .ok_or(EffectError::NotFound(id))        
    }

    pub fn iter_clip_effects(
        &self,
        start: u32,
        count: u16,
    ) -> impl Iterator<Item = (EffectId, &EffectKind, &EffectParams)> {
        let range = start as usize .. (start + count as u32) as usize;
        
        self.enabled.get(range.clone()).unwrap_or(&[]).iter()
            .zip(self.kind.get(range.clone()).unwrap_or(&[]).iter())
            .zip(self.params.get(range.clone()).unwrap_or(&[]).iter())
            .enumerate()
            .filter_map(move |(i, ((&enabled, kind), params))| {
                if enabled {
                    Some((EffectId(start + i as u32), kind, params))
                } else {
                    None
                }
            })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

#[derive(Debug)]
pub enum EffectError {
    NotFound(EffectId),
}