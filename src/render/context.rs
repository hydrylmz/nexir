// src/render/context.rs

use crate::render::resource::{ResourceId, ResolvedResource, ViewId};

pub struct RenderContext {
    resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, ViewId)>>,
}

impl RenderContext {
    /// Construct from the resolved allocations produced by CompiledGraph::execute().
    pub fn new(resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, ViewId)>>) -> Self {
        Self { resources }
    }

    /// Look up a resolved resource by its ResourceId.
    ///
    /// # Panics
    /// Panics if `id` was not declared by any node's declare_resources().
    pub fn get(&self, id: ResourceId) -> ResolvedResource<'_> {
        self.try_get(id).unwrap_or_else(|| panic!("Resource {:?} not found in context", id))
    }

    /// Check if a resource was allocated (safe alternative to get()).
    pub fn contains(&self, id: ResourceId) -> bool {
        self.resources.get(id.0 as usize).map_or(false, |opt| opt.is_some())
    }

    /// Try to get a resolved resource
    pub fn try_get(&self, id: ResourceId) -> Option<ResolvedResource<'_>> {
        self.resources.get(id.0 as usize)
            .and_then(|opt| opt.as_ref())
            .map(|(tex, view, view_id)| ResolvedResource {
                texture: tex,
                view,
                view_id: *view_id,
                format:  tex.format(),
                width:   tex.width(),
                height:  tex.height(),
            })
    }

    /// Take ownership of all allocated textures, returning them so they can be pooled
    pub fn into_resources(self) -> Vec<Option<(wgpu::Texture, wgpu::TextureView, ViewId)>> {
        self.resources
    }
}
