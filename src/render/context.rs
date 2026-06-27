// src/render/context.rs

use std::collections::HashMap;
use crate::render::resource::{ResourceId, ResolvedResource};

pub struct RenderContext {
    textures: HashMap<ResourceId, wgpu::Texture>,
    views:    HashMap<ResourceId, wgpu::TextureView>,
}

impl RenderContext {
    /// Construct from the resolved allocations produced by CompiledGraph::execute().
    pub fn new(
        textures: HashMap<ResourceId, wgpu::Texture>,
        views:    HashMap<ResourceId, wgpu::TextureView>,
    ) -> Self {
        Self { textures, views }
    }

    /// Look up a resolved resource by its ResourceId.
    ///
    /// # Panics
    /// Panics if `id` was not declared by any node's declare_resources().
    pub fn get(&self, id: ResourceId) -> ResolvedResource<'_> {
        let texture = self.textures.get(&id).unwrap_or_else(|| panic!("ResourceId {:?} not allocated", id));
        let view = self.views.get(&id).unwrap_or_else(|| panic!("ResourceId {:?} view not created", id));
        
        ResolvedResource {
            texture,
            view,
            format: texture.format(),
            width: texture.width(),
            height: texture.height(),
        }
    }

    /// Check if a resource was allocated (safe alternative to get()).
    pub fn contains(&self, id: ResourceId) -> bool {
        self.textures.contains_key(&id)
    }

    /// Optionally look up a resolved resource — returns None if not allocated.
    pub fn try_get(&self, id: ResourceId) -> Option<ResolvedResource<'_>> {
        let texture = self.textures.get(&id)?;
        let view    = self.views.get(&id)?;
        Some(ResolvedResource {
            texture,
            view,
            format: texture.format(),
            width:  texture.width(),
            height: texture.height(),
        })
    }

    /// Extract the owned textures back out for pool release.
    pub fn into_textures(self) -> HashMap<ResourceId, wgpu::Texture> {
        self.textures
    }
}
