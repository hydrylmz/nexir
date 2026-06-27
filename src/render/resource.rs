// src/render/resource.rs

use crate::render::device::GpuDevice;
use std::collections::HashMap;

/// A stable name for a texture resource within the Render Graph.
/// Nodes use ResourceIds to declare dependencies — they never own textures.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ResourceId(pub u32);

impl ResourceId {
    /// Well-known IDs for the fixed resources every graph has.
    /// Nodes can declare reads/writes against these without coordination.
    pub const FINAL_COLOR: ResourceId = ResourceId(0); // the RTT output texture
    pub const SCREEN:      ResourceId = ResourceId(1); // the swapchain image (write-only)

    /// Allocate a unique transient ResourceId at graph build time.
    /// Simply increment the graph's internal counter.
    pub fn next(id_counter: &mut u32) -> Self {
        let id = ResourceId(*id_counter);
        *id_counter += 1;
        id
    }
}

/// Describes how a texture resource should be created if it doesn't already exist.
#[derive(Clone, Debug)]
pub struct ResourceDescriptor {
    pub label:  Option<String>,
    /// Width and height. Use `ResolutionSource::Canvas` to match the project canvas size,
    /// or `ResolutionSource::Fixed(w,h)` for fixed-size intermediates.
    pub size:   ResolutionSource,
    pub format: wgpu::TextureFormat,
    pub usage:  wgpu::TextureUsages,
}

#[derive(Clone, Debug)]
pub enum ResolutionSource {
    /// Matches the project canvas (e.g. 3840×2160 for 4K). Resolved at compile time.
    Canvas,
    /// A hardcoded size. Used for LUT textures, thumbnails, etc.
    Fixed(u32, u32),
}

/// Declares the resources a node reads and writes.
/// Passed to `RenderNode::declare_resources()` during graph compilation.
pub struct ResourceBuilder {
    pub reads:   Vec<ResourceId>,
    pub writes:  Vec<ResourceId>,
    pub creates: Vec<(ResourceId, ResourceDescriptor)>,
    id_counter:  u32,
}

impl ResourceBuilder {
    pub fn new(id_counter_start: u32) -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            creates: Vec::new(),
            id_counter: id_counter_start,
        }
    }

    /// Declare that this node reads an existing resource.
    pub fn read(&mut self, id: ResourceId) {
        self.reads.push(id);
    }

    /// Declare that this node writes to an existing resource.
    pub fn write(&mut self, id: ResourceId) {
        self.writes.push(id);
    }

    /// Declare that this node creates a new transient resource and immediately writes it.
    pub fn create(&mut self, descriptor: ResourceDescriptor) -> ResourceId {
        let id = ResourceId::next(&mut self.id_counter);
        self.creates.push((id, descriptor));
        self.writes.push(id);
        id
    }
}

/// Resolved texture handle. Handed to nodes during execute().
/// Wraps a reference to a wgpu::Texture with its precomputed view.
pub struct ResolvedResource<'a> {
    pub texture: &'a wgpu::Texture,
    pub view:    &'a wgpu::TextureView,
    pub format:  wgpu::TextureFormat,
    pub width:   u32,
    pub height:  u32,
}

/// Pre-allocated pool of wgpu textures for transient resources.
/// Textures are bucketed by (format, width, height) so they can be reused
/// across frames without reallocation.
pub struct TransientTexturePool {
    /// Key: (format, width, height)
    /// Value: stack of available textures
    buckets: HashMap<TextureKey, Vec<wgpu::Texture>>,
}

#[derive(PartialEq, Eq, Hash)]
struct TextureKey {
    format: wgpu::TextureFormat,
    width:  u32,
    height: u32,
}

impl Default for TransientTexturePool {
    fn default() -> Self {
        Self::new()
    }
}

impl TransientTexturePool {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Acquire a texture of the given format and size. Reuses a pooled one or creates a new one.
    pub fn acquire(
        &mut self,
        device: &GpuDevice,
        format: wgpu::TextureFormat,
        width:  u32,
        height: u32,
    ) -> wgpu::Texture {
        let key = TextureKey { format, width, height };
        if let Some(tex) = self.buckets.get_mut(&key).and_then(|v| v.pop()) {
            tex
        } else {
            device.create_texture(
                Some("transient_texture"),
                width,
                height,
                format,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
            )
        }
    }

    /// Return a texture to the pool for reuse next frame.
    pub fn release(&mut self, tex: wgpu::Texture) {
        let key = TextureKey {
            format: tex.format(),
            width: tex.width(),
            height: tex.height(),
        };
        let bucket = self.buckets.entry(key).or_default();
        if bucket.len() < 8 {
            bucket.push(tex);
        }
    }
}
