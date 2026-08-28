// src/render/resource.rs

use crate::render::device::GpuDevice;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static VIEW_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ViewId(pub u64);

impl ViewId {
    pub fn new() -> Self {
        ViewId(VIEW_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ViewId {
    fn default() -> Self {
        Self::new()
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureAccess {
    Sampled,
    StorageRead,
    StorageWrite,
    ColorAttachment,
    CopySrc,
    CopyDst,
}

impl TextureAccess {
    pub fn to_wgpu_usage(self) -> wgpu::TextureUsages {
        match self {
            Self::Sampled         => wgpu::TextureUsages::TEXTURE_BINDING,
            Self::StorageRead     => wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            Self::StorageWrite    => wgpu::TextureUsages::STORAGE_BINDING,
            Self::ColorAttachment => wgpu::TextureUsages::RENDER_ATTACHMENT,
            Self::CopySrc         => wgpu::TextureUsages::COPY_SRC,
            Self::CopyDst         => wgpu::TextureUsages::COPY_DST,
        }
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
    pub reads:   Vec<(ResourceId, TextureAccess)>,
    pub writes:  Vec<(ResourceId, TextureAccess)>,
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
    pub fn read(&mut self, id: ResourceId, access: TextureAccess) {
        self.reads.push((id, access));
    }

    /// Declare that this node writes to an existing resource.
    pub fn write(&mut self, id: ResourceId, access: TextureAccess) {
        self.writes.push((id, access));
    }

    /// Declare that this node creates a new transient resource and immediately writes it.
    pub fn create(&mut self, descriptor: ResourceDescriptor, write_access: TextureAccess) -> ResourceId {
        let id = ResourceId::next(&mut self.id_counter);
        self.creates.push((id, descriptor));
        self.writes.push((id, write_access));
        id
    }
}

/// Resolved texture handle. Handed to nodes during execute().
/// Wraps a reference to a wgpu::Texture with its precomputed view.
pub struct ResolvedResource<'a> {
    pub texture: &'a wgpu::Texture,
    pub view:    &'a wgpu::TextureView,
    pub view_id: ViewId,
    pub format:  wgpu::TextureFormat,
    pub width:   u32,
    pub height:  u32,
}

/// Pre-allocated pool of wgpu textures for transient resources.
/// Textures are bucketed by (format, width, height) so they can be reused
/// across frames without reallocation.
pub struct TransientTexturePool {
    /// Key: (format, width, height)
    /// Value: stack of available textures, their pre-created views, and their unique ViewIds
    buckets: HashMap<TextureKey, Vec<(wgpu::Texture, wgpu::TextureView, ViewId)>>,
}

#[derive(PartialEq, Eq, Hash)]
struct TextureKey {
    format: wgpu::TextureFormat,
    usage:  wgpu::TextureUsages,
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
        usage:  wgpu::TextureUsages,
        width:  u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, ViewId) {
        let key = TextureKey { format, usage, width, height };
        if let Some(res) = self.buckets.get_mut(&key).and_then(|v| v.pop()) {
            res
        } else {
            let tex = device.create_texture(
                Some("transient_texture"),
                width,
                height,
                format,
                usage,
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view, ViewId::new())
        }
    }

    /// Return a texture to the pool for reuse next frame.
    pub fn release(&mut self, resource: (wgpu::Texture, wgpu::TextureView, ViewId)) {
        let key = TextureKey {
            format: resource.0.format(),
            usage:  resource.0.usage(),
            width:  resource.0.width(),
            height: resource.0.height(),
        };
        let bucket = self.buckets.entry(key).or_default();
        if bucket.len() < 8 {
            bucket.push(resource);
        }
    }
}
