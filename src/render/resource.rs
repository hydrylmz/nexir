// src/render/resource.rs

use crate::render::device::GpuDevice;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
    /// Resources this node reads that are supplied from OUTSIDE the graph — an
    /// NVDEC decode target, not a pooled intermediate. See [`ImportedTexture`].
    ///
    /// Separate from `creates` because the graph must not allocate one, and
    /// separate from a bare `read` because a read with no producer is the
    /// [`crate::render::graph::GraphError::MissingProducer`] diagnostic — an import
    /// *is* its own producer, and saying so here is what keeps that error meaningful
    /// for the case it was written for.
    pub imports: Vec<(ResourceId, TextureAccess)>,
    id_counter:  u32,
}

impl ResourceBuilder {
    pub fn new(id_counter_start: u32) -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
            creates: Vec::new(),
            imports: Vec::new(),
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

    /// Declare that this node reads a texture the graph neither creates nor owns,
    /// bound per frame via [`crate::render::frame_state::FrameState::imported`].
    ///
    /// The id is chosen by the caller, exactly like [`Self::read`], because the node
    /// and whoever binds the texture must agree on it — the node holds it in a field
    /// and the frame's [`ImportedResources`] is keyed by the same value.
    pub fn import(&mut self, id: ResourceId, access: TextureAccess) {
        self.imports.push((id, access));
        self.reads.push((id, access));
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

/// A texture the pool owns: handed out by [`TransientTexturePool::acquire`] and
/// given back by [`TransientTexturePool::release`].
pub type TransientResource = (wgpu::Texture, wgpu::TextureView, ViewId);

/// A texture the graph BINDS but does not own.
///
/// G2b. The interop decode path makes NVDEC's destination surface *the* Y/UV
/// texture a graph reads ([`crate::interop::decode_interop::DecodeInteropTarget`]),
/// so the texture's lifetime belongs to whoever decoded into it — `IoLayer`'s
/// per-source target, reused for every frame from that source — and not to the
/// frame that happens to sample it. Two properties follow, and both are
/// load-bearing:
///
/// - **It must never reach [`TransientTexturePool::release`].** The pool buckets by
///   (format, usage, size) and hands textures to whoever asks next, so a released
///   import would later be handed to an unrelated resource while the decoder still
///   writes into it — a torn frame, with no error and nothing in the pool's own
///   counters to show for it. [`GraphResource::into_transient`] is the single
///   funnel that decides, and it is what `CompiledGraph::execute` releases through.
/// - **Its [`ViewId`] is created ONCE, here, and cloned every frame.** Eleven nodes
///   cache their bind group on the `ViewId`s they were handed (AGENTS.md gotcha 16);
///   minting a fresh id per frame for the same underlying texture rebuilds every one
///   of those caches every frame, which is exactly the cost the FIFO pool change
///   removed.
///
/// `Arc` on both fields because a frame binds a texture the decoder owns: cloning
/// into [`ImportedResources`] must not move it, and the view is created once
/// alongside the id rather than per frame.
#[derive(Clone)]
pub struct ImportedTexture {
    texture: Arc<wgpu::Texture>,
    view:    Arc<wgpu::TextureView>,
    view_id: ViewId,
}

impl ImportedTexture {
    /// Import an externally-owned texture, creating its view and its one stable
    /// `ViewId` now.
    ///
    /// The texture must already carry the usages the reading nodes need
    /// (`TEXTURE_BINDING` for a sampled plane): the compiler aggregates usages only
    /// for resources it *creates*, so an import's flags are the importer's
    /// responsibility and a missing one surfaces as a wgpu validation error at bind
    /// time rather than as a graph error.
    pub fn new(texture: Arc<wgpu::Texture>) -> Self {
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        Self { texture, view, view_id: ViewId::new() }
    }

    /// The stable id every bind-group cache keys on. Equal across clones, and
    /// therefore across frames.
    pub fn view_id(&self) -> ViewId {
        self.view_id
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn resolved(&self) -> ResolvedResource<'_> {
        ResolvedResource {
            texture: &self.texture,
            view:    &self.view,
            view_id: self.view_id,
            format:  self.texture.format(),
            width:   self.texture.width(),
            height:  self.texture.height(),
        }
    }
}

impl std::fmt::Debug for ImportedTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedTexture")
            .field("view_id", &self.view_id)
            .field("format", &self.texture.format())
            .field("width", &self.texture.width())
            .field("height", &self.texture.height())
            .finish()
    }
}

/// One decoded frame's Y and UV planes, as textures the graph binds but does not
/// own.
///
/// G2c. The pair travels together from
/// [`crate::io::interop_decode::InteropDecodeTargets`] to whichever graph builder
/// is constructing this frame's nodes, because the two planes are meaningless
/// apart: `YuvToRgbNode` samples both, and binding one while the other is stale
/// is a frame with last frame's chroma over this frame's luma — no error, and
/// nothing in the pool's counters to show for it.
///
/// Cloning is what binding into a frame does, and it clones the `Arc`s rather
/// than the textures, so both planes keep the one [`ViewId`] they were minted
/// with (AGENTS.md gotcha 18).
#[derive(Clone, Debug)]
pub struct InteropPlanes {
    /// Luma, `R8Unorm` at the frame's full size.
    pub y:  ImportedTexture,
    /// Interleaved chroma, `Rg8Unorm` at half size.
    pub uv: ImportedTexture,
}

/// One slot of a [`crate::render::context::RenderContext`]: either a texture the
/// pool lent for this frame, or one the graph was lent from outside.
///
/// The distinction exists for exactly one reason — the release path. Everything
/// else about the two is identical from a node's point of view, which is why
/// [`Self::resolved`] erases it and no node has to know which kind it was handed.
pub enum GraphResource {
    /// Acquired from [`TransientTexturePool`] this frame; must be returned to it.
    Transient(TransientResource),
    /// Owned elsewhere, for longer than this frame; must NOT be returned.
    Imported(ImportedTexture),
}

impl GraphResource {
    pub fn resolved(&self) -> ResolvedResource<'_> {
        match self {
            Self::Transient((tex, view, view_id)) => ResolvedResource {
                texture: tex,
                view,
                view_id: *view_id,
                format:  tex.format(),
                width:   tex.width(),
                height:  tex.height(),
            },
            Self::Imported(imported) => imported.resolved(),
        }
    }

    pub fn view_id(&self) -> ViewId {
        match self {
            Self::Transient((_, _, id)) => *id,
            Self::Imported(imported) => imported.view_id,
        }
    }

    pub fn is_imported(&self) -> bool {
        matches!(self, Self::Imported(_))
    }

    /// The pool-release filter, and the whole point of this enum.
    ///
    /// `Some` only for a texture the pool lent. An import yields `None` and is
    /// dropped here, which releases this frame's `Arc` share and nothing else —
    /// the decoder's target stays alive and stays out of the pool's buckets.
    pub fn into_transient(self) -> Option<TransientResource> {
        match self {
            Self::Transient(res) => Some(res),
            Self::Imported(_) => None,
        }
    }
}

/// The imported textures one frame binds, keyed by the [`ResourceId`] the reading
/// node declared with [`ResourceBuilder::import`].
///
/// Carried on [`crate::render::frame_state::FrameState`] rather than on the
/// compiled graph, because the binding is per frame: the same graph shape reads
/// source 0's Y plane every frame, but *which* decode target currently holds that
/// plane is a property of the frame being scheduled. An empty set (the default) is
/// the CPU upload path, unchanged.
#[derive(Clone, Default, Debug)]
pub struct ImportedResources {
    entries: Vec<(ResourceId, ImportedTexture)>,
}

impl ImportedResources {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind (or rebind) `id` to an externally-owned texture for this frame.
    pub fn bind(&mut self, id: ResourceId, texture: ImportedTexture) {
        match self.entries.iter_mut().find(|(existing, _)| *existing == id) {
            Some(slot) => slot.1 = texture,
            None => self.entries.push((id, texture)),
        }
    }

    pub fn get(&self, id: ResourceId) -> Option<&ImportedTexture> {
        self.entries.iter().find(|(existing, _)| *existing == id).map(|(_, tex)| tex)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ResourceId, &ImportedTexture)> {
        self.entries.iter().map(|(id, tex)| (*id, tex))
    }
}

/// Pre-allocated pool of wgpu textures for transient resources.
/// Textures are bucketed by (format, width, height) so they can be reused
/// across frames without reallocation.
pub struct TransientTexturePool {
    /// Key: (format, width, height)
    /// Value: queue of available textures, their pre-created views, and their unique ViewIds
    ///
    /// **A queue, not a stack, and that is load-bearing.** `CompiledGraph::execute`
    /// acquires in ascending `ResourceId` order and releases in ascending order
    /// too, so a stack would pop them back out reversed — resource 1 receiving
    /// resource N's texture — and the assignment would alternate with period 2
    /// forever. Every node keys its bind-group cache on the `ViewId`s it was handed
    /// (`lut.rs:241`, `composite.rs:398`, nine more), so that reversal rebuilt all
    /// of them every frame while the pool reported a 0.6% miss rate and zero
    /// evictions. FIFO hands each resource its own texture back.
    /// `tests::a_repeated_frame_shape_reuses_each_resources_own_texture` pins it.
    buckets: HashMap<TextureKey, VecDeque<(wgpu::Texture, wgpu::TextureView, ViewId)>>,
    /// Textures handed out from a bucket, cumulative.
    hits: u64,
    /// Textures created because no pooled one was available, cumulative.
    ///
    /// Counted, not inferred: a 4K RGBA16Float texture is 66 MB, so a miss is a
    /// 66 MB allocation *and* the eventual free of whatever it replaced, in the
    /// middle of recording a frame. Whether that is happening at all was
    /// previously unanswerable from outside this file — [`Self::stats`] makes it a
    /// reading rather than a suspicion.
    misses: u64,
    /// Textures dropped on release because their bucket was full, cumulative.
    ///
    /// This is the number that explains a steady-state miss rate: a graph that
    /// releases more textures of one key per frame than the cap holds discards the
    /// surplus every frame and re-creates it the next, forever.
    evicted: u64,
    /// High-water mark of any single bucket's length. See [`PoolStats::peak_bucket`].
    peak_bucket: usize,
}

/// What a [`TransientTexturePool`] has actually done, for reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evicted: u64,
    /// Distinct (format, usage, size) buckets in the pool.
    pub buckets: usize,
    /// Textures currently parked for reuse, across all buckets.
    pub pooled: usize,
    /// The largest number of textures any single bucket has ever held.
    ///
    /// **This is the number the cap has to cover**, and it is measured rather than
    /// reasoned about: it is the observed peak of same-key textures alive in one
    /// frame. `POOL_BUCKET_CAPACITY` must be at least this, or the surplus is
    /// evicted and re-allocated every frame. Reported so the relationship between
    /// the cap and the graph is a reading instead of a comment.
    pub peak_bucket: usize,
}

impl PoolStats {
    /// Fraction of acquisitions that had to allocate, or `None` before any
    /// acquisition — a rate over zero attempts is not 0%, it is undefined.
    pub fn miss_rate(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        (total > 0).then(|| self.misses as f64 / total as f64)
    }
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

/// How many textures of one (format, usage, size) key the pool parks for reuse.
///
/// **This must cover the peak number LIVE AT ONCE in a single frame, not an
/// average.** `CompiledGraph::execute` acquires every declared resource before the
/// first node records and releases them all after the last one, so a graph with N
/// same-key intermediates holds N simultaneously and returns N at the end of the
/// frame. With a cap below N the surplus is dropped on release and re-allocated on
/// the next frame's acquire — a steady-state miss, forever, with no error and no
/// failing test.
///
/// Measured on benchmark 5 (4K60, 4 layers × YuvToRgb + CC + LUT + chroma key, plus
/// composite and tone-map targets), `target/bench_P21_final.txt` vs
/// `target/bench_P21_cap8.txt`, three repeats each:
///
/// | cap | peak bucket | miss rate | evicted / 90-frame run | FPS | steady P95 |
/// |---|---|---|---|---|---|
/// | 8  | 8 (clamped) | 31.3% | ~1150 | 43.6 | 27.48 ms |
/// | 32 | **16** (the true peak) | 0.6% | 0 | **57.3** | 18.24 ms |
///
/// The 16 textures are canvas-sized RGBA16Float, 66 MB each, so a cap of 8 asked the
/// driver to free and re-create ~12.8 of them inside every frame's recording. **The
/// only symptom was the frame interval alternating** — 17.9 ms / 25.7 ms by parity,
/// with `GPU transfer` splitting the same way (9.9 / 17.8 ms) and graph execution
/// flat at ~8.5 ms either way. No error, no warning, no failing test.
///
/// 32 covers the measured peak of 16 with room for a heavier chain, and the pool only
/// ever holds what a frame actually asked for: an unused bucket stays empty, so the
/// cap costs nothing on a 1080p single-layer graph (`peak bucket 1/32` there).
/// [`PoolStats::peak_bucket`] is how the peak is re-read after a graph change — the
/// bench prints it as `peak bucket N/CAP`, and `N == CAP` means the real peak is
/// unknown and at least the cap.
///
/// **Re-read after P2.3, which is what gotcha 14 requires of a graph change.** Fusing
/// colour correction + LUT + chroma key into one pass removes two canvas-sized
/// intermediates per layer, so benchmarks 5/7/8 now report `peak bucket 8/32` with 0
/// evicted (`target/p23_fused_3x.txt`) against `16/32` unfused
/// (`target/p23_unfused_3x.txt`). The cap stays 32: **16 is still a shape the tree
/// builds** — `NEXIR_FUSE_GRADE=0` is a supported arm and `EffectChainBuilder` emits
/// the separate nodes for a clip whose effect list is not the fusable triple — and a
/// cap is a ceiling, so lowering it to fit the cheaper shape would evict on the other
/// one for no gain (an unreached bucket costs nothing, measured: `1/32` at 1080p).
pub const POOL_BUCKET_CAPACITY: usize = 32;

impl TransientTexturePool {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            hits: 0,
            misses: 0,
            evicted: 0,
            peak_bucket: 0,
        }
    }

    /// What this pool has done so far.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            hits: self.hits,
            misses: self.misses,
            evicted: self.evicted,
            buckets: self.buckets.len(),
            pooled: self.buckets.values().map(|v| v.len()).sum(),
            peak_bucket: self.peak_bucket,
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
        if let Some(res) = self.buckets.get_mut(&key).and_then(|v| v.pop_front()) {
            self.hits += 1;
            res
        } else {
            self.misses += 1;
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
        if bucket.len() < POOL_BUCKET_CAPACITY {
            bucket.push_back(resource);
            self.peak_bucket = self.peak_bucket.max(bucket.len());
        } else {
            // Dropped here, which frees a 66 MB 4K texture — and guarantees a miss
            // for it next frame. Counted so the cost is visible instead of being a
            // silent per-frame allocation.
            self.evicted += 1;
            // The bucket is at the cap, so the peak is the cap — the real peak is
            // higher and unknown, which is exactly what `evicted > 0` reports.
            self.peak_bucket = self.peak_bucket.max(POOL_BUCKET_CAPACITY);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool must report what it did, and a rate over zero attempts must not
    /// read as 0%.
    #[test]
    fn a_fresh_pool_has_no_miss_rate_rather_than_a_zero_one() {
        let stats = TransientTexturePool::new().stats();
        assert_eq!(stats, PoolStats::default());
        assert!(
            stats.miss_rate().is_none(),
            "0 of 0 acquisitions is undefined, not 0%"
        );
        assert_eq!(
            PoolStats { hits: 3, misses: 1, ..Default::default() }.miss_rate(),
            Some(0.25)
        );
    }

    /// The bucket cap must hold a whole frame's worth of same-key textures.
    ///
    /// THE BUG THIS PINS. `CompiledGraph::execute` acquires every declared resource
    /// up front and releases them all at the end, so a graph with N same-key
    /// intermediates hands back N at once. With a cap below N the surplus is
    /// dropped and re-created on the next frame — at 4K that is 66 MB per dropped
    /// texture, every frame, and the only symptom was frame intervals alternating
    /// 17.9 / 25.7 ms.
    ///
    /// **The 16 is measured, not reasoned about.** `PoolStats::peak_bucket` on
    /// benchmark 5 reports `peak bucket 16/32`; at the old cap of 8 it reported
    /// `8/8` with a 31.3% miss rate and ~1150 evictions per 90-frame run. Raising
    /// the graph's simultaneous resource count means re-reading that line from a
    /// bench run and raising both the constant and this bound.
    ///
    /// `const { }` because both sides are constants: clippy flags a plain `assert!`
    /// over two `const`s, and a compile-time failure is strictly better here — a
    /// cap below the peak should not build, let alone run a suite to report it.
    #[test]
    fn the_bucket_cap_covers_a_whole_frames_peak() {
        /// Benchmark 5's measured high-water mark for one (format, usage, size) key.
        const MEASURED_PEAK: usize = 16;
        const _: () = assert!(
            POOL_BUCKET_CAPACITY >= MEASURED_PEAK,
            "benchmark 5's Heavy graph holds 16 canvas-sized textures of one key at \
             once (measured: `peak bucket 16/32`); this cap evicts the surplus every \
             frame — see AGENTS.md gotcha 14"
        );
        // Restated at runtime so a failure names the two numbers instead of only
        // refusing to compile with them. `black_box` defeats the const-folding that
        // makes clippy call this a constant assertion — it is one, deliberately.
        assert!(
            std::hint::black_box(POOL_BUCKET_CAPACITY) >= std::hint::black_box(MEASURED_PEAK),
            "cap {POOL_BUCKET_CAPACITY} is below the measured peak {MEASURED_PEAK}"
        );
    }

    /// A cap below the frame's peak must evict — the mechanism, not just the
    /// constant.
    ///
    /// Stated over the pool rather than over the constant, so it keeps testing
    /// something after `POOL_BUCKET_CAPACITY` is raised again: release one more
    /// than the cap of a single key and the surplus is dropped, which is a
    /// guaranteed miss on the next frame. Without this, the only thing pinning the
    /// coupling is an inequality over two numbers.
    #[test]
    fn releasing_more_than_the_cap_evicts_the_surplus() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let mut pool = TransientTexturePool::new();
        let fmt = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        const SURPLUS: usize = 3;
        let n = POOL_BUCKET_CAPACITY + SURPLUS;

        let mut frame: Vec<_> = (0..n)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        for res in frame.drain(..) {
            pool.release(res);
        }
        assert_eq!(
            pool.stats().evicted,
            SURPLUS as u64,
            "{SURPLUS} textures over the cap must be dropped, not silently kept"
        );
        assert_eq!(
            pool.stats().peak_bucket, POOL_BUCKET_CAPACITY,
            "a bucket at the cap reports the cap: the true peak is unknown and higher"
        );

        // ...and the surplus is a miss on the very next frame, which is the cost.
        let before = pool.stats().misses;
        let mut frame2: Vec<_> = (0..n)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        assert_eq!(
            pool.stats().misses - before,
            SURPLUS as u64,
            "the evicted textures must be re-allocated, every frame, forever"
        );
        for res in frame2.drain(..) {
            pool.release(res);
        }
    }

    /// A repeated frame shape must hand each resource the SAME texture as last
    /// frame, not merely *a* texture.
    ///
    /// P2.4 — this is a different property from `a_repeated_frame_shape_stops_
    /// allocating` below, and the pool satisfied that one while failing this one.
    /// `CompiledGraph::execute` acquires resources in ascending `ResourceId` order
    /// and releases them in ascending order too (`into_resources()` is indexed by
    /// id). A bucket is a stack, so ascending pushes pop descending: resource 1
    /// gets resource N's texture next frame, resource 2 gets N-1's, and the whole
    /// assignment reverses. It reverses BACK the frame after, so the mapping
    /// alternates with period 2.
    ///
    /// Nothing about that is a miss — the pool reports 0.6% and 0 evicted either
    /// way — but every node caches its bind group by the `ViewId`s it was handed
    /// (`lut.rs:241`, `composite.rs:398`, and nine more), so a reversing assignment
    /// invalidates every one of those caches on every frame. The caches then cost
    /// what they were written to save.
    #[test]
    fn a_repeated_frame_shape_reuses_each_resources_own_texture() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let mut pool = TransientTexturePool::new();
        // Benchmark 5's measured peak for one key. The count matters: with N=1 the
        // reversal is invisible, which is why the 1080p single-layer graph never
        // showed this.
        const N: usize = 16;
        let fmt = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;

        // Frame 1: acquire in ascending resource order, remember what each got.
        let frame1: Vec<_> = (0..N)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        let assigned1: Vec<ViewId> = frame1.iter().map(|(_, _, id)| *id).collect();
        // Released in ascending resource order, exactly as `execute` does.
        for res in frame1 {
            pool.release(res);
        }

        // Frame 2: the same graph, so the same ascending acquisition order.
        let frame2: Vec<_> = (0..N)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        let assigned2: Vec<ViewId> = frame2.iter().map(|(_, _, id)| *id).collect();
        for res in frame2 {
            pool.release(res);
        }

        let moved = assigned1
            .iter()
            .zip(&assigned2)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            moved, 0,
            "{moved} of {N} resources were handed a different texture than last \
             frame, so every bind group keyed on ViewId is rebuilt. assigned1={:?} \
             assigned2={:?}",
            assigned1, assigned2
        );

        // And the frame after must be stable too — a period-2 alternation passes a
        // check that only compares two consecutive frames.
        let frame3: Vec<_> = (0..N)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        let assigned3: Vec<ViewId> = frame3.iter().map(|(_, _, id)| *id).collect();
        for res in frame3 {
            pool.release(res);
        }
        assert_eq!(
            assigned2, assigned3,
            "the assignment must be stable frame after frame, not alternating"
        );
    }

    /// A second frame with the same graph shape must allocate nothing.
    ///
    /// The steady-state property the cap exists for, stated over the pool itself
    /// rather than over a GPU: acquire a frame's worth, release it, acquire the
    /// same shape again, and every acquisition must be a hit.
    #[test]
    fn a_repeated_frame_shape_stops_allocating() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let mut pool = TransientTexturePool::new();
        // 16 same-key textures, the measured peak of the Heavy graph — but small,
        // since this test is about the bookkeeping and not about bandwidth.
        const N: usize = 16;
        let fmt = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;

        let mut frame: Vec<_> = (0..N)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        assert_eq!(pool.stats().misses, N as u64, "a cold pool must allocate all of them");
        for res in frame.drain(..) {
            pool.release(res);
        }
        assert_eq!(
            pool.stats().evicted, 0,
            "a frame's peak must fit the bucket: {} evicted",
            pool.stats().evicted
        );

        let before = pool.stats().misses;
        let mut frame2: Vec<_> = (0..N)
            .map(|_| pool.acquire(&device, fmt, usage, 64, 64))
            .collect();
        assert_eq!(
            pool.stats().misses, before,
            "the second frame must reuse every texture; {} new allocations",
            pool.stats().misses - before
        );
        assert_eq!(pool.stats().hits, N as u64);
        for res in frame2.drain(..) {
            pool.release(res);
        }
    }

    // ── G2b — imported resources ────────────────────────────────────────────

    /// An imported texture must never be parked in a pool bucket.
    ///
    /// THE BUG THIS PINS, and it has no error message of its own. The pool buckets
    /// by (format, usage, size) and hands a texture to whoever asks for that key
    /// next, so a released import is handed to an *unrelated* resource one or more
    /// frames later — while the decoder that owns it keeps writing new frames into
    /// it. The symptom is a torn or wrong-content intermediate; the pool reports an
    /// ordinary hit, `evicted` stays 0, and nothing fails.
    ///
    /// Stated where it can actually fail: the import is given the **same key** as
    /// the transient beside it, so a leak lands in the same bucket and shows up as
    /// `pooled == 2`. With a different key the assertion would pass for the wrong
    /// reason.
    #[test]
    fn an_imported_resource_is_never_released_to_the_pool() {
        use crate::render::context::RenderContext;

        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let fmt = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let mut pool = TransientTexturePool::new();

        // One pooled slot and one import that would land in the very same bucket.
        let transient = pool.acquire(&device, fmt, usage, 64, 64);
        let transient_id = transient.2;
        let imported = ImportedTexture::new(Arc::new(
            device.create_texture(Some("decoder_owned"), 64, 64, fmt, usage),
        ));
        let imported_id = imported.view_id();
        assert_ne!(transient_id, imported_id);

        let ctx = RenderContext::from_slots(vec![
            Some(GraphResource::Transient(transient)),
            Some(GraphResource::Imported(imported.clone())),
        ]);
        assert!(!ctx.is_imported(ResourceId(0)));
        assert!(ctx.is_imported(ResourceId(1)));

        let released: Vec<_> = ctx.into_pooled().collect();
        assert_eq!(
            released.len(),
            1,
            "only the pool's own texture may come back; {} slots were handed to \
             release",
            released.len()
        );
        assert_eq!(released[0].2, transient_id, "the wrong slot was released");
        for res in released {
            pool.release(res);
        }

        let stats = pool.stats();
        assert_eq!(
            stats.pooled, 1,
            "the imported texture leaked into a bucket: {} textures pooled for one \
             released resource. A later frame would be handed the decoder's target \
             for an unrelated intermediate, with no error and no counter to show it.",
            stats.pooled
        );
        assert_eq!(stats.peak_bucket, 1);

        // And the import is still alive and still usable — it was dropped from the
        // frame, not freed. Its `ViewId` is unchanged, which is the property every
        // bind-group cache depends on.
        assert_eq!(imported.view_id(), imported_id);
        assert_eq!(imported.resolved().view_id, imported_id);

        // The next frame must be handed the pool's texture back, never the import's.
        let next = pool.acquire(&device, fmt, usage, 64, 64);
        assert_eq!(next.2, transient_id);
    }

    /// An import's `ViewId` is minted once and survives every per-frame clone.
    ///
    /// Eleven nodes cache their bind group on the `ViewId`s they were handed
    /// (AGENTS.md gotcha 16), so a fresh id per frame for the same underlying
    /// texture rebuilds all of them every frame — the exact cost the FIFO pool
    /// change removed, reintroduced through the other door. Cloning is what binding
    /// into a frame does, so the clone is what has to hold the id stable.
    #[test]
    fn an_imported_textures_view_id_survives_the_per_frame_clone() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let imported = ImportedTexture::new(Arc::new(device.create_texture(
            Some("decoder_owned"),
            64,
            64,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        )));
        let id = imported.view_id();

        for frame in 0..3 {
            let bound = imported.clone();
            assert_eq!(
                bound.view_id(),
                id,
                "frame {frame} was handed a new ViewId for the same texture, so every \
                 bind group keyed on it is rebuilt"
            );
            assert_eq!(GraphResource::Imported(bound).view_id(), id);
        }

        // Two separate imports of two textures are still distinct.
        let other = ImportedTexture::new(Arc::new(device.create_texture(
            Some("other"),
            64,
            64,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        )));
        assert_ne!(other.view_id(), id);
    }

    /// Rebinding an id replaces the texture rather than accumulating a second entry.
    ///
    /// The per-frame binding is a rebind of the same ids every frame (source 0's Y
    /// plane is always the same `ResourceId`), so an append-only set would grow
    /// without bound and `get` would keep answering with the first frame's texture.
    #[test]
    fn rebinding_an_import_replaces_it_rather_than_shadowing_it() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let make = |label: &'static str| {
            ImportedTexture::new(Arc::new(device.create_texture(
                Some(label),
                64,
                64,
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )))
        };
        let first = make("frame0");
        let second = make("frame1");

        let mut set = ImportedResources::new();
        assert!(set.is_empty());
        set.bind(ResourceId(7), first.clone());
        set.bind(ResourceId(7), second.clone());
        assert_eq!(set.len(), 1, "a rebind must replace, not append");
        assert_eq!(set.get(ResourceId(7)).map(|t| t.view_id()), Some(second.view_id()));
        assert!(set.get(ResourceId(8)).is_none());

        set.bind(ResourceId(8), first.clone());
        assert_eq!(set.len(), 2);
        let ids: Vec<_> = set.iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![ResourceId(7), ResourceId(8)]);
    }
}
