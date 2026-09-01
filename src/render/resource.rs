// src/render/resource.rs

use crate::render::device::GpuDevice;
use std::collections::{HashMap, VecDeque};
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
}
