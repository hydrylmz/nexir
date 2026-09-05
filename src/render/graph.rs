// src/render/graph.rs

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::render::resource::{ResourceId, ResourceBuilder, ResourceDescriptor, TransientTexturePool, ResolutionSource, TextureAccess};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;

/// Every render node implements this trait.
/// Nodes are immutable after registration — all state is passed in via FrameState + RenderContext.
pub trait RenderNode: Send + Sync {
    /// Human-readable name for debug labels and profiling.
    fn name(&self) -> &str;

    /// Called once at compile time. Declare all resource reads, writes, and creates.
    /// The compiler uses this information to build the dependency graph.
    fn declare_resources(&self, builder: &mut ResourceBuilder);

    /// Called every frame during graph execution.
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        frame:   &FrameState,
    );

    /// Dynamic downcasting for nodes that need to be mutated after compilation.
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> { None }
}

/// Compilation-time description of one node's place in the graph.
struct NodeMeta {
    reads:        Vec<(ResourceId, TextureAccess)>,
    #[allow(dead_code)]
    writes:       Vec<(ResourceId, TextureAccess)>,
    creates:      Vec<(ResourceId, ResourceDescriptor)>,
    /// Resources read from outside the graph — see [`ResourceBuilder::import`].
    imports:      Vec<(ResourceId, TextureAccess)>,
    in_degree:    usize,
    /// Indices of nodes that depend on this node's outputs.
    dependents:   Vec<usize>,
}

pub struct RenderGraphCompiler {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl Default for RenderGraphCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraphCompiler {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Register a node. Nodes are numbered 0..N in registration order.
    pub fn add_node(&mut self, node: Box<dyn RenderNode>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Compile the graph into an executable `CompiledGraph`.
    pub fn compile(
        self,
        canvas_width: u32,
        canvas_height: u32,
    ) -> Result<CompiledGraph, GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        let node_count = self.nodes.len();
        let node_names: Vec<String> = self.nodes.iter().map(|n| n.name().to_string()).collect();

        let mut node_metas = Vec::with_capacity(node_count);
        let mut id_counter = 2; // 0 and 1 are FINAL_COLOR and SCREEN

        // Step 1: Declare resources and validate single-node access compatibility
        for node in &self.nodes {
            let mut builder = ResourceBuilder::new(id_counter);
            node.declare_resources(&mut builder);

            // Validate conflicting accesses within the same node
            for &(read_id, read_access) in &builder.reads {
                for &(write_id, write_access) in &builder.writes {
                    if read_id == write_id
                        && read_access == TextureAccess::ColorAttachment
                        && write_access == TextureAccess::StorageWrite
                    {
                        return Err(GraphError::IncompatibleAccess {
                            node_name: node.name().to_string(),
                            resource: read_id,
                            access: write_access,
                            reason: "Cannot simultaneously bind resource as ColorAttachment and StorageWrite in the same pass".into(),
                        });
                    }
                }
            }

            id_counter = builder.creates.last().map_or(id_counter, |(id, _)| id.0 + 1);

            node_metas.push(NodeMeta {
                reads: builder.reads,
                writes: builder.writes,
                creates: builder.creates,
                imports: builder.imports,
                in_degree: 0,
                dependents: Vec::new(),
            });
        }

        // Step 2: Validate missing producers
        // Collect all produced (created or written) resource IDs
        let mut produced_resources = std::collections::HashSet::new();
        produced_resources.insert(ResourceId::FINAL_COLOR);
        produced_resources.insert(ResourceId::SCREEN);

        for meta in &node_metas {
            for &(id, _) in &meta.writes {
                produced_resources.insert(id);
            }
            for (id, _) in &meta.creates {
                produced_resources.insert(*id);
            }
            // An imported resource is produced OUTSIDE the graph — by the decoder
            // that wrote into it — so it satisfies the reader it was declared for
            // without any node writing it. Registering it here rather than
            // exempting imports from the check keeps `MissingProducer` meaningful
            // for the case it exists to catch: a read of a resource nobody fills.
            for &(id, _) in &meta.imports {
                produced_resources.insert(id);
            }
        }

        for (i, meta) in node_metas.iter().enumerate() {
            for &(read_id, _) in &meta.reads {
                if !produced_resources.contains(&read_id) {
                    return Err(GraphError::MissingProducer {
                        node_name: node_names[i].clone(),
                        resource: read_id,
                    });
                }
            }
        }

        // Step 3: Build dependency edges with full hazard handling:
        // - RAW (Read After Write): Reader depends on the closest preceding writer of the resource
        //   (or initial writer if backward reference).
        // - WAW (Write After Write): Subsequent writer depends on earlier writer.
        // - WAR (Write After Read): Subsequent writer depends on earlier reader.
        let mut edges: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

        // Collect sorted writer and reader indices per resource
        let mut resource_writers: HashMap<ResourceId, Vec<usize>> = HashMap::new();
        let mut resource_readers: HashMap<ResourceId, Vec<usize>> = HashMap::new();

        for (i, meta) in node_metas.iter().enumerate() {
            for &(write_id, _) in &meta.writes {
                let writers = resource_writers.entry(write_id).or_default();
                if !writers.contains(&i) {
                    writers.push(i);
                }
            }
            for (create_id, _) in &meta.creates {
                let writers = resource_writers.entry(*create_id).or_default();
                if !writers.contains(&i) {
                    writers.push(i);
                }
            }
            for &(read_id, _) in &meta.reads {
                let readers = resource_readers.entry(read_id).or_default();
                if !readers.contains(&i) {
                    readers.push(i);
                }
            }
        }

        // Ensure writers and readers lists are sorted
        for writers in resource_writers.values_mut() {
            writers.sort_unstable();
        }
        for readers in resource_readers.values_mut() {
            readers.sort_unstable();
        }

        // 1. RAW Edges: each reader depends on the most recent writer registered before it.
        // If there is no writer before it (e.g. cyclic / backward declaration), depend on the first writer.
        for (&res_id, readers) in &resource_readers {
            if let Some(writers) = resource_writers.get(&res_id) {
                for &reader in readers {
                    // Find latest writer w < reader
                    let prev_writer = writers.iter().rev().find(|&&w| w < reader);
                    if let Some(&w) = prev_writer {
                        if w != reader {
                            edges.insert((w, reader));
                        }
                    } else if let Some(&w) = writers.first() {
                        if w != reader {
                            edges.insert((w, reader));
                        }
                    }
                }
            }
        }

        // 2. WAW Edges: sequential writers of the same resource depend on each other
        for writers in resource_writers.values() {
            for window in writers.windows(2) {
                let w1 = window[0];
                let w2 = window[1];
                if w1 != w2 {
                    edges.insert((w1, w2));
                }
            }
        }

        // 3. WAR Edges: for each writer w, all readers r that read the previous version (r < w) must finish before w
        for (&res_id, writers) in &resource_writers {
            if let Some(readers) = resource_readers.get(&res_id) {
                for &writer in writers {
                    for &reader in readers {
                        if reader < writer {
                            edges.insert((reader, writer));
                        }
                    }
                }
            }
        }

        // Apply unique edges to node_metas
        for &(u, v) in &edges {
            node_metas[u].dependents.push(v);
            node_metas[v].in_degree += 1;
        }

        // Step 4: Kahn's topological sort
        let mut queue = VecDeque::new();
        for (i, meta) in node_metas.iter().enumerate() {
            if meta.in_degree == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted_order = Vec::with_capacity(node_count);
        while let Some(u) = queue.pop_front() {
            sorted_order.push(u);
            let dependents = node_metas[u].dependents.clone();
            for v in dependents {
                node_metas[v].in_degree -= 1;
                if node_metas[v].in_degree == 0 {
                    queue.push_back(v);
                }
            }
        }

        // Step 5: Cycle detection with precise cycle path extraction
        if sorted_order.len() < node_count {
            let unresolved: Vec<usize> = (0..node_count)
                .filter(|&idx| !sorted_order.contains(&idx))
                .collect();
            let dep_lists: Vec<Vec<usize>> = node_metas.iter().map(|m| m.dependents.clone()).collect();
            let cycle = extract_cycle(&unresolved, &dep_lists, &node_names);
            return Err(GraphError::CyclicDependency { cycle });
        }

        // Step 6: Build resource descriptor map and compute combined wgpu::TextureUsages
        let max_id = node_metas
            .iter()
            .flat_map(|m| {
                m.creates
                    .iter()
                    .map(|(id, _)| id.0 as usize)
                    .chain(m.reads.iter().map(|(id, _)| id.0 as usize))
                    .chain(m.writes.iter().map(|(id, _)| id.0 as usize))
            })
            .max()
            .unwrap_or(1);

        let mut descriptors: Vec<Option<(ResourceDescriptor, wgpu::TextureUsages)>> =
            (0..=max_id).map(|_| None).collect();
        for meta in &node_metas {
            for (id, desc) in &meta.creates {
                descriptors[id.0 as usize] = Some((desc.clone(), wgpu::TextureUsages::empty()));
            }
        }

        for meta in &node_metas {
            for &(id, access) in &meta.reads {
                if let Some(Some((_, ref mut usage))) = descriptors.get_mut(id.0 as usize) {
                    *usage |= access.to_wgpu_usage();
                }
            }
            for &(id, access) in &meta.writes {
                if let Some(Some((_, ref mut usage))) = descriptors.get_mut(id.0 as usize) {
                    *usage |= access.to_wgpu_usage();
                }
            }
        }

        // Step 7: Record which resources are supplied from outside.
        //
        // Held on the compiled graph rather than re-derived per frame so `execute`
        // can assert the frame bound every one of them: an import the frame forgot
        // would otherwise surface as `ctx.get` panicking inside whichever node
        // happened to sample it first, naming a ResourceId and no cause.
        //
        // Deliberately NOT given a descriptor: `descriptors[id]` stays `None` for an
        // import, which is what keeps the acquire loop from allocating one.
        let mut imported_ids: Vec<ResourceId> = node_metas
            .iter()
            .flat_map(|m| m.imports.iter().map(|(id, _)| *id))
            .collect();
        imported_ids.sort_unstable_by_key(|id| id.0);
        imported_ids.dedup();

        // A resource cannot be both. The pool would allocate a texture the frame's
        // import then shadows, and — worse — the import would be released into the
        // pool at the end of the frame, since the slot's variant is what drives the
        // release. Caught at compile time because at run time it is a torn frame
        // with no error.
        for id in &imported_ids {
            if descriptors.get(id.0 as usize).is_some_and(|d| d.is_some()) {
                let creator = node_names
                    .iter()
                    .zip(&node_metas)
                    .find(|(_, m)| m.creates.iter().any(|(cid, _)| cid == id))
                    .map(|(name, _)| name.clone())
                    .unwrap_or_else(|| "<unknown>".to_string());
                return Err(GraphError::IncompatibleAccess {
                    node_name: creator,
                    resource: *id,
                    access: TextureAccess::Sampled,
                    reason: "resource is both created by the graph and imported from \
                             outside it: the pool would allocate a texture the import \
                             shadows, then reclaim the import at end of frame"
                        .into(),
                });
            }
        }

        Ok(CompiledGraph {
            nodes: self.nodes,
            order: sorted_order,
            descriptors,
            imported_ids,
            canvas_width,
            canvas_height,
            texture_pool: Mutex::new(TransientTexturePool::new()),
        })
    }
}

/// Helper function to extract a cycle path in a directed graph.
fn extract_cycle(
    unresolved_nodes: &[usize],
    dependents: &[Vec<usize>],
    node_names: &[String],
) -> Vec<String> {
    let unresolved_set: std::collections::HashSet<usize> =
        unresolved_nodes.iter().cloned().collect();
    let mut visited = vec![false; node_names.len()];
    let mut on_stack = vec![false; node_names.len()];
    let mut stack = Vec::new();
    let mut cycle_path = Vec::new();

    fn dfs(
        u: usize,
        unresolved_set: &std::collections::HashSet<usize>,
        dependents: &[Vec<usize>],
        visited: &mut [bool],
        on_stack: &mut [bool],
        stack: &mut Vec<usize>,
        cycle_path: &mut Vec<String>,
        node_names: &[String],
    ) -> bool {
        visited[u] = true;
        on_stack[u] = true;
        stack.push(u);

        for &v in &dependents[u] {
            if !unresolved_set.contains(&v) {
                continue;
            }
            if !visited[v] {
                if dfs(
                    v,
                    unresolved_set,
                    dependents,
                    visited,
                    on_stack,
                    stack,
                    cycle_path,
                    node_names,
                ) {
                    return true;
                }
            } else if on_stack[v] {
                let start_idx = stack.iter().position(|&x| x == v).unwrap_or(0);
                for &node_idx in &stack[start_idx..] {
                    cycle_path.push(node_names[node_idx].clone());
                }
                cycle_path.push(node_names[v].clone());
                return true;
            }
        }

        stack.pop();
        on_stack[u] = false;
        false
    }

    for &start in unresolved_nodes {
        if !visited[start]
            && dfs(
                start,
                &unresolved_set,
                dependents,
                &mut visited,
                &mut on_stack,
                &mut stack,
                &mut cycle_path,
                node_names,
            )
        {
            return cycle_path;
        }
    }

    unresolved_nodes.iter().map(|&i| node_names[i].clone()).collect()
}

pub struct CompiledGraph {
    nodes: Vec<Box<dyn RenderNode>>,
    order: Vec<usize>,
    descriptors: Vec<Option<(ResourceDescriptor, wgpu::TextureUsages)>>,
    /// Resources every frame must bind from outside — see [`ResourceBuilder::import`].
    /// Empty for every graph that does not use the interop decode path.
    imported_ids: Vec<ResourceId>,
    canvas_width: u32,
    canvas_height: u32,
    texture_pool: Mutex<TransientTexturePool>,
}

impl CompiledGraph {
    pub fn nodes_mut(&mut self) -> &mut [Box<dyn RenderNode>] {
        &mut self.nodes
    }

    /// Consume the graph and return the owned nodes.
    /// Used by ExportEngine which needs both the compiled graph and node ownership.
    pub fn into_nodes(self) -> Vec<Box<dyn RenderNode>> {
        self.nodes
    }

    /// Returns the compiled execution order of nodes by their registered indices.
    pub fn execution_order(&self) -> &[usize] {
        &self.order
    }

    /// The name of the node at a registered index.
    ///
    /// Pairs with [`Self::execution_order`] so a caller can build the same
    /// index → name mapping [`Self::execute_timed`] returns, and assert the two
    /// agree — which is the only thing keeping a per-node timing report attached to
    /// the right shader.
    pub fn node_name(&self, node_idx: usize) -> &str {
        self.nodes[node_idx].name()
    }

    /// What the transient texture pool has done since this graph was compiled.
    ///
    /// Exposed so a caller can report allocation behaviour it would otherwise have
    /// to guess at. The pool's cap has to cover a whole frame's simultaneous
    /// resources — `execute` acquires them all before the first node records and
    /// releases them all after the last — so a non-zero `evicted` means every frame is
    /// dropping textures it will immediately re-allocate. At 4K that is 66 MB per
    /// texture per frame. See [`crate::render::resource::POOL_BUCKET_CAPACITY`].
    pub fn pool_stats(&self) -> crate::render::resource::PoolStats {
        self.texture_pool.lock().unwrap().stats()
    }

    /// **P2.2 — what per-resource lifetimes would buy, computed from this graph rather
    /// than estimated.**
    ///
    /// `execute` acquires every declared resource before the first node records and
    /// releases them all after the last, so a frame holds N same-key textures
    /// simultaneously whether or not their live ranges overlap. The audit's P2.2 asks
    /// whether tracking those ranges and letting non-overlapping resources SHARE a
    /// texture is worth doing — and the honest way to answer that is to compute both
    /// numbers off the compiled graph instead of writing the refactor and then
    /// measuring.
    ///
    /// Returns `(held, ideal)` per bucket key, summed over keys:
    /// * `held` — what the frame holds today: every created resource, at once. This is
    ///   the number [`crate::render::resource::PoolStats::peak_bucket`] reports and
    ///   [`crate::render::resource::POOL_BUCKET_CAPACITY`] has to cover.
    /// * `ideal` — the maximum simultaneously LIVE at any point in the execution order,
    ///   which is the floor a perfect aliasing scheme could reach. `held - ideal` is
    ///   the whole prize, and if it is zero there is nothing for P2.2 to win.
    ///
    /// **A resource's live range is `[first write … last read]` inclusive, and the
    /// inclusivity is the load-bearing part.** A node that reads A and writes B binds
    /// both in one pass, so A and B cannot share a texture even though A is "dead
    /// after" that node — wgpu would reject the bind group outright (one texture as
    /// read-storage and write-storage at once), and a scheme that ignored this would
    /// fail at the first frame rather than silently. Counting the live set as an
    /// inclusive interval is therefore not conservatism; it is the constraint.
    ///
    /// **Imports are excluded**, because they are not the pool's to alias (gotcha 18):
    /// the decoder owns that texture for longer than the frame.
    ///
    /// This is a static analysis of declared resources, not a measurement of GPU
    /// behaviour — so it says what the SHAPE permits, and it is exact for that
    /// question. Nothing here is an estimate of time.
    pub fn lifetime_bounds(&self) -> (usize, usize) {
        use std::collections::HashMap;

        // Position of each node in the recorded order, keyed by its registered index.
        let mut position = HashMap::with_capacity(self.order.len());
        for (pos, &node_idx) in self.order.iter().enumerate() {
            position.insert(node_idx, pos);
        }

        // Re-derive each resource's live range from the nodes' own declarations. Asking
        // the nodes again rather than caching it at compile time keeps this analysis
        // honest about the graph as it actually is: a node whose `declare_resources`
        // changed would change this answer too.
        let mut first_write: HashMap<ResourceId, usize> = HashMap::new();
        let mut last_read: HashMap<ResourceId, usize> = HashMap::new();
        let mut imported: std::collections::HashSet<ResourceId> =
            self.imported_ids.iter().copied().collect();

        for (&node_idx, &pos) in &position {
            let mut builder = ResourceBuilder::new(0);
            self.nodes[node_idx].declare_resources(&mut builder);
            for (id, _) in &builder.creates {
                let e = first_write.entry(*id).or_insert(pos);
                *e = (*e).min(pos);
            }
            for (id, _) in &builder.writes {
                let e = first_write.entry(*id).or_insert(pos);
                *e = (*e).min(pos);
            }
            for (id, _) in &builder.reads {
                let e = last_read.entry(*id).or_insert(pos);
                *e = (*e).max(pos);
            }
            for (id, _) in &builder.imports {
                imported.insert(*id);
            }
        }

        // Group by the same key the pool buckets on: (format, usage, resolved size).
        // Two resources of different keys never share a texture anyway, so a peak
        // computed across keys would overstate what aliasing could save.
        let mut by_key: HashMap<(wgpu::TextureFormat, wgpu::TextureUsages, u32, u32), Vec<(usize, usize)>> =
            HashMap::new();
        for (id_usize, desc_opt) in self.descriptors.iter().enumerate() {
            let Some((desc, usage)) = desc_opt else { continue };
            let id = ResourceId(id_usize as u32);
            if imported.contains(&id) {
                continue;
            }
            let (w, h) = match desc.size {
                ResolutionSource::Canvas => (self.canvas_width, self.canvas_height),
                ResolutionSource::Fixed(fw, fh) => (fw, fh),
            };
            let start = first_write.get(&id).copied().unwrap_or(0);
            // A resource nobody reads is still written, so it is live for that one
            // node — `FINAL_COLOR` on a graph with no readback is the ordinary case.
            let end = last_read.get(&id).copied().unwrap_or(start).max(start);
            by_key
                .entry((desc.format, *usage, w, h))
                .or_default()
                .push((start, end));
        }

        let mut held = 0usize;
        let mut ideal = 0usize;
        for ranges in by_key.values() {
            held += ranges.len();
            // Sweep the positions and take the largest live set. A sweep rather than an
            // interval-graph colouring because for one key the maximum clique of an
            // interval graph IS its chromatic number — so this peak is exactly the
            // number of textures a perfect scheme needs, not a bound on it.
            let mut peak = 0usize;
            for pos in 0..self.order.len() {
                let live = ranges
                    .iter()
                    .filter(|(s, e)| *s <= pos && pos <= *e)
                    .count();
                peak = peak.max(live);
            }
            ideal += peak;
        }
        (held, ideal)
    }

    /// Resolve every declared resource for one frame: pooled textures acquired from
    /// the graph's own pool, imported ones bound from the frame.
    ///
    /// Shared by all three `execute*` bodies. The acquire loop is the one part of
    /// them that must stay identical — a divergence is a resource the timed path
    /// pools and the untimed one does not, which the pool's counters would then
    /// disagree about between runs — so it lives here rather than being copied a
    /// third time.
    ///
    /// # Panics
    /// Panics if a resource declared with [`ResourceBuilder::import`] was not bound
    /// by this frame. The alternative is `ctx.get` panicking later inside whichever
    /// node sampled it first, naming a bare `ResourceId` and no cause.
    fn resolve_resources(
        &self,
        device: &GpuDevice,
        frame: &FrameState,
    ) -> Vec<Option<crate::render::resource::GraphResource>> {
        use crate::render::resource::GraphResource;

        let mut slots = crate::render::context::empty_slots(self.descriptors.len());

        let mut pool = self.texture_pool.lock().unwrap();
        for (id_usize, desc_opt) in self.descriptors.iter().enumerate() {
            if let Some((desc, usage)) = desc_opt {
                let (w, h) = match desc.size {
                    ResolutionSource::Canvas => (self.canvas_width, self.canvas_height),
                    ResolutionSource::Fixed(fw, fh) => (fw, fh),
                };
                slots[id_usize] =
                    Some(GraphResource::Transient(pool.acquire(device, desc.format, *usage, w, h)));
            }
        }
        drop(pool);

        for &id in &self.imported_ids {
            let texture = frame.imported.get(id).unwrap_or_else(|| {
                panic!(
                    "resource {:?} was declared as an import but this frame bound \
                     nothing to it (frame binds {} import(s)); the graph does not \
                     allocate imports, so there is no texture for the reading node \
                     to sample",
                    id,
                    frame.imported.len()
                )
            });
            crate::render::context::bind_import(&mut slots, id, texture);
        }

        slots
    }

    /// Execute the compiled graph for one frame.
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &GpuDevice,
        frame: &FrameState,
    ) {
        // Step 1 & 2: Acquire transient textures, bind imported ones
        let ctx = RenderContext::from_slots(self.resolve_resources(device, frame));

        // Step 3: Execute nodes
        for &node_idx in &self.order {
            let node = &self.nodes[node_idx];
            encoder.push_debug_group(node.name());
            node.record(encoder, &ctx, frame);
            encoder.pop_debug_group();
        }

        // Step 4: Return the POOLED textures — and only those. An imported texture
        // is owned by the decoder that wrote into it and is dropped by `into_pooled`
        // instead of released; parking one in a bucket would hand it to an unrelated
        // resource on a later frame while the decoder still writes to it.
        let mut pool = self.texture_pool.lock().unwrap();
        for res in ctx.into_pooled() {
            pool.release(res);
        }
    }

    /// Execute with one GPU timestamp bracket per node.
    ///
    /// Separate from [`Self::execute`] rather than a flag on it: the timing path
    /// needs a query set sized to the node count and a resolve recorded after the
    /// last node, and threading an `Option<&mut GpuTimer>` through the hot path
    /// would put a branch in every frame of the UI's preview loop for a
    /// benchmark's benefit.
    ///
    /// Returns the node names in execution order, aligned one-to-one with the
    /// timer's brackets, so the caller maps `pair[i]` → `names[i]` explicitly
    /// instead of re-deriving `self.order` and hoping the two agree.
    ///
    /// **The caller must size the timer for `execution_order().len()` brackets and
    /// call [`crate::render::gpu_timer::GpuTimer::reset`] first.** When the timer
    /// is too small, `GpuTimer::mark` silently stops writing at capacity — so this
    /// returns names for the nodes it *bracketed*, truncated to what the timer
    /// could hold, rather than a full list that would misalign every reading after
    /// the cut.
    ///
    /// **Timestamps go BETWEEN passes, not inside them.** Writing inside a pass
    /// needs `TIMESTAMP_QUERY_INSIDE_PASSES`, which nothing in this crate
    /// requests; every node opens and closes its own compute pass, so a bracket
    /// around `record` already gives per-node resolution.
    pub fn execute_timed<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        device: &GpuDevice,
        frame: &FrameState,
        timer: &mut crate::render::gpu_timer::GpuTimer,
    ) -> Vec<&'a str> {
        // Acquire exactly as `execute` does — through the same helper, so the two
        // cannot drift. That the pool interaction is identical is checkable: a timed
        // run and an untimed one must report the same hit/miss/evicted numbers.
        let ctx = RenderContext::from_slots(self.resolve_resources(device, frame));

        let mut names: Vec<&'a str> = Vec::with_capacity(self.order.len());
        for &node_idx in &self.order {
            let node = &self.nodes[node_idx];
            // Only claim a name for a node whose bracket actually fits. `written`
            // stops advancing at capacity, so comparing before and after is how a
            // dropped bracket is detected — without it, an undersized timer would
            // shift every subsequent reading onto the wrong node and the report
            // would attribute one shader's cost to another.
            let before = timer.written();
            timer.begin(encoder);
            encoder.push_debug_group(node.name());
            node.record(encoder, &ctx, frame);
            encoder.pop_debug_group();
            timer.end(encoder);
            if timer.written() == before + 2 {
                names.push(node.name());
            }
        }
        timer.record_resolve(encoder);

        let mut pool = self.texture_pool.lock().unwrap();
        for res in ctx.into_pooled() {
            pool.release(res);
        }
        names
    }

    pub fn execute_with_callback<F>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &GpuDevice,
        frame: &FrameState,
        callback: F,
    ) where
        F: FnOnce(&mut wgpu::CommandEncoder, &RenderContext),
    {
        let ctx = RenderContext::from_slots(self.resolve_resources(device, frame));

        for &node_idx in &self.order {
            let node = &self.nodes[node_idx];
            encoder.push_debug_group(node.name());
            node.record(encoder, &ctx, frame);
            encoder.pop_debug_group();
        }

        callback(encoder, &ctx);

        let mut pool = self.texture_pool.lock().unwrap();
        for res in ctx.into_pooled() {
            pool.release(res);
        }
    }
}

/// Detailed error diagnostics for RenderGraph compilation.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    EmptyGraph,
    CyclicDependency {
        cycle: Vec<String>,
    },
    MissingProducer {
        node_name: String,
        resource: ResourceId,
    },
    IncompatibleAccess {
        node_name: String,
        resource: ResourceId,
        access: TextureAccess,
        reason: String,
    },
    UnresolvableResource(ResourceId),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGraph => {
                write!(f, "RenderGraph is empty: at least one RenderNode is required")
            }
            Self::CyclicDependency { cycle } => {
                write!(
                    f,
                    "Cyclic dependency detected in render graph: {}",
                    cycle.join(" -> ")
                )
            }
            Self::MissingProducer { node_name, resource } => {
                write!(
                    f,
                    "Node \"{}\" attempted to read resource {:?}, but no node creates or produces it",
                    node_name, resource
                )
            }
            Self::IncompatibleAccess {
                node_name,
                resource,
                access,
                reason,
            } => {
                write!(
                    f,
                    "Node \"{}\" attempted invalid access {:?} on resource {:?}: {}",
                    node_name, access, resource, reason
                )
            }
            Self::UnresolvableResource(id) => {
                write!(f, "Resource {:?} could not be resolved", id)
            }
        }
    }
}

impl std::error::Error for GraphError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::resource::{ResolutionSource, ResourceDescriptor, TextureAccess};

    /// A node that creates one canvas-sized texture and optionally reads another.
    ///
    /// Deliberately minimal: [`CompiledGraph::lifetime_bounds`] is a question about
    /// DECLARATIONS, so a test node that declares and records nothing exercises exactly
    /// the code under test and needs no shaders, no pipelines and no GPU work.
    struct Stage {
        label: &'static str,
        reads: Option<ResourceId>,
        creates: ResourceId,
    }

    impl RenderNode for Stage {
        fn name(&self) -> &str {
            self.label
        }

        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.creates.push((
                self.creates,
                ResourceDescriptor {
                    label: Some(self.label.to_string()),
                    size: ResolutionSource::Canvas,
                    format: wgpu::TextureFormat::Rgba16Float,
                },
            ));
            builder.write(self.creates, TextureAccess::StorageWrite);
            if let Some(id) = self.reads {
                builder.read(id, TextureAccess::StorageRead);
            }
        }

        fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
    }

    /// **P2.2's prize, on the two shapes that matter — and it is small.**
    ///
    /// Two constraints bound what aliasing can win, and both are measured here rather
    /// than assumed:
    ///
    /// 1. **A node binds its input and its output in one pass**, so consecutive stages
    ///    can never share a texture. On a chain of any length that puts the floor at
    ///    two per bucket, not one — a reader expecting A→B→C→D to collapse to a single
    ///    texture is expecting something wgpu would reject at the first bind group.
    /// 2. **The pool buckets by (format, USAGE, size)**, so resources with different
    ///    usage flags cannot alias each other however their lifetimes fall. The chain's
    ///    last output is written and never read, so its usage is `STORAGE_BINDING`
    ///    alone while every intermediate carries `TEXTURE_BINDING | STORAGE_BINDING` —
    ///    a separate bucket, and a separate texture. That is why the chain's ideal is
    ///    3 and not 2.
    ///
    /// The second shape is the Heavy graph's: four independent chains fanning into one
    /// composite. There the composite reads all four ends in one pass, so those four
    /// are simultaneous by construction and no scheme can reduce them.
    ///
    /// **This is why P2.2 is judged before it is written.** The prize is a VRAM figure
    /// rather than a frame time — the pool already reuses textures across frames, so
    /// aliasing within a frame buys residency, not allocations.
    #[test]
    fn lifetime_bounds_reports_what_aliasing_could_save() {
        // ── A straight chain: A → B → C → D ────────────────────────────────────
        let mut c = RenderGraphCompiler::new();
        let a = ResourceId(2);
        let b = ResourceId(3);
        let d = ResourceId(4);
        let e = ResourceId(5);
        c.add_node(Box::new(Stage { label: "A", reads: None, creates: a }));
        c.add_node(Box::new(Stage { label: "B", reads: Some(a), creates: b }));
        c.add_node(Box::new(Stage { label: "C", reads: Some(b), creates: d }));
        c.add_node(Box::new(Stage { label: "D", reads: Some(d), creates: e }));
        let chain = c.compile(3840, 2160).expect("chain compiles");

        let (held, ideal) = chain.lifetime_bounds();
        assert_eq!(held, 4, "the graph holds every created texture for the frame");
        assert_eq!(
            ideal, 3,
            "two for the read-and-written intermediates (a node binds its input and \
             output together, so consecutive stages can never share) plus one for the \
             write-only final output, which the pool buckets separately because its \
             usage flags differ. An `ideal` of 1 or 2 would mean the analysis ignored \
             one of those two constraints and the refactor it justified would fail wgpu \
             validation on its first frame."
        );

        // ── The Heavy graph's shape: four chains into one composite ────────────
        //
        // The composite reads all four ends in one pass, so they ARE simultaneous and
        // aliasing cannot touch them. What it can share is each chain's own
        // intermediate.
        let mut c = RenderGraphCompiler::new();
        let mut ids = 2u32;
        let mut ends = Vec::new();
        for _ in 0..4 {
            let src = ResourceId::next(&mut ids);
            let mid = ResourceId::next(&mut ids);
            let end = ResourceId::next(&mut ids);
            c.add_node(Box::new(Stage { label: "Lsrc", reads: None, creates: src }));
            c.add_node(Box::new(Stage { label: "Lmid", reads: Some(src), creates: mid }));
            c.add_node(Box::new(Stage { label: "Lend", reads: Some(mid), creates: end }));
            ends.push(end);
        }
        // The composite: reads all four ends, writes FINAL_COLOR.
        struct Composite {
            reads: Vec<ResourceId>,
        }
        impl RenderNode for Composite {
            fn name(&self) -> &str {
                "Composite"
            }
            fn declare_resources(&self, builder: &mut ResourceBuilder) {
                builder.creates.push((
                    ResourceId::FINAL_COLOR,
                    ResourceDescriptor {
                        label: Some("FinalColor".into()),
                        size: ResolutionSource::Canvas,
                        format: wgpu::TextureFormat::Rgba16Float,
                    },
                ));
                builder.write(ResourceId::FINAL_COLOR, TextureAccess::StorageWrite);
                for id in &self.reads {
                    builder.read(*id, TextureAccess::StorageRead);
                }
            }
            fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
        }
        c.add_node(Box::new(Composite { reads: ends }));
        let fan = c.compile(3840, 2160).expect("fan-in compiles");

        let (held, ideal) = fan.lifetime_bounds();
        assert_eq!(
            held, 13,
            "four layers x three textures, plus FINAL_COLOR, all held for the frame"
        );
        // 6 is arithmetic on the order Kahn's sort produces, not a fitted number: the
        // four chains advance in lockstep (all four sources, then all four middles,
        // then all four ends), so five of the twelve same-key textures are live at the
        // busiest point — four at one stage plus the first of the next — and
        // FINAL_COLOR's own bucket adds one.
        assert_eq!(
            ideal, 6,
            "the composite reads four layer outputs in one pass, so those four are \
             simultaneous by construction; with the four chains interleaved the live \
             peak is five same-key textures plus FINAL_COLOR"
        );
        // The prize, stated as the thing P2.2 would be judged on. At 4K RGBA16Float
        // that is 7 x 66.4 MB of residency, and no frame time.
        assert_eq!(
            held - ideal,
            7,
            "aliasing could remove {} of {held} textures on this shape",
            held - ideal
        );
    }

    /// An imported resource must not appear in either figure.
    ///
    /// Gotcha 18: the decoder owns that texture across frames, so it is not the pool's
    /// to alias and counting it would inflate the prize P2.2 is judged on with memory
    /// that cannot be reclaimed.
    #[test]
    fn lifetime_bounds_ignores_imports() {
        struct Importer {
            imports: ResourceId,
            creates: ResourceId,
        }
        impl RenderNode for Importer {
            fn name(&self) -> &str {
                "Importer"
            }
            fn declare_resources(&self, builder: &mut ResourceBuilder) {
                builder.creates.push((
                    self.creates,
                    ResourceDescriptor {
                        label: Some("out".into()),
                        size: ResolutionSource::Canvas,
                        format: wgpu::TextureFormat::Rgba16Float,
                    },
                ));
                builder.write(self.creates, TextureAccess::StorageWrite);
                builder.import(self.imports, TextureAccess::StorageRead);
            }
            fn record(&self, _e: &mut wgpu::CommandEncoder, _c: &RenderContext, _f: &FrameState) {}
        }

        let mut c = RenderGraphCompiler::new();
        c.add_node(Box::new(Importer {
            imports: ResourceId(2),
            creates: ResourceId(3),
        }));
        let g = c.compile(1920, 1080).expect("import graph compiles");
        let (held, ideal) = g.lifetime_bounds();
        assert_eq!(
            (held, ideal),
            (1, 1),
            "only the created texture counts; the import is the decoder's"
        );
    }
}
