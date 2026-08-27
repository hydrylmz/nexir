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

        Ok(CompiledGraph {
            nodes: self.nodes,
            order: sorted_order,
            descriptors,
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
        if !visited[start] {
            if dfs(
                start,
                &unresolved_set,
                dependents,
                &mut visited,
                &mut on_stack,
                &mut stack,
                &mut cycle_path,
                node_names,
            ) {
                return cycle_path;
            }
        }
    }

    unresolved_nodes.iter().map(|&i| node_names[i].clone()).collect()
}

pub struct CompiledGraph {
    nodes: Vec<Box<dyn RenderNode>>,
    order: Vec<usize>,
    descriptors: Vec<Option<(ResourceDescriptor, wgpu::TextureUsages)>>,
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

    /// Execute the compiled graph for one frame.
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &GpuDevice,
        frame: &FrameState,
    ) {
        // Step 1 & 2: Acquire transient textures and create views
        let mut pool = self.texture_pool.lock().unwrap();
        let mut resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, crate::render::resource::ViewId)>> =
            (0..self.descriptors.len()).map(|_| None).collect();

        for (id_usize, desc_opt) in self.descriptors.iter().enumerate() {
            if let Some((desc, usage)) = desc_opt {
                let (w, h) = match desc.size {
                    ResolutionSource::Canvas => (self.canvas_width, self.canvas_height),
                    ResolutionSource::Fixed(fw, fh) => (fw, fh),
                };

                resources[id_usize] = Some(pool.acquire(device, desc.format, *usage, w, h));
            }
        }

        // Drop the MutexGuard
        drop(pool);

        // Step 3: Build RenderContext
        let ctx = RenderContext::new(resources);

        // Step 4: Execute nodes
        for &node_idx in &self.order {
            let node = &self.nodes[node_idx];
            encoder.push_debug_group(node.name());
            node.record(encoder, &ctx, frame);
            encoder.pop_debug_group();
        }

        // Step 5: Release transient textures
        let mut pool = self.texture_pool.lock().unwrap();
        for res in ctx.into_resources().into_iter().flatten() {
            pool.release(res);
        }
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
        let mut pool = self.texture_pool.lock().unwrap();
        let mut resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, crate::render::resource::ViewId)>> =
            (0..self.descriptors.len()).map(|_| None).collect();

        for (id_usize, desc_opt) in self.descriptors.iter().enumerate() {
            if let Some((desc, usage)) = desc_opt {
                let (w, h) = match desc.size {
                    ResolutionSource::Canvas => (self.canvas_width, self.canvas_height),
                    ResolutionSource::Fixed(fw, fh) => (fw, fh),
                };

                resources[id_usize] = Some(pool.acquire(device, desc.format, *usage, w, h));
            }
        }

        drop(pool);

        let ctx = RenderContext::new(resources);

        for &node_idx in &self.order {
            let node = &self.nodes[node_idx];
            encoder.push_debug_group(node.name());
            node.record(encoder, &ctx, frame);
            encoder.pop_debug_group();
        }

        callback(encoder, &ctx);

        let mut pool = self.texture_pool.lock().unwrap();
        for res in ctx.into_resources().into_iter().flatten() {
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
