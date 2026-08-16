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

        let mut node_metas = Vec::with_capacity(self.nodes.len());
        let mut resource_to_producer = HashMap::new();
        let mut id_counter = 2; // 0 and 1 are FINAL_COLOR and SCREEN

        // Step 1: Declare resources
        for (i, node) in self.nodes.iter().enumerate() {
            let mut builder = ResourceBuilder::new(id_counter);
            node.declare_resources(&mut builder);
            
            for &(written_id, _) in &builder.writes {
                resource_to_producer.insert(written_id, i);
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

        // Step 2: Build dependency edges
        for i in 0..self.nodes.len() {
            let reads = node_metas[i].reads.clone();
            for (r, _) in reads {
                if let Some(&producer) = resource_to_producer.get(&r) {
                    if producer != i {
                        node_metas[producer].dependents.push(i);
                        node_metas[i].in_degree += 1;
                    }
                }
            }
        }

        // Step 3: Kahn's topological sort
        let mut queue = VecDeque::new();
        for (i, meta) in node_metas.iter().enumerate() {
            if meta.in_degree == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted_order = Vec::with_capacity(self.nodes.len());
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

        if sorted_order.len() < self.nodes.len() {
            return Err(GraphError::CyclicDependency);
        }

        // Step 4: Build resource descriptor map and compute usages
        let max_id = node_metas.iter()
            .flat_map(|m| m.creates.iter().map(|(id, _)| id.0 as usize))
            .max()
            .unwrap_or(1);
            
        let mut descriptors: Vec<Option<(ResourceDescriptor, wgpu::TextureUsages)>> = (0..=max_id).map(|_| None).collect();
        for meta in &node_metas {
            for (id, desc) in &meta.creates {
                descriptors[id.0 as usize] = Some((desc.clone(), wgpu::TextureUsages::empty()));
            }
        }

        for meta in &node_metas {
            for &(id, access) in &meta.reads {
                if let Some((_, ref mut usage)) = descriptors[id.0 as usize] {
                    *usage |= access.to_wgpu_usage();
                }
            }
            for &(id, access) in &meta.writes {
                if let Some((_, ref mut usage)) = descriptors[id.0 as usize] {
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

pub struct CompiledGraph {
    nodes:          Vec<Box<dyn RenderNode>>,
    order:          Vec<usize>,
    descriptors:    Vec<Option<(ResourceDescriptor, wgpu::TextureUsages)>>,
    canvas_width:   u32,
    canvas_height:  u32,
    texture_pool:   Mutex<TransientTexturePool>,
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

    /// Execute the compiled graph for one frame.
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device:  &GpuDevice,
        frame:   &FrameState,
    ) {
        // Step 1 & 2: Acquire transient textures and create views
        let mut pool = self.texture_pool.lock().unwrap();
        let mut resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, crate::render::resource::ViewId)>> = (0..self.descriptors.len()).map(|_| None).collect();

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

        // Add external textures to the views map?
        // Actually, RenderContext handles the textures passed into the graph. Wait, how do external textures (like SCREEN or test_textures) get injected?
        // Oh, the nodes might handle SCREEN specifically. For test_textures (YUV data etc), the nodes create them or they are injected.
        // Actually, the framework handles this in Phase 4. For Phase 2, let's keep it simple.
        
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
        device:  &GpuDevice,
        frame:   &FrameState,
        callback: F,
    ) where F: FnOnce(&mut wgpu::CommandEncoder, &RenderContext)
    {
        let mut pool = self.texture_pool.lock().unwrap();
        let mut resources: Vec<Option<(wgpu::Texture, wgpu::TextureView, crate::render::resource::ViewId)>> = (0..self.descriptors.len()).map(|_| None).collect();

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

#[derive(Debug)]
pub enum GraphError {
    CyclicDependency,
    UnresolvableResource(ResourceId),
    EmptyGraph,
}
