// src/render/compute.rs

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::render::device::GpuDevice;
use crate::render::shader::registry::BuiltinShader;

/// Key into the pipeline cache.
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct PipelineKey {
    pub shader:      BuiltinShader,
    pub entry_point: &'static str,
}

/// Compile-once, thread-safe cache of wgpu compute pipelines.
/// Pipelines are expensive to create. We compile at startup and reuse the Arc every frame.
pub struct ComputePipelineCache {
    pipelines: RwLock<HashMap<PipelineKey, Arc<wgpu::ComputePipeline>>>,
}

impl Default for ComputePipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputePipelineCache {
    pub fn new() -> Self {
        Self { pipelines: RwLock::new(HashMap::new()) }
    }

    /// Get or compile a compute pipeline for the given shader + entry point.
    ///
    /// Uses double-checked locking: read first, then write only on cache miss.
    pub fn get_or_compile(
        &self,
        device:     &GpuDevice,
        key:        PipelineKey,
        layout:     &wgpu::PipelineLayout,
        shader_mod: &wgpu::ShaderModule,
    ) -> Arc<wgpu::ComputePipeline> {
        // Step 1 — Check cache under read lock (hot path, no write-lock contention)
        if let Some(p) = self.pipelines.read().unwrap().get(&key).cloned() {
            return p;
        }

        // Step 2 — Compile under write lock with double-check
        let mut map = self.pipelines.write().unwrap();
        if let Some(p) = map.get(&key).cloned() {
            return p; // another thread compiled between step 1 and 2
        }

        // Step 3 — Create compute pipeline
        let pipeline = device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key.entry_point),
            layout: Some(layout),
            module: shader_mod,
            entry_point: key.entry_point,
        });

        // Step 4 — Insert and return
        let arc = Arc::new(pipeline);
        map.insert(key, arc.clone());
        arc
    }
}

/// Workgroup size convention: all 2D image-processing shaders use 16×16.
const WG_X: u32 = 16;
const WG_Y: u32 = 16;

/// Per-dispatch helper: calculates workgroup counts and records the dispatch command.
pub struct ComputePassHelper;

impl ComputePassHelper {
    /// Calculate the number of workgroups needed to cover `width × height` pixels.
    ///
    /// Uses integer ceiling division: ceil(a/b) = (a + b - 1) / b
    ///
    /// # Panics
    /// Panics if workgroup_x or workgroup_y is zero.
    pub fn workgroup_count(
        width:       u32,
        height:      u32,
        workgroup_x: u32,
        workgroup_y: u32,
    ) -> (u32, u32) {
        assert!(workgroup_x > 0, "workgroup_x must be > 0");
        assert!(workgroup_y > 0, "workgroup_y must be > 0");
        let dispatch_x = width.div_ceil(workgroup_x);
        let dispatch_y = height.div_ceil(workgroup_y);
        (dispatch_x, dispatch_y)
    }

    /// Record a compute dispatch into `pass`.
    pub fn dispatch<'a>(
        pass:       &mut wgpu::ComputePass<'a>,
        pipeline:   &'a wgpu::ComputePipeline,
        bind_group: &'a wgpu::BindGroup,
        push_bytes: Option<&[u8]>,
        width:      u32,
        height:     u32,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        if let Some(bytes) = push_bytes {
            pass.set_push_constants(0, bytes);
        }
        let (dx, dy) = Self::workgroup_count(width, height, WG_X, WG_Y);
        pass.dispatch_workgroups(dx, dy, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workgroup_count_exact_multiple() {
        assert_eq!(ComputePassHelper::workgroup_count(32, 32, 16, 16), (2, 2));
    }

    #[test]
    fn workgroup_count_needs_ceiling() {
        assert_eq!(ComputePassHelper::workgroup_count(1, 1, 16, 16), (1, 1));
        assert_eq!(ComputePassHelper::workgroup_count(17, 17, 16, 16), (2, 2));
        assert_eq!(ComputePassHelper::workgroup_count(100, 200, 16, 16), (7, 13));
    }

    #[test]
    #[should_panic]
    fn workgroup_count_zero_panics() {
        ComputePassHelper::workgroup_count(100, 100, 0, 16);
    }
}
