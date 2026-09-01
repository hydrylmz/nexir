// src/render/nodes/yuv_to_rgb.rs

use std::sync::Arc;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId, ViewId};
use crate::render::context::RenderContext;
use std::sync::Mutex;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::render::compute::{ComputePipelineCache, ComputePassHelper, PipelineKey};
use crate::render::shader::registry::{ShaderRegistry, BuiltinShader};
use crate::colour::yuv::YuvConversion;
use crate::timeline::source::ColorInfo;

/// Push constants for the YUV→RGB compute shader.
///
/// This is [`YuvConversion`] verbatim — the colour decisions are made on the CPU
/// (see `src/colour/yuv.rs`) so they can be unit-tested without a GPU, and the
/// shader only applies the resolved numbers.
pub type YuvParams = YuvConversion;

/// Byte size of the push-constant block, kept in one place because the pipeline
/// layout and the WGSL `var<push_constant>` must agree exactly.
const YUV_PUSH_CONSTANT_BYTES: u32 = std::mem::size_of::<YuvConversion>() as u32;

pub struct YuvToRgbNode {
    /// Y-plane texture (R8Unorm or R16Unorm), produced by YuvUploadNode — or, on
    /// the interop path, imported from the decoder. See [`Self::imported_planes`].
    pub in_y:          ResourceId,
    /// UV-plane texture (Rg8Unorm or Rg16Unorm), produced by YuvUploadNode.
    pub in_uv:         ResourceId,
    /// RGBA16Float output texture written by this node.
    pub out_rgba:      ResourceId,
    pub width:         u32,
    pub height:        u32,
    /// Whether `in_y`/`in_uv` come from OUTSIDE the graph (G2c/G2d).
    ///
    /// **This changes `declare_resources`, and it has to**: with no
    /// `YuvUploadNode` in front of it there is no node creating those two ids, so a
    /// plain `builder.read` is a read with no producer and compilation fails with
    /// [`crate::render::graph::GraphError::MissingProducer`].
    /// [`ResourceBuilder::import`] registers the id as its own producer instead —
    /// which is what keeps `MissingProducer` meaningful for the case it exists to
    /// catch (gotcha 18).
    ///
    /// It must NOT be inferred from anything the node can see. A node that guessed
    /// wrong in the other direction — importing a plane the upload node creates —
    /// is rejected at compile time by the `create` + `import` check, which is the
    /// only reason that mistake is not a torn frame.
    imported_planes:   bool,
    /// The fully resolved conversion: matrix, range and bit-depth handling for
    /// this clip's actual metadata.  Computed once at graph-compile time.
    conversion:        YuvConversion,
    pipeline:          Arc<wgpu::ComputePipeline>,
    bind_group_layout: wgpu::BindGroupLayout,
    device:            Arc<wgpu::Device>,
    bg_cache:          Mutex<Option<([ViewId; 3], wgpu::BindGroup)>>,
}

impl YuvToRgbNode {
    /// Build the node for a clip whose chroma is planar (I420 / YUV420P10LE …).
    ///
    /// Prefer [`Self::new_with_layout`] and pass the real layout: for 10-bit
    /// content, planar and semi-planar differ by a factor of 64 in sample
    /// normalisation, so guessing is a visible bug rather than a nuance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_y:           ResourceId,
        in_uv:          ResourceId,
        out_rgba:       ResourceId,
        width:          u32,
        height:         u32,
        color_info:     ColorInfo,
    ) -> Self {
        Self::new_with_layout(
            device, shaders, pipeline_cache,
            in_y, in_uv, out_rgba, width, height, color_info,
            false,
        )
    }

    /// Build the node, stating whether the source's chroma is semi-planar.
    ///
    /// `is_semi_planar` is true for NV12 and P010 (one interleaved chroma plane;
    /// P010 additionally MSB-aligns its 10-bit codes) and false for planar YUV.
    /// It reaches `YuvConversion` unchanged — see the module docs there for why
    /// it matters.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_layout(
        device:         &GpuDevice,
        shaders:        &ShaderRegistry,
        pipeline_cache: &ComputePipelineCache,
        in_y:           ResourceId,
        in_uv:          ResourceId,
        out_rgba:       ResourceId,
        width:          u32,
        height:         u32,
        color_info:     ColorInfo,
        is_semi_planar: bool,
    ) -> Self {
        // Step 1 — Build bind group layout: Y, UV, RGBA(rgba16f)
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("yuv_to_rgb_bgl"),
                entries: &[
                    // binding 0: Y plane
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: UV plane
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: RGBA16Float output (storage write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            },
        );

        // Step 2 — Build pipeline layout with the YuvConversion push constants
        let pipeline_layout = device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("yuv_to_rgb_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..YUV_PUSH_CONSTANT_BYTES,
                }],
            },
        );

        // Step 3 — Get or compile compute pipeline
        let shader_mod = shaders.get(BuiltinShader::YuvToRgb);
        let pipeline = pipeline_cache.get_or_compile(
            device,
            PipelineKey { shader: BuiltinShader::YuvToRgb, entry_point: "cs_main" },
            &pipeline_layout,
            &shader_mod,
        );

        // Step 4 — Resolve the colour conversion once, here.  `YuvConversion`
        // applies ColorInfo's own resolution heuristics for unsignalled metadata
        // (SD → BT.601, UHD → BT.2020, unknown range → limited).
        let conversion = YuvConversion::new(color_info, width, height, is_semi_planar);
        log::debug!(
            "[colour] YuvToRgb {}x{}: matrix={:?} range={:?} depth={} semi_planar={} \
             (sample_scale={:.3}, luma {:.4}/{:.4}, chroma {:.4}/{:.4})",
            width, height,
            color_info.effective_matrix(width, height),
            color_info.effective_range(),
            color_info.bit_depth,
            is_semi_planar,
            conversion.sample_scale,
            conversion.luma_offset, conversion.luma_scale,
            conversion.chroma_offset, conversion.chroma_scale,
        );

        Self {
            in_y,
            in_uv,
            out_rgba,
            width,
            height,
            // Pooled planes by default: every existing caller feeds this node from a
            // `YuvUploadNode`, and `with_imported_planes` is the opt-in.
            imported_planes: false,
            conversion,
            pipeline,
            bind_group_layout,
            device: Arc::clone(&device.device),
            bg_cache: Mutex::new(None),
        }
    }

    /// State that `in_y`/`in_uv` are bound by the frame rather than created by a
    /// `YuvUploadNode` — the interop decode path (G2c/G2d).
    ///
    /// A builder-style setter rather than another `new_*` constructor: the node
    /// already takes ten arguments, and the two paths differ in exactly this one
    /// bit. The caller that sets it is the same one that omits the upload node and
    /// takes its plane ids from `FrameScheduler::interop_y_id`; those three
    /// decisions are one decision, and `src/tests/interop_graph.rs` pins that
    /// getting any of them wrong fails to compile the graph or fails a pixel check
    /// rather than rendering something plausible.
    pub fn with_imported_planes(mut self, imported: bool) -> Self {
        self.imported_planes = imported;
        self
    }

    /// Whether this node imports its planes instead of reading pooled ones.
    pub fn imports_planes(&self) -> bool {
        self.imported_planes
    }

    /// The resolved conversion this node will apply.  Exposed so tests (and the
    /// inspector) can check what colour handling a clip actually got.
    pub fn conversion(&self) -> &YuvConversion {
        &self.conversion
    }
}

impl RenderNode for YuvToRgbNode {
    fn name(&self) -> &str { "YuvToRgb" }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((self.out_rgba, ResourceDescriptor {
            label: Some(format!("YuvToRgb_{}", self.out_rgba.0)),
            size: ResolutionSource::Fixed(self.width, self.height),
            format: wgpu::TextureFormat::Rgba16Float,
        }));
        // Imported planes are declared with `import`, not `read`: nothing inside the
        // graph produces them (the upload node is absent), so a bare read would be
        // `MissingProducer`, and `import` is also what stops the acquire loop
        // allocating a pooled texture the frame's binding would then shadow —
        // AGENTS.md gotcha 18.
        if self.imported_planes {
            builder.import(self.in_y, TextureAccess::Sampled);
            builder.import(self.in_uv, TextureAccess::Sampled);
        } else {
            builder.read(self.in_y, TextureAccess::Sampled);
            builder.read(self.in_uv, TextureAccess::Sampled);
        }
        builder.write(self.out_rgba, TextureAccess::StorageWrite);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        // Step 1 — Retrieve resolved texture views
        let y_res    = ctx.get(self.in_y);
        let uv_res   = ctx.get(self.in_uv);
        let rgba_res = ctx.get(self.out_rgba);

        // Step 2 — Create per-frame bind group if cache missed
        let mut cache = self.bg_cache.lock().unwrap();
        let cache_key = [y_res.view_id, uv_res.view_id, rgba_res.view_id];

        if cache.is_none() || cache.as_ref().unwrap().0 != cache_key {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("yuv_to_rgb_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(y_res.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(uv_res.view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(rgba_res.view) },
                ],
            });
            *cache = Some((cache_key, bind_group));
        }

        let bind_group = &cache.as_ref().unwrap().1;

        // Step 3 — Hand the pre-resolved conversion to the shader.
        let push_bytes = bytemuck::bytes_of(&self.conversion);

        // Step 4 — Begin compute pass and dispatch
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("yuv_to_rgb"),
            timestamp_writes: None,
        });
        ComputePassHelper::dispatch(
            &mut pass, &self.pipeline, bind_group,
            Some(push_bytes),
            self.width, self.height,
        );
        drop(pass);
    }
}
