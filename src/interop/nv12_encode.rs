// src/interop/nv12_encode.rs
// RGB → NV12 conversion on the GPU, writing into a pitched linear buffer.
//
// WHY THIS EXISTS.  The zero-copy NVENC path hands the encoder packed RGB
// (`Abgr10RepackNode` → NV_ENC_BUFFER_FORMAT_ABGR10) and lets the DRIVER do the
// RGB→YUV conversion.  The driver applies BT.601 and there is no supported way to
// tell it otherwise — selecting the matrix means writing
// NV_ENC_CONFIG's per-codec VUI union, i.e. guessed struct offsets.  Meanwhile
// `Muxer::open` tags the stream from `job.output_color`, which is BT.709 for every
// normal export.  Samples and tags therefore disagree, and
// `ExportJob::nvenc_zero_copy_is_colour_safe` currently routes every non-BT.601
// job away from zero-copy to avoid shipping that file.
//
// Converting here removes the disagreement at its root: NVENC receives NV12 that
// already carries our matrix, performs no conversion of its own, and our tags
// become authoritative by construction.
//
// WHY A BUFFER AND NOT A TEXTURE.  NV12 is two planes — full-resolution luma
// followed by half-resolution interleaved chroma — and
// NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY takes a single CUarray, so the planes must
// share one allocation.  Measured on this machine (RTX 3050, driver API 12.2,
// standalone C probes in `nvchk/`):
//
//   * An over-tall CUarray (W × 1.5H, 8-bit single channel) DOES work and decodes
//     to the right pixels — but only with a non-zero `NV_ENC_REGISTER_RESOURCE::pitch`.
//     With `pitch = 0`, which is what `encode_interop.rs` passes today for its
//     packed ABGR10 array, `nvEncEncodePicture` kills the process outright.
//   * A D3D12 shared BUFFER imported via `cuExternalMemoryGetMappedBuffer` and
//     registered as NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR also works, with
//     documented pitch semantics, and the chroma plane is read at
//     `pitch * height`.  Verified with pitch (320) deliberately ≠ width (256),
//     with the inter-row padding filled 0xAA so a misread would have shown up as
//     garbage rather than passing quietly.
//
// The buffer route is the one this node targets: the pitch is ours to state
// rather than something CUDA infers from a D3D12 texture footprint, and it is
// what libavcodec's own nvenc wrapper uses (nvenc.c:2269-2288).
//
// SCOPE OF THIS FILE.  It converts and packs, and its test proves the packing
// bit-exactly against the CPU reference in `colour::yuv::RgbToYuv`.  It proves
// nothing about CUDA import, NVENC registration, or an exported file — those are
// the following steps.

use crate::colour::yuv::RgbToYuv;
use crate::render::device::GpuDevice;
use crate::timeline::source::ColorInfo;

/// Push constants for the NV12 encode shader: the colour conversion plus the
/// geometry of the destination buffer.
///
/// The colour half is [`RgbToYuv`] verbatim, so the matrix and range decisions
/// are made in CPU code a unit test can check without a GPU (see
/// `src/colour/yuv.rs`) and the shader only applies the resolved numbers — the
/// same split `YuvToRgbNode` uses for the decode direction.
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Nv12EncodeParams {
    /// Matrix, range and the frame's dimensions.
    pub colour: RgbToYuv,
    /// Row stride of BOTH planes, in bytes.  Must be a multiple of 4: the shader
    /// addresses the buffer as `array<u32>`, so a row that does not start on a
    /// word boundary cannot be written without a read-modify-write race against
    /// the neighbouring row.
    pub pitch: u32,
    /// Byte offset of the interleaved chroma plane from the start of the buffer.
    /// `pitch * height` for the layout NVENC expects; carried explicitly rather
    /// than recomputed so the shader and the registration can never disagree.
    pub chroma_plane_offset: u32,
    pub _pad: [u32; 2],
}

/// 96 bytes: [`RgbToYuv`]'s 80 plus four words.  A multiple of 16 as WGSL
/// requires for a struct containing `vec4`s, and inside the 128-byte
/// push-constant limit `GpuDevice` requests.
const _: () = assert!(std::mem::size_of::<Nv12EncodeParams>() == 96);
const NV12_PUSH_CONSTANT_BYTES: u32 = std::mem::size_of::<Nv12EncodeParams>() as u32;

/// How many pixels one luma invocation writes: four 8-bit codes packed into one
/// `u32`.  Also the alignment the destination pitch must satisfy.
pub const LUMA_PIXELS_PER_INVOCATION: u32 = 4;

/// Row alignment [`Nv12EncodeNode::aligned_pitch`] rounds up to.
///
/// 256 rather than the 4 the shader strictly needs — see `aligned_pitch` for
/// why the extra width is deliberate.
pub const NV12_PITCH_ALIGNMENT: u32 = 256;

pub const NV12_ENCODE_WGSL: &str = r#"
// RGB → NV12, writing into a pitched linear buffer.
//
// Two entry points rather than one: the luma plane is full resolution and the
// chroma plane is half in both axes, so a single dispatch would leave three
// quarters of its invocations idle in the chroma phase.
//
// Each invocation writes exactly one 32-bit word, which is what makes the writes
// race-free without atomics: the buffer is addressed as array<u32>, and no two
// invocations touch the same word.

// This struct must match `Nv12EncodeParams` in src/interop/nv12_encode.rs BYTE
// FOR BYTE, and that type NESTS `RgbToYuv` — which carries its own trailing
// two-word padding.  So the layout is:
//
//   offset  field
//   0..48   row_y / row_cb / row_cr        (RgbToYuv)
//   48..64  luma_scale, luma_offset, chroma_scale, chroma_offset
//   64..72  width, height
//   72..80  _colour_pad                    <-- RgbToYuv's OWN padding
//   80..88  pitch, chroma_plane_offset     (Nv12EncodeParams)
//   88..96  _tail_pad
//
// Omitting `_colour_pad` is not a cosmetic error: it shifts `pitch` and
// `chroma_plane_offset` up by 8 bytes, the shader reads pitch = 0, and every row
// writes over row 0.  The symptom was luma coming back holding Cb values.
struct Nv12Params {
    row_y:               vec4<f32>,   // [Kr, Kg, Kb, _]
    row_cb:              vec4<f32>,
    row_cr:              vec4<f32>,
    luma_scale:          f32,
    luma_offset:         f32,
    chroma_scale:        f32,
    chroma_offset:       f32,
    width:               u32,
    height:              u32,
    _colour_pad:         vec2<u32>,
    pitch:               u32,
    chroma_plane_offset: u32,
    _tail_pad:           vec2<u32>,
}

var<push_constant> p: Nv12Params;

@group(0) @binding(0) var  in_rgba: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var<storage, read_write> out_nv12: array<u32>;

/// Fetch one pixel's RGB, clamped to the image and to [0,1].
///
/// The [0,1] clamp happens HERE, before the matrix, matching
/// `RgbToYuv::apply` — clamping Y/Cb/Cr afterwards instead would shift the hue of
/// a colour that is out of range in only one channel, and would let super-white
/// occupy the codes limited range reserves.
fn load_rgb(x: u32, y: u32) -> vec3<f32> {
    let cx = min(x, p.width  - 1u);
    let cy = min(y, p.height - 1u);
    let rgba = textureLoad(in_rgba, vec2<i32>(vec2<u32>(cx, cy)));
    return clamp(rgba.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Normalised sample in [0,1] -> 8-bit code, rounding half up.
fn code8(sample: f32) -> u32 {
    return u32(clamp(sample, 0.0, 1.0) * 255.0 + 0.5);
}

fn luma_sample(rgb: vec3<f32>) -> f32 {
    return dot(p.row_y.xyz, rgb) * p.luma_scale + p.luma_offset;
}

// One invocation = 4 horizontally adjacent luma pixels = 1 word.
@compute @workgroup_size(8, 8)
fn cs_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x0 = gid.x * 4u;
    let y  = gid.y;
    if (x0 >= p.width || y >= p.height) { return; }

    var word: u32 = 0u;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        // Past the right edge the edge pixel is replicated rather than left
        // undefined: those bytes live in the row's alignment padding, and feeding
        // the encoder garbage there wastes bitrate on an invisible column.
        let code = code8(luma_sample(load_rgb(x0 + i, y)));
        word = word | (code << (i * 8u));
    }
    out_nv12[(y * p.pitch + x0) / 4u] = word;
}

// One invocation = 2 chroma samples = (U,V,U,V) = 1 word.
@compute @workgroup_size(8, 8)
fn cs_chroma(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chroma_w = (p.width  + 1u) / 2u;
    let chroma_h = (p.height + 1u) / 2u;
    let cx0 = gid.x * 2u;
    let cy  = gid.y;
    if (cx0 >= chroma_w || cy >= chroma_h) { return; }

    var word: u32 = 0u;
    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
        let cx = min(cx0 + j, chroma_w - 1u);
        // Box-average the 2x2 luma-resolution block.  Averaging in RGB and then
        // applying the matrix is identical to averaging the resulting Cb/Cr,
        // because the matrix is linear — and cheaper.  `load_rgb` clamps each
        // sample first, so the average is of in-gamut values.
        let x = cx * 2u;
        let y = cy * 2u;
        let rgb = (load_rgb(x,      y)
                 + load_rgb(x + 1u, y)
                 + load_rgb(x,      y + 1u)
                 + load_rgb(x + 1u, y + 1u)) * 0.25;

        let cb = code8(dot(p.row_cb.xyz, rgb) * p.chroma_scale + p.chroma_offset);
        let cr = code8(dot(p.row_cr.xyz, rgb) * p.chroma_scale + p.chroma_offset);
        word = word | (cb << (j * 16u)) | (cr << (j * 16u + 8u));
    }
    out_nv12[(p.chroma_plane_offset + cy * p.pitch + cx0 * 2u) / 4u] = word;
}
"#;

/// Compute node converting the render graph's RGBA16Float output into NV12 in a
/// pitched linear buffer.
///
/// Holds two pipelines sharing one bind-group layout and one push-constant block.
pub struct Nv12EncodeNode {
    luma_pipeline:     wgpu::ComputePipeline,
    chroma_pipeline:   wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// The resolved colour conversion, computed once from the job's output
    /// colour description.
    colour: RgbToYuv,
    width:  u32,
    height: u32,
}

impl Nv12EncodeNode {
    /// Build the node for a frame size and output colour description.
    pub fn new(device: &GpuDevice, color: ColorInfo, width: u32, height: u32) -> Self {
        let bind_group_layout =
            device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Nv12Encode BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shader = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nv12Encode"),
            source: wgpu::ShaderSource::Wgsl(NV12_ENCODE_WGSL.into()),
        });

        let layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nv12Encode PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..NV12_PUSH_CONSTANT_BYTES,
            }],
        });

        let make = |entry_point: &str, label: &str| {
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point,
                })
        };

        let colour = RgbToYuv::new(color, width, height);
        log::debug!(
            "[colour] Nv12Encode {}x{}: matrix={:?} range={:?} \
             (luma {:.4}/{:.4}, chroma {:.4}/{:.4})",
            width, height,
            color.effective_matrix(width, height),
            color.effective_range(),
            colour.luma_scale, colour.luma_offset,
            colour.chroma_scale, colour.chroma_offset,
        );

        Self {
            luma_pipeline:   make("cs_luma",   "Nv12Encode Luma Pipeline"),
            chroma_pipeline: make("cs_chroma", "Nv12Encode Chroma Pipeline"),
            bind_group_layout,
            colour,
            width,
            height,
        }
    }

    /// The resolved conversion this node applies.  Exposed so a test (or a log
    /// line) can state which matrix a frame actually got.
    pub fn colour(&self) -> &RgbToYuv {
        &self.colour
    }

    /// Total size in bytes of an NV12 buffer at this frame size and pitch.
    ///
    /// Chroma is `ceil(height/2)` rows, so an odd height does not lose its last
    /// chroma row.
    pub fn buffer_size(height: u32, pitch: u32) -> u64 {
        pitch as u64 * (height as u64 + (height as u64).div_ceil(2))
    }

    /// The smallest legal pitch for a width: the row length rounded up to the
    /// 4-byte word the shader writes.
    pub fn min_pitch(width: u32) -> u32 {
        width.div_ceil(LUMA_PIXELS_PER_INVOCATION) * LUMA_PIXELS_PER_INVOCATION
    }

    /// The pitch the export path actually allocates: the row length rounded up
    /// to [`NV12_PITCH_ALIGNMENT`].
    ///
    /// Wider than [`Self::min_pitch`] (which is only the 4 the shader strictly
    /// needs) so each row starts on a 256-byte boundary: the shader writes one
    /// `u32` per invocation across a row, and an aligned row start keeps a
    /// workgroup's writes inside whole cache lines instead of straddling them.
    ///
    /// **This does NOT guarantee `pitch != width`.**  Any width that is already a
    /// multiple of 256 — 1280 and 3840 among them — comes back unchanged, so the
    /// common HD/UHD cases really do run with `pitch == width`, where
    /// `pitch * height` and `width * height` are the same number and a chroma
    /// plane sited at the wrong one would be invisible.  Making the two differ in
    /// production by padding every frame further would cost real VRAM to protect a
    /// property the hardware does not care about; the offset arithmetic is instead
    /// pinned by `tests::shared_buffer::nv12_encode_writes_into_a_shared_buffer`
    /// and `tests::nv12_encode`, which pick 320 for a 256-wide frame precisely so
    /// the two expressions disagree by 8192 bytes.
    ///
    /// Measured-good: NVENC accepted a CUDADEVICEPTR NV12 registration with
    /// `pitch = 320` on a 256-wide frame (1.25x) and decoded the right pixels
    /// (`nvchk/d3d12_buf_probe.c`).  The header's only stated constraint on this
    /// field for CUDADEVICEPTR is "must be a multiple of 4".
    pub fn aligned_pitch(width: u32) -> u32 {
        width.div_ceil(NV12_PITCH_ALIGNMENT) * NV12_PITCH_ALIGNMENT
    }

    /// Record both passes: luma then chroma, into `out_buffer` at `pitch`.
    ///
    /// # Panics
    /// If `pitch` is not a multiple of 4 or is narrower than the frame. Both are
    /// programming errors in the caller's allocation, not runtime conditions, and
    /// letting them through would corrupt the row layout silently — the shader
    /// would write across row boundaries and the encoder would see a sheared
    /// image.
    pub fn record(
        &self,
        encoder:    &mut wgpu::CommandEncoder,
        device:     &GpuDevice,
        in_view:    &wgpu::TextureView,
        out_buffer: &wgpu::Buffer,
        pitch:      u32,
    ) {
        assert_eq!(
            pitch % LUMA_PIXELS_PER_INVOCATION, 0,
            "NV12 pitch {pitch} must be a multiple of {LUMA_PIXELS_PER_INVOCATION}: \
             the shader addresses the buffer as array<u32> and a row that does not \
             start on a word boundary cannot be written race-free"
        );
        assert!(
            pitch >= Self::min_pitch(self.width),
            "NV12 pitch {pitch} is narrower than the {}px frame needs ({})",
            self.width, Self::min_pitch(self.width)
        );

        let params = Nv12EncodeParams {
            colour: self.colour,
            pitch,
            chroma_plane_offset: pitch * self.height,
            _pad: [0; 2],
        };
        let push_bytes = bytemuck::bytes_of(&params);

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nv12Encode BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(in_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Nv12Encode"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);

        // `set_push_constants` requires a pipeline to already be bound, and the
        // constants do NOT survive a `set_pipeline`, so each pass sets its own.
        // Hoisting this above the two `set_pipeline` calls fails validation with
        // "Compute pipeline must be set".

        // Luma: one invocation per 4 pixels across, one per row down.
        pass.set_pipeline(&self.luma_pipeline);
        pass.set_push_constants(0, push_bytes);
        pass.dispatch_workgroups(
            self.width.div_ceil(LUMA_PIXELS_PER_INVOCATION).div_ceil(8),
            self.height.div_ceil(8),
            1,
        );

        // Chroma: half resolution both ways, two chroma samples per invocation.
        let chroma_w = self.width.div_ceil(2);
        let chroma_h = self.height.div_ceil(2);
        pass.set_pipeline(&self.chroma_pipeline);
        pass.set_push_constants(0, push_bytes);
        pass.dispatch_workgroups(chroma_w.div_ceil(2).div_ceil(8), chroma_h.div_ceil(8), 1);
    }
}
