// src/bin/bench.rs
// Nexir frame-timing benchmark suite.
//
// Run with: cargo run -p nexir --bin bench --release
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT THIS MEASURES, AND WHAT IT REFUSES TO CLAIM
//
// Every number printed here is timed with `Instant::now()` around work that
// actually ran, or counted from a byte size this process really allocated or
// transferred. Nothing is estimated.
//
// That is worth stating because this file used to do the opposite. It reported
// `gpu_utilization: 88.5`, `nvenc_utilization: 94.0` and `ram_used_bytes: 420 MB`
// as hardcoded literals, and its "Zero-Copy NVENC" execution path was a
// `std::thread::yield_now()` with a comment claiming it simulated ~0.2 ms of
// encoder overhead — so the NVENC column of every run was a measurement of how
// long it takes to yield a thread, printed next to a fabricated utilisation
// figure. A benchmark that invents its results is worse than no benchmark: it
// gets quoted.
//
// So:
//   * `ExecutionPath::GpuNvencZeroCopy` opens a real NVENC session through
//     `EncodeInterop`, converts each rendered frame to NV12 with
//     `Nv12EncodeNode` straight into the CUDA/D3D12 shared buffer NVENC reads,
//     submits the picture, and reports the bitstream it got back. If the
//     hardware or CUDA interop is missing, the benchmark SKIPS with the reason
//     printed — it does not fall back to timing nothing.
//   * GPU utilisation, NVENC utilisation, CPU utilisation, driver VRAM and
//     process RSS need NVML / `cuMemGetInfo` / an OS query, none of which this
//     process performs. They are left unmeasured and print as `n/a`.
//
// Adding a real driver query later means filling in the corresponding
// `SystemMetrics` field. Do not fill one in from arithmetic.
// ─────────────────────────────────────────────────────────────────────────────

use std::sync::Arc;

use nexir::colour::lut_parser::Lut3D;
use nexir::export::job::{
    AudioCodec, Container, CpuPreset, ExportJob, VideoCodec, VideoQuality,
};
use nexir::interop::capability::InteropCapability;
use nexir::interop::cuda_context::CudaContext;
use nexir::interop::encode_interop::EncodeInterop;
use nexir::interop::nv12_encode::Nv12EncodeNode;
use nexir::profiling::{FrameProfile, PipelineStage, ProfilingSession, SystemMetrics};
use nexir::render::compute::ComputePipelineCache;
use nexir::render::device::GpuDevice;
use nexir::render::frame_state::{ClipRenderEntry, FrameState};
use nexir::render::graph::RenderGraphCompiler;
use nexir::render::nodes::chroma_key::ChromaKeyNode;
use nexir::render::nodes::color_correction::{ColorCorrectionNode, ColorCorrectionParams};
use nexir::render::nodes::composite::CompositeNode;
use nexir::render::nodes::lut::LutNode;
use nexir::render::nodes::tonemap::{
    GamutConversion, InputTransferFn, ToneMapMode, ToneMapNode, ToneMapPushConstants,
};
use nexir::render::nodes::yuv_to_rgb::YuvToRgbNode;
use nexir::render::nodes::yuv_upload::YuvUploadNode;
use nexir::render::resource::ResourceId;
use nexir::render::shader::registry::ShaderRegistry;
use nexir::timeline::ids::SourceId;
use nexir::timeline::rational::Rational;
use nexir::timeline::source::{
    ColorInfo, ColorPrimaries, ColorRange, MatrixCoefficients, TransferFunction,
};
use nexir::timeline::store::ClipKind;
use nexir::timeline::transform::ClipTransform;

#[derive(Debug, Clone, Copy)]
pub enum WorkloadComplexity {
    /// 1080p, 1 video layer, composite only.
    Simple,
    /// 1080p, 3 video layers with colour correction.
    Medium,
    /// 4K, 4 video layers with colour correction + LUT + chroma key + tone map.
    Heavy,
}

impl WorkloadComplexity {
    /// How many upload/composite layers this workload builds.
    fn layer_count(self) -> usize {
        match self {
            Self::Simple => 1,
            Self::Medium => 3,
            Self::Heavy => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPath {
    /// Render graph only; the result stays on the GPU and is never read.
    GpuRender,
    /// Render graph → `Nv12EncodeNode` → shared buffer → a real NVENC session.
    GpuNvencZeroCopy,
    /// Render graph → `copy_texture_to_buffer` → map and touch on the CPU.
    CpuReadback,
}

struct BenchmarkConfig {
    name: &'static str,
    complexity: WorkloadComplexity,
    path: ExecutionPath,
    canvas_w: u32,
    canvas_h: u32,
    target_fps: f64,
    frame_count: usize,
}

/// Why a benchmark could not run, for honest reporting instead of a zero row.
struct Skipped(String);

fn align256(n: u32) -> u32 {
    (n + 255) & !255
}

/// A static NV12 test frame: eight vertical luma bars with alternating chroma,
/// shifted horizontally by `phase` pixels.
///
/// Not flat grey, which is what this used to upload. Flat input makes the whole
/// pipeline unrepresentative at the one place it matters most — the encoder: a
/// constant frame costs NVENC almost nothing to compress, so its bitstream says
/// nothing about real footage. Bars give it spatial detail, and varying `phase`
/// across a pre-built set of frames gives it motion, so inter-prediction has to
/// do actual work.
///
/// Still not real footage: a rigid horizontal pan is the easiest motion a motion
/// estimator can face, and there is no noise or grain. Treat the NVENC timings as
/// a floor rather than a prediction of an export.
fn make_nv12(w: u32, h: u32, phase: u32) -> Vec<u8> {
    let y_size = (w * h) as usize;
    let mut buf = vec![0u8; y_size + y_size / 2];

    for y in 0..h {
        for x in 0..w {
            let bar = ((x + phase) % w * 8 / w) as u8;
            // 16..235 is the limited-range luma span; step through it per bar.
            buf[(y * w + x) as usize] = 16 + bar * 31;
        }
    }

    let chroma_w = w / 2;
    let chroma_h = h / 2;
    for cy in 0..chroma_h {
        for cx in 0..chroma_w {
            let bar = ((cx + phase / 2) % chroma_w * 8 / chroma_w) as u8;
            let base = y_size + (cy * w + cx * 2) as usize;
            // Push Cb and Cr in opposite directions so the bars differ in hue as
            // well as brightness.
            buf[base] = 128u8.wrapping_add(bar.wrapping_mul(12));
            buf[base + 1] = 128u8.wrapping_sub(bar.wrapping_mul(12));
        }
    }

    buf
}

/// How many distinct frames the pattern cycles through.
///
/// Built once before timing starts, then indexed per frame: generating a frame
/// inside the loop would charge NV12 synthesis to the Upload stage, which
/// measures `queue.write_texture` and nothing else.
const PATTERN_FRAMES: usize = 8;

/// The export job an NVENC session is opened against: H.264, BT.709 limited
/// 8-bit, at the benchmark's own canvas size and frame rate.
fn nvenc_job(config: &BenchmarkConfig) -> ExportJob {
    let tb = Rational::TIMEBASE_90K;
    let fps = Rational::new(config.target_fps.round() as i64, 1);
    let frame_dur = tb.den / fps.num;
    ExportJob {
        // Nothing is muxed here — `EncodeInterop` never opens the file — but
        // `ExportJob::validate` insists on a plausible one.
        output_path: std::env::temp_dir().join("nexir_bench_unused.mp4"),
        container: Container::Mp4,
        video_codec: VideoCodec::H264,
        audio_codec: AudioCodec::Aac,
        quality: VideoQuality::Crf(23),
        audio_bitrate: 128_000,
        pts_in: 0,
        pts_out: frame_dur * config.frame_count as i64,
        width: config.canvas_w,
        height: config.canvas_h,
        frame_rate: fps,
        project_tb: tb,
        render_threads: 1,
        cpu_preset: CpuPreset::Medium,
        output_color: ColorInfo::bt709(),
        hdr10: None,
    }
}

fn run_benchmark(
    device: &Arc<GpuDevice>,
    config: &BenchmarkConfig,
) -> Result<ProfilingSession, Skipped> {
    let shaders = ShaderRegistry::compile_all(device).expect("Shader compilation failed");
    let compute = Arc::new(ComputePipelineCache::new());
    let color_info = ColorInfo {
        matrix: MatrixCoefficients::Bt709,
        range: ColorRange::Limited,
        transfer_fn: TransferFunction::Bt709,
        primaries: ColorPrimaries::Bt709,
        bit_depth: 8,
    };

    // ── NVENC session, opened BEFORE any timing ───────────────────────────────
    // Opened first so an unavailable encoder is a skip with a reason rather than
    // a benchmark that reports timings for a path it never took.
    let mut nvenc: Option<(EncodeInterop, Nv12EncodeNode)> = None;
    if config.path == ExecutionPath::GpuNvencZeroCopy {
        let capability = InteropCapability::probe(device);
        if !capability.is_available() {
            return Err(Skipped(format!(
                "CUDA interop is unavailable on this machine (transport={:?}), so the \
                 zero-copy NVENC path cannot be exercised at all",
                capability.transport
            )));
        }
        let cuda_ctx = Arc::new(
            CudaContext::new(&capability)
                .map_err(|e| Skipped(format!("CudaContext::new failed: {e:?}")))?,
        );
        let job = nvenc_job(config);
        job.validate()
            .map_err(|e| Skipped(format!("the benchmark's own ExportJob is invalid: {e:?}")))?;
        let interop = EncodeInterop::open(
            Arc::clone(&cuda_ctx),
            device,
            &job,
            capability.transport,
            job.video_codec,
        )
        .map_err(|e| Skipped(format!("EncodeInterop::open failed: {e:?}")))?;

        let nv12 = Nv12EncodeNode::new(device, job.output_color, job.width, job.height);
        nvenc = Some((interop, nv12));
    }

    let mut compiler = RenderGraphCompiler::new();
    let mut id_counter = 2u32;
    let mut uploader_indices = Vec::new();
    let mut final_composite_inputs = Vec::new();

    // ── Build the render graph according to complexity ────────────────────────

    match config.complexity {
        WorkloadComplexity::Simple => {
            // 1 video clip → YUV upload → YUV to RGB → composite
            let rgba_id = ResourceId::next(&mut id_counter);
            let y_id = ResourceId::next(&mut id_counter);
            let uv_id = ResourceId::next(&mut id_counter);

            let u_idx = compiler.add_node(Box::new(YuvUploadNode::new_with_layout(
                device, 0, config.canvas_w, config.canvas_h, y_id, uv_id,
                nexir::timeline::source::FrameLayout::NV12,
            )));
            uploader_indices.push(u_idx);

            compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                device,
                &shaders,
                &compute,
                y_id,
                uv_id,
                rgba_id,
                config.canvas_w,
                config.canvas_h,
                color_info,
                // The bench uploads NV12 (semi-planar) sample data.
                true,
            )));

            let mut composite = CompositeNode::new(
                device,
                &shaders,
                ResourceId::FINAL_COLOR,
                1,
                wgpu::TextureFormat::Rgba16Float,
            );
            composite.input_textures.push(rgba_id);
            compiler.add_node(Box::new(composite));
        }
        WorkloadComplexity::Medium => {
            // 3 layers with colour correction
            for i in 0..3 {
                let y_id = ResourceId::next(&mut id_counter);
                let uv_id = ResourceId::next(&mut id_counter);
                let rgb_id = ResourceId::next(&mut id_counter);
                let cc_id = ResourceId::next(&mut id_counter);

                let u_idx = compiler.add_node(Box::new(YuvUploadNode::new_with_layout(
                    device, i, config.canvas_w, config.canvas_h, y_id, uv_id,
                    nexir::timeline::source::FrameLayout::NV12,
                )));
                uploader_indices.push(u_idx);

                compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                    device,
                    &shaders,
                    &compute,
                    y_id,
                    uv_id,
                    rgb_id,
                    config.canvas_w,
                    config.canvas_h,
                    color_info,
                    true,
                )));

                let mut params = ColorCorrectionParams::identity(config.canvas_w, config.canvas_h);
                params.saturation = 1.05 + 0.02 * (i as f32);

                compiler.add_node(Box::new(ColorCorrectionNode::new(
                    device,
                    &shaders,
                    &compute,
                    rgb_id,
                    cc_id,
                    params,
                )));

                final_composite_inputs.push(cc_id);
            }

            let mut composite = CompositeNode::new(
                device,
                &shaders,
                ResourceId::FINAL_COLOR,
                3,
                wgpu::TextureFormat::Rgba16Float,
            );
            composite.input_textures = final_composite_inputs.clone();
            compiler.add_node(Box::new(composite));
        }
        WorkloadComplexity::Heavy => {
            // 4 layers with colour correction + LUT + chroma key, then tone map
            let n = 17u32;
            let mut lut_data = Vec::with_capacity((n * n * n) as usize);
            for b in 0..n {
                for g in 0..n {
                    for r in 0..n {
                        lut_data.push([
                            r as f32 / (n - 1) as f32,
                            g as f32 / (n - 1) as f32,
                            b as f32 / (n - 1) as f32,
                        ]);
                    }
                }
            }
            let identity_lut = Lut3D {
                size: n,
                data: lut_data,
                domain_min: [0.0, 0.0, 0.0],
                domain_max: [1.0, 1.0, 1.0],
            };

            for i in 0..4 {
                let y_id = ResourceId::next(&mut id_counter);
                let uv_id = ResourceId::next(&mut id_counter);
                let rgb_id = ResourceId::next(&mut id_counter);
                let cc_id = ResourceId::next(&mut id_counter);
                let lut_id = ResourceId::next(&mut id_counter);
                let key_id = ResourceId::next(&mut id_counter);

                let u_idx = compiler.add_node(Box::new(YuvUploadNode::new_with_layout(
                    device, i, config.canvas_w, config.canvas_h, y_id, uv_id,
                    nexir::timeline::source::FrameLayout::NV12,
                )));
                uploader_indices.push(u_idx);

                compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                    device,
                    &shaders,
                    &compute,
                    y_id,
                    uv_id,
                    rgb_id,
                    config.canvas_w,
                    config.canvas_h,
                    color_info,
                    true,
                )));

                compiler.add_node(Box::new(ColorCorrectionNode::new(
                    device,
                    &shaders,
                    &compute,
                    rgb_id,
                    cc_id,
                    ColorCorrectionParams::identity(config.canvas_w, config.canvas_h),
                )));

                // `LutNode::new` only sees the LUT cube, so the frame size must be
                // set explicitly. Omitting it is why benchmark 5 used to abort with
                // wgpu's `Dimension X is zero` before printing anything.
                let mut lut_node = LutNode::new(
                    device,
                    &shaders,
                    &compute,
                    &identity_lut,
                    cc_id,
                    lut_id,
                    1.0,
                );
                lut_node.set_size(config.canvas_w, config.canvas_h);
                compiler.add_node(Box::new(lut_node));

                compiler.add_node(Box::new(ChromaKeyNode::new(
                    device,
                    &shaders,
                    &compute,
                    lut_id,
                    key_id,
                    nexir::render::nodes::chroma_key::ChromaKeyParams::green_screen(
                        config.canvas_w,
                        config.canvas_h,
                    ),
                )));

                final_composite_inputs.push(key_id);
            }

            let pre_tonemap_id = ResourceId::next(&mut id_counter);
            let mut composite = CompositeNode::new(
                device,
                &shaders,
                pre_tonemap_id,
                4,
                wgpu::TextureFormat::Rgba16Float,
            );
            composite.input_textures = final_composite_inputs.clone();
            compiler.add_node(Box::new(composite));

            compiler.add_node(Box::new(ToneMapNode::new(
                device,
                &shaders,
                &compute,
                pre_tonemap_id,
                ResourceId::FINAL_COLOR,
                ToneMapPushConstants::for_sdr_preview(
                    InputTransferFn::Linear,
                    GamutConversion::None,
                    ToneMapMode::AcesFilmic,
                    1000.0,
                    config.canvas_w,
                    config.canvas_h,
                ),
            )));
        }
    }

    let mut graph = compiler
        .compile(config.canvas_w, config.canvas_h)
        .expect("Graph compile failed");

    // ── Buffers this benchmark allocates itself ───────────────────────────────
    // Counted, not estimated: these are the sizes actually requested.
    let mut allocated_gpu_bytes: u64 = 0;

    let readback_stride = align256(config.canvas_w * 8);
    let readback_size = readback_stride as u64 * config.canvas_h as u64;
    let readback_buffer: Option<wgpu::Buffer> = match config.path {
        ExecutionPath::CpuReadback => {
            allocated_gpu_bytes += readback_size;
            Some(device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bench_readback"),
                size: readback_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
        }
        _ => None,
    };
    if let Some((interop, _)) = &nvenc {
        // The NV12 shared buffers `EncodeInterop::open` allocated on our behalf,
        // one per pipeline slot, at the pitch the registration declared.
        allocated_gpu_bytes += Nv12EncodeNode::buffer_size(config.canvas_h, interop.pitch())
            * interop.slot_count() as u64;
    }

    // Built up front, outside every timed region: synthesising NV12 inside the
    // loop would land in the Upload stage, which exists to measure
    // `queue.write_texture` and nothing else.
    let nv12_frames: Vec<Vec<u8>> = (0..PATTERN_FRAMES)
        .map(|i| {
            // Shift by a whole number of chroma samples so the two planes stay in
            // step, and by enough per frame that motion estimation cannot treat
            // successive frames as identical.
            let phase = (i as u32 * 16) % config.canvas_w;
            make_nv12(config.canvas_w, config.canvas_h, phase)
        })
        .collect();
    let upload_bytes_per_frame = nv12_frames[0].len() as u64 * uploader_indices.len() as u64;

    let mut frame_clips = Vec::new();
    for i in 0..uploader_indices.len() {
        frame_clips.push(ClipRenderEntry {
            source_id: SourceId::new(i as u32),
            texture_slot: i as u32,
            layer_order: i as u16,
            clip_width: config.canvas_w,
            clip_height: config.canvas_h,
            transform: ClipTransform::identity(),
            opacity: 1.0,
            blend_mode: nexir::timeline::transform::BlendMode::Normal,
            crop: nexir::timeline::transform::CropRect::full(),
            corner_pin: nexir::timeline::transform::CornerPin::identity(),
            matte_mode: nexir::timeline::transform::MatteMode::None,
            effects: Default::default(),
            // The bench feeds `make_nv12` data, so describe it as NV12 —
            // `ClipSignature`/`YuvUploadNode` both key off this.
            frame_meta: nexir::timeline::source::DecodedFrameMeta {
                layout: nexir::timeline::source::FrameLayout::NV12,
                color: color_info,
            },
            kind: ClipKind::Video,
        });
    }

    let frame_state = FrameState {
        pts: 0,
        canvas_width: config.canvas_w,
        canvas_height: config.canvas_h,
        clips: frame_clips,
        test_textures: vec![],
    };

    // ── GPU timing ────────────────────────────────────────────────────────────
    // One bracket around the whole graph submission. This is the measurement
    // that makes `GPU wait` interpretable: that stage is CPU wall time spent
    // blocked in `poll`, which is an upper bound on GPU execution and says
    // nothing about how much of it the GPU was busy for. Disabled devices report
    // nothing rather than zero.
    let mut gpu_timer = nexir::render::gpu_timer::GpuTimer::new(device, 1);

    // ── Warm-up: 3 frames, untimed ────────────────────────────────────────────
    // Shader/pipeline creation and the first bind-group cache miss would
    // otherwise land in frame 0 and skew its P99.
    for w in 0..3 {
        for &u_idx in &uploader_indices {
            if let Some(u) = graph.nodes_mut()[u_idx]
                .as_any_mut()
                .and_then(|n| n.downcast_mut::<YuvUploadNode>())
            {
                u.upload_frame(
                    &nv12_frames[w % PATTERN_FRAMES],
                    true,
                    config.canvas_w,
                    config.canvas_h,
                );
            }
        }
        let mut enc = device.begin_frame();
        graph.execute(&mut enc, device, &frame_state);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
    }

    // ── Measurement loop ──────────────────────────────────────────────────────
    let session = ProfilingSession::new(config.target_fps);
    let job = nvenc.as_ref().map(|_| nvenc_job(config));
    let mut bitstream_bytes: u64 = 0;
    let mut packets_out: usize = 0;
    let mut download_bytes: u64 = 0;

    for f in 0..config.frame_count {
        let pts = job.as_ref().map(|j| j.frame_pts(f)).unwrap_or(f as i64);
        let mut profile = FrameProfile::new(f, pts);

        // Stage: upload. Named Upload rather than Decode because no decoder ran —
        // the pattern frames were all built before the loop.
        profile.measure(PipelineStage::Upload, || {
            let frame_bytes = &nv12_frames[f % PATTERN_FRAMES];
            for &u_idx in &uploader_indices {
                if let Some(u) = graph.nodes_mut()[u_idx]
                    .as_any_mut()
                    .and_then(|n| n.downcast_mut::<YuvUploadNode>())
                {
                    u.upload_frame(frame_bytes, true, config.canvas_w, config.canvas_h);
                }
            }
        });

        // Stage: effects + composite (the whole graph), plus whichever
        // per-frame consumer this path attaches to FINAL_COLOR.
        let mut encoder = device.begin_frame();
        let slot = nvenc
            .as_ref()
            .map(|(interop, _)| f % interop.slot_count())
            .unwrap_or(0);

        // Backpressure before recording into a slot's buffer: NVENC must be done
        // reading it. Timed under Nvenc — it is encoder wait, not render time.
        if let Some((interop, _)) = &mut nvenc {
            let reclaimed = profile.measure(PipelineStage::Nvenc, || interop.reclaim_slot(slot));
            let reclaimed = reclaimed.expect("NVENC reclaim_slot failed mid-benchmark");
            packets_out += reclaimed.len();
            bitstream_bytes += reclaimed.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
        }

        // Open the GPU bracket before the graph records anything, and close it
        // after. Both calls sit OUTSIDE the `measure` closure because that
        // closure borrows `encoder` mutably and the timer needs it too — the
        // timestamps are two encoder commands, so their CPU cost is negligible
        // and their placement in the command stream is what matters.
        gpu_timer.reset();
        gpu_timer.begin(&mut encoder);

        profile.measure(PipelineStage::Composite, || {
            match (&mut nvenc, &readback_buffer) {
                (Some((interop, nv12)), _) => {
                    // The pitch comes from the encoder, never from a second
                    // computation here: the chroma plane sits at pitch*height and
                    // a stride disagreement is a silent hue shift.
                    let pitch = interop.pitch();
                    let nv12_buffer = interop.nv12_buffer_for_slot(slot);
                    graph.execute_with_callback(
                        &mut encoder,
                        device,
                        &frame_state,
                        |enc, ctx| {
                            let in_view = ctx
                                .get(ResourceId::FINAL_COLOR)
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            nv12.record(enc, device, &in_view, nv12_buffer, pitch);
                        },
                    );
                }
                (None, Some(buf)) => {
                    graph.execute_with_callback(&mut encoder, device, &frame_state, |enc, ctx| {
                        if ctx.contains(ResourceId::FINAL_COLOR) {
                            enc.copy_texture_to_buffer(
                                wgpu::ImageCopyTexture {
                                    texture: ctx.get(ResourceId::FINAL_COLOR).texture,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d::ZERO,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::ImageCopyBuffer {
                                    buffer: buf,
                                    layout: wgpu::ImageDataLayout {
                                        offset: 0,
                                        bytes_per_row: Some(readback_stride),
                                        rows_per_image: Some(config.canvas_h),
                                    },
                                },
                                wgpu::Extent3d {
                                    width: config.canvas_w,
                                    height: config.canvas_h,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                    });
                }
                (None, None) => {
                    graph.execute(&mut encoder, device, &frame_state);
                }
            }
        });

        gpu_timer.end(&mut encoder);
        gpu_timer.record_resolve(&mut encoder);

        let submission_id = profile.measure(PipelineStage::GpuSubmit, || device.submit(encoder));

        // Stage: GPU wait / readback.
        profile.measure(PipelineStage::GpuWait, || {
            if let Some(buf) = &readback_buffer {
                device
                    .device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_id));
                let slice = buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = tx.send(res);
                });
                device.device.poll(wgpu::Maintain::Wait);
                let _ = rx.recv();
                let mapped = slice.get_mapped_range();
                // Touch the mapping so the read is not optimised away.
                std::hint::black_box(mapped[0]);
                drop(mapped);
                buf.unmap();
            } else {
                // NVENC reads the NV12 buffer through CUDA, so the conversion
                // pass must have completed before the picture is submitted.
                device
                    .device
                    .poll(wgpu::Maintain::WaitForSubmissionIndex(submission_id));
            }
        });
        if readback_buffer.is_some() {
            download_bytes += readback_size;
        }

        // Read the GPU bracket. The `GpuWait` stage above has already polled this
        // submission to completion, so the resolved timestamps are ready and this
        // adds no stall of its own.
        //
        // Recorded against Composite because that is the stage whose CPU time
        // covers the same commands. The two are NOT alternatives: Composite is
        // how long the CPU took to record the graph, this is how long the GPU
        // took to run it.
        if let Some(ns) = gpu_timer.resolve_last() {
            profile.record_gpu(
                PipelineStage::Composite,
                std::time::Duration::from_nanos(ns as u64),
            );
        }

        // Stage: NVENC. A real `nvEncEncodePicture` against a real session,
        // plus whatever bitstream the driver handed back.
        if let Some((interop, _)) = &mut nvenc {
            let packets = profile.measure(PipelineStage::Nvenc, || interop.encode_frame(pts, slot));
            let packets = packets.expect("NVENC encode_frame failed mid-benchmark");
            packets_out += packets.len();
            bitstream_bytes += packets.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
        }

        session.push_frame(profile);
    }

    // Drain whatever the encoder was still holding. Deliberately outside the
    // per-frame timing: this cost belongs to no single frame.
    if let Some((interop, _)) = &mut nvenc {
        let tail = interop.flush().expect("NVENC flush failed");
        packets_out += tail.len();
        bitstream_bytes += tail.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
    }

    session.update_system_metrics(SystemMetrics {
        // Unmeasured: these need NVML, `cuMemGetInfo`, and an OS RSS query
        // respectively. Left as None so the report says so.
        gpu_utilization: None,
        cpu_utilization: None,
        nvenc_utilization: None,
        vram_used_bytes: None,
        ram_used_bytes: None,
        // Measured/counted.
        allocated_gpu_bytes: Some(allocated_gpu_bytes),
        gpu_upload_bytes: Some(upload_bytes_per_frame * config.frame_count as u64),
        gpu_download_bytes: Some(download_bytes),
        // The NVENC pipeline depth is a real property of the session.
        frame_queue_depth: nvenc.as_ref().map(|(i, _)| i.slot_count()),
        decode_queue_depth: None,
        encode_queue_depth: nvenc.as_ref().map(|(i, _)| i.slot_count()),
    });

    if nvenc.is_some() {
        println!(
            "    NVENC: {} packet(s), {:.2} MB of bitstream from {} frame(s)",
            packets_out,
            bitstream_bytes as f64 / (1024.0 * 1024.0),
            config.frame_count
        );
        if packets_out == 0 {
            // Not a panic — the timings above are still real — but it means the
            // encoder produced nothing, so say it loudly rather than printing a
            // clean-looking table.
            println!(
                "    WARNING: the NVENC session accepted every picture but returned no \
                 bitstream. The NVENC column below is submit cost only."
            );
        }
    }

    Ok(session)
}

const BENCHMARKS: &[BenchmarkConfig] = &[
    BenchmarkConfig {
        name: "1. Simple 1080p60 (1 layer, composite) — GPU render only",
        complexity: WorkloadComplexity::Simple,
        path: ExecutionPath::GpuRender,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 240,
    },
    BenchmarkConfig {
        name: "2. Simple 1080p60 (1 layer, composite) — zero-copy NVENC",
        complexity: WorkloadComplexity::Simple,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 240,
    },
    BenchmarkConfig {
        name: "3. Simple 1080p60 (1 layer, composite) — CPU readback",
        complexity: WorkloadComplexity::Simple,
        path: ExecutionPath::CpuReadback,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 120,
    },
    BenchmarkConfig {
        name: "4. Medium 1080p60 (3 layers + colour correction) — zero-copy NVENC",
        complexity: WorkloadComplexity::Medium,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 180,
    },
    BenchmarkConfig {
        name: "5. Heavy 4K60 (4 layers + LUT + chroma key + tone map) — zero-copy NVENC",
        complexity: WorkloadComplexity::Heavy,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 3840,
        canvas_h: 2160,
        target_fps: 60.0,
        frame_count: 90,
    },
];

fn main() {
    println!("\n========================================================================");
    println!("                     NEXIR FRAME-TIMING BENCHMARKS                      ");
    println!("========================================================================\n");

    let device = Arc::new(
        pollster::block_on(GpuDevice::new_headless()).expect("Headless GPU initialization failed"),
    );
    let info = device.adapter.get_info();
    println!("GPU Adapter : {} ({:?})", info.name, info.backend);
    println!("Driver Info : {}", info.driver_info);
    println!("Binding arrays : {}", device.has_binding_arrays);
    println!(
        "Timestamp queries : {}",
        if device.has_timestamp_queries {
            "available"
        } else {
            "UNAVAILABLE (GPU execution times print n/a)"
        }
    );
    let capability = InteropCapability::probe(&device);
    println!(
        "CUDA interop   : {} (transport={:?})",
        if capability.is_available() { "available" } else { "UNAVAILABLE" },
        capability.transport
    );
    println!(
        "\nEvery figure below is timed or counted. Anything this process does not \
         measure\nprints as n/a rather than as an estimate.\n"
    );

    let mut skipped = Vec::new();

    for bench in BENCHMARKS {
        println!(">>> Running: {}", bench.name);
        println!("    Layers: {}", bench.complexity.layer_count());
        match run_benchmark(&device, bench) {
            Ok(session) => println!("{}", session.generate_report().format_table()),
            Err(Skipped(reason)) => {
                println!("    SKIPPED: {reason}\n");
                skipped.push((bench.name, reason));
            }
        }
    }

    if skipped.is_empty() {
        println!("All benchmarks ran.\n");
    } else {
        println!("{} of {} benchmarks did NOT run:", skipped.len(), BENCHMARKS.len());
        for (name, reason) in &skipped {
            println!("  - {name}\n      {reason}");
        }
        println!();
    }
}
