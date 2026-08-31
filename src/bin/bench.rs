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
    /// How many DISTINCT video sources the layers are drawn from.
    ///
    /// Layers are assigned round-robin, so `1` means every layer shows the same
    /// clip and `layer_count()` means they all show different ones. The engine
    /// uploads once per distinct source, so this is exactly the multiplier on
    /// bytes-per-frame — and it is the difference between an honest 4K row and a
    /// flattering one.
    ///
    /// The Heavy workload is run BOTH ways on purpose. Four layers sharing one
    /// source is a real editing pattern (the same clip on two tracks, one source
    /// feeding several effect chains) and de-duplicating it is a real
    /// optimisation. It is also not what "4 layers of 4K" means to a reader, and
    /// a 4K60 claim measured on it would be measuring a smaller workload. Both
    /// rows are printed; the distinct one is the one the audit's 4K60 target is
    /// judged against.
    distinct_sources: usize,
}

impl BenchmarkConfig {
    /// Distinct sources, clamped to something buildable.
    fn source_count(&self) -> usize {
        self.distinct_sources.clamp(1, self.complexity.layer_count())
    }
}

/// One `YuvUploadNode` in the compiled graph, and which source feeds it.
///
/// The mapping is not the identity once layers share sources: four layers over
/// one source produce ONE uploader, and the loop that pushes pixels iterates
/// uploaders rather than layers. Getting this backwards is how a "de-duplicated"
/// pipeline uploads exactly as much as before while reporting that it did not.
struct Uploader {
    node_idx: usize,
    source_idx: usize,
}

/// The Y/UV texture pair a source's decoded frame lands in.
///
/// Created once per distinct source and handed to every layer that reads it, so
/// two layers sharing a clip share the upload, the staging buffer and the
/// textures — the bytes cross the bus once.
#[derive(Clone, Copy)]
struct SourcePlanes {
    y: ResourceId,
    uv: ResourceId,
}

/// Get (or create) the upload node and planes for one source.
///
/// The first layer to ask for a source builds its `YuvUploadNode`; later layers
/// with the same source get the same `ResourceId`s back and add no upload.
#[allow(clippy::too_many_arguments)]
fn planes_for_source(
    device: &GpuDevice,
    compiler: &mut RenderGraphCompiler,
    id_counter: &mut u32,
    planes: &mut [Option<SourcePlanes>],
    uploaders: &mut Vec<Uploader>,
    source_idx: usize,
    width: u32,
    height: u32,
) -> SourcePlanes {
    if let Some(existing) = planes[source_idx] {
        return existing;
    }
    let y = ResourceId::next(id_counter);
    let uv = ResourceId::next(id_counter);
    let node_idx = compiler.add_node(Box::new(YuvUploadNode::new_with_layout(
        device,
        source_idx as u32,
        width,
        height,
        y,
        uv,
        nexir::timeline::source::FrameLayout::NV12,
    )));
    uploaders.push(Uploader {
        node_idx,
        source_idx,
    });
    let created = SourcePlanes { y, uv };
    planes[source_idx] = Some(created);
    created
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

/// How many frames the benchmark keeps in flight when there is no encoder to set
/// the depth for it.
///
/// Bounded rather than unbounded on purpose: with no limit the CPU can run
/// arbitrarily far ahead of the GPU and the reported FPS stops being "how fast
/// frames complete" and becomes "how fast this loop records commands" — a number
/// that improves when you make the pipeline worse.
const DEFAULT_PIPELINE_DEPTH: usize = 4;

/// How long the untimed warm-up runs before measurement starts.
///
/// **Three frames is not enough, and the difference is not small.** Benchmark 5
/// measured 55.3 FPS when it ran after benchmarks 1-4 and 45.5 FPS when run
/// alone — a 21% swing on identical code. The GPU's own graph execution was
/// unchanged (8.2 ms both ways); what moved was CPU recording time (0.6 -> 4.6 ms)
/// and the transfer span (9.0 -> 13.1 ms), i.e. CPU and GPU clocks ramping from
/// idle. The earlier benchmarks were acting as a warm-up for the later ones, so
/// each row's figure depended on its position in the list.
///
/// A duration rather than a frame count, because that is what clock ramping is
/// measured in: 90 frames of 4K is 1.6 s, so a frame-count warm-up that is
/// adequate at 1080p is a fraction of a second at 4K.
const WARMUP_DURATION: std::time::Duration = std::time::Duration::from_millis(1200);

/// Minimum warm-up frames regardless of duration, so pipeline/bind-group creation
/// is always off the measured path even on an implausibly fast device.
const WARMUP_MIN_FRAMES: usize = 3;

/// How many times each benchmark is measured by default.
///
/// **A single reading of the 4K row is not a result on this hardware.** Measured
/// across repeats it moved between 39 and 56 FPS on identical code, with the GPU's
/// own graph execution steady at ~9 ms — so the variance is in transfer and CPU
/// scheduling, not in the shaders. Quoting one run's number as "the" figure would
/// be picking a sample from a distribution 40% wide.
///
/// Three is the minimum that yields a median rather than a midpoint, and the
/// summary prints the full spread so a wide one cannot hide behind it.
const DEFAULT_REPEATS: usize = 3;

/// One measured run's headline figures.
struct RunReading {
    fps: f64,
    lat_avg: f64,
    lat_p95: f64,
    lat_p99: f64,
}

/// Every reading for one benchmark, plus what workload produced them.
struct BenchSummary {
    name: &'static str,
    layers: usize,
    sources: usize,
    runs: Vec<RunReading>,
}

/// Median of a sample, or 0.0 when empty.
///
/// Median rather than mean because the 4K distribution has a long slow tail — one
/// repeat catching a background task would pull a mean down and the reported figure
/// would then depend on machine noise rather than on the code.
fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

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

/// One frame the GPU is working on but which has not been retired yet.
///
/// The [`FrameProfile`] travels with the frame rather than being pushed to the
/// session at submit time: its NVENC cost and its GPU execution time are only
/// known once the submission has completed, and a profile pushed early would
/// report a frame whose encode had not happened.
struct InFlight {
    pts: i64,
    submission: wgpu::SubmissionIndex,
    slot: usize,
    profile: FrameProfile,
}

/// Wait for one in-flight frame, read what it cost, hand it to the encoder and
/// record it.
///
/// This is where the two genuinely unavoidable synchronisations live:
///
///   * The GPU must have finished writing the NV12 buffer before NVENC reads it
///     through CUDA. That is a real ordering requirement across two APIs that
///     share no timeline, not a benchmark artefact.
///   * A CPU readback is a synchronisation point by definition.
///
/// Everything else about a frame — upload, recording, submission — now happens
/// while earlier frames are still executing.
#[allow(clippy::too_many_arguments)]
fn retire_frame(
    frame: InFlight,
    device: &Arc<GpuDevice>,
    nvenc: &mut Option<(EncodeInterop, Nv12EncodeNode)>,
    readback_buffer: &Option<wgpu::Buffer>,
    readback_size: u64,
    gpu_timers: &mut [nexir::render::gpu_timer::GpuTimer],
    session: &ProfilingSession,
    packets_out: &mut usize,
    bitstream_bytes: &mut u64,
    download_bytes: &mut u64,
    last_retire: &mut Option<std::time::Instant>,
    last_graph_end_tick: &mut Option<u64>,
) {
    let InFlight {
        pts,
        submission,
        slot,
        mut profile,
    } = frame;

    // Stage: GPU wait / readback. Named for what it measures — this thread
    // blocked — and it is CPU time, not GPU execution time. The GPU column
    // reports the latter separately.
    profile.measure(PipelineStage::GpuWait, || {
        if let Some(buf) = readback_buffer {
            device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
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
            // NVENC reads the NV12 buffer through CUDA, so the conversion pass
            // must have completed before the picture is submitted. For the
            // render-only path nothing consumes the texture, but the wait is
            // still what makes this frame's timestamps readable below.
            device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        }
    });
    if readback_buffer.is_some() {
        *download_bytes += readback_size;
    }

    // Read this frame's GPU bracket. Its submission has just been waited on, so
    // the resolve is already in the readback buffer and this adds no stall —
    // which is the whole reason the timers are per-slot and read here rather than
    // immediately after submitting.
    //
    // Raw ticks rather than `resolve_all`, because two spans come out of them and
    // only one is inside this frame's bracket:
    //
    //   ticks[0] ─ graph begins        ticks[1] ─ graph ends
    //   ─────────── Composite (GPU) ────────────
    //   previous frame's ticks[1] ── to ── this frame's ticks[0]
    //   ────────────── GPU transfer ──────────────
    //
    // The second span is the one Phase C could only infer. wgpu's `write_buffer`
    // does not write the destination buffer from the CPU: it stages the bytes and
    // records a `copy_buffer_to_buffer` into its own `pending_writes` encoder,
    // which `pre_submit()` prepends to the NEXT `queue.submit`
    // (wgpu-core-0.19.4 `device/queue.rs:231-242`, `:1443`). Those copies run in a
    // command buffer this process never encodes, so no bracket of ours can contain
    // them — but they execute on the same queue between the two frames' graphs,
    // and the queue's own counter dates them.
    //
    // Both stages are recorded against the GPU column only. Neither is CPU time,
    // and `Composite`'s CPU row is the cost of *recording* the graph.
    if let (Some(ticks), Some(period_ns)) = (
        gpu_timers[slot].resolve_raw_ticks(),
        gpu_timers[slot].period_ns(),
    ) {
        if let [begin, end, ..] = ticks[..] {
            profile.record_gpu(
                PipelineStage::Composite,
                std::time::Duration::from_nanos((end.saturating_sub(begin) as f64 * period_ns) as u64),
            );
            if let Some(prev_end) = *last_graph_end_tick {
                // `saturating_sub` guards the one case that would otherwise print
                // an astronomical figure: a driver that reports ticks out of order
                // across a submission boundary. A zero gap is a believable claim
                // (the copies were free or overlapped); a negative one is not, and
                // an underflowed u64 would read as ~5 million ms.
                let gap = begin.saturating_sub(prev_end);
                profile.record_gpu(
                    PipelineStage::GpuTransfer,
                    std::time::Duration::from_nanos((gap as f64 * period_ns) as u64),
                );
            }
            *last_graph_end_tick = Some(end);
        }
    }

    // Stage: NVENC. A real `nvEncEncodePicture` against a real session, plus
    // whatever bitstream the driver handed back.
    if let Some((interop, _)) = nvenc {
        let packets = profile.measure(PipelineStage::Nvenc, || interop.encode_frame(pts, slot));
        let packets = packets.expect("NVENC encode_frame failed mid-benchmark");
        *packets_out += packets.len();
        *bitstream_bytes += packets.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
    }

    // ── Frame latency ─────────────────────────────────────────────────────────
    // Stamped HERE, at the retire point, because this is where the frame is
    // actually finished: waited on, read back or encoded, done. The interval
    // between consecutive stamps is the rate frames are delivered at, and it is
    // the only quantity a "P95 ≤ 16.67 ms" target can be checked against.
    //
    // It is deliberately not the sum of this frame's CPU stages. Since frames
    // went in flight those two diverged by a factor of three at 4K — 5.0 ms of
    // CPU work per frame on a pipeline delivering one every 18.0 ms — so a
    // percentile over the CPU sum reports a frame rate nothing achieved.
    //
    // The `gpu_lookahead` frames retired in the drain loop retire back-to-back
    // rather than at the loop's pace, so they read near zero — that is why the
    // `Min` column on this row is ~0.00 ms and why the **mean** is the figure to
    // read, cross-checked against `frames / wall time` by the report itself.
    let now = std::time::Instant::now();
    if let Some(prev) = last_retire.replace(now) {
        profile.record_latency(now.duration_since(prev));
    }

    session.push_frame(profile);
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
    let mut final_composite_inputs = Vec::new();

    // ── Source sharing ────────────────────────────────────────────────────────
    // Layers are assigned to sources round-robin, and each source is uploaded
    // exactly once per frame. With `distinct_sources == layer_count()` this is the
    // old one-uploader-per-layer shape; with fewer, the layers that share a source
    // share its upload and its textures.
    let source_count = config.source_count();
    let layer_count = config.complexity.layer_count();
    let mut source_planes: Vec<Option<SourcePlanes>> = vec![None; source_count];
    let mut uploaders: Vec<Uploader> = Vec::with_capacity(source_count);

    // ── Build the render graph according to complexity ────────────────────────

    match config.complexity {
        WorkloadComplexity::Simple => {
            // 1 video clip → YUV upload → YUV to RGB → composite
            let rgba_id = ResourceId::next(&mut id_counter);
            let planes = planes_for_source(
                device,
                &mut compiler,
                &mut id_counter,
                &mut source_planes,
                &mut uploaders,
                0,
                config.canvas_w,
                config.canvas_h,
            );

            compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                device,
                &shaders,
                &compute,
                planes.y,
                planes.uv,
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
            for i in 0..layer_count {
                let rgb_id = ResourceId::next(&mut id_counter);
                let cc_id = ResourceId::next(&mut id_counter);

                let planes = planes_for_source(
                    device,
                    &mut compiler,
                    &mut id_counter,
                    &mut source_planes,
                    &mut uploaders,
                    i % source_count,
                    config.canvas_w,
                    config.canvas_h,
                );

                compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                    device,
                    &shaders,
                    &compute,
                    planes.y,
                    planes.uv,
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

            for i in 0..layer_count {
                let rgb_id = ResourceId::next(&mut id_counter);
                let cc_id = ResourceId::next(&mut id_counter);
                let lut_id = ResourceId::next(&mut id_counter);
                let key_id = ResourceId::next(&mut id_counter);

                let planes = planes_for_source(
                    device,
                    &mut compiler,
                    &mut id_counter,
                    &mut source_planes,
                    &mut uploaders,
                    i % source_count,
                    config.canvas_w,
                    config.canvas_h,
                );

                compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
                    device,
                    &shaders,
                    &compute,
                    planes.y,
                    planes.uv,
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

    // The de-duplication must be real, not merely intended: with layers sharing a
    // source there must be FEWER upload nodes than layers, and the graph must
    // contain exactly one per distinct source. If this ever fires, the round-robin
    // above stopped hitting the memo in `planes_for_source` and the bench is
    // uploading the same bytes several times while claiming it does not.
    assert_eq!(
        uploaders.len(),
        source_count,
        "one upload node per distinct source: {} layers over {} sources produced {} \
         uploaders",
        layer_count,
        source_count,
        uploaders.len()
    );

    let mut graph = compiler
        .compile(config.canvas_w, config.canvas_h)
        .expect("Graph compile failed");

    // Which upload mechanism the nodes are using, read once from the first
    // uploader rather than re-read per frame.
    //
    // `UploadPath::from_env` is per-node, so every node in a run agrees by
    // construction; asserting that here would be asserting that `std::env::var`
    // is deterministic. What matters is that the printed banner names the
    // mechanism the numbers were produced by — a bench run whose upload path is
    // invisible is a bench run whose FPS cannot be compared to another.
    let upload_mechanism = uploaders
        .first()
        .and_then(|up| {
            graph.nodes_mut()[up.node_idx]
                .as_any_mut()
                .and_then(|n| n.downcast_mut::<YuvUploadNode>())
        })
        .map(|u| u.upload_path())
        .unwrap_or(nexir::render::nodes::yuv_upload::UploadPath::Auto);

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
    let nv12_frames: Vec<std::sync::Arc<Vec<u8>>> = (0..PATTERN_FRAMES)
        .map(|i| {
            // Shift by a whole number of chroma samples so the two planes stay in
            // step, and by enough per frame that motion estimation cannot treat
            // successive frames as identical.
            let phase = (i as u32 * 16) % config.canvas_w;
            std::sync::Arc::new(make_nv12(config.canvas_w, config.canvas_h, phase))
        })
        .collect();

    // Which pattern frame each source shows, as an offset into the cycle.
    //
    // Distinct sources must carry distinct PIXELS, not just distinct node
    // indices. Handing every source the same bytes would leave a
    // four-distinct-source run uploading four copies of one image, which is a
    // workload the driver's own caches may treat differently from four real
    // clips — and the whole point of the distinct variant is to be the pessimistic
    // case. Offsetting into the shared cycle costs no extra memory: at 4K a
    // per-source frame set would be ~400 MB of host RAM.
    let source_phase_offset = |source_idx: usize| source_idx * 2;

    // Bytes handed to the queue per frame, ACCUMULATED FROM THE UPLOADS
    // THEMSELVES rather than computed from the frame size.
    //
    // The two are not the same number, and the difference is not rounding: on the
    // re-strided path the node writes padded rows, so 1080p 8-bit luma pads 1920
    // to 2048 and a 3.11 MB frame becomes 3.32 MB on the bus. A computed figure
    // would have understated 1080p traffic by 6.7% and fed that error straight
    // into the GB/s divisor. Counted, this cannot drift from what happened.
    let mut upload_bytes_per_frame: u64 = 0;

    // One composite input per LAYER; several layers may name the same source.
    let mut frame_clips = Vec::new();
    for i in 0..layer_count {
        let source_idx = i % source_count;
        frame_clips.push(ClipRenderEntry {
            source_id: SourceId::new(source_idx as u32),
            texture_slot: source_idx as u32,
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

    // ── Pipeline depth ────────────────────────────────────────────────────────
    // How many frames may be in flight. With an encoder this is the session's own
    // slot count, because that is what bounds it in reality: rendering into a
    // slot NVENC is still reading is the one thing `reclaim_slot` exists to
    // prevent. Without one, a fixed bound — see `DEFAULT_PIPELINE_DEPTH`.
    let pipeline_depth = nvenc
        .as_ref()
        .map(|(interop, _)| interop.slot_count())
        .unwrap_or(DEFAULT_PIPELINE_DEPTH);

    // How many submissions may be outstanding without having been retired.
    //
    // Strictly below `pipeline_depth` so the slot about to be rendered into is
    // never one still sitting in `inflight` — the same reasoning as
    // `src/export/renderer.rs:864-869`, which this loop deliberately mirrors
    // rather than reinventing.
    let gpu_lookahead = pipeline_depth.saturating_sub(2).max(1);

    // ── GPU timing ────────────────────────────────────────────────────────────
    // One timer per in-flight frame, NOT one shared timer.
    //
    // A shared timer would have frame N+1 overwrite the query set before frame
    // N's timestamps were read, and the resulting number would be the difference
    // between two unrelated frames' clocks — plausible-looking and meaningless.
    // Per-slot timers also mean the resolve is read at the retire point, where
    // the submission has already been waited on, so reading it adds no stall of
    // its own. Getting that wrong is how a pipelined benchmark measures no
    // improvement: the profiler serialises what the pipeline parallelised.
    let mut gpu_timers: Vec<nexir::render::gpu_timer::GpuTimer> = (0..pipeline_depth)
        .map(|_| nexir::render::gpu_timer::GpuTimer::new(device, 1))
        .collect();

    // ── Warm-up, untimed ──────────────────────────────────────────────────────
    // Runs until `WARMUP_DURATION` has elapsed (and at least `WARMUP_MIN_FRAMES`
    // frames), doing exactly the work the measured loop does.
    //
    // The frame count is not the point; see `WARMUP_DURATION`. A 3-frame warm-up
    // left the 4K row reading 55.3 FPS in the full suite and 45.5 FPS alone,
    // because benchmarks 1-4 were warming the clocks for it and its number
    // therefore depended on its position in the list.
    let warmup_start = std::time::Instant::now();
    let mut w = 0usize;
    while w < WARMUP_MIN_FRAMES || warmup_start.elapsed() < WARMUP_DURATION {
        for up in &uploaders {
            if let Some(u) = graph.nodes_mut()[up.node_idx]
                .as_any_mut()
                .and_then(|n| n.downcast_mut::<YuvUploadNode>())
            {
                u.upload_frame_shared(
                    std::sync::Arc::clone(
                        &nv12_frames[(w + source_phase_offset(up.source_idx)) % PATTERN_FRAMES],
                    ),
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
        w += 1;
    }
    println!(
        "    Warm-up: {} frame(s) in {:.2} s (untimed)",
        w,
        warmup_start.elapsed().as_secs_f64()
    );

    // ── Measurement loop ──────────────────────────────────────────────────────
    //
    // FRAMES IN FLIGHT (P0.2/P0.5). The loop is split into a submit half and a
    // retire half so the CPU records frame N+1 while the GPU executes frame N.
    // Before this, every frame ended in `poll(WaitForSubmissionIndex)`, which put
    // the CPU's upload work and the GPU's render work strictly end-to-end: the
    // 4K row spent 12.66 ms — 71% of the frame — blocked, of which only 8.4 ms
    // was the GPU actually busy.
    //
    // Structure mirrors `src/export/renderer.rs:856-988`, which already does this
    // correctly for real exports, rather than being a second design:
    //
    //   submit:  reclaim slot -> upload -> record -> submit -> push to inflight
    //   retire:  while inflight > lookahead: wait oldest -> read GPU time ->
    //            encode -> push profile
    //
    // The FrameProfile travels WITH the frame in `inflight`. Pushing it to the
    // session at submit time would report a frame whose encode had not happened
    // yet and whose GPU time was unknown.
    let session = ProfilingSession::new(config.target_fps);
    let job = nvenc.as_ref().map(|_| nvenc_job(config));
    let mut bitstream_bytes: u64 = 0;
    let mut packets_out: usize = 0;
    let mut download_bytes: u64 = 0;

    let mut inflight: std::collections::VecDeque<InFlight> =
        std::collections::VecDeque::with_capacity(gpu_lookahead + 1);

    // Last retirement instant, for the frame-latency interval. Owned by the loop
    // rather than by the session so that the drain below continues the same
    // series.
    //
    // SEEDED at the loop's start, not left `None`, so frame 0's interval covers
    // the pipeline fill — the submit work for the first `gpu_lookahead` frames
    // that happens before anything retires. Leaving it unseeded drops that time
    // from the series entirely, and the mean then reads below `frames / wall
    // time` for no reason a reader could see: the 4K row reported an 18.25 ms
    // mean latency on a run delivering a frame every 21.1 ms. The two must agree,
    // because they are measuring the same seconds.
    let mut last_retire: Option<std::time::Instant> = Some(std::time::Instant::now());

    // Closing GPU tick of the previously retired frame's graph, for the
    // `GpuTransfer` span. Valid to carry across frames because every timer draws
    // its ticks from the same queue counter, and frames retire in submission
    // order — so consecutive retirements are consecutive submissions and the gap
    // between them is real queue time, not a reordering artefact.
    let mut last_graph_end_tick: Option<u64> = None;

    for f in 0..config.frame_count {
        let pts = job.as_ref().map(|j| j.frame_pts(f)).unwrap_or(f as i64);
        let mut profile = FrameProfile::new(f, pts);
        let slot = f % pipeline_depth;

        // ── Backpressure ──────────────────────────────────────────────────────
        // NVENC must be finished reading this slot's NV12 buffer before anything
        // records into it again. A no-op until the pipeline is full; after that it
        // blocks on the oldest outstanding picture, which is the correct thing to
        // wait for. Timed under Nvenc: it is encoder wait, not render time.
        if let Some((interop, _)) = &mut nvenc {
            let reclaimed = profile.measure(PipelineStage::Nvenc, || interop.reclaim_slot(slot));
            let reclaimed = reclaimed.expect("NVENC reclaim_slot failed mid-benchmark");
            packets_out += reclaimed.len();
            bitstream_bytes += reclaimed.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
        }

        // Stage: upload. Named Upload rather than Decode because no decoder ran —
        // the pattern frames were all built before the loop.
        //
        // Recorded as three stages: `Upload` is the total, and prepare/submit
        // decompose it. They are breakdown stages, so `total_time` counts only
        // the parent — see `PipelineStage::is_breakdown`.
        let upload_start = std::time::Instant::now();
        let mut upload_cost = nexir::render::nodes::yuv_upload::UploadCost::zero();
        {
            // Iterates UPLOADERS, one per distinct source — not layers. When four
            // layers share a source there is one upload node, so the bytes cross
            // the bus once no matter how many layers read them.
            //
            // Indexed rather than iterated because the body needs `graph`
            // mutably while `uploaders` is borrowed.
            #[allow(clippy::needless_range_loop)]
            for up_i in 0..uploaders.len() {
                let (node_idx, source_idx) =
                    (uploaders[up_i].node_idx, uploaders[up_i].source_idx);
                let frame_bytes =
                    &nv12_frames[(f + source_phase_offset(source_idx)) % PATTERN_FRAMES];
                if let Some(u) = graph.nodes_mut()[node_idx]
                    .as_any_mut()
                    .and_then(|n| n.downcast_mut::<YuvUploadNode>())
                {
                    // `_shared` rather than `upload_frame` so the node can take the
                    // `write_texture` path when configured to: that path needs the
                    // bytes to survive until `record`, and an `Arc` is how they do
                    // that without a copy. On the default staging path this is
                    // exactly `upload_frame` — pinned by
                    // `tests::yuv_upload::shared_upload_on_the_staging_path_is_the_ordinary_upload`.
                    upload_cost.add(u.upload_frame_shared(
                        std::sync::Arc::clone(frame_bytes),
                        true,
                        config.canvas_w,
                        config.canvas_h,
                    ));
                }
            }
        }
        profile.record_stage(PipelineStage::Upload, upload_start.elapsed());
        profile.record_stage(PipelineStage::UploadPrepare, upload_cost.prepare);
        profile.record_stage(PipelineStage::UploadSubmit, upload_cost.submit);
        if f == 0 {
            // Every frame uploads the same amount, so record the counted figure
            // once and use it as the per-frame bandwidth divisor.
            upload_bytes_per_frame = upload_cost.bytes;
            // Printed once per benchmark, because which path the upload took is
            // the difference between a memcpy and a no-op and is not otherwise
            // visible in the table — and because the layer:source ratio is the
            // single most important caveat on any FPS figure below it.
            println!(
                "    Upload: {} ({:.1} MB/frame handed to the queue; {} layer(s) over \
                 {} distinct source(s))",
                match upload_mechanism {
                    nexir::render::nodes::yuv_upload::UploadPath::WriteTexture =>
                        "write_texture direct to texture (wgpu re-strides internally)"
                            .to_string(),
                    nexir::render::nodes::yuv_upload::UploadPath::StagingBuffer =>
                        format!(
                            "write_buffer staging, {}",
                            if upload_cost.contiguous {
                                "contiguous — source rows already at the staging stride, \
                                 no repack"
                            } else {
                                "re-strided — rows repacked into reused scratch"
                            }
                        ),
                    // Unreachable in practice: `YuvUploadNode::new_with_layout`
                    // resolves `Auto` from its own dimensions, so a live node never
                    // reports it. Printed rather than `unreachable!()` because a
                    // benchmark should not panic over a label.
                    nexir::render::nodes::yuv_upload::UploadPath::Auto =>
                        "UNRESOLVED Auto (a node did not resolve its path — report this)"
                            .to_string(),
                },
                upload_cost.bytes as f64 / (1024.0 * 1024.0),
                layer_count,
                source_count,
            );
        } else {
            // Constant per frame by construction; if it ever is not, the GB/s
            // divisor computed from frame 0 is wrong for every other frame.
            debug_assert_eq!(
                upload_cost.bytes, upload_bytes_per_frame,
                "upload bytes changed mid-run"
            );
        }

        // Stage: effects + composite (the whole graph), plus whichever
        // per-frame consumer this path attaches to FINAL_COLOR.
        let mut encoder = device.begin_frame();

        // Open the GPU bracket before the graph records anything, and close it
        // after. Both calls sit OUTSIDE the `measure` closure because that
        // closure borrows `encoder` mutably and the timer needs it too — the
        // timestamps are two encoder commands, so their CPU cost is negligible
        // and their placement in the command stream is what matters.
        let gpu_timer = &mut gpu_timers[slot];
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

        let submission = profile.measure(PipelineStage::GpuSubmit, || device.submit(encoder));

        // Non-blocking: lets the driver make progress without stalling this
        // thread. `Maintain::Poll` is explicitly NOT a wait.
        device.device.poll(wgpu::Maintain::Poll);

        inflight.push_back(InFlight {
            pts,
            submission,
            slot,
            profile,
        });

        // ── Retire ────────────────────────────────────────────────────────────
        // Only once more than `gpu_lookahead` frames are outstanding, so the CPU
        // stays ahead of the GPU rather than being paced by it.
        while inflight.len() > gpu_lookahead {
            let frame = inflight
                .pop_front()
                .expect("inflight is non-empty: len > gpu_lookahead >= 1");
            retire_frame(
                frame,
                device,
                &mut nvenc,
                &readback_buffer,
                readback_size,
                &mut gpu_timers,
                &session,
                &mut packets_out,
                &mut bitstream_bytes,
                &mut download_bytes,
                &mut last_retire,
                &mut last_graph_end_tick,
            );
        }
    }

    // ── Drain ─────────────────────────────────────────────────────────────────
    // Every frame still in flight, retired the same way. Skipping this would drop
    // the last `gpu_lookahead` frames from both the report and the encoder.
    while let Some(frame) = inflight.pop_front() {
        retire_frame(
            frame,
            device,
            &mut nvenc,
            &readback_buffer,
            readback_size,
            &mut gpu_timers,
            &session,
            &mut packets_out,
            &mut bitstream_bytes,
            &mut download_bytes,
            &mut last_retire,
            &mut last_graph_end_tick,
        );
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
        distinct_sources: 1,
    },
    BenchmarkConfig {
        name: "2. Simple 1080p60 (1 layer, composite) — zero-copy NVENC",
        complexity: WorkloadComplexity::Simple,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 240,
        distinct_sources: 1,
    },
    BenchmarkConfig {
        name: "3. Simple 1080p60 (1 layer, composite) — CPU readback",
        complexity: WorkloadComplexity::Simple,
        path: ExecutionPath::CpuReadback,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 120,
        distinct_sources: 1,
    },
    BenchmarkConfig {
        name: "4. Medium 1080p60 (3 layers, 3 sources + colour correction) — zero-copy NVENC",
        complexity: WorkloadComplexity::Medium,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 1920,
        canvas_h: 1080,
        target_fps: 60.0,
        frame_count: 180,
        // Three distinct sources: this is the row the 1080p multi-layer target is
        // judged against, so it must not be the cheaper shared-source case.
        distinct_sources: 3,
    },
    BenchmarkConfig {
        name: "5. Heavy 4K60 (4 layers, 4 DISTINCT sources + LUT + chroma key + tone map) \
               — zero-copy NVENC",
        complexity: WorkloadComplexity::Heavy,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 3840,
        canvas_h: 2160,
        target_fps: 60.0,
        frame_count: 90,
        // FOUR distinct sources: 47.5 MB/frame across the bus.
        //
        // **This is the row the audit's 4K60 target is judged against.** A real
        // 4-layer 4K timeline has four different clips, so this is what "4 layers
        // of 4K" means. Benchmark 6 runs the same graph with all four layers
        // sharing one source — a real editing pattern, a real optimisation, and a
        // materially smaller workload. Reporting only that one would be claiming
        // 4K60 on a quarter of the bytes.
        distinct_sources: 4,
    },
    BenchmarkConfig {
        name: "6. Heavy 4K60 (4 layers SHARING 1 source + LUT + chroma key + tone map) \
               — zero-copy NVENC",
        complexity: WorkloadComplexity::Heavy,
        path: ExecutionPath::GpuNvencZeroCopy,
        canvas_w: 3840,
        canvas_h: 2160,
        target_fps: 60.0,
        frame_count: 90,
        // ONE source feeding all four layers: 11.9 MB/frame instead of 47.5.
        //
        // The same clip on two tracks, or one source feeding several effect
        // chains, is ordinary in real projects, and uploading it once is the
        // engine doing the right thing. It is NOT the 4K60 target's workload —
        // benchmark 5 is. Both are printed so the difference is visible rather
        // than chosen.
        distinct_sources: 1,
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
    // Every benchmark's repeated readings, for the summary table.
    //
    // A Vec of runs per benchmark rather than one figure each, because one reading
    // of the 4K row is a sample and not a result — see `DEFAULT_REPEATS`.
    let mut summary: Vec<BenchSummary> = Vec::new();

    // Optional filter: `bench.exe 5` or `bench.exe 5 6` runs only those.
    //
    // Exists because benchmarks share one process and one GPU, so a row can be
    // affected by what ran before it — thermals, driver state, the texture pool's
    // buckets. Being able to run one in isolation is how that gets checked rather
    // than assumed.
    let only: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse::<usize>().ok())
        .collect();

    // `NEXIR_BENCH_REPEATS=1` for a quick smoke run; the default is what any
    // reported figure must come from.
    let repeats: usize = std::env::var("NEXIR_BENCH_REPEATS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(DEFAULT_REPEATS);
    if repeats != DEFAULT_REPEATS {
        println!(
            "Repeats: {repeats} (overridden from {DEFAULT_REPEATS}). A single run of the \
             4K row\nspans ~40% between repeats, so treat any figure from repeats=1 as \
             a sample.\n"
        );
    } else {
        println!("Repeats: {repeats} per benchmark; the summary reports the median and spread.\n");
    }

    for (i, bench) in BENCHMARKS.iter().enumerate() {
        let number = i + 1;
        if !only.is_empty() && !only.contains(&number) {
            continue;
        }
        println!(">>> Running: {}", bench.name);
        println!(
            "    Layers: {} over {} distinct source(s)",
            bench.complexity.layer_count(),
            bench.source_count()
        );

        let mut runs: Vec<RunReading> = Vec::with_capacity(repeats);
        let mut skip_reason: Option<String> = None;
        for r in 0..repeats {
            if repeats > 1 {
                println!("    --- repeat {}/{} ---", r + 1, repeats);
            }
            match run_benchmark(&device, bench) {
                Ok(session) => {
                    let report = session.generate_report();
                    println!("{}", report.format_table());
                    let lat = report.frame_latency_stats.unwrap_or_default();
                    runs.push(RunReading {
                        fps: report.average_fps,
                        lat_avg: lat.avg_ms,
                        lat_p95: lat.p95_ms,
                        lat_p99: lat.p99_ms,
                    });
                }
                Err(Skipped(reason)) => {
                    println!("    SKIPPED: {reason}\n");
                    skip_reason = Some(reason);
                    break;
                }
            }
        }

        if let Some(reason) = skip_reason {
            skipped.push((bench.name, reason));
        } else if !runs.is_empty() {
            summary.push(BenchSummary {
                name: bench.name,
                layers: bench.complexity.layer_count(),
                sources: bench.source_count(),
                runs,
            });
        }
    }

    if !summary.is_empty() {
        // Throughput and the latency percentiles, side by side.
        //
        // Printed because the per-benchmark tables are long enough that the two 4K
        // rows end up screens apart, and the whole reason both exist is to be
        // compared. `layers:sources` is in the table because an FPS figure without
        // it is not comparable to anything.
        //
        // Medians with an explicit spread, not means: the 4K distribution is wide
        // enough that a mean would be dragged around by whichever repeat happened
        // to hit a slow patch, and a figure printed without its spread invites
        // being quoted as if it were repeatable.
        println!("========================================================================");
        println!("SUMMARY — median of {repeats} run(s), with spread");
        println!("  P95/P99 are frame LATENCY (wall-clock spacing between frames), which is");
        println!("  the quantity a 60 FPS target is stated in. 60 FPS = 16.67 ms.");
        println!("  FPS spread is (max-min)/median: anything above ~10% means one run's");
        println!("  number is not a result on its own.");
        println!("------------------------------------------------------------------------");
        println!(
            "{:<40} {:>5} {:>8} {:>8} {:>9} {:>9} {:>9}",
            "Benchmark", "L:S", "FPS", "spread", "Lat avg", "Lat P95", "Lat P99"
        );
        for s in &summary {
            let short: String = s.name.chars().take(38).collect();
            let fps = median(s.runs.iter().map(|r| r.fps));
            let spread = if fps > 0.0 {
                let lo = s.runs.iter().map(|r| r.fps).fold(f64::MAX, f64::min);
                let hi = s.runs.iter().map(|r| r.fps).fold(f64::MIN, f64::max);
                (hi - lo) / fps * 100.0
            } else {
                0.0
            };
            println!(
                "{:<40} {:>5} {:>8.1} {:>7.0}% {:>6.2} ms {:>6.2} ms {:>6.2} ms",
                short,
                format!("{}:{}", s.layers, s.sources),
                fps,
                spread,
                median(s.runs.iter().map(|r| r.lat_avg)),
                median(s.runs.iter().map(|r| r.lat_p95)),
                median(s.runs.iter().map(|r| r.lat_p99)),
            );
        }
        println!("========================================================================\n");
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
