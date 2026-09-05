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

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nexir::bench_media::{self, MediaClass};
use nexir::colour::lut_parser::Lut3D;
use nexir::io::decoder::Decoder;
use nexir::io::demuxer::Demuxer;
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
use nexir::render::nodes::fused_grade::{FusedGradeNode, FusedGradeParams};
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
    ColorInfo, ColorPrimaries, ColorRange, DecodedFrameMeta, FrameLayout, MatrixCoefficients,
    TransferFunction,
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

/// Which decode path one pass drives. Two arms of the same loop.
///
/// Declared up here rather than beside `--interop`'s machinery because
/// [`WorkloadSource`] carries one too: benchmarks 7 and 8 are the same pipelined
/// loop differing only in this, which is what makes their comparison a measurement
/// rather than a git checkout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodePath {
    /// `InteropDecodeTargets::with_context` — NVDEC writes into the graph's textures.
    Interop,
    /// `InteropDecodeTargets::disabled` — the CPU upload path, unchanged, as the
    /// control.
    Cpu,
}

impl DecodePath {
    fn label(self) -> &'static str {
        match self {
            Self::Interop => "interop",
            Self::Cpu => "cpu",
        }
    }
}

/// Where a benchmark's pixels come from.
///
/// **The six synthetic rows and the real-media rows are not interchangeable, and
/// this field is what keeps them from being confused for each other.**
/// `make_nv12`'s bars come out of host memory with no decoder in the process, so no
/// run of them can touch the interop decode path however the host is configured;
/// the real-media rows drive `FrameScheduler::schedule_frame` through an `IoLayer`
/// and therefore measure a decode as well as a render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkloadSource {
    /// `make_nv12` bars, uploaded from host memory. Benchmarks 1-6.
    Synthetic,
    /// Real coded 4K60 files through `IoLayer`, one per distinct source, on the
    /// given decode path. Benchmarks 7-8.
    RealMedia4K60(DecodePath),
}

struct BenchmarkConfig {
    name: &'static str,
    complexity: WorkloadComplexity,
    path: ExecutionPath,
    /// Where the pixels come from — see [`WorkloadSource`].
    source: WorkloadSource,
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

/// An identity 3D LUT of `n` points per axis.
///
/// Shared by the Heavy synthetic graph and benchmark 7's real-media graph so the two
/// carry the SAME effect chain: benchmark 7's whole purpose is to be benchmark 5's
/// graph fed from a different decode path, and a LUT that differed between them
/// would make every comparison a mix of two changes.
fn identity_lut(n: u32) -> Lut3D {
    let mut data = Vec::with_capacity((n * n * n) as usize);
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                data.push([
                    r as f32 / (n - 1) as f32,
                    g as f32 / (n - 1) as f32,
                    b as f32 / (n - 1) as f32,
                ]);
            }
        }
    }
    Lut3D {
        size: n,
        data,
        domain_min: [0.0, 0.0, 0.0],
        domain_max: [1.0, 1.0, 1.0],
    }
}

/// Whether the graph fuses colour correction + LUT + chroma key into one pass —
/// **P2.3, and the switch is what makes it a measurement rather than a git checkout.**
///
/// One [`FusedGradeNode`] where the Heavy chain otherwise emits `ColorCorrection` →
/// `Lut3D` → `ChromaKey`. `NEXIR_FUSE_GRADE=0` takes the unfused arm, from the same
/// binary on the same fixtures — one argument different, the discipline
/// `NEXIR_UPLOAD_PATH` and `--interop`'s two arms already follow.
///
/// **ON by default, because the win is measured on BOTH content sets and neither is
/// the compressible one.** Gotcha 27's rule is that a bandwidth saving must be checked
/// against real and low-entropy content, since the latter makes the graph ~20% cheaper
/// with no code change and would flatter exactly this kind of optimisation. Medians of
/// 3, `target/p23_unfused_3x.txt` vs `target/p23_fused_3x.txt` (repo fixtures) and
/// `target/p23_bars_*_3x.txt` (the `nexir_media_bars` control):
///
/// | row | graph TOTAL unfused → fused | FPS |
/// |---|---|---|
/// | 5 — synthetic 4K60 | 6.783 → 4.814 ms (−29%) | 57.5 → 67.1 |
/// | 7 — real media, interop | 7.387 → 4.682 ms (−37%) | 27.8 → 28.6 |
/// | 8 — real media, CPU upload | same graph | 18.9 → 21.9 |
/// | 7 — bars control | 5.881 → 3.867 ms (−34%) | 45.0 → 47.6 |
///
/// The three passes' 0.861 ms/pass become one at 0.367, and `peak bucket` falls 16/32
/// → 8/32 with 0 evicted — the re-read gotcha 14 requires after a graph change, and a
/// reduction rather than a risk. **It does NOT close the 4K60 gate and is not claimed
/// to**: benchmark 7 still prints `ALTERNATING` (12.9 / 52.4 ms), which fails the target
/// on its own, and Task I's floor puts the four-source decode at 32.0 ms against a
/// 16.67 ms frame.
fn fuse_grade() -> bool {
    std::env::var("NEXIR_FUSE_GRADE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true)
}

/// The Heavy chain's per-layer grade, built either fused or as three nodes.
///
/// **One function so the two graph builders cannot drift.** Benchmark 5 (synthetic) and
/// benchmark 7/8 (real media) exist to be compared with each other, and their whole
/// claim is that the graph is identical and only the decode path differs — so a fusion
/// applied to one and not the other would silently make every 5-vs-7 comparison a mix
/// of two changes. Both call this.
///
/// Returns the `ResourceId` the composite should read. The unfused arm allocates the
/// two intermediates it needs from `id_counter`; the fused arm allocates ONE, which is
/// the saving (`tests::fused_grade::fusing_removes_two_intermediates_per_layer` pins
/// that it is two canvas-sized textures per layer, i.e. what `peak bucket` reports).
///
/// **The id counter therefore advances by a different amount on the two arms**, which is
/// fine for both callers — neither derives a later id arithmetically — but it is why
/// `FrameScheduler::interop_id_counter_start`'s reservation must stay BELOW the
/// counter's start rather than being computed as an offset into it.
#[allow(clippy::too_many_arguments)]
fn add_grade_chain(
    device: &GpuDevice,
    shaders: &ShaderRegistry,
    compute: &ComputePipelineCache,
    compiler: &mut RenderGraphCompiler,
    id_counter: &mut u32,
    lut: &Lut3D,
    in_rgba: ResourceId,
    width: u32,
    height: u32,
) -> ResourceId {
    let cc = ColorCorrectionParams::identity(width, height);
    let key = nexir::render::nodes::chroma_key::ChromaKeyParams::green_screen(width, height);

    if fuse_grade() {
        let out = ResourceId::next(id_counter);
        compiler.add_node(Box::new(FusedGradeNode::new(
            device,
            shaders,
            compute,
            lut,
            in_rgba,
            out,
            // The same three parameter structs the unfused arm passes, through the one
            // constructor that translates them — so the two arms cannot differ in what
            // grade they apply, only in how many passes apply it.
            FusedGradeParams::from_parts(cc, 1.0, key),
        )));
        return out;
    }

    let cc_id = ResourceId::next(id_counter);
    let lut_id = ResourceId::next(id_counter);
    let key_id = ResourceId::next(id_counter);

    compiler.add_node(Box::new(ColorCorrectionNode::new(
        device, shaders, compute, in_rgba, cc_id, cc,
    )));

    // `LutNode::new` only sees the cube, so the frame size must be set explicitly —
    // gotcha 10. Omitting it asks wgpu for a zero-sized texture and fails with
    // `Dimension X is zero`, naming neither the node nor the missing call.
    let mut lut_node = LutNode::new(device, shaders, compute, lut, cc_id, lut_id, 1.0);
    lut_node.set_size(width, height);
    compiler.add_node(Box::new(lut_node));

    compiler.add_node(Box::new(ChromaKeyNode::new(
        device, shaders, compute, lut_id, key_id, key,
    )));
    key_id
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

/// The ONE CUDA primary context for this process, retained once and never
/// released.
///
/// **One owner, for the reason `src/tests/mod.rs` records in the same words.**
/// `CudaContext::new` calls `cuDevicePrimaryCtxRetain` and `Drop` calls the
/// matching release, and every `CudaContext` in a process wraps the SAME
/// `CUcontext` (AGENTS.md gotcha 4) — so a context per benchmark, per class, per
/// repeat is a stream of retain/release pairs on one resource, and the symptom of
/// getting it wrong is not a failure where the second owner was added: in the test
/// binary it was three unrelated NVENC tests reporting *"the export engine selected
/// the FFmpeg backend"*, because `EncodeInterop::open` was refused against a
/// context a second owner had disturbed.
///
/// This binary now has four callers that need one (the NVENC benchmarks, the media
/// profile's encode pass, the export profile, and the interop profile below), and
/// the interop profile is the one that makes the sharing load-bearing: its
/// `DecodeInteropTarget`s hold CUDA imports for the whole pass, so a release from
/// somewhere else mid-run would tear down memory NVDEC is still writing into.
///
/// `None` when the host has no interop capability, which every caller must turn
/// into a printed skip rather than a zero row.
fn shared_cuda_ctx(capability: &InteropCapability) -> Option<Arc<CudaContext>> {
    static CUDA: std::sync::OnceLock<Option<Arc<CudaContext>>> = std::sync::OnceLock::new();
    CUDA.get_or_init(|| {
        if !capability.is_available() {
            return None;
        }
        match CudaContext::new(capability) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                println!("    CudaContext::new failed: {e:?}");
                None
            }
        }
    })
    .clone()
}

/// One measured run's headline figures.
struct RunReading {
    fps: f64,
    /// Mean frame interval over the WHOLE run, including the pipeline fill — the
    /// figure that must agree with `fps`, since both cover the same seconds.
    lat_avg: f64,
    /// Percentiles over the STEADY-STATE intervals, i.e. after the pipeline has
    /// filled.
    ///
    /// Read off `steady_latency_stats` rather than the whole-run series because a
    /// 90-frame run has ~90 intervals, so the P99 index lands on the largest
    /// sample — and at 4K the largest sample is the 67-91 ms pipeline fill. The
    /// 54.6 ms "latency P99" this plan opened with was that: the first frame
    /// waiting for a cold pipeline, reported as a frame arriving late.
    lat_p95: f64,
    lat_p99: f64,
    /// The pipeline-fill interval itself, printed beside the percentiles so it is
    /// separated rather than hidden.
    fill: f64,
}

/// Every reading for one benchmark, plus what workload produced them.
struct BenchSummary {
    name: &'static str,
    layers: usize,
    sources: usize,
    runs: Vec<RunReading>,
    /// What the last repeat's real-media workload can attest to, or `None` for the
    /// synthetic sweep.
    ///
    /// Carried up to the summary rather than only printed per repeat because the
    /// acceptance criterion for benchmarks 7 and 8 is read off it: a row headed
    /// "4 DISTINCT files — INTEROP decode" is only that row if four targets were
    /// allocated and every measured frame bound four of them (G2f.4).
    media: Option<RealMediaProvenance>,
}

/// One measured run: the profiling session, plus what the run's own workload can
/// attest to.
///
/// The second half exists because a real-media row's FPS figure means nothing on
/// its own: `live targets 0` under an `INTEROP decode` heading is the CPU path with
/// a misleading label, and nothing in a `ProfileReport` would say so.
struct BenchRun {
    session: ProfilingSession,
    /// `None` for the synthetic sweep — no decoder ran, so there is nothing to
    /// attest.
    media: Option<RealMediaProvenance>,
}

/// What a real-media benchmark row counted about its own workload.
///
/// Every field is counted or timed by the run itself (gotcha 9). The pairing rules
/// are the same ones `--interop` reports: counted upload bytes are only meaningful
/// beside `live_targets`, and `live_targets` is only meaningful beside how many of
/// the frame's clips actually bound one.
#[derive(Clone)]
struct RealMediaProvenance {
    decode_path: DecodePath,
    /// Frames the measured loop actually rendered — not what was requested. The two
    /// differ when the scheduler runs dry, and dividing wall time by the request
    /// would report a rate for frames that never rendered.
    frames: usize,
    /// DISTINCT sources the frame was composited from.
    sources: usize,
    /// Clips the leanest measured frame carried, and how many of them imported.
    clips: usize,
    interop_clips: usize,
    /// Y/UV texture pairs the registry allocated. `0` on the CPU arm by
    /// construction; `0` on the interop arm means every source fell back.
    live_targets: usize,
    /// VRAM those pairs occupy, counted from each target's own dimensions and a
    /// LOWER BOUND — never the NVML row (G2f.2).
    target_bytes: u64,
    /// Bytes `UploadCost` counted going into `YuvUploadNode`, per frame.
    upload_bytes_per_frame: f64,
    /// The registry's counters, `None` on the CPU arm.
    interop_stats: Option<nexir::io::interop_decode::InteropDecodeStats>,
    /// The same counters split by source — the G2f.3 reading.
    per_source: Vec<(u32, nexir::io::interop_decode::InteropDecodeStats)>,
    /// Sources that fell back, with the reason. Printed, never hidden.
    rejections: Vec<(u32, &'static str)>,
    /// The pool's counters for the graph this row compiled — the `peak bucket`
    /// reading gotcha 14 says to take rather than assume, on a graph shape neither
    /// the synthetic sweep nor the single-source `--interop` rows compile.
    pool: nexir::render::resource::PoolStats,
    /// How many times the graph had to be recompiled. >1 means a source changed
    /// shape or fell back mid-run.
    compiles: usize,
    /// One sampled picture from AFTER the measured loop, so the two real-media rows
    /// can be compared pixel-wise.
    ///
    /// Read outside the timed span: it is a full 4K `copy_texture_to_buffer` plus a
    /// map, and the correctness check must not pay for itself out of the number it
    /// validates.
    sample: Option<(usize, FrameSample)>,
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
    nvenc_job_for(
        config.canvas_w,
        config.canvas_h,
        config.target_fps,
        config.frame_count,
    )
}

/// The same job, from bare dimensions.
///
/// Split out because the real-media profile (`--media`) needs one too and has no
/// [`BenchmarkConfig`] — its geometry comes from the fixture. One constructor so
/// the two profiles cannot drift into opening sessions with different settings and
/// then having their NVENC figures compared.
fn nvenc_job_for(width: u32, height: u32, target_fps: f64, frame_count: usize) -> ExportJob {
    let tb = Rational::TIMEBASE_90K;
    let fps = Rational::new(target_fps.round() as i64, 1);
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
        pts_out: frame_dur * frame_count as i64,
        width,
        height,
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

/// Print what the graph's transient texture pool did over a run.
///
/// Printed because a pool that evicts is allocating a canvas-sized texture per
/// eviction per frame — 66 MB at 4K — inside the frame's own recording, and
/// nothing else in a report would show it. Counted by the pool itself, not
/// inferred from timings.
///
/// Shared by both profiles: the real-media path (`--media`) compiles a DIFFERENT
/// graph shape from the synthetic benchmarks, so it is exactly the caller whose
/// `peak bucket` reading might disagree with `POOL_BUCKET_CAPACITY` (AGENTS.md
/// gotcha 14). A second copy of this print that drifted from the first is how one
/// profile keeps warning and the other stops.
fn print_pool_stats(pool: nexir::render::resource::PoolStats) {
    println!(
        "    Texture pool: {} hit / {} miss ({}), {} evicted, {} bucket(s), {} pooled, \
         peak bucket {}/{}",
        pool.hits,
        pool.misses,
        pool.miss_rate()
            .map(|r| format!("{:.1}% miss", r * 100.0))
            .unwrap_or_else(|| "n/a".into()),
        pool.evicted,
        pool.buckets,
        pool.pooled,
        // The measured high-water mark of one bucket, against the cap. This is the
        // pair that decides whether the cap is right — a peak equal to the cap
        // means the real peak is unknown and at least this, which is what
        // `evicted > 0` then reports.
        pool.peak_bucket,
        nexir::render::resource::POOL_BUCKET_CAPACITY,
    );
    if pool.evicted > 0 {
        // Loud, because it is a per-frame allocation with no other symptom. The
        // measured one alternated the 4K frame interval 17.6/25.4 ms.
        println!(
            "    WARNING: the pool evicted {} texture(s) — its per-key cap is below \
             this graph's\n    simultaneous peak, so those are re-allocated every \
             frame (66 MB each at 4K).",
            pool.evicted
        );
    }
}

/// **P2.2 — what per-resource texture lifetimes could save on THIS graph.**
///
/// Printed beside the pool stats because the two answer the same question from
/// opposite ends: `peak bucket N/CAP` is what the frame holds, and this is what it
/// would have to hold if non-overlapping resources shared textures.
/// [`CompiledGraph::lifetime_bounds`] computes both off the compiled graph's own
/// declarations, so the figure is exact for the shape rather than an estimate — and if
/// `save` is 0 there is nothing for P2.2 to win on this row and the task can be closed
/// on evidence instead of attempted.
///
/// **It is a VRAM figure, not a frame time, and the line says so.** The pool already
/// reuses textures across frames (0.5-0.8% miss rate, 0 evicted), so aliasing within a
/// frame removes residency rather than allocations: nothing in the frame's recording
/// gets shorter. Quoting it as a speed-up would be the mistake gotcha 9 exists to
/// prevent.
fn print_lifetime_bounds(graph: &nexir::render::graph::CompiledGraph, canvas_w: u32, canvas_h: u32) {
    let (held, ideal) = graph.lifetime_bounds();
    let save = held.saturating_sub(ideal);
    // One canvas-sized RGBA16Float texture, which is what the Heavy graph's buckets
    // hold. Counted from the geometry rather than hardcoded, and labelled a lower bound
    // for the same reason `allocated_gpu_bytes` is: it counts the resources the graph
    // declares and knows nothing about driver-side padding.
    let per_texture = canvas_w as u64 * canvas_h as u64 * 8;
    println!(
        "    Lifetimes: holds {held} texture(s) simultaneously; {ideal} would suffice if \
         non-overlapping\n    resources shared one (P2.2's ceiling) — {save} fewer, \
         {:.1} MB of 4K residency (lower bound).",
        save as f64 * per_texture as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    That is VRAM, NOT frame time: the pool already reuses across frames, so \
         aliasing\n    inside a frame changes residency and not the recording."
    );
}

/// Open the NVENC session a config asks for, BEFORE any timing starts.
///
/// Shared by the synthetic sweep and the real-media rows, for the same reason
/// [`nvenc_job_for`] is shared: two copies of this would drift into opening
/// sessions with different settings, and their NVENC columns would then be
/// compared as if they were the same encoder. `Ok(None)` means this config does not
/// use the encoder at all; `Err` means it does and could not, which is a printed
/// skip rather than a row of zeros.
fn open_nvenc_session(
    device: &Arc<GpuDevice>,
    config: &BenchmarkConfig,
) -> Result<Option<(EncodeInterop, Nv12EncodeNode)>, Skipped> {
    if config.path != ExecutionPath::GpuNvencZeroCopy {
        return Ok(None);
    }
    let capability = InteropCapability::probe(device);
    if !capability.is_available() {
        return Err(Skipped(format!(
            "CUDA interop is unavailable on this machine (transport={:?}), so the \
             zero-copy NVENC path cannot be exercised at all",
            capability.transport
        )));
    }
    let cuda_ctx = shared_cuda_ctx(&capability)
        .ok_or_else(|| Skipped("CudaContext::new failed on this host".to_string()))?;
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
    Ok(Some((interop, nv12)))
}

/// How many submissions may be outstanding, and the `NEXIR_GPU_LOOKAHEAD` override.
///
/// Strictly below `pipeline_depth` so the slot about to be rendered into is never
/// one still sitting in `inflight` — the same reasoning as
/// `src/export/renderer.rs:864-869`, which both measured loops deliberately mirror
/// rather than reinventing.
///
/// `NEXIR_GPU_LOOKAHEAD` lowers it, and only lowers it: 1 is a serial pipeline.
/// That exists because the 4K row's intervals alternate (17.6 / 25.4 ms measured),
/// and the two candidate explanations — a device property vs. an artefact of how
/// two in-flight frames' uploads interleave on the queue — are told apart by
/// running the same code with the pipeline collapsed. Clamped from above rather
/// than replacing the derivation, because exceeding the encoder's slot count is the
/// one thing `reclaim_slot` exists to prevent.
fn gpu_lookahead_for(pipeline_depth: usize) -> usize {
    let derived = pipeline_depth.saturating_sub(2).max(1);
    let chosen = std::env::var("NEXIR_GPU_LOOKAHEAD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .map(|n| n.min(derived))
        .unwrap_or(derived);
    if chosen != derived {
        println!(
            "    Pipeline: {chosen} frame(s) outstanding (overridden from {derived})"
        );
    }
    chosen
}

/// Run one benchmark, on whichever workload its config names.
///
/// **The branch on `config.source` is the whole reason that field exists.** Without
/// it `bench.exe 7` ran `make_nv12`'s bars and printed their numbers under a
/// heading that says `REAL MEDIA … INTEROP decode` — a fabricated *attribution*
/// rather than a fabricated number, which is if anything harder to catch than the
/// hardcoded `gpu_utilization: 88.5` gotcha 9 was written for. A row must measure
/// the workload it names or not exist.
fn run_benchmark(
    device: &Arc<GpuDevice>,
    config: &BenchmarkConfig,
) -> Result<BenchRun, Skipped> {
    match config.source {
        WorkloadSource::Synthetic => run_synthetic_benchmark(device, config),
        WorkloadSource::RealMedia4K60(decode_path) => {
            run_real_media_benchmark(device, config, decode_path)
        }
    }
}

fn run_synthetic_benchmark(
    device: &Arc<GpuDevice>,
    config: &BenchmarkConfig,
) -> Result<BenchRun, Skipped> {
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
    let mut nvenc = open_nvenc_session(device, config)?;

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
            let identity_lut = identity_lut(17);

            for i in 0..layer_count {
                let rgb_id = ResourceId::next(&mut id_counter);

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

                // Colour correction → LUT → chroma key, or one fused pass — see
                // `add_grade_chain` and `NEXIR_FUSE_GRADE`. Shared with benchmark
                // 7/8's builder so the two graphs cannot differ in anything but the
                // decode path.
                let graded = add_grade_chain(
                    device,
                    &shaders,
                    &compute,
                    &mut compiler,
                    &mut id_counter,
                    &identity_lut,
                    rgb_id,
                    config.canvas_w,
                    config.canvas_h,
                );

                final_composite_inputs.push(graded);
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
            // The synthetic sweep uploads from host memory, so nothing is imported.
            interop_planes: None,
        });
    }

    let frame_state = FrameState {
        pts: 0,
        canvas_width: config.canvas_w,
        canvas_height: config.canvas_h,
        clips: frame_clips,
        test_textures: vec![],
        // The synthetic sweep drives the CPU upload path, so nothing is imported.
        imported: Default::default(),
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

    // How many submissions may be outstanding without having been retired — see
    // [`gpu_lookahead_for`], shared with the real-media rows so the two pipelines
    // are the same pipeline.
    let gpu_lookahead = gpu_lookahead_for(pipeline_depth);

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

    // ── Per-node GPU timings (E1b) ────────────────────────────────────────────
    // Deliberately BEFORE `ProfilingSession::new`, and this placement is
    // load-bearing.
    //
    // `generate_report` divides the frame count by `start_time.elapsed()`, so any
    // work between the session's creation and the report lands in the throughput
    // divisor. Running this pass after the drain instead cost benchmark 5 a
    // measured 41.8 FPS against a 17.55 ms mean interval — i.e. the FPS row was
    // reporting the extra 30 bracketed frames as if they were part of the run.
    // The report caught it (`mean latency implies 57.0 FPS but throughput is
    // 41.8`), which is exactly what gotcha 13's cross-check exists for; the fix is
    // to keep the session's seconds covering only the measured loop.
    //
    // Behind `NEXIR_NODE_TIMINGS` because bracketing every node adds two encoder
    // commands per node per frame, so a run with it on is not the run the headline
    // figures come from. Its frames double as extra warm-up, which is harmless —
    // they are untimed either way.
    //
    // This is for understanding, not for the 4K60 target: the whole graph is
    // ~7.4 ms measured against a 16.7 ms budget, so even a free graph leaves the
    // ~10 ms transfer row. It exists so Phase H can be scoped from data.
    if let Some(timing_frames) = node_timing_frames() {
        let uploads: Vec<(usize, usize)> = uploaders
            .iter()
            .map(|u| (u.node_idx, u.source_idx))
            .collect();
        let frames_ref = &nv12_frames;
        print_node_timings(device, &mut graph, &frame_state, timing_frames, |g, f| {
            for &(node_idx, source_idx) in &uploads {
                if let Some(u) = g.nodes_mut()[node_idx]
                    .as_any_mut()
                    .and_then(|n| n.downcast_mut::<YuvUploadNode>())
                {
                    u.upload_frame_shared(
                        std::sync::Arc::clone(
                            &frames_ref[(f + source_phase_offset(source_idx)) % PATTERN_FRAMES],
                        ),
                        true,
                        config.canvas_w,
                        config.canvas_h,
                    );
                }
            }
        });
    }

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

    // CPU utilisation and RSS are read from the OS across the measured loop only,
    // never from arithmetic over the stage timings (AGENTS.md gotcha 9). Started
    // here rather than before the warm-up so the interval it divides by is the
    // same interval the FPS and latency rows cover.
    let cpu_sampler = nexir::profiling::sysinfo::CpuSampler::start();

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

    // ── What the texture pool did ─────────────────────────────────────────────
    // Counted after the node-timing pass as well as the measured loop, when both
    // ran: the pool is per-graph and its high-water mark is what gotcha 14's cap
    // is read off, so it must cover every frame this graph executed.
    print_pool_stats(graph.pool_stats());
    // ...and what per-resource lifetimes could reduce it to — P2.2's ceiling, off this
    // graph's own declarations rather than estimated. VRAM, not frame time.
    print_lifetime_bounds(&graph, config.canvas_w, config.canvas_h);

    // NVML, read once here rather than per frame: the driver samples utilisation
    // over its own window (200 ms on this card, per nvchk/nvml_probe.c), so polling
    // it per frame would return the same window repeatedly and cost a driver call
    // per frame for it. Taken before the loop's textures are dropped, so the VRAM
    // figure is one taken while this workload was resident.
    //
    // `unwrap_or_default` because every field of `GpuMetrics` is already `Option`:
    // "no nvml.dll on this machine" and "the driver declined this query" both end
    // as `None`, which the report prints as `n/a`. Never a zero row.
    let gpu = nexir::profiling::ffi::nvml::read_device_0().unwrap_or_default();

    session.update_system_metrics(SystemMetrics {
        // Measured from OS queries over the timed loop — `GetProcessTimes` and
        // `GetProcessMemoryInfo`, not arithmetic over the stage timings. `None` on
        // a platform without the query, which prints `n/a`.
        cpu_utilization: cpu_sampler.utilisation(),
        ram_used_bytes: nexir::profiling::sysinfo::process_rss_bytes(),
        // Measured from NVML, read at the end of the timed loop. Instantaneous by
        // nature — NVML reports a sampling window, not an average over our run — so
        // it is a reading taken while the workload was still resident, not a
        // characterisation of the whole run, and the report labels the row plainly.
        // `None` on a machine without `nvml.dll`, which prints `n/a` rather than 0.
        gpu_utilization: gpu.gpu_utilization,
        nvenc_utilization: gpu.encoder_utilization,
        vram_used_bytes: gpu.vram_used_bytes,
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

    Ok(BenchRun {
        session,
        // Nothing decoded: `make_nv12`'s bars come out of host memory, so there is
        // no decode path to attest to and a provenance block here would be a claim
        // about a decoder that never ran.
        media: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK G2f.1 STEP 4 — THE PIPELINED REAL-MEDIA ROWS (benchmarks 7 and 8)
//
// **What these are, and what they are not.** Benchmark 5 is the pipelined 4K60
// baseline the audit's target is judged against, and it is fed `make_nv12` bars
// from host memory — no decoder in the process, so no run of it can touch the
// interop decode path. `--interop` measures the decode path on real files, but
// SERIALLY (one frame in flight) and on a `Simple` chain, so its rows are a frame
// COST and cannot be read against a percentile target.
//
// Benchmarks 7 and 8 are the missing cell: benchmark 5's exact pipeline — the same
// Heavy chain, the same NVENC session, the same `gpu_lookahead`, the same per-slot
// `GpuTimer`s, the same `retire_frame` — fed from four real 4K60 files through
// `IoLayer`. 7 and 8 differ by ONE argument, the `InteropDecodeTargets`, so the
// pair is a measurement of the decode path rather than of two different builds.
//
// Five things in here are load-bearing rather than incidental, each with its
// failure mode recorded elsewhere:
//
//   * `mark_submitted(&frame.clips, &submission)` one line after `device.submit`,
//     or the next decode overwrites textures the submitted graph is still sampling
//     — a torn frame with no error and no counter (gotcha 18).
//   * The graph is rebuilt when `ClipShape` moves, INCLUDING on a fallback: a
//     cached graph that keeps importing planes the frame no longer binds panics in
//     `resolve_resources`.
//   * Upload bytes are pushed only for clips that HAVE an upload node. `uploads[i]
//     == None` is an interop clip, and feeding it reads tier 0 slot 0 — an
//     unrelated source's frame over the top of NVDEC's output (the G2d failure).
//   * `WARMUP_DURATION` is respected, and the first `DecodeInteropTarget`
//     allocation happens inside the warm-up, so a `CreateCommittedResource` +
//     `CreateSharedHandle` + `cuImportExternalMemory` per plane is off the measured
//     path.
//   * A missing fixture or absent CUDA is `Err(Skipped(reason))`, never a zero row.
// ─────────────────────────────────────────────────────────────────────────────

/// A pipelined real-media frame's per-frame state, carried alongside `InFlight`.
///
/// The interop registry must not decode into a source's textures again until the
/// submission that read them has retired, and the retire happens in a different
/// loop iteration from the submit — so the clips that frame bound have to travel
/// with it. `mark_submitted` is called at submit time (that is the guard), and this
/// exists so the *upload* bookkeeping can be attributed to the right frame.
fn run_real_media_benchmark(
    device: &Arc<GpuDevice>,
    config: &BenchmarkConfig,
    decode_path: DecodePath,
) -> Result<BenchRun, Skipped> {
    use nexir::io::interop_decode::InteropDecodeTargets;

    // ── The inputs, before anything is allocated ──────────────────────────────
    // A missing `ffmpeg` or an unencodable class is a property of the HOST, so it
    // is a printed skip. A skip here is not a zero row: the summary omits the
    // benchmark entirely and prints the reason.
    let ffmpeg = bench_media::ffmpeg_binary().ok_or_else(|| {
        Skipped(
            "no `ffmpeg` binary on PATH (set NEXIR_FFMPEG). The real-media rows \
             generate their inputs with the CLI, so nothing could be measured."
                .to_string(),
        )
    })?;
    let want_sources = config.source_count();
    let mut inputs: Vec<std::path::PathBuf> = Vec::with_capacity(want_sources);
    for class in bench_media::MULTI_4K60_SOURCES.iter().take(want_sources) {
        match bench_media::ensure_fixture(&ffmpeg, class) {
            Ok(p) => inputs.push(p),
            // A missing source fails the ROW rather than shrinking it: three
            // sources under a four-source heading is a different workload, and the
            // source count is the whole point of this row.
            Err(why) => {
                return Err(Skipped(format!(
                    "{} could not be generated: {why}. A row with fewer sources than \
                     its heading claims would be a smaller workload.",
                    class.label
                )))
            }
        }
    }
    if inputs.len() != want_sources {
        return Err(Skipped(format!(
            "this row needs {want_sources} distinct 4K60 source(s) but \
             MULTI_4K60_SOURCES only declares {}",
            bench_media::MULTI_4K60_SOURCES.len()
        )));
    }

    let capability = InteropCapability::probe(device);
    if decode_path == DecodePath::Interop && !capability.is_available() {
        return Err(Skipped(format!(
            "CUDA interop is unavailable on this host (transport={:?}), so the \
             interop decode row cannot be exercised at all",
            capability.transport
        )));
    }
    let interop = Arc::new(match decode_path {
        DecodePath::Interop => InteropDecodeTargets::with_context(
            Arc::clone(device),
            capability.clone(),
            shared_cuda_ctx(&capability),
        ),
        // The control arm, and `disabled` rather than `with_context(None)` so the
        // CPU row cannot accidentally allocate a target on a host that has CUDA.
        DecodePath::Cpu => InteropDecodeTargets::disabled(Arc::clone(device)),
    });
    if decode_path == DecodePath::Interop && !interop.is_available() {
        return Err(Skipped(
            "the interop registry declined to initialise (CudaContext::new failed), \
             so this row would be the CPU path under an interop heading"
                .to_string(),
        ));
    }

    // ── NVENC, opened before any timing, exactly as the synthetic rows do ─────
    let mut nvenc = open_nvenc_session(device, config)?;

    // ── The project: N sources on N video tracks ──────────────────────────────
    let paths: Vec<&Path> = inputs.iter().map(|p| p.as_path()).collect();
    let (harness, streams, duration_pts) =
        build_multi_clip_harness(device, &paths, Arc::clone(&interop))
            .map_err(|why| Skipped(format!("could not build the clip harness: {why}")))?;
    let stream = &streams[0];
    let width = stream
        .width
        .ok_or_else(|| Skipped("the input stream has no width".to_string()))?;
    let height = stream
        .height
        .ok_or_else(|| Skipped("the input stream has no height".to_string()))?;
    // The canvas is the FIXTURE's geometry, and it must match what the config
    // claims: the heading says 4K and the composite scales nothing, so a mismatch
    // would be a differently-sized measurement under a 4K label.
    if (width, height) != (config.canvas_w, config.canvas_h) {
        return Err(Skipped(format!(
            "the fixtures are {width}x{height} but this row is declared \
             {}x{} — the figures would not be comparable with benchmark 5",
            config.canvas_w, config.canvas_h
        )));
    }
    let fps = stream.frame_rate.unwrap_or(Rational { num: 30, den: 1 });
    let tb = Rational::TIMEBASE_90K;
    let pts_of = |i: usize| -> i64 { nexir::timeline::rational::frame_to_pts(i as i64, fps, tb) };
    // How many frames the shortest source can serve. `-1` because the last frame's
    // pts sits exactly on `duration_pts`, where no clip is active.
    let available =
        (nexir::timeline::rational::pts_to_frame(duration_pts, fps, tb) as usize).saturating_sub(1);
    if available == 0 {
        return Err(Skipped(
            "the inputs decode to no frames, so there is no span to measure".to_string(),
        ));
    }

    let compute = Arc::clone(&harness.compute);
    let shaders = Arc::clone(&harness.shaders);

    // ── Buffers this benchmark allocates itself, counted ─────────────────────
    let mut allocated_gpu_bytes: u64 = 0;
    if let Some((interop_enc, _)) = &nvenc {
        allocated_gpu_bytes += Nv12EncodeNode::buffer_size(config.canvas_h, interop_enc.pitch())
            * interop_enc.slot_count() as u64;
    }

    let pipeline_depth = nvenc
        .as_ref()
        .map(|(i, _)| i.slot_count())
        .unwrap_or(DEFAULT_PIPELINE_DEPTH);
    let gpu_lookahead = gpu_lookahead_for(pipeline_depth);
    let mut gpu_timers: Vec<nexir::render::gpu_timer::GpuTimer> = (0..pipeline_depth)
        .map(|_| nexir::render::gpu_timer::GpuTimer::new(device, 1))
        .collect();

    // The graph, cached across frames and rebuilt when the clip shape moves.
    let mut cached: Option<InteropGraph> = None;
    let mut compiles = 0usize;
    // The leanest measured frame's import count, and the largest clip count, as
    // `run_interop_pass` counts them: a min rather than a last-frame reading, so one
    // frame that fell back is visible.
    let mut clips_max = 0usize;
    let mut interop_clips_min = usize::MAX;
    let mut upload_bytes_total: u64 = 0;

    // One shared readback node, used for a single sampled frame AFTER the measured
    // loop — never inside it.
    let readback = SampleReadbackNode::new(device, width, height);

    // ── One frame, submitted ─────────────────────────────────────────────────
    //
    // A closure so the warm-up runs exactly the code the measurement runs. It
    // returns the submission and the clips it bound, or `None` when the scheduler
    // ran dry; the caller decides whether that is a clean stop or a failure.
    #[allow(clippy::too_many_arguments)]
    fn submit_frame(
        device: &Arc<GpuDevice>,
        harness: &ClipHarness,
        shaders: &ShaderRegistry,
        compute: &Arc<ComputePipelineCache>,
        interop: &InteropDecodeTargets,
        cached: &mut Option<InteropGraph>,
        compiles: &mut usize,
        chain: GraphChain,
        pts: i64,
        readback: Option<SampleReadbackNode>,
        nvenc: Option<(&EncodeInterop, &Nv12EncodeNode, usize)>,
        gpu_timer: Option<&mut nexir::render::gpu_timer::GpuTimer>,
    ) -> Result<Option<SubmittedFrame>, String> {
        // ── Schedule: decode (interop or CPU) and bind this frame ────────────
        let t = Instant::now();
        let frame = {
            let store = harness.timeline.read().unwrap();
            let tracks = harness.tracks.read().unwrap();
            let sources = harness.sources.read().unwrap();
            harness
                .scheduler
                .schedule_frame(pts, &store, &tracks, &sources)
        };
        let schedule = t.elapsed();
        if frame.clips.is_empty() {
            // A clean stop, not a failure: `IoLayer`'s reachable span is a property
            // of the fixture and FFmpeg's threading, and the caller reports the
            // frames it measured rather than a rate for a count it did not reach.
            return Ok(None);
        }

        // ── Graph: reuse unless the shape moved ─────────────────────────────
        //
        // `readback.is_some()` forces a rebuild because the sampled frame needs the
        // readback node in the graph and no measured frame may carry it.
        let shape: Vec<ClipShape> = frame.clips.iter().map(ClipShape::of).collect();
        let want_sample = readback.is_some();
        if want_sample || cached.as_ref().map(|g| g.shape != shape).unwrap_or(true) {
            *cached = Some(build_interop_graph(
                device, shaders, compute, &frame, chain, readback,
            )?);
            *compiles += 1;
        }
        let g = cached.as_mut().expect("just built");

        // ── Upload: only the clips that HAVE an upload node ─────────────────
        //
        // `uploads[i]` is `None` for an interop clip. Feeding one is the G2d
        // failure: its planes are imported, `texture_slot` is 0, and reading tier 0
        // slot 0 uploads an unrelated source's frame over them.
        let t = Instant::now();
        let mut upload_cost = nexir::render::nodes::yuv_upload::UploadCost::zero();
        for (slot, clip) in frame.clips.iter().enumerate() {
            let Some(node_idx) = g.uploads.get(slot).copied().flatten() else {
                continue;
            };
            let tier = (clip.texture_slot >> 16) as u8;
            let index = (clip.texture_slot & 0xFFFF) as u16;
            let slot_id = nexir::io::slot_pool::FrameSlotId { tier, index };
            let io = harness.scheduler.io_layer();
            if let Some(u) = g.graph.nodes_mut()[node_idx]
                .as_any_mut()
                .and_then(|n| n.downcast_mut::<YuvUploadNode>())
            {
                upload_cost.add(io.pool.with_buffer_read(slot_id, |data| {
                    u.upload_frame(
                        data,
                        clip.frame_meta.layout.semi_planar,
                        clip.clip_width,
                        clip.clip_height,
                    )
                }));
            }
        }
        let upload = t.elapsed();

        // ── Record ───────────────────────────────────────────────────────────
        let mut encoder = device.begin_frame();
        if let Some(timer) = gpu_timer {
            timer.reset();
            timer.begin(&mut encoder);
            match nvenc {
                Some((enc_session, nv12, slot)) => {
                    // The pitch comes from the encoder, never from a second
                    // computation here — gotcha 6.
                    let pitch = enc_session.pitch();
                    let nv12_buffer = enc_session.nv12_buffer_for_slot(slot);
                    g.graph.execute_with_callback(
                        &mut encoder,
                        device,
                        &frame,
                        |enc, ctx| {
                            let in_view = ctx
                                .get(ResourceId::FINAL_COLOR)
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            nv12.record(enc, device, &in_view, nv12_buffer, pitch);
                        },
                    );
                }
                None => g.graph.execute(&mut encoder, device, &frame),
            }
            timer.end(&mut encoder);
            timer.record_resolve(&mut encoder);
        } else {
            match nvenc {
                Some((enc_session, nv12, slot)) => {
                    let pitch = enc_session.pitch();
                    let nv12_buffer = enc_session.nv12_buffer_for_slot(slot);
                    g.graph.execute_with_callback(
                        &mut encoder,
                        device,
                        &frame,
                        |enc, ctx| {
                            let in_view = ctx
                                .get(ResourceId::FINAL_COLOR)
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            nv12.record(enc, device, &in_view, nv12_buffer, pitch);
                        },
                    );
                }
                None => g.graph.execute(&mut encoder, device, &frame),
            }
        }
        let submission = device.submit(encoder);

        // THE GUARD. One line after submit, on the WHOLE clip list, exactly as the
        // two production callers do it. Without this the next decode into any of
        // these textures races the graph still reading them (gotcha 18).
        interop.mark_submitted(&frame.clips, &submission);
        device.device.poll(wgpu::Maintain::Poll);

        let interop_clips = frame.clips.iter().filter(|c| c.is_interop()).count();
        Ok(Some(SubmittedFrame {
            submission,
            clips: frame.clips.len(),
            interop_clips,
            upload_bytes: upload_cost.bytes,
            schedule,
            upload,
            upload_prepare: upload_cost.prepare,
            upload_submit: upload_cost.submit,
        }))
    }

    // ── Warm-up, untimed ─────────────────────────────────────────────────────
    //
    // The same code the measured loop runs, for `WARMUP_DURATION` — which is what
    // puts the first `DecodeInteropTarget` allocation, the decoders' start-up and
    // the pipeline/bind-group creation off the measured path. Serial here
    // deliberately: warming the pipeline is not what this is for, and a warm-up
    // that left frames in flight would have them retire inside the measured span.
    let warmup_start = std::time::Instant::now();
    let mut w = 0usize;
    while w < WARMUP_MIN_FRAMES || warmup_start.elapsed() < WARMUP_DURATION {
        let slot = w % pipeline_depth;
        if let Some((enc, _)) = &mut nvenc {
            // Warm-up frames are encoded too, so the session's slots are in the
            // same state the measured loop expects. Their packets are discarded:
            // this is untimed work and its bitstream is not the row's.
            let _ = enc.reclaim_slot(slot);
        }
        let submitted = {
            let (enc_ref, nv12_ref) = match &nvenc {
                Some((e, n)) => (Some(e), Some(n)),
                None => (None, None),
            };
            let nvenc_args = enc_ref.zip(nv12_ref).map(|(e, n)| (e, n, slot));
            submit_frame(
                device,
                &harness,
                &shaders,
                &compute,
                &interop,
                &mut cached,
                &mut compiles,
                GraphChain::Heavy,
                pts_of(w % available),
                None,
                nvenc_args,
                None,
            )
            .map_err(|why| Skipped(format!("the warm-up could not render a frame: {why}")))?
        };
        let Some(submitted) = submitted else {
            // The warm-up ran out of span. Not fatal — it means the fixture is
            // shorter than the warm-up wants — so stop warming and let the measured
            // loop report what it can reach.
            break;
        };
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submitted.submission));
        if let Some((enc, _)) = &mut nvenc {
            let _ = enc.encode_frame(pts_of(w % available), slot);
        }
        w += 1;
    }
    println!(
        "    Warm-up: {} frame(s) in {:.2} s (untimed), {} live interop target(s)",
        w,
        warmup_start.elapsed().as_secs_f64(),
        interop.live_targets()
    );
    // The graph compiled during the warm-up is not a measured recompile.
    compiles = 0;

    // ── Per-node GPU timings, when asked for ─────────────────────────────────
    //
    // BEFORE `ProfilingSession::new`, for the reason the synthetic path documents at
    // its own call site: `generate_report` divides the frame count by the session's
    // own elapsed time, so a pass run inside that span lands in the throughput
    // divisor. This one is also bracketing every node, which is 42 extra encoder
    // commands per frame on the Heavy graph — not the run the headline figures come
    // from.
    //
    // **It re-executes ONE scheduled frame rather than scheduling 30.** The fixture's
    // reachable span is what bounds the measured loop (`available`), and scheduling
    // frames here would spend it on an untimed pass; the graph shape and its imported
    // planes are stable, so re-recording the same frame measures the same shaders.
    // `push` is therefore a no-op: on the interop path there is no upload node to
    // feed, and on the CPU path the frame's slot-pool buffer is still the one the
    // warm-up filled.
    if let Some(timing_frames) = node_timing_frames() {
        let frame = {
            let store = harness.timeline.read().unwrap();
            let tracks = harness.tracks.read().unwrap();
            let sources = harness.sources.read().unwrap();
            harness
                .scheduler
                .schedule_frame(pts_of(0), &store, &tracks, &sources)
        };
        match cached.as_mut() {
            Some(g) if !frame.clips.is_empty() => {
                print_node_timings(device, &mut g.graph, &frame, timing_frames, |_, _| {});
            }
            _ => println!(
                "    Per-node GPU timings SKIPPED: no graph was compiled during the \
                 warm-up, so\n    there is nothing to bracket. Never a zero row."
            ),
        }
    }

    // ── Measurement loop ─────────────────────────────────────────────────────
    //
    // Structurally identical to the synthetic sweep's: submit half, retire half,
    // `retire_frame` unchanged. That is the point of the row — it is benchmark 5's
    // pipeline with a different pixel source, so the two are comparable.
    let session = ProfilingSession::new(config.target_fps);
    let job = nvenc.as_ref().map(|_| nvenc_job(config));
    let mut bitstream_bytes: u64 = 0;
    let mut packets_out: usize = 0;
    let mut download_bytes: u64 = 0;
    let mut inflight: std::collections::VecDeque<InFlight> =
        std::collections::VecDeque::with_capacity(gpu_lookahead + 1);
    let mut last_retire: Option<std::time::Instant> = Some(std::time::Instant::now());
    let cpu_sampler = nexir::profiling::sysinfo::CpuSampler::start();
    let mut last_graph_end_tick: Option<u64> = None;
    // No readback buffer on this path: the row is the preview pipeline, and the
    // correctness sample is taken after the loop through the graph's own node.
    let readback_buffer: Option<wgpu::Buffer> = None;

    let mut measured = 0usize;
    let mut ran_dry = false;
    for f in 0..config.frame_count {
        if f >= available {
            ran_dry = true;
            break;
        }
        let pts = pts_of(f);
        let mut profile = FrameProfile::new(f, job.as_ref().map(|j| j.frame_pts(f)).unwrap_or(pts));
        let slot = f % pipeline_depth;

        // Backpressure: NVENC must be finished reading this slot's NV12 buffer.
        if let Some((enc, _)) = &mut nvenc {
            let reclaimed = profile.measure(PipelineStage::Nvenc, || enc.reclaim_slot(slot));
            let reclaimed = reclaimed.expect("NVENC reclaim_slot failed mid-benchmark");
            packets_out += reclaimed.len();
            bitstream_bytes += reclaimed.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
        }

        let composite_start = std::time::Instant::now();
        let submitted = {
            let (enc_ref, nv12_ref) = match &nvenc {
                Some((e, n)) => (Some(e), Some(n)),
                None => (None, None),
            };
            let nvenc_args = enc_ref.zip(nv12_ref).map(|(e, n)| (e, n, slot));
            submit_frame(
                device,
                &harness,
                &shaders,
                &compute,
                &interop,
                &mut cached,
                &mut compiles,
                GraphChain::Heavy,
                pts,
                None,
                nvenc_args,
                Some(&mut gpu_timers[slot]),
            )
            .map_err(|why| Skipped(format!("frame {f} could not be rendered: {why}")))?
        };
        let Some(submitted) = submitted else {
            // Ran dry mid-loop. Everything already in flight is still retired
            // below, and the reported frame count is what was measured.
            ran_dry = true;
            break;
        };
        // ── Stage attribution ────────────────────────────────────────────────
        //
        // `Decode` really is a decode here, unlike the synthetic sweep where the
        // equivalent stage is an upload from host memory and is named `Upload` for
        // exactly that reason. `Composite` is the cost of RECORDING the graph on
        // both paths, which is why it is the submit span minus the two stages
        // already accounted for — a `Composite` figure that included the decode
        // would make the two rows' graph-recording columns incomparable.
        profile.record_stage(PipelineStage::Decode, submitted.schedule);
        profile.record_stage(PipelineStage::Upload, submitted.upload);
        profile.record_stage(PipelineStage::UploadPrepare, submitted.upload_prepare);
        profile.record_stage(PipelineStage::UploadSubmit, submitted.upload_submit);
        profile.record_stage(
            PipelineStage::Composite,
            composite_start
                .elapsed()
                .saturating_sub(submitted.schedule)
                .saturating_sub(submitted.upload),
        );

        clips_max = clips_max.max(submitted.clips);
        interop_clips_min = interop_clips_min.min(submitted.interop_clips);
        upload_bytes_total += submitted.upload_bytes;
        measured += 1;

        inflight.push_back(InFlight {
            pts: job.as_ref().map(|j| j.frame_pts(f)).unwrap_or(pts),
            submission: submitted.submission,
            slot,
            profile,
        });

        while inflight.len() > gpu_lookahead {
            let frame = inflight
                .pop_front()
                .expect("inflight is non-empty: len > gpu_lookahead >= 1");
            retire_frame(
                frame,
                device,
                &mut nvenc,
                &readback_buffer,
                0,
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

    // Drain: every frame still in flight, retired the same way.
    while let Some(frame) = inflight.pop_front() {
        retire_frame(
            frame,
            device,
            &mut nvenc,
            &readback_buffer,
            0,
            &mut gpu_timers,
            &session,
            &mut packets_out,
            &mut bitstream_bytes,
            &mut download_bytes,
            &mut last_retire,
            &mut last_graph_end_tick,
        );
    }
    if let Some((enc, _)) = &mut nvenc {
        let tail = enc.flush().expect("NVENC flush failed");
        packets_out += tail.len();
        bitstream_bytes += tail.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
    }

    if measured == 0 {
        return Err(Skipped(format!(
            "the scheduler produced no clips at all over {} frame(s) of a {available}-frame \
             span, so there is no rate to report",
            config.frame_count
        )));
    }
    if ran_dry {
        // Printed, never hidden: a row measuring fewer frames than it asked for is
        // comparable with another row only if the reader knows.
        println!(
            "    NOTE: the span ran dry after {measured} of {} requested frame(s) \
             ({available} reachable).\n    The figures below are for the frames actually \
             rendered.",
            config.frame_count
        );
    }

    // ── The pool's counters, and the correctness sample ──────────────────────
    //
    // Read BEFORE the sample pass, because that pass compiles a different graph
    // (with the readback node) and a `CompiledGraph` owns its pool — reading
    // afterwards would report a pool that saw one frame. Same rule
    // `run_interop_pass` follows.
    let pool = cached
        .as_ref()
        .map(|g| g.graph.pool_stats())
        .unwrap_or_default();
    // P2.2's ceiling on the real-media graph shape, taken from the same cached graph
    // and for the same reason: the sample pass below compiles a different one (with the
    // readback node), whose lifetimes are not this row's.
    let lifetimes = cached.as_ref().map(|g| g.graph.lifetime_bounds());

    // One sampled frame, decoded FORWARD from where the measured loop stopped and
    // outside the timed span. Forward rather than re-visiting, for
    // `run_interop_pass`'s reason: the CPU arm answers a repeat from `FrameCache`
    // while a one-frame interop target must seek and re-decode, and a seek into an
    // open-GOP stream can legitimately produce a differently-referenced frame.
    let mut sample = None;
    if measured < available {
        let idx = measured;
        let submitted = submit_frame(
            device,
            &harness,
            &shaders,
            &compute,
            &interop,
            &mut cached,
            &mut compiles,
            GraphChain::Heavy,
            pts_of(idx),
            Some(readback.attached()),
            None,
            None,
        )
        .unwrap_or(None);
        if let Some(s) = submitted {
            device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(s.submission));
            sample = read_frame_sample(device, &readback, None).map(|fs| (idx, fs));
        }
    }

    print_pool_stats(pool);
    // Printed from the `(held, ideal)` pair taken above rather than by re-reading the
    // graph, which by now is the sample pass's. `n/a` when no graph was ever cached,
    // because a row that rendered nothing has no shape to analyse — never a zero.
    match lifetimes {
        Some((held, ideal)) => {
            let save = held.saturating_sub(ideal);
            let per_texture = config.canvas_w as u64 * config.canvas_h as u64 * 8;
            println!(
                "    Lifetimes: holds {held} texture(s) simultaneously; {ideal} would \
                 suffice if non-overlapping\n    resources shared one (P2.2's ceiling) \
                 — {save} fewer, {:.1} MB of residency (lower bound). VRAM, not frame \
                 time.",
                save as f64 * per_texture as f64 / (1024.0 * 1024.0)
            );
        }
        None => println!("    Lifetimes: n/a (no graph was compiled for this row)"),
    }

    let gpu = nexir::profiling::ffi::nvml::read_device_0().unwrap_or_default();
    session.update_system_metrics(SystemMetrics {
        cpu_utilization: cpu_sampler.utilisation(),
        ram_used_bytes: nexir::profiling::sysinfo::process_rss_bytes(),
        gpu_utilization: gpu.gpu_utilization,
        nvenc_utilization: gpu.encoder_utilization,
        vram_used_bytes: gpu.vram_used_bytes,
        // Counted: the NVENC buffers this row allocated, plus the interop targets'
        // own NV12 pairs. Both are lower bounds on total GPU memory (the graph's
        // texture pool is not counted), which is what the field is documented as.
        allocated_gpu_bytes: Some(allocated_gpu_bytes + interop.target_bytes()),
        // Counted by `UploadCost`, so it is 0 on the interop row because nothing
        // crossed the bus — which is the result, not an unmeasured stage.
        gpu_upload_bytes: Some(upload_bytes_total),
        gpu_download_bytes: Some(download_bytes),
        frame_queue_depth: nvenc.as_ref().map(|(i, _)| i.slot_count()),
        // A real property of this path: one frame per source, by construction.
        decode_queue_depth: Some(1),
        encode_queue_depth: nvenc.as_ref().map(|(i, _)| i.slot_count()),
    });

    if nvenc.is_some() {
        println!(
            "    NVENC: {} packet(s), {:.2} MB of bitstream from {measured} frame(s)",
            packets_out,
            bitstream_bytes as f64 / (1024.0 * 1024.0),
        );
        if packets_out == 0 {
            println!(
                "    WARNING: the NVENC session accepted every picture but returned no \
                 bitstream. The NVENC column below is submit cost only."
            );
        }
    }

    let provenance = RealMediaProvenance {
        decode_path,
        frames: measured,
        sources: want_sources,
        clips: clips_max,
        interop_clips: if interop_clips_min == usize::MAX {
            0
        } else {
            interop_clips_min
        },
        live_targets: interop.live_targets(),
        target_bytes: interop.target_bytes(),
        upload_bytes_per_frame: upload_bytes_total as f64 / measured as f64,
        interop_stats: (decode_path == DecodePath::Interop).then(|| interop.stats()),
        per_source: if decode_path == DecodePath::Interop {
            interop
                .per_source_stats()
                .into_iter()
                .map(|(id, s)| (id.index() as u32, s))
                .collect()
        } else {
            Vec::new()
        },
        rejections: interop
            .rejections()
            .into_iter()
            .map(|(id, reason)| (id.index() as u32, reason))
            .collect(),
        pool,
        compiles,
        sample,
    };
    print_real_media_provenance(&provenance);

    Ok(BenchRun {
        session,
        media: Some(provenance),
    })
}

/// What one submitted real-media frame bound and what it cost the CPU.
///
/// The stage timings are returned rather than written into a `FrameProfile` inside
/// `submit_frame`, so the warm-up can call exactly the same function without a
/// profile to write into — a warm-up that ran different code would warm something
/// else, which is what `WARMUP_DURATION`'s own comment is about.
struct SubmittedFrame {
    submission: wgpu::SubmissionIndex,
    clips: usize,
    interop_clips: usize,
    /// Bytes `UploadCost` counted for this frame. 0 on the interop path because
    /// there is no upload node — the result, not an absent measurement.
    upload_bytes: u64,
    /// `FrameScheduler::schedule_frame`, i.e. the decode. On the CPU arm this also
    /// contains `av_hwframe_transfer_data`'s copy to host memory; on the interop arm
    /// it contains the device→array copy instead. That is the pair of stages this
    /// row exists to compare.
    schedule: Duration,
    /// `YuvUploadNode::upload_frame`, summed over the clips that have one. Zero on
    /// the interop arm because there is no upload node — the stage the interop path
    /// DELETES, so the CPU arm's figure is the upper bound on what removing it can
    /// save.
    upload: Duration,
    upload_prepare: Duration,
    upload_submit: Duration,
}

/// Print what a real-media row's own counters say about the workload it measured.
///
/// **This is what stops benchmark 7 from being benchmark 8 with a different
/// heading.** The FPS row above it is identical in shape either way; only these
/// counters can say whether four sources really decoded into their own textures. The
/// pairings are the ones `--interop` reports for the same reason: `0 bytes` alone is
/// indistinguishable from a pass that never ran, and `live targets` alone cannot see
/// a frame that fell back.
fn print_real_media_provenance(p: &RealMediaProvenance) {
    println!(
        "    Decode path: {} — {} frame(s) measured, {} distinct source(s), \
         {}/{} clip(s) imported",
        p.decode_path.label(),
        p.frames,
        p.sources,
        p.interop_clips,
        p.clips,
    );
    println!(
        "    Upload: {:.2} MB/frame (counted), live targets {} ({}), graph compiles {}",
        p.upload_bytes_per_frame / (1024.0 * 1024.0),
        p.live_targets,
        if p.target_bytes > 0 {
            format!(
                "{:.1} MB VRAM, lower bound",
                p.target_bytes as f64 / (1024.0 * 1024.0)
            )
        } else {
            "no VRAM counted".to_string()
        },
        p.compiles,
    );
    if let Some(s) = p.interop_stats {
        println!(
            "    Interop: decodes {}, cached {}, pending {}, decode {} — waits {} ({})",
            s.decodes,
            s.cached,
            s.pending,
            s.decode_ms_per_frame()
                .map(|ms| format!("{ms:.2} ms/frame"))
                .unwrap_or_else(|| "n/a".into()),
            s.waits,
            s.wait_ms_per_decode()
                .map(|ms| format!("{ms:.3} ms/decode"))
                .unwrap_or_else(|| "n/a".into()),
        );
        match s.copy_ms_per_call() {
            Some((barrier, issue, sync)) => println!(
                "             copies {}: decode barrier {barrier:.3} ms, issue {issue:.3} \
                 ms, transfer sync {sync:.3} ms",
                s.copy.calls
            ),
            None => println!("             copies 0: no device→array copy ran, so its phases are n/a"),
        }
        // Per source, for the same reason `--interop` prints it: a registry-wide
        // decode mean of 6 ms is equally consistent with four sources at 1.5 ms and
        // with three at 0.5 plus one at 4.5, and those lead to opposite fixes.
        if p.per_source.len() > 1 {
            println!("             per source:");
            for (id, ps) in &p.per_source {
                println!(
                    "               src {id}: {} decode(s), {} cached, {} — waits {} ({}), \
                     copies {}",
                    ps.decodes,
                    ps.cached,
                    ps.decode_ms_per_frame()
                        .map(|ms| format!("{ms:.2} ms/frame"))
                        .unwrap_or_else(|| "n/a".into()),
                    ps.waits,
                    ps.wait_ms_per_decode()
                        .map(|ms| format!("{ms:.3} ms/decode"))
                        .unwrap_or_else(|| "n/a".into()),
                    ps.copy.calls,
                );
            }
        }
    }
    if p.decode_path == DecodePath::Interop && p.live_targets == 0 {
        println!(
            "    WARNING: no interop target was allocated, so this row measured the CPU\n\
             \x20   upload path under an `INTEROP decode` heading."
        );
    }
    for (source, reason) in &p.rejections {
        println!("             source {source} FELL BACK: {reason}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK F1b — THE REAL-MEDIA EXECUTION PATH (`bench --media`)
//
// Every figure in the six benchmarks above was measured on `make_nv12`'s eight
// vertical bars with a rigid horizontal pan — the easiest motion a motion
// estimator can face, with no grain and no coded frames at all. Until this path
// runs, "4K at 56.8 FPS" means "4K at 56.8 FPS on synthetic bars".
//
// So this profile drives the SAME graph and the SAME upload/NVENC machinery from
// real H.264 files (`bench_media::MEDIA_CLASSES`, generated once with the ffmpeg
// binary), and reports four separately-measured rates per class:
//
//   decode      demux + decode only, nothing else running
//   render      upload + graph, fed from frames this decoder produced
//   encode      the same plus a real NVENC session
//   end-to-end  all of it in one loop, which is what a user experiences
//
// The four are measured in four passes rather than derived from one another: the
// end-to-end rate is NOT `1/(1/decode + 1/render)`, because the passes overlap
// differently, and printing a computed figure next to measured ones is exactly
// what gotcha 9 forbids.
//
// A missing `ffmpeg`, an unencodable class, or absent NVENC is a printed skip
// with its reason. `NEXIR_REQUIRE_MEDIA=1` turns a skip into a non-zero exit.
// ─────────────────────────────────────────────────────────────────────────────

/// How many decoded frames are kept in RAM to feed the render and encode passes.
///
/// Small on purpose: a 2-second 4K60 class is 120 frames and one 4K NV12 frame is
/// 12.4 MB, so caching the clip would be 1.5 GB of host RAM and the measurement
/// would be of the allocator. Eight real frames cycled is the same shape as
/// `PATTERN_FRAMES` in the synthetic path — the difference this profile exists to
/// make is that these eight came out of a decoder.
const MEDIA_CACHE_FRAMES: usize = 8;

/// How many frames the render/encode/end-to-end passes measure per class.
///
/// Enough that the pipeline-fill interval is a small share of the run (gotcha 15)
/// without making a five-class sweep take minutes.
const MEDIA_PASS_FRAMES: usize = 90;

/// Real decoded frames, plus what the decoder said they are.
///
/// `layout`/`color` come from the DecodedFrame rather than from the container, so
/// the upload node and `YuvToRgbNode` are built from what the decoder actually
/// wrote — the same rule `src/io/io_layer.rs` follows, and the reason gotcha 11
/// exists.
struct DecodedCache {
    frames: Vec<Arc<Vec<u8>>>,
    meta: DecodedFrameMeta,
    width: u32,
    height: u32,
}

/// What one pass measured. `frames / elapsed` and nothing else.
struct PassReading {
    frames: usize,
    elapsed: Duration,
}

impl PassReading {
    fn fps(&self) -> Option<f64> {
        let secs = self.elapsed.as_secs_f64();
        if self.frames == 0 || secs <= 0.0 {
            // Zero frames is not zero FPS. Gotcha 9's rule, applied to a rate.
            None
        } else {
            Some(self.frames as f64 / secs)
        }
    }
}

/// Decode `path` from the start, timing the decode and keeping the first few
/// frames.
///
/// Hardware decode is left ENABLED (`Decoder::open(.., true)`) because that is
/// what `IoLayer` does in production: on this machine NVDEC is selected and the
/// frame is then transferred to host memory, which is precisely the CPU
/// round-trip Task G2 is about. Forcing `open_sw` here would report a cost no
/// user pays.
fn media_decode_pass(path: &Path, max_frames: usize) -> Result<(PassReading, DecodedCache), String> {
    let mut demuxer = Demuxer::open(path).map_err(|e| format!("Demuxer::open failed: {e:?}"))?;
    let stream = demuxer
        .video_stream
        .clone()
        .ok_or_else(|| "the container reports no video stream".to_string())?;
    let cw = stream.width.ok_or("the stream has no width")? as usize;
    let ch = stream.height.ok_or("the stream has no height")? as usize;
    let mut decoder = Decoder::open(&stream, stream.codecpar, true)
        .map_err(|e| format!("Decoder::open failed: {e:?}"))?;

    // Room for 4:4:4 16-bit, so no decoder choice can overflow it.
    let mut buf = vec![0u8; cw * ch * 6 + 256];
    let mut cache: Vec<Arc<Vec<u8>>> = Vec::with_capacity(MEDIA_CACHE_FRAMES);
    let mut last: Option<(DecodedFrameMeta, u32, u32)> = None;
    let mut decoded = 0usize;

    // Accumulated per-call, so the frame COPY into the cache below is excluded:
    // that copy exists for the later passes and charging it to decode would
    // overstate the decoder's cost on exactly the first eight frames.
    let mut decode_time = Duration::ZERO;

    while decoded < max_frames {
        let pkt = match demuxer.next_video_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(format!("next_video_packet failed: {e:?}")),
        };
        let t = Instant::now();
        let got = decoder.decode_into(&pkt, &mut buf, None);
        let dt = t.elapsed();
        match got {
            Ok(Some(frame)) => {
                decode_time += dt;
                decoded += 1;
                let plane_bytes = frame_bytes_for(frame.meta.layout, frame.width, frame.height);
                if cache.len() < MEDIA_CACHE_FRAMES && plane_bytes <= buf.len() {
                    cache.push(Arc::new(buf[..plane_bytes].to_vec()));
                }
                last = Some((frame.meta, frame.width, frame.height));
            }
            // A packet the decoder swallowed without emitting is real work, but not
            // a frame: timing it into a per-frame figure would charge a frame that
            // never arrived. Counted nowhere, which is why the reported rate is
            // frames ÷ decode time and not packets ÷ anything.
            Ok(None) => decode_time += dt,
            Err(e) => return Err(format!("decode_into failed: {e:?}")),
        }
    }
    // Frame-threading holds back roughly one frame per core, so a short clip can
    // yield almost nothing without this — see `media_compat.rs`.
    while decoded < max_frames {
        let t = Instant::now();
        let got = decoder.drain_into(&mut buf);
        let dt = t.elapsed();
        match got {
            Ok(Some(frame)) => {
                decode_time += dt;
                decoded += 1;
                let plane_bytes = frame_bytes_for(frame.meta.layout, frame.width, frame.height);
                if cache.len() < MEDIA_CACHE_FRAMES && plane_bytes <= buf.len() {
                    cache.push(Arc::new(buf[..plane_bytes].to_vec()));
                }
                last = Some((frame.meta, frame.width, frame.height));
            }
            Ok(None) => break,
            Err(e) => return Err(format!("drain_into failed: {e:?}")),
        }
    }

    let (meta, w, h) = last.ok_or("the file produced no decodable frames")?;
    if cache.is_empty() {
        return Err("no frame could be cached for the render pass".to_string());
    }
    Ok((
        PassReading {
            frames: decoded,
            elapsed: decode_time,
        },
        DecodedCache {
            frames: cache,
            meta,
            width: w,
            height: h,
        },
    ))
}

/// Bytes one decoded frame occupies for a given layout — luma plus both chroma
/// planes at 4:2:0.
///
/// Computed from the layout the DECODER reported rather than assumed: at 10-bit
/// this is twice the 8-bit figure, and slicing the staging buffer to the wrong
/// length would hand `YuvUploadNode` a short frame, which it reports as a warning
/// and then skips a plane over.
fn frame_bytes_for(layout: FrameLayout, width: u32, height: u32) -> usize {
    let bpp = layout.bytes_per_sample();
    let luma = width as usize * height as usize * bpp;
    // 4:2:0 chroma is half the luma samples in total, whether planar (U + V) or
    // semi-planar (one interleaved plane).
    luma + luma / 2
}

/// Build the one-layer graph the media passes render through: upload → YUV→RGB →
/// composite into `FINAL_COLOR`.
///
/// Deliberately the SIMPLE shape rather than the Heavy one. The question this
/// profile answers is what real coded frames cost relative to synthetic bars, and
/// the effect chain is identical either way — it never sees the source's
/// provenance. Adding four LUT passes here would make each class's number a mix of
/// two changes at once.
///
/// The upload node and the converter are both built from `cache.meta`, i.e. from
/// what the decoder reported — not from the container and not from a constant.
/// That is the gotcha 11 rule: a fixture whose metadata is dropped somewhere would
/// otherwise still render plausibly.
fn media_graph(
    device: &Arc<GpuDevice>,
    shaders: &ShaderRegistry,
    compute: &Arc<ComputePipelineCache>,
    cache: &DecodedCache,
) -> Result<(nexir::render::graph::CompiledGraph, usize, FrameState), String> {
    let mut compiler = RenderGraphCompiler::new();
    let mut id_counter = 2u32;
    let y = ResourceId::next(&mut id_counter);
    let uv = ResourceId::next(&mut id_counter);
    let rgba = ResourceId::next(&mut id_counter);

    let upload_idx = compiler.add_node(Box::new(YuvUploadNode::new_with_layout(
        device,
        0,
        cache.width,
        cache.height,
        y,
        uv,
        cache.meta.layout,
    )));
    compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
        device,
        shaders,
        compute,
        y,
        uv,
        rgba,
        cache.width,
        cache.height,
        cache.meta.color,
        cache.meta.layout.semi_planar,
    )));
    let mut composite = CompositeNode::new(
        device,
        shaders,
        ResourceId::FINAL_COLOR,
        1,
        wgpu::TextureFormat::Rgba16Float,
    );
    composite.input_textures.push(rgba);
    compiler.add_node(Box::new(composite));

    let graph = compiler
        .compile(cache.width, cache.height)
        .map_err(|e| format!("graph compile failed: {e:?}"))?;

    let frame_state = FrameState {
        pts: 0,
        canvas_width: cache.width,
        canvas_height: cache.height,
        clips: vec![ClipRenderEntry {
            source_id: SourceId::new(0),
            texture_slot: 0,
            layer_order: 0,
            clip_width: cache.width,
            clip_height: cache.height,
            transform: ClipTransform::identity(),
            opacity: 1.0,
            blend_mode: nexir::timeline::transform::BlendMode::Normal,
            crop: nexir::timeline::transform::CropRect::full(),
            corner_pin: nexir::timeline::transform::CornerPin::identity(),
            matte_mode: nexir::timeline::transform::MatteMode::None,
            effects: Default::default(),
            // From the decoder, not a literal — see the doc comment.
            frame_meta: cache.meta,
            kind: ClipKind::Video,
            // The media profile decodes to host memory and uploads, which is what
            // its `GPU transfer` row measures.
            interop_planes: None,
        }],
        test_textures: vec![],
        // Real coded frames, still through the CPU upload path (G2c is what makes
        // this class of row import its planes instead).
        imported: Default::default(),
    };
    Ok((graph, upload_idx, frame_state))
}

/// Push one cached frame into the upload node.
///
/// `upload_frame_shared` rather than `upload_frame`, for the same reason the
/// synthetic loop uses it: the node may have resolved to `UploadPath::WriteTexture`
/// (1080p classes do — 1920 pads to 2048), and that path needs the bytes to
/// survive until `record`. On the staging path it is exactly `upload_frame`.
fn media_push_frame(
    graph: &mut nexir::render::graph::CompiledGraph,
    upload_idx: usize,
    cache: &DecodedCache,
    frame: usize,
) {
    if let Some(u) = graph.nodes_mut()[upload_idx]
        .as_any_mut()
        .and_then(|n| n.downcast_mut::<YuvUploadNode>())
    {
        u.upload_frame_shared(
            Arc::clone(&cache.frames[frame % cache.frames.len()]),
            cache.meta.layout.semi_planar,
            cache.width,
            cache.height,
        );
    }
}

/// Upload + graph, from real decoded frames, with no decoder and no encoder in the
/// loop.
///
/// Serial (`poll(Wait)` per frame) on purpose: this pass exists to isolate the GPU
/// cost of a real frame, and the six benchmarks above already report the pipelined
/// figure. Keeping it serial also means its FPS is comparable across classes
/// without the pipeline depth as a hidden variable.
fn media_render_pass(
    device: &Arc<GpuDevice>,
    graph: &mut nexir::render::graph::CompiledGraph,
    upload_idx: usize,
    cache: &DecodedCache,
    frame_state: &FrameState,
    frames: usize,
) -> PassReading {
    // Untimed warm-up: first-frame pipeline and bind-group creation is not a
    // per-frame cost, and at 4K it is tens of milliseconds.
    for f in 0..WARMUP_MIN_FRAMES {
        media_push_frame(graph, upload_idx, cache, f);
        let mut enc = device.begin_frame();
        graph.execute(&mut enc, device, frame_state);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
    }

    let start = Instant::now();
    for f in 0..frames {
        media_push_frame(graph, upload_idx, cache, f);
        let mut enc = device.begin_frame();
        graph.execute(&mut enc, device, frame_state);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
    }
    PassReading {
        frames,
        elapsed: start.elapsed(),
    }
}

/// Upload + graph + `Nv12EncodeNode` into a real NVENC session.
///
/// `Err` is a per-class skip with its reason — a machine without CUDA interop or
/// NVENC must print that rather than a zero row.
///
/// The bitstream size is returned alongside the rate because it is the one figure
/// that says whether the encoder was given real work: the whole point of this
/// profile is that grain and non-translational motion cost bits, and the synthetic
/// bars' bitstream is a floor.
#[allow(clippy::too_many_arguments)]
fn media_encode_pass(
    device: &Arc<GpuDevice>,
    graph: &mut nexir::render::graph::CompiledGraph,
    upload_idx: usize,
    cache: &DecodedCache,
    frame_state: &FrameState,
    frames: usize,
    target_fps: f64,
) -> Result<(PassReading, usize, u64), String> {
    let capability = InteropCapability::probe(device);
    if !capability.is_available() {
        return Err(format!(
            "CUDA interop unavailable (transport={:?}), so no NVENC session can be \
             opened",
            capability.transport
        ));
    }
    let cuda_ctx = shared_cuda_ctx(&capability)
        .ok_or_else(|| "CudaContext::new failed on this host".to_string())?;
    let job = nvenc_job_for(cache.width, cache.height, target_fps, frames);
    job.validate()
        .map_err(|e| format!("the bench's own ExportJob is invalid: {e:?}"))?;
    let mut interop = EncodeInterop::open(
        Arc::clone(&cuda_ctx),
        device,
        &job,
        capability.transport,
        job.video_codec,
    )
    .map_err(|e| format!("EncodeInterop::open failed: {e:?}"))?;
    let nv12 = Nv12EncodeNode::new(device, job.output_color, job.width, job.height);

    let slots = interop.slot_count();
    let mut packets = 0usize;
    let mut bitstream = 0u64;

    let mut run = |frames: usize, timed: bool| -> Result<Duration, String> {
        let start = Instant::now();
        for f in 0..frames {
            let slot = f % slots;
            let reclaimed = interop
                .reclaim_slot(slot)
                .map_err(|e| format!("reclaim_slot failed: {e:?}"))?;
            if timed {
                packets += reclaimed.len();
                bitstream += reclaimed.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
            }
            media_push_frame(graph, upload_idx, cache, f);
            let mut enc = device.begin_frame();
            // The pitch comes from the encoder and is never recomputed here: the
            // chroma plane sits at pitch*height and a disagreement is a silent hue
            // shift (gotcha 6).
            let pitch = interop.pitch();
            let nv12_buffer = interop.nv12_buffer_for_slot(slot);
            graph.execute_with_callback(&mut enc, device, frame_state, |e, ctx| {
                let view = ctx
                    .get(ResourceId::FINAL_COLOR)
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                nv12.record(e, device, &view, nv12_buffer, pitch);
            });
            let sid = device.submit(enc);
            // NVENC reads the NV12 buffer through CUDA, so the conversion must have
            // completed before the picture is submitted. Two APIs sharing no
            // timeline — a real ordering requirement, not a benchmark artefact.
            device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
            let out = interop
                .encode_frame(job.frame_pts(f), slot)
                .map_err(|e| format!("encode_frame failed: {e:?}"))?;
            if timed {
                packets += out.len();
                bitstream += out.iter().map(|p| p.bytes.len() as u64).sum::<u64>();
            }
        }
        Ok(start.elapsed())
    };

    run(WARMUP_MIN_FRAMES, false)?;
    let elapsed = run(frames, true)?;

    let tail = interop
        .flush()
        .map_err(|e| format!("NVENC flush failed: {e:?}"))?;
    packets += tail.len();
    bitstream += tail.iter().map(|p| p.bytes.len() as u64).sum::<u64>();

    Ok((PassReading { frames, elapsed }, packets, bitstream))
}

/// Demux → decode → upload → graph, in ONE loop, which is what a user watching
/// playback experiences.
///
/// Measured rather than derived. The three passes above each isolate a stage, and
/// the temptation is to combine them arithmetically — but decode is CPU work and
/// the graph is GPU work, so they overlap by an amount only a run can report. A
/// computed end-to-end figure would be an estimate printed beside measurements.
///
/// The decoder is re-opened from the start of the file, so this pass decodes the
/// same coded frames the decode pass did rather than continuing from wherever it
/// stopped.
fn media_end_to_end_pass(
    path: &Path,
    device: &Arc<GpuDevice>,
    graph: &mut nexir::render::graph::CompiledGraph,
    upload_idx: usize,
    cache: &DecodedCache,
    frame_state: &FrameState,
    frames: usize,
) -> Result<PassReading, String> {
    let mut demuxer = Demuxer::open(path).map_err(|e| format!("Demuxer::open failed: {e:?}"))?;
    let stream = demuxer
        .video_stream
        .clone()
        .ok_or_else(|| "no video stream".to_string())?;
    let mut decoder = Decoder::open(&stream, stream.codecpar, true)
        .map_err(|e| format!("Decoder::open failed: {e:?}"))?;
    let mut buf = vec![0u8; cache.width as usize * cache.height as usize * 6 + 256];

    let plane_bytes = frame_bytes_for(cache.meta.layout, cache.width, cache.height);
    let mut rendered = 0usize;
    let start = Instant::now();
    while rendered < frames {
        let pkt = match demuxer.next_video_packet() {
            Ok(Some(p)) => p,
            // Out of coded frames before the target count: report what was
            // actually rendered rather than looping the file, which would report a
            // decode rate for frames that were decoded once.
            Ok(None) => break,
            Err(e) => return Err(format!("next_video_packet failed: {e:?}")),
        };
        let got = decoder
            .decode_into(&pkt, &mut buf, None)
            .map_err(|e| format!("decode_into failed: {e:?}"))?;
        let Some(frame) = got else { continue };
        let bytes = frame_bytes_for(frame.meta.layout, frame.width, frame.height).min(plane_bytes);

        // One copy per frame, into an `Arc` the node can hold. This is the cost the
        // production path pays too — `IoLayer` decodes into a pooled slot and hands
        // it on — so it belongs inside the timed region rather than outside it.
        if let Some(u) = graph.nodes_mut()[upload_idx]
            .as_any_mut()
            .and_then(|n| n.downcast_mut::<YuvUploadNode>())
        {
            u.upload_frame(
                &buf[..bytes],
                frame.meta.layout.semi_planar,
                frame.width.min(cache.width),
                frame.height.min(cache.height),
            );
        }
        let mut enc = device.begin_frame();
        graph.execute(&mut enc, device, frame_state);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        rendered += 1;
    }
    Ok(PassReading {
        frames: rendered,
        elapsed: start.elapsed(),
    })
}

/// One class's four measured rates, for the summary table.
struct MediaReading {
    label: &'static str,
    width: u32,
    height: u32,
    layout: FrameLayout,
    decode: Option<f64>,
    render: Option<f64>,
    encode: Option<f64>,
    end_to_end: Option<f64>,
    /// Bitstream the encoder actually produced, `None` when NVENC was skipped.
    bitstream_bytes: Option<u64>,
}

/// Run all four passes for one class.
fn run_media_class(
    device: &Arc<GpuDevice>,
    class: &'static MediaClass,
    path: &Path,
) -> Result<MediaReading, String> {
    let shaders = ShaderRegistry::compile_all(device)
        .map_err(|e| format!("shader compilation failed: {e:?}"))?;
    let compute = Arc::new(ComputePipelineCache::new());

    // Decode first: everything below is built from what it reported.
    let (decode, cache) = media_decode_pass(path, class.frame_count().min(MEDIA_PASS_FRAMES))?;
    println!(
        "    Decoded {}x{} {:?} ({} bit{}), {} frame(s) cached for the render passes",
        cache.width,
        cache.height,
        cache.meta.color.matrix,
        cache.meta.layout.bit_depth,
        if cache.meta.layout.semi_planar {
            ", semi-planar"
        } else {
            ", planar"
        },
        cache.frames.len(),
    );

    let (mut graph, upload_idx, frame_state) = media_graph(device, &shaders, &compute, &cache)?;
    let render = media_render_pass(
        device,
        &mut graph,
        upload_idx,
        &cache,
        &frame_state,
        MEDIA_PASS_FRAMES,
    );

    let encoded = media_encode_pass(
        device,
        &mut graph,
        upload_idx,
        &cache,
        &frame_state,
        MEDIA_PASS_FRAMES,
        class.fps as f64,
    );
    let (encode, bitstream) = match encoded {
        Ok((reading, packets, bytes)) => {
            println!(
                "    NVENC: {} packet(s), {:.2} MB of bitstream from {} frame(s)",
                packets,
                bytes as f64 / (1024.0 * 1024.0),
                reading.frames
            );
            if packets == 0 {
                println!(
                    "    WARNING: the NVENC session accepted every picture but returned \
                     no bitstream."
                );
            }
            (reading.fps(), Some(bytes))
        }
        Err(why) => {
            // A printed skip, never a zero row.
            println!("    NVENC pass SKIPPED: {why}");
            (None, None)
        }
    };

    let end_to_end = media_end_to_end_pass(
        path,
        device,
        &mut graph,
        upload_idx,
        &cache,
        &frame_state,
        MEDIA_PASS_FRAMES,
    )?;
    // Per-node timings on a real-media graph too, when asked for: the shape here
    // is one layer rather than four, so the per-pass figures are directly
    // comparable with the Heavy graph's without its multiplicity.
    if let Some(timing_frames) = node_timing_frames() {
        print_node_timings(device, &mut graph, &frame_state, timing_frames, |g, f| {
            media_push_frame(g, upload_idx, &cache, f)
        });
    }

    // The pool's own counters, on a graph shape the six synthetic benchmarks never
    // compile — which is exactly the reading gotcha 14 says to take rather than
    // assume.
    print_pool_stats(graph.pool_stats());

    Ok(MediaReading {
        label: class.label,
        width: cache.width,
        height: cache.height,
        layout: cache.meta.layout,
        decode: decode.fps(),
        render: render.fps(),
        encode,
        end_to_end: end_to_end.fps(),
        bitstream_bytes: bitstream,
    })
}

/// The `--media` profile: generate (or reuse) each class's fixture and run it.
///
/// Returns whether anything was actually measured, so `main` can honour
/// `NEXIR_REQUIRE_MEDIA` — a run where every class skipped must not exit 0 with a
/// clean-looking summary on a machine that is supposed to have the tooling.
fn run_media_profile(device: &Arc<GpuDevice>) -> bool {
    println!("========================================================================");
    println!("REAL-MEDIA PROFILE — coded frames, not synthetic bars");
    println!("  Each class is decoded with this crate's own demuxer/decoder, then");
    println!("  rendered through the same upload + graph the six benchmarks use, then");
    println!("  encoded through a real NVENC session. The four rates are measured in");
    println!("  four passes; none is computed from the others.");
    println!("========================================================================\n");

    let ffmpeg = match bench_media::ffmpeg_binary() {
        Some(p) => p,
        None => {
            println!(
                "SKIPPED: no `ffmpeg` binary on PATH (set NEXIR_FFMPEG to point at one).\n\
                 The FFmpeg DLLs this crate links against do not imply the CLI is\n\
                 installed, and the fixtures are generated with the CLI. No class ran.\n"
            );
            return false;
        }
    };
    println!(
        "ffmpeg      : {}\nFixture dir : {}\n",
        ffmpeg.display(),
        bench_media::fixture_dir().display()
    );

    let mut readings: Vec<MediaReading> = Vec::new();
    let mut skipped: Vec<(&str, String)> = Vec::new();

    for class in bench_media::MEDIA_CLASSES {
        println!(
            ">>> {} — {}x{}@{} ({})",
            class.label, class.width, class.height, class.fps, class.why
        );
        let path = match bench_media::ensure_fixture(&ffmpeg, class) {
            Ok(p) => p,
            Err(why) => {
                println!("    SKIPPED: {why}\n");
                skipped.push((class.label, why));
                continue;
            }
        };
        match run_media_class(device, class, &path) {
            Ok(r) => {
                readings.push(r);
                println!();
            }
            Err(why) => {
                println!("    SKIPPED: {why}\n");
                skipped.push((class.label, why));
            }
        }
    }

    if !readings.is_empty() {
        fn fps(v: Option<f64>) -> String {
            // `n/a` for a pass that did not run. A skipped NVENC session is not an
            // encoder that managed 0 FPS.
            v.map(|x| format!("{x:>8.1}"))
                .unwrap_or_else(|| format!("{:>8}", "n/a"))
        }
        println!("------------------------------------------------------------------------");
        println!("REAL-MEDIA SUMMARY — FPS, one run per class");
        println!("  Decode is demux+decode alone; Render is upload+graph from decoded");
        println!("  frames; Encode adds a real NVENC session; E2E is all of it in one");
        println!("  loop. E2E is MEASURED, not 1/(1/decode + 1/render) — the stages");
        println!("  overlap, and by how much is what the pass reports.");
        println!("  All four passes are serial (one frame in flight), so these are NOT");
        println!("  comparable to the pipelined figures in the summary above; they are");
        println!("  comparable to each other and across classes.");
        println!("------------------------------------------------------------------------");
        println!(
            "{:<16} {:>11} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Class", "Geometry", "Decode", "Render", "Encode", "E2E", "Bitstream"
        );
        for r in &readings {
            println!(
                "{:<16} {:>11} {} {} {} {} {:>10}",
                r.label,
                format!("{}x{}/{}b", r.width, r.height, r.layout.bit_depth),
                fps(r.decode),
                fps(r.render),
                fps(r.encode),
                fps(r.end_to_end),
                r.bitstream_bytes
                    .map(|b| format!("{:.1} MB", b as f64 / (1024.0 * 1024.0)))
                    .unwrap_or_else(|| "n/a".into()),
            );
        }
        println!("------------------------------------------------------------------------\n");
    }

    if !skipped.is_empty() {
        println!("{} class(es) did NOT run:", skipped.len());
        for (label, why) in &skipped {
            println!("  - {label}\n      {why}");
        }
        println!();
    }

    !readings.is_empty()
}

/// How many frames the per-node GPU timing pass measures, or `None` when it is
/// off.
///
/// `NEXIR_NODE_TIMINGS=1` gives the default 30; any other number sets it.
///
/// Behind a flag, and run AFTER the measured loop, because it changes what is
/// being measured: `execute_timed` writes two timestamps around every node, so a
/// 21-node graph gets 42 extra encoder commands per frame and a resolve. Folding
/// it into the main loop would mean the headline FPS came from a run nobody
/// reproduces without the flag.
fn node_timing_frames() -> Option<usize> {
    let raw = std::env::var("NEXIR_NODE_TIMINGS").ok()?;
    match raw.trim() {
        "" | "0" | "false" | "off" => None,
        "1" | "true" | "on" => Some(30),
        n => n.parse::<usize>().ok().filter(|v| *v > 0).or(Some(30)),
    }
}

/// One node's GPU cost, aggregated across the timing frames.
struct NodeTiming {
    name: String,
    /// How many brackets contributed — `frames × instances of this node`.
    samples: usize,
    /// Instances of this node in one frame, e.g. 4 for `YuvUpload` in the Heavy
    /// graph.
    per_frame: usize,
    /// Median of the individual brackets, POOLED across every instance.
    ///
    /// **Only comparable with `frame_total_ms / per_frame` when the instances cost
    /// the same**, which on a multi-source row they need not — see
    /// [`NodeTiming::instance_spread_percent`].
    median_ms: f64,
    /// Median of this node's TOTAL per frame, i.e. all its instances summed.
    ///
    /// The figure that matters for "how much of the frame is this shader": one LUT
    /// pass at 0.4 ms is cheap, four of them are 1.6 ms of a 16.7 ms budget. A
    /// report that printed only the per-bracket median would understate every
    /// repeated node by its own multiplicity.
    frame_total_ms: f64,
    /// Each instance's own median, in execution order within the frame.
    ///
    /// **This is what makes `median_ms` falsifiable.** `median_ms` pools every
    /// bracket of a name, so on a row whose four layers carry different pictures it
    /// is the median of a multi-modal sample and `per_frame × median_ms` understates
    /// the frame total — measured 22-30% low on benchmark 7 (gotcha 27). Keeping the
    /// instances apart is the only way "this shader costs 0.3 ms" and "one of its
    /// four passes costs 0.6" can be told apart.
    instance_ms: Vec<f64>,
}

impl NodeTiming {
    /// How far this node's dearest instance is from its cheapest, as a percentage of
    /// the median instance.
    ///
    /// `None` for a single-instance node, where the question does not arise.
    fn instance_spread_percent(&self) -> Option<f64> {
        if self.instance_ms.len() < 2 {
            return None;
        }
        let mid = median(self.instance_ms.iter().copied());
        if mid <= 0.0 {
            return None;
        }
        let lo = self.instance_ms.iter().copied().fold(f64::MAX, f64::min);
        let hi = self.instance_ms.iter().copied().fold(f64::MIN, f64::max);
        Some((hi - lo) / mid * 100.0)
    }
}

/// Above this, a node's instances are reported individually rather than as one
/// pooled median.
///
/// Read off the measurement rather than chosen: three repeats of the same row
/// reproduce a per-pass median to within 0.2-1.1%
/// (`target/taskH_composite_3x.txt`, `target/taskH_mixed_3x.txt`), so 10% is an
/// order of magnitude outside the row's own noise and cannot fire on it. Benchmark
/// 7's real spread is 90-110%.
const NODE_INSTANCE_SPREAD_WARN: f64 = 10.0;

/// Aggregate one bracketed pass into per-node rows.
///
/// `frames` is one entry per timed frame, each the `(node name, milliseconds)`
/// brackets **in execution order** — which is what makes the per-instance split
/// possible: the Nth occurrence of a name within a frame is the same graph node
/// every frame, because the graph shape is fixed for the pass.
///
/// Split out of [`print_node_timings`] so the aggregation is testable without a
/// GPU. The rule it exists to enforce is gotcha 27's: a pooled median over
/// instances that differ is not a per-pass cost, and the report must say so rather
/// than let the reader multiply it by the count.
fn aggregate_node_timings(frames: &[Vec<(String, f64)>]) -> Vec<NodeTiming> {
    let mut per_bracket: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut per_frame_totals: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut per_instance: std::collections::BTreeMap<String, Vec<Vec<f64>>> = Default::default();
    let mut instances: std::collections::BTreeMap<String, usize> = Default::default();

    for brackets in frames {
        let mut frame_sums: std::collections::BTreeMap<String, f64> = Default::default();
        let mut seen: std::collections::BTreeMap<String, usize> = Default::default();
        for (name, ms) in brackets {
            per_bracket.entry(name.clone()).or_default().push(*ms);
            *frame_sums.entry(name.clone()).or_default() += *ms;
            // Which occurrence of this name within the frame — the instance index.
            let idx = seen.entry(name.clone()).or_insert(0);
            let slots = per_instance.entry(name.clone()).or_default();
            if slots.len() <= *idx {
                slots.resize(*idx + 1, Vec::new());
            }
            slots[*idx].push(*ms);
            *idx += 1;
        }
        for (name, sum) in frame_sums {
            per_frame_totals.entry(name.clone()).or_default().push(sum);
            // Recorded from the frame rather than from the compile-time graph, so a
            // node that failed to bracket is not counted as an instance.
            let n = seen[&name];
            instances
                .entry(name)
                .and_modify(|v| *v = (*v).max(n))
                .or_insert(n);
        }
    }

    let mut rows: Vec<NodeTiming> = per_bracket
        .iter()
        .map(|(name, samples)| NodeTiming {
            name: name.clone(),
            samples: samples.len(),
            per_frame: instances.get(name).copied().unwrap_or(1),
            median_ms: median(samples.iter().copied()),
            frame_total_ms: median(
                per_frame_totals
                    .get(name)
                    .map(|v| v.iter().copied())
                    .unwrap_or_default(),
            ),
            instance_ms: per_instance
                .get(name)
                .map(|slots| {
                    slots
                        .iter()
                        .map(|s| median(s.iter().copied()))
                        .collect::<Vec<f64>>()
                })
                .unwrap_or_default(),
        })
        .collect();
    // Costliest first: the point of this table is what to look at next.
    rows.sort_by(|a, b| {
        b.frame_total_ms
            .partial_cmp(&a.frame_total_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

/// Bracket every node for `frames` frames and print what each shader cost.
///
/// Aggregated by node NAME, so the Heavy graph's four LUT passes report as one row
/// with a count rather than four indistinguishable rows.
///
/// `push` uploads one frame's pixels; it is a closure because the two profiles
/// feed their upload nodes differently and this pass must drive whichever one the
/// caller built.
fn print_node_timings(
    device: &Arc<GpuDevice>,
    graph: &mut nexir::render::graph::CompiledGraph,
    frame_state: &FrameState,
    frames: usize,
    mut push: impl FnMut(&mut nexir::render::graph::CompiledGraph, usize),
) {
    if !device.has_timestamp_queries {
        println!(
            "    Per-node GPU timings SKIPPED: this adapter does not report \
             TIMESTAMP_QUERY, so\n    there is nothing to read. Never a zero row."
        );
        return;
    }

    let node_count = graph.execution_order().len();
    // One bracket per node. `QUERY_SET_MAX_QUERIES` is 8192, so the Heavy graph's
    // 21 nodes (42 queries) is not close to the limit — but assert rather than
    // assume, because an undersized timer drops brackets silently and
    // `execute_timed` would then return fewer names than nodes.
    assert!(
        node_count as u32 * 2 <= wgpu::QUERY_SET_MAX_QUERIES,
        "{node_count} nodes need {} queries, over the {} limit",
        node_count * 2,
        wgpu::QUERY_SET_MAX_QUERIES
    );
    let mut timer = nexir::render::gpu_timer::GpuTimer::new(device, node_count as u32);

    // One entry per timed frame: its brackets in execution order. Aggregated
    // afterwards by `aggregate_node_timings`, which is where the per-instance split
    // lives — see gotcha 27.
    let mut collected: Vec<Vec<(String, f64)>> = Vec::with_capacity(frames);
    let mut incomplete = 0usize;

    for f in 0..frames {
        push(graph, f);
        let mut enc = device.begin_frame();
        // MUST reset first: the timer writes from index 0 and `written` is what
        // decides whether a bracket fitted.
        timer.reset();
        let names = graph.execute_timed(&mut enc, device, frame_state, &mut timer);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));

        // The names `execute_timed` returns are the nodes it BRACKETED, in
        // execution order. Cross-checked against the graph's own order rather than
        // trusted: `node_name(idx)` rebuilds the same mapping independently, and if
        // the two ever disagree every row below names the wrong shader.
        let expected: Vec<&str> = graph
            .execution_order()
            .iter()
            .map(|&i| graph.node_name(i))
            .collect();
        if names.len() != expected.len() {
            incomplete += 1;
        }
        assert_eq!(
            names.as_slice(),
            &expected[..names.len()],
            "execute_timed's names disagree with execution_order()'s — every \
             per-node figure would be attributed to the wrong shader"
        );

        let Some(pairs) = timer.resolve_all() else {
            continue;
        };
        collected.push(
            names
                .iter()
                .zip(pairs.iter())
                .map(|(name, ns)| ((*name).to_string(), ns / 1.0e6))
                .collect(),
        );
    }

    let rows = aggregate_node_timings(&collected);
    if rows.is_empty() {
        println!(
            "    Per-node GPU timings: no bracket resolved over {frames} frame(s) — \
             nothing measured, so nothing printed."
        );
        return;
    }

    let graph_total: f64 = rows.iter().map(|r| r.frame_total_ms).sum();
    println!(
        "    Per-node GPU timings over {frames} frame(s) — medians, GPU execution \
         only:"
    );
    println!(
        "      {:<18} {:>5} {:>11} {:>12} {:>7}",
        "Node", "×/fr", "per pass", "per frame", "share"
    );
    // Nodes whose instances disagree by more than the row's own noise. Collected
    // while printing and expanded below, because the pooled `per pass` median is
    // NOT `per frame ÷ ×/fr` for these and a reader who multiplies is off by the
    // spread — 22-30% on benchmark 7 (gotcha 27).
    let mut uneven: Vec<&NodeTiming> = Vec::new();
    for r in &rows {
        let spread = r.instance_spread_percent();
        println!(
            "      {:<18} {:>5} {:>8.3} ms {:>9.3} ms {:>6.1}%{}",
            r.name,
            r.per_frame,
            r.median_ms,
            r.frame_total_ms,
            if graph_total > 0.0 {
                r.frame_total_ms / graph_total * 100.0
            } else {
                0.0
            },
            match spread {
                Some(s) if s > NODE_INSTANCE_SPREAD_WARN => "  UNEVEN",
                _ => "",
            },
        );
        if spread.map(|s| s > NODE_INSTANCE_SPREAD_WARN).unwrap_or(false) {
            uneven.push(r);
        }
        debug_assert!(r.samples >= r.per_frame);
    }
    println!(
        "      {:<18} {:>5} {:>11} {:>9.3} ms {:>6.1}%",
        "TOTAL", "", "", graph_total, 100.0
    );
    if !uneven.is_empty() {
        println!(
            "    UNEVEN: these nodes' instances do NOT cost the same, so `per pass` is a\n    \
             pooled median and `×/fr × per pass` UNDERSTATES `per frame`. Read the\n    \
             per-frame column, or the per-instance medians below (execution order):"
        );
        for r in &uneven {
            let each: Vec<String> = r.instance_ms.iter().map(|ms| format!("{ms:.3}")).collect();
            println!(
                "      {:<18} {} ms  (spread {:.0}% of the median instance)",
                r.name,
                each.join(" / "),
                r.instance_spread_percent().unwrap_or(0.0),
            );
        }
    }
    if incomplete > 0 {
        // A dropped bracket means the timer was too small for the graph, which
        // would otherwise show up only as a missing row.
        println!(
            "    WARNING: {incomplete} frame(s) bracketed fewer nodes than the graph \
             has — the rows above\n    are incomplete and the TOTAL is a lower bound."
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK F2 — THE END-TO-END EXPORT BENCHMARK (`bench --export`)
//
// Input MP4 → demux → decode → upload → graph → NV12 → NVENC → mux → output MP4,
// driving the REAL `ExportEngine`. Not a reimplementation of it: the engine
// already wires the decode workers, the renderer's in-flight pipeline, the encoder
// and the muxer, and a benchmark that rebuilt that would be measuring a pipeline
// no user has.
//
// Reported: input duration, export wall time, the realtime multiplier, average
// FPS, peak process RSS, and the output's bitrate counted from its own size and
// duration.
//
// VERIFIED, not assumed. A "fast export" that wrote a broken file would otherwise
// score best: the output is reopened with this crate's own demuxer/decoder and the
// benchmark fails the row unless the frames decode and the count is right. That is
// the same contract `src/tests/export_validation.rs` holds the engine to.
// ─────────────────────────────────────────────────────────────────────────────

/// How long to wait for an export before calling it wedged.
///
/// Generous — a 4K60 clip through a real encoder is not fast — but bounded, so a
/// stuck encoder is a printed failure rather than a benchmark that never returns.
const EXPORT_TIMEOUT: Duration = Duration::from_secs(300);

/// What one class's export measured, all of it counted or timed.
struct ExportReading {
    label: &'static str,
    width: u32,
    height: u32,
    /// Frames the job asked for.
    frames_requested: usize,
    /// Frames the OUTPUT file actually decodes to. The two must agree.
    frames_decoded: usize,
    input_seconds: f64,
    export_seconds: f64,
    output_bytes: u64,
    /// Peak RSS observed while the export ran, from the OS. `None` where the query
    /// is unavailable.
    peak_rss_bytes: Option<u64>,
    /// Whether the engine's own backend probe said NVENC. Reported rather than
    /// assumed: `ExportEngine` falls back to FFmpeg silently, and an export timed
    /// without knowing which encoder ran is not comparable to one that does.
    nvenc: bool,
}

impl ExportReading {
    /// Frames divided by elapsed seconds — the only FPS this row claims.
    fn fps(&self) -> Option<f64> {
        if self.frames_decoded == 0 || self.export_seconds <= 0.0 {
            None
        } else {
            Some(self.frames_decoded as f64 / self.export_seconds)
        }
    }

    /// Output seconds produced per second of wall time.
    fn realtime(&self) -> Option<f64> {
        if self.export_seconds <= 0.0 || self.input_seconds <= 0.0 {
            None
        } else {
            Some(self.input_seconds / self.export_seconds)
        }
    }

    /// Bits per second, counted from the file's size and its own duration — not
    /// from the job's requested quality, which is a setting rather than a result.
    fn bitrate_bps(&self) -> Option<f64> {
        if self.input_seconds <= 0.0 || self.output_bytes == 0 {
            None
        } else {
            Some(self.output_bytes as f64 * 8.0 / self.input_seconds)
        }
    }
}

/// One class's repeats, aggregated the way the synthetic sweep aggregates its own:
/// median with an explicit spread.
///
/// A separate type rather than medians computed inline at the print site, because
/// every figure here has to be reduced the same way and the per-repeat `Option`s
/// have to survive the reduction — a class whose bitrate is unavailable must print
/// `n/a` after three repeats just as it does after one (gotcha 9), not `0.0`
/// because `unwrap_or(0.0)` crept into a median.
struct ExportRow {
    /// At least one; a class with no successful repeat is not pushed.
    runs: Vec<ExportReading>,
}

impl ExportRow {
    /// The first repeat, for the fields that are properties of the job rather than
    /// of the run: label, geometry, frame count, encoder, input duration.
    ///
    /// Those are identical across repeats by construction — the same class, the
    /// same file, the same job — and `verify_exported_file` has already failed the
    /// class if a repeat's frame count differed.
    fn first(&self) -> &ExportReading {
        &self.runs[0]
    }

    fn median_seconds(&self) -> f64 {
        median(self.runs.iter().map(|r| r.export_seconds))
    }

    /// Median of the per-repeat FPS figures, or `None` when no repeat produced one.
    ///
    /// Median of the rates rather than frames ÷ median time: the two differ, and
    /// this one is a median of things that were each measured.
    fn median_fps(&self) -> Option<f64> {
        let v: Vec<f64> = self.runs.iter().filter_map(|r| r.fps()).collect();
        (!v.is_empty()).then(|| median(v.into_iter()))
    }

    fn median_realtime(&self) -> Option<f64> {
        let v: Vec<f64> = self.runs.iter().filter_map(|r| r.realtime()).collect();
        (!v.is_empty()).then(|| median(v.into_iter()))
    }

    fn median_bitrate_bps(&self) -> Option<f64> {
        let v: Vec<f64> = self.runs.iter().filter_map(|r| r.bitrate_bps()).collect();
        (!v.is_empty()).then(|| median(v.into_iter()))
    }

    /// `(max - min) / median` over the repeats' FPS, as a percentage.
    ///
    /// The figure that says whether the median above is a result or a sample. One
    /// repeat has no spread to report, so this is `None` rather than 0% — a single
    /// run is not a run with zero variance.
    fn fps_spread_percent(&self) -> Option<f64> {
        if self.runs.len() < 2 {
            return None;
        }
        let v: Vec<f64> = self.runs.iter().filter_map(|r| r.fps()).collect();
        if v.len() < 2 {
            return None;
        }
        let med = median(v.iter().copied());
        if med <= 0.0 {
            return None;
        }
        let lo = v.iter().copied().fold(f64::MAX, f64::min);
        let hi = v.iter().copied().fold(f64::MIN, f64::max);
        Some((hi - lo) / med * 100.0)
    }

    /// The highest RSS peak any repeat observed, or `None` where the OS query is
    /// unavailable.
    ///
    /// A max rather than a median: the question a peak answers is "how much did
    /// this need at worst", and a median of high-water marks answers neither that
    /// nor anything else.
    fn peak_rss_bytes(&self) -> Option<u64> {
        self.runs.iter().filter_map(|r| r.peak_rss_bytes).max()
    }
}

/// Everything `ExportEngine::new` needs, built around one real input file.
///
/// Mirrors `src/tests/export_validation.rs`'s harness rather than inventing a
/// second way to assemble a project: the source registry, the track, the clip and
/// the `FrameScheduler` are wired exactly as a real project is, so the export
/// under measurement is the one users run.
///
/// Shared with the interop profile (`--interop`, Task G2e), which needs the same
/// assembly with one field different — the `IoLayer`'s `InteropDecodeTargets`. A
/// second copy of this function is how the two profiles would drift into
/// measuring differently-wired pipelines and then having their numbers compared.
struct ClipHarness {
    device: Arc<GpuDevice>,
    scheduler: Arc<nexir::scheduler::frame_scheduler::FrameScheduler>,
    timeline: Arc<std::sync::RwLock<nexir::timeline::store::TimelineStore>>,
    tracks: Arc<std::sync::RwLock<nexir::timeline::track::TrackList>>,
    sources: Arc<std::sync::RwLock<nexir::timeline::source::SourceRegistry>>,
    shaders: Arc<ShaderRegistry>,
    compute: Arc<ComputePipelineCache>,
    /// Kept alive so the prefetch worker's channel does not disconnect, and set on
    /// drop so the worker actually stops — see [`ClipHarness::drop`].
    shutdown: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for ClipHarness {
    /// Stop this harness's prefetch worker.
    ///
    /// Not tidiness: `PrefetchWorker::run` polls its channel with a 500 µs sleep
    /// and calls `decode_blocking` on whatever it is handed, so a harness left
    /// undropped-but-idle keeps a thread and a decoder alive for the rest of the
    /// process. One harness is built per class per repeat, so without this the
    /// fifth class of the third repeat is measured against fourteen surviving
    /// workers — the later rows would be slower for a reason that is the
    /// benchmark's own bookkeeping rather than the pipeline's.
    fn drop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Build a one-clip project over `path`, reading its geometry from the file.
///
/// A thin wrapper over [`build_multi_clip_harness`] with one source, so the
/// single-clip callers (`--export`, the single-source `--interop` rows) and the
/// four-source ones assemble the SAME pipeline. A second copy of the assembly is
/// how the two would drift into measuring differently-wired engines and then having
/// their numbers compared.
fn build_clip_harness(
    device: &Arc<GpuDevice>,
    path: &Path,
    interop: Arc<nexir::io::interop_decode::InteropDecodeTargets>,
) -> Result<(ClipHarness, nexir::io::demuxer::StreamInfo, i64), String> {
    let (harness, mut streams, duration_pts) =
        build_multi_clip_harness(device, std::slice::from_ref(&path), interop)?;
    Ok((harness, streams.remove(0), duration_pts))
}

/// Build an N-clip project over N files, one video track each.
///
/// Geometry, frame rate and duration come from the containers rather than from the
/// `MediaClass` declarations. The two should agree — `declared_geometry_matches_the_filter`
/// pins that — but the measurement is judged against the files, so the files are
/// what the project is built from.
///
/// **N tracks rather than N clips on one track**, because `build_islands` groups by
/// track and `query_active` needs the clips to overlap in time: four clips on one
/// track are four *consecutive* clips and only one is ever active, which would
/// silently make a four-source row a one-source row.
///
/// Returns the streams in the order the paths were given and the SHORTEST duration
/// across them: a span longer than the shortest source would have that source run
/// dry first, and a frame missing one of its four clips is not the workload the row
/// claims.
///
/// `interop` is the `IoLayer`'s decode-target registry, and it is a PARAMETER
/// rather than a constant because it is the one thing the callers disagree about,
/// and each of them is right:
///
/// * `--export` passes `disabled`. `ExportDecodeWorker` decodes 16 frames ahead
///   into the CPU `FrameCache` and a one-frame-per-source interop target cannot
///   serve a lookahead (settled — see the plan's table). Stated in code rather
///   than left to whether this host happens to have CUDA.
/// * `--interop` passes a live registry for its interop arm and `disabled` for its
///   CPU arm, because measuring the difference between exactly those two is the
///   entire point of that profile.
fn build_multi_clip_harness(
    device: &Arc<GpuDevice>,
    paths: &[&Path],
    interop: Arc<nexir::io::interop_decode::InteropDecodeTargets>,
) -> Result<(ClipHarness, Vec<nexir::io::demuxer::StreamInfo>, i64), String> {
    use nexir::io::frame_cache::FrameCache;
    use nexir::io::io_layer::IoLayer;
    use nexir::io::slot_pool::FrameSlotPool;
    use nexir::project::Project;
    use nexir::scheduler::frame_scheduler::FrameScheduler;
    use nexir::timeline::mutation::ClipInsertParams;
    use nexir::timeline::source::{PixelFormat, VideoRotation, VideoStreamInfo};

    if paths.is_empty() {
        return Err("a harness needs at least one source".to_string());
    }
    let tb = Rational::TIMEBASE_90K;

    // Read every container first, so a mismatched set fails before anything is
    // allocated.
    let mut streams: Vec<nexir::io::demuxer::StreamInfo> = Vec::with_capacity(paths.len());
    let mut durations: Vec<i64> = Vec::with_capacity(paths.len());
    for path in paths {
        let demuxer = Demuxer::open(path).map_err(|e| format!("Demuxer::open failed: {e:?}"))?;
        let stream = demuxer
            .video_stream
            .clone()
            .ok_or_else(|| format!("{} has no video stream", path.display()))?;
        let duration_pts = tb.from_pts(stream.duration, stream.time_base);
        if duration_pts <= 0 {
            return Err(format!(
                "{} reports a {duration_pts}-tick duration, so there is no span to \
                 measure",
                path.display()
            ));
        }
        // `Demuxer` owns FFmpeg pointers and implements `Drop`; the `StreamInfo`
        // clone carries the `codecpar` the decoder needs, and the engine reopens the
        // file itself, so the demuxer is dropped here rather than held.
        drop(demuxer);
        streams.push(stream);
        durations.push(duration_pts);
    }

    let width = streams[0].width.ok_or("the first stream has no width")?;
    let height = streams[0].height.ok_or("the first stream has no height")?;
    // One canvas, so mixed geometries would have the composite scale some clips and
    // not others — a different workload, and one no row here claims to measure.
    for (i, s) in streams.iter().enumerate() {
        if (s.width, s.height) != (Some(width), Some(height)) {
            return Err(format!(
                "source {i} is {:?}x{:?} but source 0 is {width}x{height}; a \
                 multi-source row must be one geometry or the composite is scaling \
                 some clips",
                s.width, s.height
            ));
        }
    }
    let duration_pts = *durations.iter().min().expect("non-empty");

    // Assembled through `Project` rather than by poking `TimelineStore` directly:
    // that is the API the UI uses, it owns the layer-order bookkeeping, and
    // `TrackId`'s constructor is crate-private anyway. The three halves are then
    // taken out of it because `ExportEngine::new` wants each behind its own lock.
    let mut project = Project::new("bench-export");
    for (i, (path, stream)) in paths.iter().zip(&streams).enumerate() {
        let track_id = project
            .add_video_track(format!("V{}", i + 1))
            .map_err(|e| format!("could not add a video track: {e:?}"))?;
        let source_id = project.register_source(
            path.to_path_buf(),
            Some(VideoStreamInfo {
                width,
                height,
                frame_rate: stream.frame_rate.unwrap_or(Rational { num: 30, den: 1 }),
                pixel_fmt: if stream.color_info.bit_depth >= 10 {
                    PixelFormat::P010
                } else {
                    PixelFormat::Yuv420p
                },
                color_info: stream.color_info,
                duration_pts,
                is_vfr: stream.is_vfr,
                time_base: stream.time_base,
                rotation: VideoRotation::None,
            }),
            None,
        );
        project
            .insert_clip(ClipInsertParams {
                track_id,
                source_id,
                kind: ClipKind::Video,
                pts_in: 0,
                pts_out: duration_pts,
                ..Default::default()
            })
            .map_err(|e| format!("could not insert the clip: {e:?}"))?;
    }

    let sources = Arc::clone(&project.sources);
    let tracks = Arc::new(std::sync::RwLock::new(std::mem::take(&mut project.tracks)));
    let timeline = Arc::new(std::sync::RwLock::new(std::mem::replace(
        &mut project.clips,
        nexir::timeline::store::TimelineStore::new(),
    )));

    let pool = Arc::new(FrameSlotPool::new(device));
    let cache = Arc::new(FrameCache::new(Arc::clone(&pool), 32));
    let (prefetch_tx, prefetch_rx) = std::sync::mpsc::sync_channel(16);
    let io_layer = Arc::new(IoLayer::new(
        Arc::clone(&device.device),
        Arc::clone(&pool),
        Arc::clone(&cache),
        Arc::clone(&sources),
        prefetch_tx,
        tb,
        // Chosen by the CALLER — see this function's doc comment. `--export` is on
        // the CPU upload path by design; `--interop` is the profile that measures
        // the other one.
        interop,
    ));
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let _ = nexir::io::prefetch::spawn_prefetch_worker(nexir::io::prefetch::PrefetchWorker::new(
        prefetch_rx,
        Arc::clone(&io_layer),
        Arc::clone(&cache),
        Arc::clone(&shutdown),
    ));

    let scheduler = Arc::new(FrameScheduler::new(io_layer, width, height));
    let shaders = Arc::new(
        ShaderRegistry::compile_all(device)
            .map_err(|e| format!("shader compilation failed: {e:?}"))?,
    );

    Ok((
        ClipHarness {
            device: Arc::clone(device),
            scheduler,
            timeline,
            tracks,
            sources,
            shaders,
            compute: Arc::new(ComputePipelineCache::new()),
            shutdown,
        },
        streams,
        duration_pts,
    ))
}

/// Decode the exported file and count its frames.
///
/// This is what stops a broken-but-fast export from scoring well. `Err` means the
/// output does not decode, which is a failure of the export rather than a property
/// of the host — so the row is reported as failed rather than skipped.
///
/// Software decode (`open_sw`): the question is whether the FILE is well formed,
/// and a hardware decoder's tolerance for a malformed bitstream is its own
/// property.
fn verify_exported_file(path: &Path, expect_frames: usize) -> Result<usize, String> {
    let mut demuxer =
        Demuxer::open(path).map_err(|e| format!("the exported file will not open: {e:?}"))?;
    let stream = demuxer
        .video_stream
        .clone()
        .ok_or_else(|| "the exported file has no video stream".to_string())?;
    let w = stream.width.ok_or("the exported stream has no width")? as usize;
    let h = stream.height.ok_or("the exported stream has no height")? as usize;
    let mut decoder = Decoder::open_sw(&stream, stream.codecpar)
        .map_err(|e| format!("the exported file's codec will not open: {e:?}"))?;

    let mut buf = vec![0u8; w * h * 6 + 256];
    let mut frames = 0usize;
    loop {
        match demuxer.next_video_packet() {
            Ok(Some(pkt)) => match decoder.decode_into(&pkt, &mut buf, None) {
                Ok(Some(_)) => frames += 1,
                Ok(None) => continue,
                Err(e) => return Err(format!("the exported file failed to decode: {e:?}")),
            },
            Ok(None) => break,
            Err(e) => return Err(format!("demuxing the exported file failed: {e:?}")),
        }
    }
    // Frame-level threading holds frames back; without the drain a short export
    // decodes to almost nothing and the count assertion below would fire on a
    // perfectly good file.
    loop {
        match decoder.drain_into(&mut buf) {
            Ok(Some(_)) => frames += 1,
            Ok(None) => break,
            Err(e) => return Err(format!("draining the exported file failed: {e:?}")),
        }
    }

    if frames == 0 {
        return Err("the exported file decoded to zero frames".to_string());
    }
    // A tolerance of one, for the same reason `media_compat` allows one: an open
    // GOP can legitimately shift a leading frame. Anything wider than that is
    // frames going missing.
    if frames + 1 < expect_frames || frames > expect_frames + 1 {
        return Err(format!(
            "the exported file decoded to {frames} frame(s) but the job asked for \
             {expect_frames} — frames were lost, so the export time above is for a \
             different amount of work"
        ));
    }
    Ok(frames)
}

/// Export one class end-to-end through the real `ExportEngine`, then verify the
/// file it wrote.
fn run_export_class(
    device: &Arc<GpuDevice>,
    class: &'static MediaClass,
    input: &Path,
) -> Result<ExportReading, String> {
    use nexir::export::engine::ExportEngine;
    use nexir::export::progress::ExportPhase;

    let (harness, stream, duration_pts) = build_clip_harness(
        device,
        input,
        // The export profile is deliberately on the CPU upload path: the engine's
        // decode workers fill the CPU `FrameCache` 16 frames ahead of the render
        // cursor, and a one-frame-per-source interop target cannot serve a
        // lookahead. See `build_clip_harness`.
        Arc::new(nexir::io::interop_decode::InteropDecodeTargets::disabled(
            Arc::clone(device),
        )),
    )?;
    let tb = Rational::TIMEBASE_90K;
    let fps = stream.frame_rate.unwrap_or(Rational { num: 30, den: 1 });
    let out_path = bench_media::fixture_dir().join(format!("{}_export.mp4", class.label));
    let _ = std::fs::remove_file(&out_path);

    let mut job = nvenc_job_for(
        stream.width.unwrap_or(1920),
        stream.height.unwrap_or(1080),
        fps.num as f64 / fps.den.max(1) as f64,
        1,
    );
    job.output_path = out_path.clone();
    job.pts_in = 0;
    job.pts_out = duration_pts;
    job.frame_rate = fps;
    job.project_tb = tb;
    job.validate()
        .map_err(|e| format!("the bench's own ExportJob is invalid: {e:?}"))?;
    let frames_requested = job.total_frames();

    let capability = InteropCapability::probe(device);
    let cuda_ctx = if capability.is_available() {
        // One context per process — see `shared_cuda_ctx`. A per-repeat context was
        // a retain/release pair per repeat against a resource the whole process
        // shares.
        let ctx = shared_cuda_ctx(&capability);
        if ctx.is_none() {
            // Not fatal: the engine falls back to the FFmpeg encoder, and an
            // export that used it is still a real export. What must not happen
            // is reporting the row as if NVENC ran.
            println!("    CUDA context unavailable — the engine will use FFmpeg");
        }
        ctx
    } else {
        println!(
            "    CUDA interop unavailable (transport={:?}) — the engine will use FFmpeg",
            capability.transport
        );
        None
    };

    // Ask which backend `select` picks BEFORE starting, using the engine's own
    // inputs: the engine consumes the backend on its dispatch thread, so afterwards
    // there is nothing left to probe. Reported rather than assumed — an export timed
    // without knowing which encoder ran is not comparable to one that does.
    let nvenc = match nexir::export::video_encoder::VideoEncoderBackend::select(
        &job,
        &capability,
        cuda_ctx.as_ref(),
        device,
    ) {
        Ok(b) => {
            let is_gpu = matches!(
                b,
                nexir::export::video_encoder::VideoEncoderBackend::CudaNvenc { .. }
            );
            // Release the probe's session before the engine opens its own: NVENC
            // allows only a few concurrent sessions.
            drop(b);
            is_gpu
        }
        Err(e) => {
            println!("    backend probe failed ({e:?}) — reporting the row as non-NVENC");
            false
        }
    };

    let engine = ExportEngine::new(
        Arc::clone(&harness.device),
        job,
        Arc::clone(&harness.scheduler),
        Arc::clone(&harness.timeline),
        Arc::clone(&harness.tracks),
        Arc::clone(&harness.sources),
        capability,
        cuda_ctx,
        false,
    );

    // The clock starts at `start`, which is where the engine's own threads begin —
    // so the measured span covers demux, decode, upload, graph, NV12, encode and
    // mux, i.e. everything a user waits for.
    let started = Instant::now();
    let rx = engine
        .start(Arc::clone(&harness.shaders), Arc::clone(&harness.compute))
        .map_err(|e| format!("ExportEngine::start failed: {e:?}"))?;

    // Peak RSS is sampled while the export runs, not read once at the end: the
    // interesting figure is the high-water mark, and by the time the export is done
    // the renderer's buffers may already be freed. `None` where the OS query is
    // unavailable — never a zero.
    let mut peak_rss = nexir::profiling::sysinfo::process_rss_bytes();
    let deadline = started + EXPORT_TIMEOUT;
    let mut last = ExportPhase::Rendering;
    let terminal = loop {
        if Instant::now() >= deadline {
            return Err(format!(
                "the export did not finish within {EXPORT_TIMEOUT:?} (last phase: \
                 {last:?})"
            ));
        }
        match rx.try_recv() {
            Some(update) => {
                last = update.phase.clone();
                if let Some(rss) = nexir::profiling::sysinfo::process_rss_bytes() {
                    peak_rss = Some(peak_rss.map_or(rss, |p: u64| p.max(rss)));
                }
                match update.phase {
                    ExportPhase::Done => break ExportPhase::Done,
                    ExportPhase::Cancelled | ExportPhase::Failed(_) => break last.clone(),
                    _ => {}
                }
            }
            None => std::thread::sleep(Duration::from_millis(5)),
        }
    };
    let export_seconds = started.elapsed().as_secs_f64();
    if terminal != ExportPhase::Done {
        return Err(format!("the export ended in {terminal:?} rather than Done"));
    }

    let output_bytes = std::fs::metadata(&out_path)
        .map_err(|e| format!("the export reported Done but its output is unreadable: {e}"))?
        .len();
    // Verified before any figure is reported: a fast export that wrote a broken
    // file must fail rather than score well.
    let frames_decoded = verify_exported_file(&out_path, frames_requested)?;

    Ok(ExportReading {
        label: class.label,
        width: stream.width.unwrap_or(0),
        height: stream.height.unwrap_or(0),
        frames_requested,
        frames_decoded,
        input_seconds: duration_pts as f64 / tb.den as f64,
        export_seconds,
        output_bytes,
        peak_rss_bytes: peak_rss,
        nvenc,
    })
}

/// The `--export` profile: export every media class end-to-end and verify each
/// output.
///
/// Returns whether anything ran, for `NEXIR_REQUIRE_MEDIA`.
fn run_export_profile(device: &Arc<GpuDevice>) -> bool {
    println!("========================================================================");
    println!("END-TO-END EXPORT PROFILE — the real ExportEngine");
    println!("  Input MP4 -> demux -> decode -> upload -> graph -> NV12 -> encoder ->");
    println!("  mux -> output MP4, driven through ExportEngine::start rather than a");
    println!("  reimplementation of it. Every output is then reopened and decoded:");
    println!("  a fast export that wrote a broken file FAILS instead of scoring well.");
    println!("========================================================================\n");

    let ffmpeg = match bench_media::ffmpeg_binary() {
        Some(p) => p,
        None => {
            println!(
                "SKIPPED: no `ffmpeg` binary on PATH (set NEXIR_FFMPEG). The inputs are\n\
                 generated with the CLI, so no export ran.\n"
            );
            return false;
        }
    };

    // Same knob and same default as the synthetic sweep, deliberately: an export
    // row is a wall-clock time on a shared machine, so it carries the same
    // run-to-run spread the 4K synthetic row does and a single reading of it is a
    // sample rather than a result. `NEXIR_BENCH_REPEATS=1` is the smoke path and
    // says so in the header, in the same words the synthetic path uses.
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
        println!("Repeats: {repeats} per class; the summary reports the median and spread.\n");
    }

    let mut rows: Vec<ExportRow> = Vec::new();
    let mut skipped: Vec<(&str, String)> = Vec::new();

    for class in bench_media::MEDIA_CLASSES {
        println!(
            ">>> {} — {}x{}@{} ({})",
            class.label, class.width, class.height, class.fps, class.why
        );
        let input = match bench_media::ensure_fixture(&ffmpeg, class) {
            Ok(p) => p,
            Err(why) => {
                println!("    SKIPPED: {why}\n");
                skipped.push((class.label, why));
                continue;
            }
        };

        let mut runs: Vec<ExportReading> = Vec::with_capacity(repeats);
        for r in 0..repeats {
            if repeats > 1 {
                println!("    --- repeat {}/{} ---", r + 1, repeats);
            }
            match run_export_class(device, class, &input) {
                Ok(reading) => {
                    println!(
                        "    Exported {} frame(s) in {:.2} s; the output decodes to {} \
                         frame(s), {:.1} MB",
                        reading.frames_requested,
                        reading.export_seconds,
                        reading.frames_decoded,
                        reading.output_bytes as f64 / (1024.0 * 1024.0),
                    );
                    runs.push(reading);
                }
                Err(why) => {
                    // A failed repeat fails the whole class rather than being
                    // dropped from the median: the failures this can return are
                    // "the file does not decode" and "frames went missing", and a
                    // median over the repeats that happened to succeed would
                    // report a rate for a pipeline that is not reliably producing
                    // the file.
                    println!("    FAILED: {why}\n");
                    skipped.push((
                        class.label,
                        if runs.is_empty() {
                            why
                        } else {
                            format!("{why} (after {} repeat(s) had succeeded)", runs.len())
                        },
                    ));
                    runs.clear();
                    break;
                }
            }
        }
        if !runs.is_empty() {
            println!();
            rows.push(ExportRow { runs });
        }
    }

    if !rows.is_empty() {
        println!("------------------------------------------------------------------------");
        println!("EXPORT SUMMARY — median of {repeats} run(s) per class, with spread");
        println!("  Realtime is output seconds per second of wall time; >1.0 x means the");
        println!("  export is faster than playback. FPS is the DECODED frame count over");
        println!("  the same wall time, so a row can only score well on a file that");
        println!("  actually decoded. Spread is (max-min)/median over the repeats:");
        println!("  anything above ~10% means one run's number is not a result on its");
        println!("  own. Bitrate is counted from the output's own size and duration, not");
        println!("  from the requested CRF. RSS is the highest peak sampled across the");
        println!("  repeats.");
        println!("------------------------------------------------------------------------");
        println!(
            "{:<16} {:>11} {:>6} {:>8} {:>9} {:>8} {:>7} {:>9} {:>9}",
            "Class", "Geometry", "Enc", "Frames", "Export", "FPS", "spread", "Realtime", "Bitrate"
        );
        for row in &rows {
            let first = row.first();
            println!(
                "{:<16} {:>11} {:>6} {:>8} {:>7.2} s {:>8} {:>7} {:>8} {:>9}",
                first.label,
                format!("{}x{}", first.width, first.height),
                if first.nvenc { "NVENC" } else { "cpu" },
                first.frames_decoded,
                row.median_seconds(),
                row.median_fps()
                    .map(|f| format!("{f:.1}"))
                    .unwrap_or_else(|| "n/a".into()),
                // `n/a` rather than `0%` on a single repeat: one run is not a run
                // with zero variance, and printing 0% there would read as the
                // strongest possible claim about repeatability from the weakest
                // possible evidence.
                row.fps_spread_percent()
                    .map(|s| format!("{s:.0}%"))
                    .unwrap_or_else(|| "n/a".into()),
                row.median_realtime()
                    .map(|x| format!("{x:.2} x"))
                    .unwrap_or_else(|| "n/a".into()),
                row.median_bitrate_bps()
                    .map(|b| format!("{:.1} Mb/s", b / 1.0e6))
                    .unwrap_or_else(|| "n/a".into()),
            );
        }
        // RSS on its own line: it is a process-wide figure, so it belongs beside the
        // rows rather than inside one, and it is `n/a` where the OS query is absent.
        for row in &rows {
            let first = row.first();
            println!(
                "  {:<16} input {:.2} s, {} run(s), peak RSS {}",
                first.label,
                first.input_seconds,
                row.runs.len(),
                row.peak_rss_bytes()
                    .map(|b| format!("{:.1} MB", b as f64 / (1024.0 * 1024.0)))
                    .unwrap_or_else(|| "n/a".into()),
            );
        }
        println!("------------------------------------------------------------------------\n");
    }

    if !skipped.is_empty() {
        println!("{} class(es) did NOT produce a verified export:", skipped.len());
        for (label, why) in &skipped {
            println!("  - {label}\n      {why}");
        }
        println!();
    }

    !rows.is_empty()
}

// ─────────────────────────────────────────────────────────────────────────────
// TASK G2e — MEASURING THE INTEROP DECODE PATH (`bench --interop`)
//
// G2b/G2c/G2d built the path where NVDEC decodes straight into the textures the
// render graph binds. Nothing had measured it: `--media` builds a `YuvUploadNode`
// unconditionally from a host-memory cache and constructs no `IoLayer` at all, so
// no run of it can touch the interop path, and the six synthetic benchmarks upload
// `make_nv12` bytes by design.
//
// So this profile drives real files through `FrameScheduler::schedule_frame` — the
// production entry point — and runs each class TWICE from one binary:
//
//   interop  IoLayer holding `InteropDecodeTargets::with_context`
//   cpu      the same harness, the same loop, `InteropDecodeTargets::disabled`
//
// The two arms differ by that one argument, which is what makes the comparison a
// measurement rather than a git checkout: same scheduler, same graph builder, same
// serial loop, same file, interleaved in one process.
//
// SERIAL, one frame in flight, and that is a property of the path rather than a
// simplification. A target holds one frame per source, so the next decode waits on
// the submission that last read it (`InteropDecodeTargets::decode_into_target`) —
// a deeper pipeline would spend its lookahead inside that wait. The CPU arm is
// serial too, so the arms stay comparable, and the figures here are comparable to
// `--media`'s serial rows rather than to the pipelined synthetic sweep.
//
// WHY THERE IS NO `GPU transfer` ROW HERE. That row is the gap between consecutive
// frames' GPU timestamps (settled: `write_buffer`'s copies live in wgpu's own
// `pending_writes` command buffer, which this process never encodes). On a serial
// loop that gap also contains the CPU's decode, so printing it as a transfer figure
// would be attributing decode time to the bus. What the transfer costs shows up
// here as `Wait` — the time this thread blocks in `poll(WaitForSubmissionIndex)`,
// which on a serial loop is the GPU's whole frame including those prepended copies
// — and as `Upload B/f`, the bytes `UploadCost` counted on the way in. Both are
// measured; neither is derived from the other.
// ─────────────────────────────────────────────────────────────────────────────

/// A downsampled picture of one rendered frame, for comparing the two paths'
/// OUTPUT rather than only their timings.
///
/// **This is what makes "while frames still decode correctly" a measurement.** A
/// path that rendered black, stale or torn frames would post the best numbers in
/// the table: nothing else in a timing row can tell "the transfer is gone" from
/// "the picture is gone", which is precisely the failure mode gotchas 14, 16 and 18
/// all share — no error, no counter, no failing test.
///
/// A grid rather than a hash: two different decoders' output is allowed to differ
/// by a level or two, so the comparison needs a magnitude, and `max_delta` is a
/// number the row can print. `mean_luma` is carried separately because it catches
/// the one case a delta cannot — both arms rendering black would agree perfectly.
///
/// `Clone` because the pipelined real-media rows (benchmarks 7 and 8) carry one
/// sample up to the summary in `RealMediaProvenance`, where the two rows' frames are
/// compared after both have run. The serial `--interop` profile compares within one
/// class and does not need it.
#[derive(Clone)]
struct FrameSample {
    /// `GRID × GRID` RGB samples, in row-major order.
    grid: Vec<[u8; 3]>,
    /// Mean of the sampled luma, 0..255.
    mean_luma: f64,
}

impl FrameSample {
    /// How many samples across; 16 × 16 = 256 points, which is enough to catch a
    /// sheared or half-decoded frame without reading the whole 66 MB texture back
    /// into a comparison.
    const GRID: u32 = 16;

    /// How many consecutive frames at the end of a pass are sampled.
    ///
    /// **More than one, and that is the whole point.** A single sample per arm can
    /// only say "these two pictures differ"; it cannot say whether they differ
    /// because one is corrupt or because the two arms are one frame apart in the
    /// stream. Those are unrelated diagnoses — the first invalidates the timing row,
    /// the second is a pts-handling difference between two decode paths — and
    /// sampling a short run makes them distinguishable by searching for the offset
    /// that lines up.
    const RUN: usize = 3;

    /// Largest per-channel difference between two samples of the same frame, or
    /// `None` when the grids are not comparable.
    fn max_delta(&self, other: &Self) -> Option<u8> {
        if self.grid.len() != other.grid.len() || self.grid.is_empty() {
            return None;
        }
        Some(
            self.grid
                .iter()
                .zip(&other.grid)
                .flat_map(|(a, b)| (0..3).map(move |c| a[c].abs_diff(b[c])))
                .max()
                .unwrap_or(0),
        )
    }
}

/// How two arms' sampled frames line up.
struct SampleAlignment {
    /// Frames the CPU arm is ahead of (positive) or behind (negative) the interop
    /// arm at the best match.
    offset: i64,
    /// The largest per-channel difference at that alignment.
    delta: u8,
    /// The same at offset 0, i.e. comparing the frames both arms *think* are the
    /// same one.
    delta_aligned: u8,
}

/// Find the frame offset at which the two arms' pictures agree best.
///
/// Compares every interop sample against the CPU sample `offset` frames away and
/// keeps the alignment with the smallest worst-case difference. `offset == 0` with a
/// small delta is the outcome that says the two paths render the same frame the same
/// way; a non-zero offset with a small delta says the pictures are fine and the
/// *indexing* differs, which is a different bug and must not be reported as
/// corruption.
fn align_samples(
    interop: &[(usize, FrameSample)],
    cpu: &[(usize, FrameSample)],
) -> Option<SampleAlignment> {
    let mut best: Option<SampleAlignment> = None;
    let mut aligned: Option<u8> = None;
    let span = FrameSample::RUN as i64;
    for offset in -span..=span {
        let mut worst: Option<u8> = None;
        for (idx, sample) in interop {
            let want = *idx as i64 + offset;
            let Some((_, other)) = cpu.iter().find(|(i, _)| *i as i64 == want) else {
                continue;
            };
            let Some(d) = sample.max_delta(other) else { continue };
            worst = Some(worst.map_or(d, |w: u8| w.max(d)));
        }
        let Some(worst) = worst else { continue };
        if offset == 0 {
            aligned = Some(worst);
        }
        if best.as_ref().is_none_or(|b| worst < b.delta) {
            best = Some(SampleAlignment { offset, delta: worst, delta_aligned: worst });
        }
    }
    let mut best = best?;
    // The offset-0 figure is reported alongside the best one, so a row cannot show
    // only the flattering number.
    best.delta_aligned = aligned.unwrap_or(best.delta);
    Some(best)
}

/// One clip's shape, for deciding when the graph must be recompiled.
///
/// `interop` is in here for the reason `ExportRenderer::ClipSignature` carries it:
/// a source that falls back mid-timeline changes the graph's *structure* (an upload
/// node appears, and `YuvToRgbNode` stops importing), and a cached graph would keep
/// importing planes the frame no longer binds — which `resolve_resources` turns
/// into a panic naming the resource rather than a silently black clip.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ClipShape {
    source_id: SourceId,
    width: u32,
    height: u32,
    semi_planar: bool,
    bit_depth: u8,
    interop: bool,
}

impl ClipShape {
    fn of(clip: &ClipRenderEntry) -> Self {
        Self {
            source_id: clip.source_id,
            width: clip.clip_width,
            height: clip.clip_height,
            semi_planar: clip.frame_meta.layout.semi_planar,
            bit_depth: clip.frame_meta.layout.bit_depth,
            interop: clip.is_interop(),
        }
    }
}

/// A compiled graph for one frame shape, plus where its upload nodes are.
struct InteropGraph {
    graph: nexir::render::graph::CompiledGraph,
    /// Clip index → its `YuvUploadNode`, or `None` for an interop clip.
    ///
    /// `None` is load-bearing (G2d): an interop clip has NO upload node, so there
    /// is nothing to feed and no slot-pool buffer to read — `texture_slot` is 0 on
    /// that path and reading tier 0 slot 0 would upload an unrelated source's
    /// frame.
    uploads: Vec<Option<usize>>,
    shape: Vec<ClipShape>,
}

/// Which effect chain a real-media graph carries.
///
/// Two shapes, and the choice is load-bearing in both directions:
///
/// * `Simple` — upload-or-import → YUV→RGB → composite. What `--interop`'s serial
///   rows use, because the question there is what the DECODE PATH costs and an
///   effect chain is identical either way; adding one would make each row a mix of
///   two changes.
/// * `Heavy` — the same, plus colour correction + LUT + chroma key per layer and a
///   tone map at the end. **Benchmark 5's graph exactly**, which is the whole point
///   of benchmark 7: the row is only comparable with the synthetic 4K60 baseline if
///   the graph is the same and only the decode path differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphChain {
    Simple,
    Heavy,
}

/// Build the graph for one scheduled frame, branching per clip exactly as
/// `ExportRenderer::ensure_graph` does.
///
/// `chain` selects the effect chain — see [`GraphChain`] for why there are two and
/// which row uses which. The upload-or-import branch is identical in both, which is
/// the reason this is one function rather than two: that branch is the G2b/G2d
/// invariant (an interop clip gets NO upload node, not one that is fed nothing), and
/// a second copy of it is how one of them would drift.
///
/// `readback` appends a `FINAL_COLOR` → buffer copy, for the one frame per pass that
/// is sampled for the correctness comparison. `None` for every measured frame: the
/// copy is a full 4K `copy_texture_to_buffer` and would otherwise be charged to the
/// per-frame figure it exists to validate.
fn build_interop_graph(
    device: &Arc<GpuDevice>,
    shaders: &ShaderRegistry,
    compute: &Arc<ComputePipelineCache>,
    frame: &FrameState,
    chain: GraphChain,
    readback: Option<SampleReadbackNode>,
) -> Result<InteropGraph, String> {
    use nexir::scheduler::frame_scheduler::FrameScheduler;

    let mut compiler = RenderGraphCompiler::new();
    // Two ids per clip are reserved below this for interop planes whether or not a
    // clip uses them: `FrameScheduler::interop_y_id` picks the same numbers when it
    // binds the frame's imports, and a counter that skipped the CPU-path clips
    // would shift every later clip's ids the moment one source fell back.
    let mut id_counter = FrameScheduler::interop_id_counter_start(frame.clips.len());

    // The Heavy chain's LUT, built once for the whole graph rather than per layer:
    // `LutNode::new` copies the cube into its own texture, so one `Lut3D` feeds all
    // four layers exactly as benchmark 5's does.
    let lut = (chain == GraphChain::Heavy).then(|| identity_lut(17));

    // Heavy composites into an intermediate and tone-maps that into FINAL_COLOR;
    // Simple composites straight into FINAL_COLOR. Allocated before the per-clip
    // loop so the id ordering matches benchmark 5's.
    let composite_out = match chain {
        GraphChain::Simple => ResourceId::FINAL_COLOR,
        GraphChain::Heavy => ResourceId::next(&mut id_counter),
    };
    let mut composite = CompositeNode::new(
        device,
        shaders,
        composite_out,
        (frame.clips.len() as u32).max(1),
        wgpu::TextureFormat::Rgba16Float,
    );

    let mut uploads: Vec<Option<usize>> = Vec::with_capacity(frame.clips.len());
    for (slot, clip) in frame.clips.iter().enumerate() {
        let rgba_id = ResourceId::next(&mut id_counter);
        let layout = clip.frame_meta.layout;

        let (y_id, uv_id) = if clip.is_interop() {
            (
                FrameScheduler::interop_y_id(slot),
                FrameScheduler::interop_uv_id(slot),
            )
        } else {
            (
                ResourceId::next(&mut id_counter),
                ResourceId::next(&mut id_counter),
            )
        };

        if clip.is_interop() {
            uploads.push(None);
        } else {
            let node = YuvUploadNode::new_with_layout(
                device,
                slot as u32,
                clip.clip_width,
                clip.clip_height,
                y_id,
                uv_id,
                layout,
            );
            uploads.push(Some(compiler.add_node(Box::new(node))));
        }

        compiler.add_node(Box::new(
            YuvToRgbNode::new_with_layout(
                device,
                shaders,
                compute,
                y_id,
                uv_id,
                rgba_id,
                clip.clip_width,
                clip.clip_height,
                // From the DECODER, never the container — gotcha 11, and the whole
                // reason the interop path needed a colour test of its own.
                clip.frame_meta.color,
                layout.semi_planar,
            )
            .with_imported_planes(clip.is_interop()),
        ));

        // The per-layer effect chain, at the CLIP's own dimensions because that is
        // what `rgba_id` was created at. Benchmark 5's order exactly: colour
        // correction, then LUT, then chroma key — through the SAME `add_grade_chain`
        // the synthetic builder calls, so `NEXIR_FUSE_GRADE` cannot apply to one row
        // and not the other and make 5-vs-7 a mix of two changes.
        let composite_input = match (&lut, chain) {
            (Some(cube), GraphChain::Heavy) => add_grade_chain(
                device,
                shaders,
                compute,
                &mut compiler,
                &mut id_counter,
                cube,
                rgba_id,
                clip.clip_width,
                clip.clip_height,
            ),
            _ => rgba_id,
        };
        composite.input_textures.push(composite_input);
    }
    compiler.add_node(Box::new(composite));

    if chain == GraphChain::Heavy {
        compiler.add_node(Box::new(ToneMapNode::new(
            device,
            shaders,
            compute,
            composite_out,
            ResourceId::FINAL_COLOR,
            ToneMapPushConstants::for_sdr_preview(
                InputTransferFn::Linear,
                GamutConversion::None,
                ToneMapMode::AcesFilmic,
                1000.0,
                frame.canvas_width,
                frame.canvas_height,
            ),
        )));
    }

    if let Some(node) = readback {
        compiler.add_node(Box::new(node));
    }

    let graph = compiler
        .compile(frame.canvas_width, frame.canvas_height)
        .map_err(|e| format!("graph compile failed: {e:?}"))?;

    Ok(InteropGraph {
        graph,
        uploads,
        shape: frame.clips.iter().map(ClipShape::of).collect(),
    })
}

/// Copies `FINAL_COLOR` into a mappable buffer so a frame's PIXELS can be compared
/// between the two arms.
///
/// A graph node rather than a `copy_texture_to_buffer` after `execute`, because
/// `FINAL_COLOR` is a pooled resource that only exists inside the graph's own
/// acquire/release window — the same reason `tests::colour_plumbing` reads back
/// through a node.
///
/// Declared `CopySrc` on a resource the composite already creates, so it adds no
/// allocation and no pool bucket of its own.
struct SampleReadbackNode {
    buf: Arc<wgpu::Buffer>,
    width: u32,
    height: u32,
    /// 256-aligned row stride, as `copy_texture_to_buffer` requires. Computed once
    /// here and used again when the bytes are read, because a disagreement between
    /// the two shears the picture exactly like a wrong NVENC pitch (gotcha 6).
    stride: u32,
}

impl SampleReadbackNode {
    /// Bytes per texel of `Rgba16Float`.
    const BPP: u32 = 8;

    fn new(device: &GpuDevice, width: u32, height: u32) -> Self {
        let stride = align256(width * Self::BPP);
        let buf = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interop_sample_readback"),
            size: stride as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        Self { buf, width, height, stride }
    }

    /// A second node writing into the SAME buffer, for handing to a graph.
    ///
    /// The graph takes `Box<dyn RenderNode>` by value while the caller keeps the
    /// buffer to map afterwards, so the two share one `Arc` rather than the caller
    /// allocating a second 66 MB staging buffer per pass.
    fn attached(&self) -> Self {
        Self {
            buf: Arc::clone(&self.buf),
            width: self.width,
            height: self.height,
            stride: self.stride,
        }
    }
}

impl nexir::render::graph::RenderNode for SampleReadbackNode {
    fn name(&self) -> &str {
        "InteropSampleReadback"
    }

    fn declare_resources(&self, builder: &mut nexir::render::resource::ResourceBuilder) {
        builder.read(
            ResourceId::FINAL_COLOR,
            nexir::render::resource::TextureAccess::CopySrc,
        );
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &nexir::render::context::RenderContext,
        _frame: &FrameState,
    ) {
        if !ctx.contains(ResourceId::FINAL_COLOR) {
            return;
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: ctx.get(ResourceId::FINAL_COLOR).texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.stride),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Map the readback buffer and reduce it to a [`FrameSample`].
///
/// The caller must already have waited on the submission that recorded the copy.
///
/// `dump` writes the whole frame as a PNG as well, which is what turns "the two
/// arms disagree by 146/255" into a diagnosis: a 16×16 grid can detect a difference
/// but cannot say whether it is corruption, a spatial shift or a different frame of
/// the same clip, and those have nothing in common. Behind an env var because it is
/// a full-resolution encode per sample.
fn read_frame_sample(
    device: &GpuDevice,
    node: &SampleReadbackNode,
    dump: Option<&Path>,
) -> Option<FrameSample> {
    let slice = node.buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;

    let mut grid = Vec::with_capacity((FrameSample::GRID * FrameSample::GRID) as usize);
    let mut luma_sum = 0.0f64;
    // Only allocated when a dump was asked for: a 4K RGB buffer is 24 MB.
    let mut rgb8: Vec<u8> = if dump.is_some() {
        Vec::with_capacity((node.width * node.height * 3) as usize)
    } else {
        Vec::new()
    };
    {
        let view = slice.get_mapped_range();
        for gy in 0..FrameSample::GRID {
            // Sampled at the centre of each cell rather than at its corner: a
            // corner grid over a frame whose left column is intact would miss a
            // half-decoded picture.
            let y = (gy * 2 + 1) * node.height / (FrameSample::GRID * 2);
            for gx in 0..FrameSample::GRID {
                let x = (gx * 2 + 1) * node.width / (FrameSample::GRID * 2);
                let off = (y * node.stride + x * SampleReadbackNode::BPP) as usize;
                if off + 6 > view.len() {
                    return None;
                }
                let mut rgb = [0u8; 3];
                for c in 0..3 {
                    let h = half::f16::from_le_bytes([view[off + c * 2], view[off + c * 2 + 1]]);
                    rgb[c] = (h.to_f32() * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
                // Rec.709 luma, only ever compared against itself, so the exact
                // coefficients matter less than using one set consistently.
                luma_sum +=
                    0.2126 * rgb[0] as f64 + 0.7152 * rgb[1] as f64 + 0.0722 * rgb[2] as f64;
                grid.push(rgb);
            }
        }
        if dump.is_some() {
            for y in 0..node.height {
                for x in 0..node.width {
                    let off = (y * node.stride + x * SampleReadbackNode::BPP) as usize;
                    if off + 6 > view.len() {
                        break;
                    }
                    for c in 0..3 {
                        let h =
                            half::f16::from_le_bytes([view[off + c * 2], view[off + c * 2 + 1]]);
                        rgb8.push((h.to_f32() * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                    }
                }
            }
        }
    }
    node.buf.unmap();
    if let Some(path) = dump {
        match image::save_buffer(
            path,
            &rgb8,
            node.width,
            node.height,
            image::ColorType::Rgb8,
        ) {
            Ok(()) => println!("               dumped {}", path.display()),
            Err(e) => println!("               WARNING: could not write {}: {e}", path.display()),
        }
    }
    let n = grid.len().max(1) as f64;
    Some(FrameSample { grid, mean_luma: luma_sum / n })
}

/// What one arm of one class measured. Every field timed or counted.
struct InteropReading {
    path: DecodePath,
    frames: usize,
    /// Wall time of the measured loop.
    elapsed: Duration,
    /// Sum of the per-frame scheduler time — the decode, and on the CPU arm the
    /// `av_hwframe_transfer_data` copy to host memory inside it.
    ///
    /// **The upload is NOT in here** — see [`Self::upload_total`]. The two were one
    /// field and that made `grain_1080p60` unattributable.
    schedule_total: Duration,
    /// Sum of the per-frame `YuvUploadNode::upload_frame` time. Zero on the interop
    /// arm because there is no upload node, which is the point.
    upload_total: Duration,
    /// Sum of the per-frame `poll(WaitForSubmissionIndex)` time. On a serial loop
    /// this is the GPU's whole frame, including the staging copies wgpu prepends to
    /// the submission (which is why the CPU arm's figure is the one that carries
    /// the transfer).
    wait_total: Duration,
    /// Bytes `UploadCost` counted going into `YuvUploadNode`. **Zero on the interop
    /// arm because there is no upload node to count**, which is only meaningful
    /// beside `live_targets > 0` — see the acceptance criterion.
    upload_bytes: u64,
    /// Y/UV texture pairs the registry allocated. `0` on the CPU arm by
    /// construction; `0` on the interop arm means every source fell back and the
    /// row is measuring the CPU path.
    live_targets: usize,
    /// VRAM those pairs occupy, counted from each target's own dimensions — G2f.2.
    ///
    /// A lower bound (it excludes alignment padding and driver overhead), and NOT a
    /// driver query: NVML's `vram_used_bytes` is the row that is one, and this must
    /// never be printed as if it were.
    target_bytes: u64,
    /// Clips in the frame the loop actually scheduled, and how many of them carried
    /// imported planes.
    ///
    /// **The pairing that says a multi-source row really was one.** `live_targets`
    /// counts allocations; this counts what the *frame* bound, and the two differ
    /// exactly when a source fell back mid-run or the timeline produced fewer clips
    /// than the harness has sources — either of which turns a "4 sources" row into a
    /// smaller workload with no other symptom.
    clips: usize,
    interop_clips: usize,
    /// The registry's own counters, `None` on the CPU arm.
    interop_stats: Option<nexir::io::interop_decode::InteropDecodeStats>,
    /// The same counters split by source — G2f.1 step 3, G2f.3.
    ///
    /// Registry-wide figures sum four sources into one, so a single slow source is
    /// invisible and the attribution G2e built stops working at exactly the point it
    /// becomes interesting. Empty on the CPU arm.
    per_source: Vec<(u32, nexir::io::interop_decode::InteropDecodeStats)>,
    /// Sources that fell back, with the reason. Printed, never hidden.
    rejections: Vec<(u32, &'static str)>,
    /// The pool's counters for the graph this pass compiled — the `peak bucket`
    /// reading gotcha 14 says to take rather than assume.
    pool: nexir::render::resource::PoolStats,
    /// How many times the graph had to be recompiled. >1 means a source changed
    /// shape or fell back mid-run, which is worth seeing.
    compiles: usize,
    /// A sampled picture of the last few frames, for the correctness comparison.
    ///
    /// Several rather than one, keyed by frame index, because that is what separates
    /// "the picture is corrupt" from "the two arms are one frame apart" — two
    /// diagnoses with nothing in common, and a single sample cannot tell them apart.
    samples: Vec<(usize, FrameSample)>,
}

impl InteropReading {
    fn fps(&self) -> Option<f64> {
        let secs = self.elapsed.as_secs_f64();
        (self.frames > 0 && secs > 0.0).then(|| self.frames as f64 / secs)
    }

    fn ms_per_frame(&self) -> Option<f64> {
        let secs = self.elapsed.as_secs_f64();
        (self.frames > 0 && secs > 0.0).then(|| secs * 1000.0 / self.frames as f64)
    }

    fn schedule_ms(&self) -> Option<f64> {
        (self.frames > 0)
            .then(|| self.schedule_total.as_secs_f64() * 1000.0 / self.frames as f64)
    }

    /// Mean upload time per frame, or `None` before any frame.
    ///
    /// `Some(0.0)` on the interop arm is correct and deliberate: there is no upload
    /// node, so zero is a measurement of an absent stage rather than an unmeasured
    /// one.
    fn upload_ms(&self) -> Option<f64> {
        (self.frames > 0).then(|| self.upload_total.as_secs_f64() * 1000.0 / self.frames as f64)
    }

    fn wait_ms(&self) -> Option<f64> {
        (self.frames > 0).then(|| self.wait_total.as_secs_f64() * 1000.0 / self.frames as f64)
    }

    /// Counted upload bytes per frame, or `None` when no frame was measured.
    ///
    /// Deliberately `Some(0)` rather than `None` on the interop arm: zero bytes is
    /// the *result*, and printing `n/a` there would hide the very thing the task is
    /// measuring.
    fn upload_bytes_per_frame(&self) -> Option<f64> {
        (self.frames > 0).then(|| self.upload_bytes as f64 / self.frames as f64)
    }
}

/// Run one arm of one class: schedule `frames` frames through `FrameScheduler` and
/// render each one.
///
/// `paths` is one entry per SOURCE, each becoming its own video track, its own
/// `SourceId`, its own decoder and its own interop target — which is what makes a
/// four-source row four sources rather than one file read four times (G2f.1 step 1).
/// One path is the single-source case the earlier rows measure.
///
/// Five things in here are load-bearing rather than incidental:
///
/// 1. **`mark_submitted` is called once per frame on the WHOLE clip list**, one line
///    after `submit`. Skipping it is the read-after-write hazard
///    `io::interop_decode`'s header documents: the next decode overwrites textures
///    the submitted graph is still sampling, giving a torn frame with no error, no
///    counter and no failing test. Calling it per clip would stamp the same
///    submission N times for no benefit; it takes `&[ClipRenderEntry]` precisely so
///    the caller cannot pass the wrong set (G2f.1 step 2). It is a no-op on the CPU
///    arm.
/// 2. **The graph is rebuilt when the clip shape changes**, including when a source
///    falls back — see [`ClipShape`]. A cached graph that kept importing planes the
///    frame no longer binds panics in `resolve_resources` naming the resource.
/// 3. **The warm-up is untimed and runs the same code**, so first-frame pipeline
///    creation, the first `DecodeInteropTarget` allocation (a
///    `CreateCommittedResource` + `CreateSharedHandle` + `cuImportExternalMemory`
///    per plane) and the decoder's own start-up are out of the measured span.
/// 4. **The readback node is in the graph for the LAST frame only.** It is a full
///    4K `copy_texture_to_buffer` plus a map, which would otherwise dominate the
///    per-frame figure — the correctness check must not pay for itself out of the
///    number it is validating.
/// 5. **What the FRAME bound is counted, not just what the registry allocated.**
///    `live_targets` says how many texture pairs exist; `interop_clips` says how many
///    of this frame's clips actually read one. A row where those disagree is a
///    smaller workload than its heading claims, and nothing else would show it.
fn run_interop_pass(
    device: &Arc<GpuDevice>,
    paths: &[&Path],
    label: &str,
    decode_path: DecodePath,
    chain: GraphChain,
    capability: &InteropCapability,
    frames: usize,
) -> Result<InteropReading, String> {
    let interop = Arc::new(match decode_path {
        DecodePath::Interop => nexir::io::interop_decode::InteropDecodeTargets::with_context(
            Arc::clone(device),
            capability.clone(),
            shared_cuda_ctx(capability),
        ),
        DecodePath::Cpu => {
            nexir::io::interop_decode::InteropDecodeTargets::disabled(Arc::clone(device))
        }
    });
    if decode_path == DecodePath::Interop && !interop.is_available() {
        return Err(format!(
            "CUDA interop is unavailable on this host (transport={:?}), so the \
             interop arm cannot be exercised at all",
            capability.transport
        ));
    }

    let (harness, streams, duration_pts) =
        build_multi_clip_harness(device, paths, Arc::clone(&interop))?;
    let stream = &streams[0];
    let fps = stream.frame_rate.unwrap_or(Rational { num: 30, den: 1 });
    let tb = Rational::TIMEBASE_90K;
    let width = stream.width.ok_or("the input stream has no width")?;
    let height = stream.height.ok_or("the input stream has no height")?;

    // The pts of frame `i`, in the project timebase. Quantised by the scheduler
    // itself, so this only has to be monotonic and on the frame grid.
    let pts_of = |i: usize| -> i64 { nexir::timeline::rational::frame_to_pts(i as i64, fps, tb) };
    let available = (nexir::timeline::rational::pts_to_frame(duration_pts, fps, tb) as usize)
        .saturating_sub(1);
    if available == 0 {
        return Err("the input decodes to no frames".to_string());
    }

    // HOW LONG A PASS CAN BE, and why it is not simply the file's frame count.
    //
    // Historically `IoLayer` never drained the decoder, so FFmpeg's frame-level
    // threading (roughly one frame held back per core) made the last few frames of
    // any file unreachable through `schedule_frame` — measured on `cam_4k30` (60
    // coded frames): frame 57 was the first the scheduler could not produce, and
    // asking for `MEDIA_PASS_FRAMES` (90) from it failed the whole class with *"the
    // scheduler produced no clips at pts 171000"*. **Task G3 fixed that**:
    // `decode_blocking` now drains at EOF, and `decode_interop` returns `None` there
    // so the same fallback serves those frames.
    //
    // The span calculation stays regardless, and deliberately: a benchmark that
    // fails a whole class because of a decoder property is a benchmark bug either
    // way, and the last frames of the interop arm are served by the CPU fallback by
    // design (G3 step 2) — which would count upload bytes on the arm whose whole
    // point is that they are zero. Stopping short of the tail keeps the two arms
    // measuring the same thing.
    //
    // So the span is DISCOVERED rather than asserted: the loop below stops cleanly
    // when the scheduler runs dry, and each arm reports the frames it actually
    // measured. What is reserved up front is only the correctness sample run, which
    // decodes FORWARD from where the measured loop stopped — without that reserve a
    // short class is measured without ever being pixel-verified, and `Δpx` prints
    // `n/a` for the row that most needed it.
    let requested = frames.min(available);
    let frames = if available >= requested + FrameSample::RUN {
        requested
    } else {
        requested.saturating_sub(FrameSample::RUN).max(1)
    };

    // The accumulators, in a struct rather than as captured locals.
    //
    // Not cosmetic: the loop below is a closure so the warm-up and the measured run
    // execute *the same code*, and a closure that captured these by mutable
    // reference would hold that borrow for its whole lifetime — the reads after the
    // last call would not compile. Passing the state in per call scopes the borrow
    // to the call.
    struct PassState {
        cached: Option<InteropGraph>,
        compiles: usize,
        schedule_total: Duration,
        /// Time in `YuvUploadNode::upload_frame`, on the CPU arm only.
        ///
        /// **Separate from `schedule_total`, because the grain row needs the split.**
        /// With the two summed, a CPU row reads as one "schedule" figure covering
        /// FFmpeg's decode, `av_hwframe_transfer_data` and the upload — so a class
        /// where the interop arm loses cannot be attributed: "the decode got more
        /// expensive" and "the upload it replaced was cheap here" produce the same
        /// number. `grain_1080p60` is that class.
        upload_total: Duration,
        wait_total: Duration,
        upload_bytes: u64,
        /// Most clips any timed frame carried, and the FEWEST of them that were on
        /// the interop path in any one timed frame.
        ///
        /// A min rather than a last-frame reading, so one frame that fell back is
        /// visible: `4/4` means every measured frame had all four clips importing,
        /// and `3/4` means at least one frame did not — which is a smaller workload
        /// than the row's heading and has no other symptom.
        clips_max: usize,
        interop_clips_min: usize,
    }
    let mut st = PassState {
        cached: None,
        compiles: 0,
        schedule_total: Duration::ZERO,
        upload_total: Duration::ZERO,
        wait_total: Duration::ZERO,
        upload_bytes: 0,
        clips_max: 0,
        interop_clips_min: usize::MAX,
    };

    // One frame of every pass ends in a readback, and it is the LAST one, so the
    // copy is outside the timed loop below.
    let readback = SampleReadbackNode::new(device, width, height);

    // ── The loop, shared by the warm-up and the measured run ──────────────────
    //
    // A closure rather than two copies: the warm-up must run exactly the code the
    // measurement runs, or it warms something else. `timed` gates only the
    // accumulators.
    let run_frames = |st: &mut PassState,
                      first: usize,
                      count: usize,
                      timed: bool,
                      want_sample: bool|
     -> Result<usize, String> {
        let mut done = 0usize;
        for i in first..first + count {
            let pts = pts_of(i);

            // ── Schedule: decode (interop or CPU) and bind this frame ─────────
            let t = Instant::now();
            let frame = {
                let store = harness.timeline.read().unwrap();
                let tracks = harness.tracks.read().unwrap();
                let sources = harness.sources.read().unwrap();
                harness
                    .scheduler
                    .schedule_frame(pts, &store, &tracks, &sources)
            };
            let schedule = t.elapsed();
            if frame.clips.is_empty() {
                // THE SPAN RAN OUT — a clean stop, not a failure, and the caller
                // reports the frames that were measured rather than a rate for a
                // count it did not reach. `IoLayer` never drains the decoder, so the
                // last frames a file holds are unreachable through the scheduler
                // (see the span calculation above); returning an error here failed
                // the whole `cam_4k30` class for a property of FFmpeg's threading.
                break;
            }

            // ── Graph: reuse unless the shape moved ──────────────────────────
            //
            // `want_sample` forces a rebuild because the sampled frame needs the
            // readback node in the graph and no other frame may carry it.
            let shape: Vec<ClipShape> = frame.clips.iter().map(ClipShape::of).collect();
            if want_sample || st.cached.as_ref().map(|g| g.shape != shape).unwrap_or(true) {
                st.cached = Some(build_interop_graph(
                    device,
                    &harness.shaders,
                    &harness.compute,
                    &frame,
                    chain,
                    want_sample.then(|| readback.attached()),
                )?);
                if timed {
                    st.compiles += 1;
                }
            }
            let g = st.cached.as_mut().expect("just built");

            // ── Upload: only the clips that have an upload node ──────────────
            //
            // `uploads[i]` is `None` for an interop clip. Feeding one would be the
            // G2d failure: its planes are imported, `texture_slot` is 0, and reading
            // tier 0 slot 0 uploads an unrelated source's frame over them.
            let t = Instant::now();
            for (slot, clip) in frame.clips.iter().enumerate() {
                let Some(node_idx) = g.uploads.get(slot).copied().flatten() else {
                    continue;
                };
                let tier = (clip.texture_slot >> 16) as u8;
                let index = (clip.texture_slot & 0xFFFF) as u16;
                let slot_id = nexir::io::slot_pool::FrameSlotId { tier, index };
                let io = harness.scheduler.io_layer();
                if let Some(u) = g.graph.nodes_mut()[node_idx]
                    .as_any_mut()
                    .and_then(|n| n.downcast_mut::<YuvUploadNode>())
                {
                    let cost = io.pool.with_buffer_read(slot_id, |data| {
                        u.upload_frame(
                            data,
                            clip.frame_meta.layout.semi_planar,
                            clip.clip_width,
                            clip.clip_height,
                        )
                    });
                    if timed {
                        st.upload_bytes += cost.bytes;
                    }
                }
            }
            let upload = t.elapsed();

            // ── Record, submit, stamp, wait ──────────────────────────────────
            let mut enc = device.begin_frame();
            g.graph.execute(&mut enc, device, &frame);
            let submission = device.submit(enc);

            // THE GUARD. One line after submit, exactly as the two production
            // callers do it. Without this the next decode into any of these
            // textures races the graph still reading them.
            interop.mark_submitted(&frame.clips, &submission);

            let t = Instant::now();
            device
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
            let wait = t.elapsed();

            if timed {
                st.schedule_total += schedule;
                st.upload_total += upload;
                st.wait_total += wait;
                // COUNTED PER FRAME, not read off the last one: `live_targets` says
                // how many texture pairs the registry allocated, and this says how
                // many of THIS frame's clips actually bound one. A row where the two
                // disagree measured a smaller workload than its heading claims —
                // either a source fell back mid-run or the timeline produced fewer
                // clips than the harness has sources — and there is no other symptom.
                let interop_clips = frame.clips.iter().filter(|c| c.is_interop()).count();
                st.clips_max = st.clips_max.max(frame.clips.len());
                st.interop_clips_min = st.interop_clips_min.min(interop_clips);
            }
            done += 1;
        }
        Ok(done)
    };

    // Warm-up: untimed, same code. Bounded by what the file holds.
    let warm = WARMUP_MIN_FRAMES.min(frames);
    run_frames(&mut st, 0, warm, false, false)?;

    // The measured span, wall-clock. Deliberately NOT `schedule_total +
    // wait_total`: those two are stage sums and would silently exclude anything
    // between them (recording the graph, submitting, `mark_submitted`), so an FPS
    // derived from them would be a rate for less work than the loop did. Same rule
    // `average_fps` follows — frames ÷ elapsed seconds, from one clock.
    let loop_start = Instant::now();
    let measured = run_frames(&mut st, 0, frames, true, false)?;
    let elapsed = loop_start.elapsed();
    if measured == 0 {
        return Err(
            "the scheduler produced no clips at all, so no frame was rendered and \
             there is no rate to report"
                .to_string(),
        );
    }

    // The pool's counters BEFORE the sample pass, because that pass compiles a
    // different graph (with the readback node) and a `CompiledGraph` owns its pool
    // — reading afterwards would report a pool that saw one frame.
    let pool = st
        .cached
        .as_ref()
        .map(|g| g.graph.pool_stats())
        .unwrap_or_default();

    // The correctness samples: a short run of frames decoded FORWARD from where the
    // measured loop stopped.
    //
    // Forward, and continuing rather than re-visiting, because a backwards jump is
    // not a like-for-like comparison between the two arms: the CPU arm answers a
    // repeat request from `FrameCache`'s 32 slots, while a one-frame interop target
    // has to seek and re-decode, and a seek into an open-GOP H.264 stream can
    // legitimately produce a frame built from a different reference set. Sampling
    // frames the earlier run already passed measured that difference and reported it
    // as corruption. Both arms now decode these frames the same way: forward, once,
    // in order.
    let mut samples: Vec<(usize, FrameSample)> = Vec::new();
    let dump_dir = interop_dump_dir();
    let sample_end = (measured + FrameSample::RUN).min(available);
    for i in measured..sample_end {
        match run_frames(&mut st, i, 1, false, true) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }

        // `NEXIR_INTEROP_DUMP=<dir>` writes each sampled frame as a PNG, named by
        // class, arm and frame index, so the two arms' pictures can be looked at
        // side by side. That is the only thing that turns a delta into a diagnosis.
        let dump = dump_dir
            .as_ref()
            .map(|d| d.join(format!("{label}_{}_{i:04}.png", decode_path.label())));
        if let Some(s) = read_frame_sample(device, &readback, dump.as_deref()) {
            samples.push((i, s));
        }
    }

    Ok(InteropReading {
        path: decode_path,
        // What was MEASURED, not what was asked for. The two differ whenever the
        // file's reachable span is shorter than the request, and dividing wall time
        // by the request would report a rate for frames that never rendered.
        frames: measured,
        elapsed,
        schedule_total: st.schedule_total,
        upload_total: st.upload_total,
        wait_total: st.wait_total,
        upload_bytes: st.upload_bytes,
        live_targets: interop.live_targets(),
        // Counted from the targets' own dimensions, and a lower bound — G2f.2.
        target_bytes: interop.target_bytes(),
        clips: st.clips_max,
        // `usize::MAX` means no timed frame was seen, which `measured == 0` has
        // already turned into an error above; 0 is then the honest answer.
        interop_clips: if st.interop_clips_min == usize::MAX {
            0
        } else {
            st.interop_clips_min
        },
        interop_stats: (decode_path == DecodePath::Interop).then(|| interop.stats()),
        per_source: if decode_path == DecodePath::Interop {
            interop
                .per_source_stats()
                .into_iter()
                .map(|(id, s)| (id.index() as u32, s))
                .collect()
        } else {
            Vec::new()
        },
        rejections: interop
            .rejections()
            .into_iter()
            .map(|(id, reason)| (id.index() as u32, reason))
            .collect(),
        pool,
        compiles: st.compiles,
        samples,
    })
}

/// Both arms of one class, with every repeat kept.
struct InteropRow {
    label: String,
    width: u32,
    height: u32,
    /// How many DISTINCT sources this row's frames were composited from.
    ///
    /// In the table because an FPS or ms/frame figure without it is not comparable
    /// to anything — the same reason the synthetic summary prints `L:S` (benchmark 5
    /// is 4:4 and benchmark 6 is 4:1 on the same graph, at a quarter of the bytes).
    sources: usize,
    /// One entry per repeat, interop arm.
    interop: Vec<InteropReading>,
    /// One entry per repeat, CPU arm — the baseline the interop arm is read against.
    cpu: Vec<InteropReading>,
}

/// Median and spread of one arm's per-frame time.
///
/// Spread is `(max-min)/median`, the same definition the synthetic summary uses, and
/// `None` on a single repeat rather than `0%` — one run is not a run with zero
/// variance (gotcha 17's rule, applied here).
fn median_spread(runs: &[InteropReading]) -> (Option<f64>, Option<f64>) {
    let values: Vec<f64> = runs.iter().filter_map(|r| r.ms_per_frame()).collect();
    if values.is_empty() {
        return (None, None);
    }
    let med = median(values.iter().copied());
    let spread = if values.len() < 2 || med <= 0.0 {
        None
    } else {
        let lo = values.iter().copied().fold(f64::MAX, f64::min);
        let hi = values.iter().copied().fold(f64::MIN, f64::max);
        Some((hi - lo) / med * 100.0)
    };
    (Some(med), spread)
}

/// Print one arm's detail: the counters that decide whether its timing row means
/// anything.
fn print_interop_detail(r: &InteropReading) {
    println!(
        "      {:<8} {:>7.2} ms/frame ({:>6.1} FPS) over {} frame(s) — schedule {:>6.2} ms, \
         upload {:>5.2} ms, wait {:>6.2} ms",
        r.path.label(),
        r.ms_per_frame().unwrap_or(f64::NAN),
        r.fps().unwrap_or(f64::NAN),
        // Printed because it is not always what was asked for: `IoLayer` never
        // drains the decoder, so a file's last few frames are unreachable and the
        // pass stops there. A row measuring fewer frames than its neighbours is a
        // fact about the fixture, and hiding it would make the medians look
        // like-for-like when they are not.
        r.frames,
        r.schedule_ms().unwrap_or(f64::NAN),
        // Split out of `schedule` so a class the interop arm loses can be
        // attributed: on the CPU arm this is the stage the interop path DELETES, so
        // it is the upper bound on what removing it could ever save.
        r.upload_ms().unwrap_or(f64::NAN),
        r.wait_ms().unwrap_or(f64::NAN),
    );
    // THE PAIRING THE ACCEPTANCE CRITERION RESTS ON. `0 bytes` alone is
    // indistinguishable from "the pass never ran", so the counted bytes are printed
    // beside the number of texture pairs that were actually allocated — and beside
    // how many of the frame's clips actually bound one, because at four sources
    // `live targets 4` with `3/4 clips` imported is a smaller workload than the
    // heading claims.
    println!(
        "               upload {:>9.2} MB/frame (counted), live targets {} ({}), \
         imported {}/{} clip(s), graph compiles {}",
        r.upload_bytes_per_frame().unwrap_or(f64::NAN) / (1024.0 * 1024.0),
        r.live_targets,
        // G2f.2: counted from each target's own dimensions, and labelled a lower
        // bound — it excludes alignment padding and driver overhead, and it is NOT
        // the NVML VRAM row.
        if r.target_bytes > 0 {
            format!(
                "{:.1} MB VRAM, lower bound",
                r.target_bytes as f64 / (1024.0 * 1024.0)
            )
        } else {
            "no VRAM counted".to_string()
        },
        r.interop_clips,
        r.clips,
        r.compiles,
    );
    if let Some(s) = r.interop_stats {
        println!(
            "               decodes {}, cached {}, pending {}, decode {} — waits {} ({})",
            s.decodes,
            s.cached,
            s.pending,
            s.decode_ms_per_frame()
                .map(|ms| format!("{ms:.2} ms/frame"))
                .unwrap_or_else(|| "n/a".into()),
            s.waits,
            s.wait_ms_per_decode()
                .map(|ms| format!("{ms:.3} ms/decode"))
                .unwrap_or_else(|| "n/a".into()),
        );
        // WHERE THE COPY'S TIME WENT, per copy, split three ways. A class that loses
        // to the CPU path loses inside one of these three, and a single
        // "decode ms/frame" cannot say which — `grain_1080p60` is the class that made
        // the split necessary.
        match s.copy_ms_per_call() {
            Some((barrier, issue, sync)) => println!(
                "               copies {}: decode barrier {barrier:.3} ms, issue \
                 {issue:.3} ms, transfer sync {sync:.3} ms",
                s.copy.calls,
            ),
            None => println!(
                "               copies 0: no device→array copy ran, so its phases are n/a"
            ),
        }
        // AND WHAT IS LEFT, which is the figure the grain row turns on: everything in
        // `decode_into_target` that is NOT our three copy phases and not the
        // read-after-write guard. That residue is `avcodec_send_packet` +
        // `avcodec_receive_frame` — NVDEC itself.
        //
        // A subtraction of measured quantities rather than a fourth timer, and
        // labelled as one: the two sides come from different counters (per decode vs
        // per copy) and a read-forward performs more copies than decodes, so this is
        // an attribution, not a bracket. It is printed because it is the only way to
        // tell "our copy is expensive" from "the decode itself got slower once we
        // started synchronising the context every frame".
        if let (Some(total), Some((barrier, issue, sync))) =
            (s.decode_ms_per_frame(), s.copy_ms_per_call())
        {
            let copies_per_decode = s.copy.calls as f64 / s.decodes.max(1) as f64;
            let ours = (barrier + issue + sync) * copies_per_decode;
            println!(
                "               of {total:.2} ms/decode, {ours:.2} ms is ours (copy x{:.2}) \
                 and {:.2} ms is\n               inside NVDEC (send/receive)",
                copies_per_decode,
                (total - ours).max(0.0),
            );
        }

        // ── PER SOURCE — G2f.1 step 3 and G2f.3 ───────────────────────────────
        //
        // The figures above are registry-wide, so four sources sum into one number
        // and a single slow source is invisible: 6 ms/decode is equally consistent
        // with four sources at 1.5 ms and with three at 0.5 plus one at 4.5. Those
        // lead to opposite fixes, which is exactly the attribution G2e built for one
        // source and would otherwise lose at the point it becomes interesting.
        //
        // Printed only for a genuinely multi-source pass: at one source this is the
        // aggregate row again, and repeating it would bury the rows that matter.
        if r.per_source.len() > 1 {
            println!("               per source:");
            for (id, ps) in &r.per_source {
                println!(
                    "                 src {id}: {} decode(s), {} cached, {} pending, \
                     {} — waits {} ({}), copies {}{}",
                    ps.decodes,
                    ps.cached,
                    ps.pending,
                    ps.decode_ms_per_frame()
                        .map(|ms| format!("{ms:.2} ms/frame"))
                        .unwrap_or_else(|| "n/a".into()),
                    ps.waits,
                    ps.wait_ms_per_decode()
                        .map(|ms| format!("{ms:.3} ms/decode"))
                        .unwrap_or_else(|| "n/a".into()),
                    ps.copy.calls,
                    ps.copy_ms_per_call()
                        .map(|(b, i, s)| format!(
                            " (barrier {b:.3} + issue {i:.3} + sync {s:.3} ms)"
                        ))
                        // Never `0.000` for a source whose target was evicted before
                        // the read: a mean over no copies is undefined, and printing
                        // zero there is how an unexercised path reads as a fast one.
                        .unwrap_or_else(|| " (phases n/a)".into()),
                );
            }
            // WHICH OF THE TWO FIXES THE READING CALLS FOR, stated from the numbers
            // rather than left to a reader. G2f.3's table:
            //   waits large + wait_total large  → G4, double buffering
            //   waits small, barrier grows w/ N → per-source CUDA streams
            // The two are not interchangeable, and starting the wrong one costs VRAM
            // or complexity and moves nothing.
            let worst_wait = r
                .per_source
                .iter()
                .filter_map(|(id, ps)| ps.wait_ms_per_decode().map(|ms| (*id, ms)))
                .fold(None, |acc: Option<(u32, f64)>, (id, ms)| {
                    Some(match acc {
                        Some((bid, best)) if best >= ms => (bid, best),
                        _ => (id, ms),
                    })
                });
            let mean_barrier = {
                let v: Vec<f64> = r
                    .per_source
                    .iter()
                    .filter_map(|(_, ps)| ps.copy_ms_per_call().map(|(b, _, _)| b))
                    .collect();
                (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
            };
            match (worst_wait, mean_barrier) {
                (Some((id, ms)), Some(barrier)) if ms >= 0.5 => println!(
                    "               G2f.3: source {id} waits {ms:.3} ms/decode on the \
                     read-after-write\n               guard — that is G4's case (a second \
                     target per source), not the shared\n               stream's (mean \
                     barrier {barrier:.3} ms)."
                ),
                (_, Some(barrier)) if barrier >= 0.5 => println!(
                    "               G2f.3: the guard is free but the decode barrier \
                     averages {barrier:.3} ms\n               per copy across {} \
                     source(s) — that is the SHARED STREAM, and per-source\n               \
                     CUDA streams are the fix. A second target would not touch it.",
                    r.per_source.len()
                ),
                (_, Some(barrier)) => println!(
                    "               G2f.3: neither contention appeared — the guard's worst \
                     source is\n               {} and the mean barrier is {barrier:.3} ms \
                     per copy. G4 stays UNSTARTED.",
                    worst_wait
                        .map(|(id, ms)| format!("src {id} at {ms:.3} ms/decode"))
                        .unwrap_or_else(|| "no source that polled at all".into()),
                ),
                _ => println!(
                    "               G2f.3: no copy was timed, so neither contention can be \
                     read from this run."
                ),
            }
        }
    }
    if r.path == DecodePath::Interop && r.live_targets == 0 {
        println!(
            "      WARNING: no interop target was allocated, so this row measured the \
             CPU\n      upload path under an `interop` label."
        );
    }
    for (source, reason) in &r.rejections {
        // A fallback is printed, never hidden: a row whose sources all fell back is
        // measuring the path it meant to replace.
        println!("               source {source} FELL BACK: {reason}");
    }
    print_pool_stats(r.pool);
}

/// Run both arms of one row, `repeats` times, and reduce them to an [`InteropRow`].
///
/// Shared by the single-source classes and the four-source 4K60 row (G2f.1), so the
/// two cannot drift into being measured differently and then compared — the whole
/// reason `--interop` interleaves its arms in one process rather than reading two
/// git checkouts.
///
/// `Err` fails the WHOLE row rather than dropping a repeat, for gotcha 17's reason:
/// the failures `run_interop_pass` returns are "the graph would not compile" and
/// "the scheduler produced nothing at all", and a median over the repeats that
/// happened to work would report a rate for a pipeline that is not reliably
/// producing frames. **Running out of frames is not one of them** — that is a
/// property of the fixture, handled by the span calculation inside the pass and
/// reported per row.
fn run_interop_class(
    device: &Arc<GpuDevice>,
    label: &str,
    inputs: &[&Path],
    width: u32,
    height: u32,
    capability: &InteropCapability,
    repeats: usize,
    frames: usize,
) -> Result<InteropRow, String> {
    let mut interop_runs = Vec::with_capacity(repeats);
    let mut cpu_runs = Vec::with_capacity(repeats);
    for r in 0..repeats {
        if repeats > 1 {
            println!("    --- repeat {}/{} ---", r + 1, repeats);
        }
        // Interleaved arm by arm within a repeat rather than all of one then all of
        // the other: the two then see the same thermal and driver state, so a clock
        // ramp cannot land entirely on one arm and be read as a win.
        //
        // The INTEROP arm goes first and its measured count caps the CPU arm's
        // request. The two arms do not run out of frames at the same index:
        // `decode_blocking` answers a pts past the decoder's reach out of
        // `FrameCache`'s 32 slots while a one-frame interop target has nothing to fall
        // back on, so the CPU arm would keep going — re-rendering a stale frame — past
        // the point the interop arm stopped. Capping makes the two like-for-like by
        // construction; the check in the caller is the backstop.
        let mut cap = frames;
        for path in [DecodePath::Interop, DecodePath::Cpu] {
            let reading =
                run_interop_pass(device, inputs, label, path, GraphChain::Simple, capability, cap)
                    .map_err(|why| format!("{} arm: {why}", path.label()))?;
            print_interop_detail(&reading);
            match path {
                DecodePath::Interop => {
                    cap = reading.frames;
                    interop_runs.push(reading);
                }
                DecodePath::Cpu => cpu_runs.push(reading),
            }
        }
    }
    if interop_runs.is_empty() || cpu_runs.is_empty() {
        return Err("no repeat completed both arms".to_string());
    }

    // THE ARMS MUST HAVE MEASURED THE SAME FRAMES. They stop independently when the
    // scheduler runs dry, and the CPU arm can run further than the interop one
    // because `decode_blocking` falls back to `FrameCache` while a one-frame interop
    // target has nothing to fall back on. A ratio of two medians over different
    // amounts of work is not a gain, so this is checked and printed rather than
    // assumed.
    let i_frames: Vec<usize> = interop_runs.iter().map(|r| r.frames).collect();
    let c_frames: Vec<usize> = cpu_runs.iter().map(|r| r.frames).collect();
    if i_frames != c_frames {
        println!(
            "    WARNING: the two arms measured different frame counts (interop \
             {i_frames:?} vs cpu\n    {c_frames:?}). The gain column below divides two \
             medians over different amounts of work."
        );
    }
    println!();
    Ok(InteropRow {
        label: label.to_string(),
        width,
        height,
        sources: inputs.len(),
        interop: interop_runs,
        cpu: cpu_runs,
    })
}

/// The `--interop` profile: every class, both arms, medians with spread.
///
/// Returns whether anything was measured, so `main` can honour
/// `NEXIR_REQUIRE_MEDIA`.
fn run_interop_profile(device: &Arc<GpuDevice>) -> bool {
    println!("========================================================================");
    println!("INTEROP DECODE PROFILE — the interop decode path vs the CPU upload path");
    println!("  Both arms drive real files through FrameScheduler::schedule_frame and");
    println!("  build the graph from frame.clips, exactly as ExportRenderer does. They");
    println!("  differ by ONE argument: the IoLayer's InteropDecodeTargets. Same loop,");
    println!("  same file, same process, interleaved per class — so the comparison is a");
    println!("  measurement rather than a git checkout.");
    println!("  Serial, one frame in flight: a target holds one frame per source, so a");
    println!("  deeper pipeline would spend its lookahead inside the read-after-write");
    println!("  wait. Comparable to `--media`'s serial rows, NOT to the pipelined sweep.");
    println!("  The last row is G2f.1(b): FOUR DISTINCT 4K60 sources in one frame, which");
    println!("  is benchmark 5's shape. It measures the frame's COST, not the pipeline's");
    println!("  throughput, so it cannot produce an \"avg FPS >= 60\" verdict and must not");
    println!("  be quoted as if it could — that is what benchmark 7 (`bench.exe 7`) is.");
    println!("========================================================================\n");

    let capability = InteropCapability::probe(device);
    println!(
        "CUDA interop   : {} (transport={:?}, driver {})",
        if capability.is_available() { "available" } else { "UNAVAILABLE" },
        capability.transport,
        if capability.driver_version.is_empty() {
            "n/a"
        } else {
            &capability.driver_version
        },
    );
    if !capability.is_available() {
        println!(
            "\nSKIPPED: without CUDA interop the interop arm cannot run at all, and a\n\
             CPU-only table would be `--media` under a different name. Nothing measured.\n"
        );
        return false;
    }

    let ffmpeg = match bench_media::ffmpeg_binary() {
        Some(p) => p,
        None => {
            println!(
                "\nSKIPPED: no `ffmpeg` binary on PATH (set NEXIR_FFMPEG). The inputs are\n\
                 generated with the CLI, so no class ran.\n"
            );
            return false;
        }
    };
    println!("Fixture dir    : {}\n", bench_media::fixture_dir().display());

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
        println!("Repeats: {repeats} per arm; the summary reports the median and spread.\n");
    }

    let frames = interop_pass_frames();
    let mut rows: Vec<InteropRow> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for class in bench_media::MEDIA_CLASSES {
        println!(
            ">>> {} — {}x{}@{} ({})",
            class.label, class.width, class.height, class.fps, class.why
        );
        let input = match bench_media::ensure_fixture(&ffmpeg, class) {
            Ok(p) => p,
            Err(why) => {
                println!("    SKIPPED: {why}\n");
                skipped.push((class.label.to_string(), why));
                continue;
            }
        };
        match run_interop_class(
            device,
            class.label,
            &[input.as_path()],
            class.width,
            class.height,
            &capability,
            repeats,
            frames,
        ) {
            Ok(row) => rows.push(row),
            Err(why) => {
                println!("    FAILED: {why}\n");
                skipped.push((class.label.to_string(), why));
            }
        }
    }

    // ── G2f.1(b): four distinct 4K60 sources in one frame ─────────────────────
    //
    // **The cheap falsification, and it is the reason this runs before benchmark 7.**
    // Four 4K sources decoding serially either fit inside ~16.7 ms or they do not,
    // and if they do not then no pipelined row can reach 60 FPS however deep the
    // pipeline is — the decode is on the critical path either way. That makes this an
    // hour that can save building against a target already out of reach, which is the
    // standard order for any optimisation whose payoff is uncertain: measure the bound
    // before building the machine.
    //
    // Four DISTINCT files, not one file opened four times: one `SourceId` means one
    // target and one decoder mutex, which is the single-source case measured above.
    let multi_label = "multi4_4k60";
    println!(
        ">>> {multi_label} — 4 DISTINCT 4K60 sources in one frame (benchmark 5's shape, \
         serial)"
    );
    let mut multi_inputs: Vec<std::path::PathBuf> = Vec::new();
    let mut multi_skip: Option<String> = None;
    for class in bench_media::MULTI_4K60_SOURCES {
        match bench_media::ensure_fixture(&ffmpeg, class) {
            Ok(p) => multi_inputs.push(p),
            Err(why) => {
                // A missing source fails the ROW rather than shrinking it: three
                // sources under a four-source heading is a different workload, and
                // the whole point of this row is the source count.
                multi_skip = Some(format!("{} could not be generated: {why}", class.label));
                break;
            }
        }
    }
    match multi_skip {
        Some(why) => {
            println!("    SKIPPED: {why}\n");
            skipped.push((multi_label.to_string(), why));
        }
        None => {
            let paths: Vec<&Path> = multi_inputs.iter().map(|p| p.as_path()).collect();
            match run_interop_class(
                device,
                multi_label,
                &paths,
                3840,
                2160,
                &capability,
                repeats,
                frames,
            ) {
                Ok(row) => rows.push(row),
                Err(why) => {
                    println!("    FAILED: {why}\n");
                    skipped.push((multi_label.to_string(), why));
                }
            }
        }
    }

    print_interop_summary(&rows, repeats, frames);

    if !skipped.is_empty() {
        println!("{} row(s) did NOT run both arms:", skipped.len());
        for (label, why) in &skipped {
            println!("  - {label}\n      {why}");
        }
        println!();
    }

    !rows.is_empty()
}

/// Where sampled frames are dumped as PNGs, or `None` when nothing asked for them.
///
/// `NEXIR_INTEROP_DUMP=<dir>`. The directory is created if it does not exist, and a
/// failure to create it disables the dump with a printed reason rather than failing
/// the profile — the pictures are a diagnostic, not the measurement.
fn interop_dump_dir() -> Option<std::path::PathBuf> {
    let raw = std::env::var("NEXIR_INTEROP_DUMP").ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    let dir = std::path::PathBuf::from(raw);
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(dir),
        Err(e) => {
            println!("    WARNING: could not create {} ({e}); no frames dumped", dir.display());
            None
        }
    }
}

/// How many frames each arm measures per repeat.
///
/// `MEDIA_PASS_FRAMES` by default, so the rows are directly comparable with
/// `--media`'s. `NEXIR_INTEROP_FRAMES` overrides it, because this profile runs
/// 2 arms × 5 classes × 3 repeats and a shorter pass is the difference between a
/// minute and ten while investigating.
fn interop_pass_frames() -> usize {
    std::env::var("NEXIR_INTEROP_FRAMES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(MEDIA_PASS_FRAMES)
}

/// The summary table: both arms per class, medians with spread, and the pixel
/// comparison that keeps a fast-but-broken path from scoring well.
fn print_interop_summary(rows: &[InteropRow], repeats: usize, frames: usize) {
    if rows.is_empty() {
        return;
    }
    println!("------------------------------------------------------------------------");
    println!("INTEROP SUMMARY — median of {repeats} run(s) per arm, up to {frames} frames each");
    println!("  ms/frame is wall clock over the measured loop; FPS is its reciprocal.");
    println!("  `S` is DISTINCT sources per frame — a ms/frame figure without it is not");
    println!("  comparable to anything, and the four-source row is ~4x the decode work.");
    println!("  `f` is the frames the interop arm actually MEASURED, which is not always");
    println!("  what was asked for: the pass stops cleanly when the scheduler runs dry.");
    println!("  `Upload` is bytes COUNTED by UploadCost, so the interop arm's 0.00 MB is");
    println!("  a count of nothing uploaded — read it beside `tgt` (live targets), or");
    println!("  0 bytes is indistinguishable from a pass that never ran.");
    println!("  `VRAM` is the interop targets' own NV12 size, counted per target and a");
    println!("  LOWER BOUND (no alignment padding, no driver overhead). It is not the");
    println!("  NVML driver row.");
    println!("  `Δpx` is the largest per-channel difference between the two arms' last");
    println!("  rendered frame, sampled on a 16x16 grid: it is what makes \"frames still");
    println!("  decode correctly\" a measurement rather than an assumption.");
    println!("  spread is (max-min)/median; n/a on a single repeat, never 0%.");
    println!("------------------------------------------------------------------------");
    println!(
        "{:<16} {:>11} {:>2} {:>4} {:>10} {:>7} {:>10} {:>7} {:>7} {:>4} {:>9} {:>6}",
        "Class", "Geometry", "S", "f", "interop", "spread", "cpu", "spread", "gain", "tgt",
        "VRAM", "Δpx"
    );
    for row in rows {
        let (i_ms, i_spread) = median_spread(&row.interop);
        let (c_ms, c_spread) = median_spread(&row.cpu);
        // The gain is a ratio of two medians, and it is only printed when both
        // exist: a "1.00x" standing in for a missing arm would be the strongest
        // possible claim from no evidence.
        let gain = match (i_ms, c_ms) {
            (Some(i), Some(c)) if i > 0.0 => Some(c / i),
            _ => None,
        };
        let align = row
            .interop
            .last()
            .zip(row.cpu.last())
            .and_then(|(i, c)| align_samples(&i.samples, &c.samples));
        println!(
            "{:<16} {:>11} {:>2} {:>4} {:>7} ms {:>7} {:>7} ms {:>7} {:>7} {:>4} {:>9} {:>6}",
            row.label,
            format!("{}x{}", row.width, row.height),
            row.sources,
            row.interop.last().map(|r| r.frames).unwrap_or(0),
            i_ms.map(|v| format!("{v:.2}")).unwrap_or_else(|| "n/a".into()),
            i_spread.map(|v| format!("{v:.0}%")).unwrap_or_else(|| "n/a".into()),
            c_ms.map(|v| format!("{v:.2}")).unwrap_or_else(|| "n/a".into()),
            c_spread.map(|v| format!("{v:.0}%")).unwrap_or_else(|| "n/a".into()),
            gain.map(|v| format!("{v:.2}x")).unwrap_or_else(|| "n/a".into()),
            row.interop.last().map(|r| r.live_targets).unwrap_or(0),
            // Counted from the targets, not `tgt * a constant` — G2f.2.
            row.interop
                .last()
                .map(|r| format!("{:.1} MB", r.target_bytes as f64 / (1024.0 * 1024.0)))
                .unwrap_or_else(|| "n/a".into()),
            // The best-alignment delta with its offset: `4@0` is "agrees to 4 levels
            // on the same frame", `3@-1` is "agrees to 3 levels one frame apart",
            // which is a completely different finding.
            align
                .as_ref()
                .map(|a| format!("{}@{:+}", a.delta, a.offset))
                .unwrap_or_else(|| "n/a".into()),
        );
    }
    println!("------------------------------------------------------------------------");
    print_interop_verdicts(rows);
    println!();
}

/// The acceptance criterion, checked and printed rather than left to a reader.
///
/// G2e.2 asks for two things: the counted upload bytes fall to zero on the interop
/// arm **while frames still decode correctly**. Both are cheap to check from what
/// the run already counted, and neither is worth leaving as an exercise — the whole
/// failure mode this task guards against is a row that looks fast because it
/// rendered nothing.
///
/// Each line is a claim about THIS run, so it is printed with the numbers that
/// support it. Nothing here is a threshold on performance: the plan says to state
/// what the figure falls to rather than assert a target.
fn print_interop_verdicts(rows: &[InteropRow]) {
    /// How far apart two decoders' output may be before it is a difference in
    /// PICTURE rather than in rounding.
    ///
    /// The two arms decode the same coded frames with the same decoder; they differ
    /// only in where the frame lands (a CUDA array vs host memory) and therefore in
    /// how the sampler reads it. 8 levels of 255 allows for the chroma siting a
    /// hardware and a software path can disagree on while being far below any real
    /// corruption — a torn, stale or black frame is tens to hundreds of levels out.
    const PIXEL_TOLERANCE: u8 = 8;

    for row in rows {
        let Some(last) = row.interop.last() else { continue };
        let bytes = last.upload_bytes_per_frame().unwrap_or(f64::NAN);

        // 1. Upload bytes gone, and a target actually allocated.
        if last.live_targets == 0 {
            println!(
                "  {}: NOT MEASURED on the interop path — 0 live targets, so the row is \
                 the CPU path.",
                row.label
            );
        } else if bytes == 0.0 {
            println!(
                "  {}: upload bytes 0 with {} live target(s), {:.1} MB of counted VRAM \
                 (lower\n    bound) — nothing crossed the bus, and the pass did run.",
                row.label,
                last.live_targets,
                last.target_bytes as f64 / (1024.0 * 1024.0),
            );
        } else {
            println!(
                "  {}: WARNING: the interop arm still counted {:.2} MB/frame of upload. \
                 Some\n    clip kept its YuvUploadNode; check the fallback lines above.",
                row.label,
                bytes / (1024.0 * 1024.0)
            );
        }

        // 1b. THE SOURCE COUNT THE ROW CLAIMS IS THE SOURCE COUNT IT MEASURED.
        //
        // G2f.4 reads `live_targets == 4` off this row, and three separate things have
        // to line up for that to mean what it says: the harness built N sources, the
        // registry allocated N targets, and every measured FRAME bound N of them. A
        // gap in any of the three is a smaller workload printed under a four-source
        // heading, with no other symptom — the same class of silent shrinkage gotcha 9
        // is about.
        if last.live_targets != row.sources || last.interop_clips != row.sources {
            println!(
                "    WARNING: this row claims {} distinct source(s) but allocated {} \
                 target(s) and\n    the leanest measured frame imported {} of {} clip(s). \
                 The figures above are\n    for a smaller workload than the heading.",
                row.sources, last.live_targets, last.interop_clips, last.clips,
            );
        } else if row.sources > 1 {
            println!(
                "    all {} source(s) decoded into their own textures on every measured \
                 frame\n    ({}/{} clips imported), so this row really is a {}-source \
                 frame.",
                row.sources, last.interop_clips, last.clips, row.sources,
            );
        }

        // 2. The picture survived — and if it differs, WHY.
        let cpu_last = row.cpu.last();
        match cpu_last.and_then(|c| align_samples(&last.samples, &c.samples)) {
            Some(a) => {
                let luma = last
                    .samples
                    .last()
                    .map(|(_, s)| s.mean_luma)
                    .unwrap_or(f64::NAN);
                let cpu_luma = cpu_last
                    .and_then(|c| c.samples.last())
                    .map(|(_, s)| s.mean_luma)
                    .unwrap_or(f64::NAN);
                if a.delta_aligned <= PIXEL_TOLERANCE {
                    println!(
                        "    pixels agree with the CPU path to {}/255 on the same frame \
                         (mean luma\n    {luma:.1} vs {cpu_luma:.1}).",
                        a.delta_aligned
                    );
                } else if a.offset != 0 && a.delta <= PIXEL_TOLERANCE {
                    // The important distinction, and the reason the samples are a run
                    // rather than one frame: the PICTURES are fine, the two paths
                    // simply resolved a pts to different frames. That is a real
                    // difference worth recording, and it is not corruption.
                    println!(
                        "    pixels agree to {}/255 at an offset of {:+} frame(s), but \
                         differ by\n    {}/255 on the frame both arms call the same one. \
                         The pictures are intact;\n    the two paths resolve a pts to \
                         different frames (mean luma {luma:.1} vs {cpu_luma:.1}).",
                        a.delta, a.offset, a.delta_aligned
                    );
                } else {
                    println!(
                        "    WARNING: pixels differ by {}/255 from the CPU path and no \
                         offset within\n    ±{} frame(s) lines them up (best {}/255 at \
                         {:+}). Mean luma {luma:.1} vs {cpu_luma:.1}.\n    The timing row \
                         above is not a like-for-like comparison until this is explained.",
                        a.delta_aligned,
                        FrameSample::RUN,
                        a.delta,
                        a.offset,
                    );
                }
                // Both arms black would agree perfectly, so luma is checked on its
                // own. Every fixture is `testsrc2`, which is nowhere near black.
                if luma < 1.0 {
                    println!(
                        "    WARNING: the interop arm's mean luma is {luma:.2} — it \
                         rendered black, and\n    a black frame agrees with anything."
                    );
                }
            }
            None => println!("    pixels: n/a (no comparable sample was read back)."),
        }

        // 3. The read-after-write guard's own cost — the plan's G2e.4 suspect.
        if let Some(s) = last.interop_stats {
            match (s.waits, s.wait_ms_per_decode()) {
                (0, _) => println!(
                    "    the read-after-write guard never polled: {} decode(s), 0 wait(s).",
                    s.decodes
                ),
                (n, Some(ms)) if ms < 0.5 => println!(
                    "    the read-after-write guard polled on {n} of {} decode(s) at \
                     {ms:.3} ms each\n    — not the bottleneck.",
                    s.decodes
                ),
                (n, Some(ms)) => println!(
                    "    ATTENTION: the read-after-write guard polled on {n} of {} \
                     decode(s) at\n    {ms:.3} ms each. This is G2e.4's suspect; a second \
                     target per source (ping-pong)\n    is the structural fix, at one more \
                     Y/UV pair of VRAM per source.",
                    s.decodes
                ),
                (n, None) => println!("    the guard polled {n} time(s); no decode to divide by."),
            }
        }

        // 4. WHEN THE INTEROP ARM LOSES, say what it lost to.
        //
        // A row where the CPU path is faster is the one result this profile must not
        // leave to a reader, because "the interop path is slower here" has three
        // unrelated causes and they lead opposite ways: the copy we added, the
        // context-wide barrier ahead of it, or an upload that was cheap in this class
        // to begin with. The two arms' own stage figures answer it, so the answer is
        // printed with them.
        let (i_med, _) = median_spread(&row.interop);
        let (c_med, _) = median_spread(&row.cpu);
        let (Some(i_ms), Some(c_ms)) = (i_med, c_med) else {
            continue;
        };
        if i_ms <= c_ms {
            continue;
        }
        let saved = row
            .cpu
            .last()
            .and_then(|c| c.upload_ms())
            .unwrap_or(f64::NAN);
        let cost = last
            .interop_stats
            .and_then(|s| s.copy_ms_per_call())
            .map(|(b, i, s)| (b, i, s, b + i + s));
        match cost {
            Some((barrier, issue, sync, total)) => println!(
                "    SLOWER than the CPU path by {:.2} ms/frame. The upload it removed was \
                 {saved:.2} ms;\n    the copy it added is {total:.2} ms (barrier {barrier:.3} \
                 + issue {issue:.3} + sync {sync:.3}).\n    The remainder is NVDEC's own \
                 decode, which both arms pay — see the per-decode\n    split above.",
                i_ms - c_ms
            ),
            None => println!(
                "    SLOWER than the CPU path by {:.2} ms/frame, and no copy was timed, so \
                 it\n    cannot be attributed from this run.",
                i_ms - c_ms
            ),
        }
    }
}

const BENCHMARKS: &[BenchmarkConfig] = &[
    BenchmarkConfig {
        name: "1. Simple 1080p60 (1 layer, composite) — GPU render only",
        complexity: WorkloadComplexity::Simple,
        path: ExecutionPath::GpuRender,
        source: WorkloadSource::Synthetic,
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
        source: WorkloadSource::Synthetic,
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
        source: WorkloadSource::Synthetic,
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
        source: WorkloadSource::Synthetic,
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
        source: WorkloadSource::Synthetic,
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
        source: WorkloadSource::Synthetic,
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
    // ── G2f.1(a): the pipelined real-media rows ───────────────────────────────
    //
    // **These are new rows beside benchmark 5, never a rewrite of it.** Benchmark 5
    // is the synthetic control and the 56.8 FPS baseline; losing it would leave the
    // comparison to a git checkout, which is the whole thing `--interop`'s
    // interleaved arms exist to avoid.
    //
    // Same Heavy graph, same NVENC path, same `gpu_lookahead`, same 90 frames — fed
    // from four real 4K60 files through `IoLayer` instead of from `make_nv12`. 7 and
    // 8 differ by ONE argument, the `InteropDecodeTargets`, so the pair is a
    // measurement of the decode path rather than of two different builds.
    //
    // **BENCHMARK 7 IS THE 4K60 GATE, and the target is a frame-time budget with a
    // percentile rather than an average**: steady-state interval ≤ 16.67 ms at P95
    // and ≤ 20 ms at P99, median of ≥3 repeats, read off the `↳ steady` row
    // (gotcha 15). Benchmark 5 cannot be the gate however fast it gets — it uploads
    // `make_nv12` bars from host memory with no decoder in the process — and it now
    // PASSES that budget while 7 sits at P95 72.87 ms, which is exactly why the
    // distinction is written down rather than assumed.
    //
    // Both rows print `ALTERNATING`, and per gotcha 15 that fails the target on its
    // own. It is attributed and it is NOT this crate's pipeline: the cycle is the
    // decoder's reorder buffer, it survives `NEXIR_GPU_LOOKAHEAD=1`, and it
    // disappears on the same fixtures re-encoded with `-bf 0` (gotcha 24). The four
    // fixtures are also not four comparable decodes — `cam_4k60_grain` carries ~35×
    // its neighbours' bitrate (gotcha 25 and `bench_media::MULTI_4K60_SOURCES`).
    BenchmarkConfig {
        name: "7. Heavy 4K60 REAL MEDIA (4 layers, 4 DISTINCT files) — INTEROP decode, \
               zero-copy NVENC",
        complexity: WorkloadComplexity::Heavy,
        path: ExecutionPath::GpuNvencZeroCopy,
        source: WorkloadSource::RealMedia4K60(DecodePath::Interop),
        canvas_w: 3840,
        canvas_h: 2160,
        target_fps: 60.0,
        frame_count: 90,
        distinct_sources: 4,
    },
    BenchmarkConfig {
        name: "8. Heavy 4K60 REAL MEDIA (4 layers, 4 DISTINCT files) — CPU upload decode, \
               zero-copy NVENC",
        complexity: WorkloadComplexity::Heavy,
        path: ExecutionPath::GpuNvencZeroCopy,
        source: WorkloadSource::RealMedia4K60(DecodePath::Cpu),
        canvas_w: 3840,
        canvas_h: 2160,
        target_fps: 60.0,
        frame_count: 90,
        distinct_sources: 4,
    },
];

/// How many outlier frames the per-frame dump prints, or `None` when it is off.
///
/// `NEXIR_FRAME_DUMP=1` gives the default 8; any other number sets it. The dump
/// exists for one question — does the frame with the worst latency have the worst
/// upload? — and that question is settled by looking at the pairing, which a
/// percentile cannot show. See `profiling::format_frame_dump`.
fn frame_dump_rows() -> Option<usize> {
    let raw = std::env::var("NEXIR_FRAME_DUMP").ok()?;
    match raw.trim() {
        "" | "0" | "false" | "off" => None,
        "1" | "true" | "on" => Some(8),
        n => n.parse::<usize>().ok().filter(|v| *v > 0).or(Some(8)),
    }
}

/// Where to write a per-frame CSV of this run, or `None` when not requested.
///
/// `NEXIR_FRAME_CSV=path` writes one row per frame: index, latency, every CPU
/// stage, and the GPU spans. Separate from `NEXIR_FRAME_DUMP` because the two
/// answer different questions — the dump ranks outliers and prints coefficients,
/// while the CSV keeps ARRIVAL ORDER, which is the only way a periodic pattern is
/// visible. A tail that alternates every other frame and a tail that stalls once
/// have the same distribution and the same correlations; they differ only in
/// sequence.
///
/// A `%d` in the path is replaced by `<benchmark>_<run>`, so a multi-benchmark
/// multi-repeat invocation does not have every row overwrite the last. Without a
/// `%d` that is exactly what happens, which is why the placeholder is documented
/// rather than optional in practice.
fn frame_csv_path(bench: usize, run: usize) -> Option<std::path::PathBuf> {
    let raw = std::env::var("NEXIR_FRAME_CSV").ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(
        raw.replace("%d", &format!("{bench}_{run}")),
    ))
}

/// Write the per-frame series in arrival order.
///
/// Failure is printed, not swallowed and not fatal: the benchmark's own numbers are
/// unaffected by a CSV that could not be written, but a silently absent file would
/// have the reader analysing a stale one from a previous run.
fn write_frame_csv(path: &std::path::Path, frames: &[FrameProfile]) {
    use std::io::Write;
    let s = frame_csv_text(frames);
    // The shape check runs on every write rather than only in a test, because the
    // file's whole use is to be read column-wise by something outside this process
    // — see `frame_csv_columns_agree`.
    if let Err(why) = frame_csv_columns_agree(&s) {
        println!("    WARNING: not writing {}: {why}", path.display());
        return;
    }
    match std::fs::File::create(path).and_then(|mut fh| fh.write_all(s.as_bytes())) {
        Ok(()) => println!("    Per-frame CSV: {}", path.display()),
        Err(e) => println!("    WARNING: could not write {}: {e}", path.display()),
    }
}

/// The CSV's text, split out from the write so the shape can be checked without a
/// filesystem.
fn frame_csv_text(frames: &[FrameProfile]) -> String {
    let mut s = String::from(
        "frame,latency_ms,cpu_sum_ms,decode_ms,upload_ms,upload_prepare_ms,upload_submit_ms,\
         composite_ms,gpu_submit_ms,gpu_wait_ms,nvenc_ms,gpu_graph_ms,gpu_transfer_ms\n",
    );
    // Empty rather than 0 for anything unmeasured, so a spreadsheet cannot average
    // a missing GPU tick into the series as a fast frame.
    fn o(v: Option<f64>) -> String {
        v.map(|x| format!("{x:.4}")).unwrap_or_default()
    }
    for f in frames {
        s.push_str(&format!(
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{}\n",
            f.frame_index,
            o(f.latency_ms()),
            f.total_time_ms(),
            // `Decode` READ FROM THE RECORDED STAGE, never reconstructed as
            // `cpu_sum - composite - upload`. Task B's whole question is which CPU
            // stage alternates, and a subtraction cannot answer it: `Composite` on
            // the real-media rows is *itself* defined as the submit span minus the
            // accounted stages, so deriving the decode from the same identity would
            // make the two columns algebraically dependent and the answer
            // circular. Gotcha 9's rule — print what was measured.
            f.stage_time_ms(PipelineStage::Decode),
            f.stage_time_ms(PipelineStage::Upload),
            f.stage_time_ms(PipelineStage::UploadPrepare),
            f.stage_time_ms(PipelineStage::UploadSubmit),
            f.stage_time_ms(PipelineStage::Composite),
            f.stage_time_ms(PipelineStage::GpuSubmit),
            f.stage_time_ms(PipelineStage::GpuWait),
            f.stage_time_ms(PipelineStage::Nvenc),
            o(f.gpu_time_ms(PipelineStage::Composite)),
            o(f.gpu_time_ms(PipelineStage::GpuTransfer)),
        ));
    }
    s
}

/// Every row of a frame CSV must carry exactly as many fields as the header names.
///
/// **The failure this catches has no symptom in the file itself.** Adding a column
/// to the header without adding it to the `format!` (or the reverse) shifts every
/// column after the insertion point by one, so a reader analyses `decode_ms` under
/// the heading `upload_ms` and every conclusion drawn from the series is about the
/// wrong stage — which is exactly the analysis the CSV exists for (task B read the
/// alternation off it). A comma count is enough because every field is a number or
/// empty, never a quoted string.
fn frame_csv_columns_agree(csv: &str) -> Result<usize, String> {
    let mut lines = csv.lines();
    let header = lines.next().ok_or("the CSV is empty")?;
    let want = header.matches(',').count() + 1;
    for (i, line) in lines.enumerate() {
        let got = line.matches(',').count() + 1;
        if got != want {
            return Err(format!(
                "row {i} has {got} field(s) but the header names {want}: every column \
                 after the mismatch is attributed to the wrong stage"
            ));
        }
    }
    Ok(want)
}

/// What the real-media rows reached, and whether they measured what they claim.
///
/// **G2f.4's acceptance criterion, printed rather than left to a reader.** The
/// summary table above cannot tell benchmark 7 from benchmark 8: their FPS rows are
/// the same shape, and the only difference is which decode path produced them. So
/// this restates each real-media row against the three things that have to line up
/// for its heading to be true (targets allocated, every frame importing, upload
/// bytes gone) and — when both rows ran — prints the pair's ratio with the pixel
/// comparison beside it.
///
/// **It deliberately does not assert 60 FPS.** The plan's G2f.4 says to state what
/// the figure reaches; G5 is what turns a number into a verdict, and it is stated
/// against `↳ steady` P95/P99 rather than an average (gotcha 15).
fn print_real_media_verdicts(summary: &[BenchSummary]) {
    /// How far apart the two rows' pictures may be before it is a difference in
    /// PICTURE rather than in rounding — the same tolerance and the same reasoning
    /// as `print_interop_verdicts`.
    const PIXEL_TOLERANCE: u8 = 8;

    let media: Vec<(&BenchSummary, &RealMediaProvenance)> = summary
        .iter()
        .filter_map(|s| s.media.as_ref().map(|m| (s, m)))
        .collect();
    if media.is_empty() {
        return;
    }

    println!("REAL-MEDIA ROWS — what each one measured, and what it reaches");
    println!("  These rows drive four real 4K60 files through IoLayer on benchmark 5's");
    println!("  Heavy graph and pipeline. The FPS row above cannot distinguish the two");
    println!("  decode paths, so the counters that can are restated here.");
    println!("  No 60 FPS verdict is asserted: read P95/P99 off the `steady` row of each");
    println!("  table above (gotcha 15), never off an average.");
    println!("------------------------------------------------------------------------");
    for (s, m) in &media {
        let fps = median(s.runs.iter().map(|r| r.fps));
        let p95 = median(s.runs.iter().map(|r| r.lat_p95));
        let p99 = median(s.runs.iter().map(|r| r.lat_p99));
        println!(
            "  {} decode: {fps:.1} FPS median, steady P95 {p95:.2} ms / P99 {p99:.2} ms \
             over\n    {} frame(s) x {} repeat(s).",
            m.decode_path.label(),
            m.frames,
            s.runs.len(),
        );
        // The provenance check, in the same three parts `--interop` uses.
        if m.decode_path == DecodePath::Interop {
            if m.live_targets != m.sources || m.interop_clips != m.sources {
                println!(
                    "    WARNING: this row claims {} distinct source(s) but allocated {} \
                     target(s) and\n    the leanest measured frame imported {} of {} clip(s). \
                     The figures above are\n    for a smaller workload than the heading.",
                    m.sources, m.live_targets, m.interop_clips, m.clips,
                );
            } else {
                println!(
                    "    all {} source(s) decoded into their own textures on every measured \
                     frame\n    ({}/{} clips imported), {:.1} MB of counted VRAM (lower bound).",
                    m.sources,
                    m.interop_clips,
                    m.clips,
                    m.target_bytes as f64 / (1024.0 * 1024.0),
                );
            }
            if m.upload_bytes_per_frame == 0.0 && m.live_targets > 0 {
                println!(
                    "    upload bytes 0 beside {} live target(s) — nothing crossed the bus, \
                     and the\n    pass did run.",
                    m.live_targets
                );
            } else if m.upload_bytes_per_frame > 0.0 {
                println!(
                    "    WARNING: the interop row still counted {:.2} MB/frame of upload. \
                     Some clip\n    kept its YuvUploadNode; check the fallback lines above.",
                    m.upload_bytes_per_frame / (1024.0 * 1024.0)
                );
            }
        }
        // Gotcha 14: the cap is re-read on this graph shape, which nothing else
        // compiles. `peak == cap` means the true peak is unknown and at least the
        // cap, which is exactly when evictions start.
        println!(
            "    pool: peak bucket {}/{}, {} evicted, {} bucket(s){}",
            m.pool.peak_bucket,
            nexir::render::resource::POOL_BUCKET_CAPACITY,
            m.pool.evicted,
            m.pool.buckets,
            if m.pool.peak_bucket >= nexir::render::resource::POOL_BUCKET_CAPACITY {
                " — AT THE CAP, so the real peak is unknown and POOL_BUCKET_CAPACITY \
                 must be re-read (gotcha 14)"
            } else {
                ""
            },
        );
    }

    // ── The pair, when both rows ran ──────────────────────────────────────────
    //
    // A ratio of two medians is only a gain if the two measured the same amount of
    // work, so the frame counts are checked and the disagreement printed rather
    // than assumed — the same rule `run_interop_class` applies to its two arms.
    let interop = media
        .iter()
        .find(|(_, m)| m.decode_path == DecodePath::Interop);
    let cpu = media.iter().find(|(_, m)| m.decode_path == DecodePath::Cpu);
    if let (Some((si, mi)), Some((sc, mc))) = (interop, cpu) {
        let i_fps = median(si.runs.iter().map(|r| r.fps));
        let c_fps = median(sc.runs.iter().map(|r| r.fps));
        println!("------------------------------------------------------------------------");
        if mi.frames != mc.frames {
            println!(
                "  WARNING: the two rows measured different frame counts (interop {} vs \
                 cpu {}).\n  The ratio below divides two medians over different amounts of \
                 work.",
                mi.frames, mc.frames
            );
        }
        if c_fps > 0.0 && i_fps > 0.0 {
            println!(
                "  interop vs CPU upload, pipelined 4K60 real media: {i_fps:.1} vs \
                 {c_fps:.1} FPS\n  ({:.2}x), {:.2} vs {:.2} ms mean frame interval.",
                i_fps / c_fps,
                1000.0 / i_fps,
                1000.0 / c_fps,
            );
        }
        // The pixel check, so a fast-but-broken row cannot score well. One sample
        // per row rather than a run, so an offset cannot be searched for — the two
        // rows sample the same frame index by construction (both stop where their
        // own span runs out, and the count disagreement above is what reports it if
        // they did not).
        match (&mi.sample, &mc.sample) {
            (Some((ii, is)), Some((ci, cs))) if ii == ci => match is.max_delta(cs) {
                Some(d) if d <= PIXEL_TOLERANCE => println!(
                    "  pixels agree to {d}/255 on frame {ii} (mean luma {:.1} vs {:.1}).",
                    is.mean_luma, cs.mean_luma
                ),
                Some(d) => println!(
                    "  WARNING: pixels differ by {d}/255 on frame {ii} (mean luma {:.1} vs \
                     {:.1}).\n  The timing rows are not a like-for-like comparison until \
                     this is explained.",
                    is.mean_luma, cs.mean_luma
                ),
                None => println!("  pixels: n/a (the two samples are not comparable)."),
            },
            (Some((ii, _)), Some((ci, _))) => println!(
                "  pixels: n/a — the two rows sampled different frames ({ii} vs {ci}), so \
                 a\n  difference would say nothing about correctness."
            ),
            _ => println!("  pixels: n/a (no comparable sample was read back)."),
        }
    } else {
        println!("------------------------------------------------------------------------");
        println!(
            "  Only one real-media row ran, so there is no interop-vs-CPU ratio. Run \
             both\n  (`bench.exe 5 7 8`) for the comparison."
        );
    }
    println!("========================================================================\n");
}

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
    // The grade path, printed in the banner for the same reason the upload mechanism
    // is: a figure whose graph shape is invisible cannot be compared with another
    // run's. `NEXIR_FUSE_GRADE=0` is how every pre-P2.3 number is reproduced.
    println!(
        "Grade path   : {} (NEXIR_FUSE_GRADE=0 for the three separate passes)",
        if fuse_grade() {
            "FUSED — colour correction + LUT + chroma key in one compute pass"
        } else {
            "UNFUSED — ColorCorrection -> Lut3D -> ChromaKey, three passes"
        }
    );
    println!();
    if node_timing_frames().is_some() {
        println!(
            "NEXIR_NODE_TIMINGS is set: each benchmark prints per-node GPU timings from \
             an\nEXTRA pass after its measured loop. Those frames are not in the FPS or \
             latency\nrows — bracketing every node changes what is being measured.\n"
        );
    }

    let mut skipped = Vec::new();
    // `--media` and `--export` each replace the synthetic sweep rather than
    // appending to it: the profiles share one GPU and one process, and running the
    // 4K60 rows first would leave the later profile's clocks warmed by them —
    // exactly the position-dependence `WARMUP_DURATION` documents. One profile per
    // invocation, and each is measured from the same starting state.
    let args: Vec<String> = std::env::args().collect();
    let want_media = args.iter().any(|a| a == "--media");
    let want_export = args.iter().any(|a| a == "--export");
    let want_interop = args.iter().any(|a| a == "--interop");
    if want_media || want_export || want_interop {
        let ran = if want_export {
            run_export_profile(&device)
        } else if want_interop {
            run_interop_profile(&device)
        } else {
            run_media_profile(&device)
        };
        if !ran && bench_media::require() {
            eprintln!(
                "NEXIR_REQUIRE_MEDIA is set but nothing ran. Every figure the profile\n\
                 would have reported is absent, so this is a failure rather than a skip."
            );
            std::process::exit(1);
        }
        return;
    }
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
        // What the last completed repeat's workload can attest to. `None` for the
        // synthetic sweep; for a real-media row it is what the summary's provenance
        // check is read off, and it must come from a repeat that actually ran rather
        // than from the config's declaration.
        let mut media: Option<RealMediaProvenance> = None;
        for r in 0..repeats {
            if repeats > 1 {
                println!("    --- repeat {}/{} ---", r + 1, repeats);
            }
            match run_benchmark(&device, bench) {
                Ok(BenchRun { session, media: m }) => {
                    let report = session.generate_report();
                    println!("{}", report.format_table());
                    // P2.1 step 1: the pairing, not the percentiles.
                    //
                    // Behind an env flag because it is per-frame output — 8 rows and
                    // several coefficients per repeat — and because the question it
                    // answers ("is the late frame the slow upload?") is asked while
                    // investigating, not on every run.
                    if frame_dump_rows().is_some() || frame_csv_path(number, r + 1).is_some() {
                        let frames = session.frames_snapshot();
                        if let Some(worst) = frame_dump_rows() {
                            println!(
                                "{}",
                                nexir::profiling::format_frame_dump(&frames, worst)
                            );
                        }
                        if let Some(path) = frame_csv_path(number, r + 1) {
                            write_frame_csv(&path, &frames);
                        }
                    }
                    let lat = report.frame_latency_stats.unwrap_or_default();
                    // Percentiles from the steady-state series, mean from the
                    // whole-run one — see `RunReading::lat_p95`. Falling back to the
                    // whole-run series when there is no steady one (a 1-frame run)
                    // rather than reporting zeros.
                    let steady = report.steady_latency_stats.unwrap_or(lat);
                    runs.push(RunReading {
                        fps: report.average_fps,
                        lat_avg: lat.avg_ms,
                        lat_p95: steady.p95_ms,
                        lat_p99: steady.p99_ms,
                        fill: report.pipeline_fill_ms.unwrap_or(0.0),
                    });
                    media = m;
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
                media,
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
        println!("  Lat avg is the WHOLE-RUN mean interval (it must agree with FPS; both");
        println!("  cover the same seconds). P95/P99 are over the STEADY-STATE intervals,");
        println!("  i.e. after the pipeline fills — the `Fill` column is that first");
        println!("  interval, separated rather than hidden. A ~90-frame run has ~90");
        println!("  intervals, so a whole-run P99 lands on the largest sample, and at 4K");
        println!("  that sample is the fill: reporting it as a frame's latency attributes");
        println!("  a cold-start cost to steady playback. 60 FPS = 16.67 ms.");
        println!("  FPS spread is (max-min)/median: anything above ~10% means one run's");
        println!("  number is not a result on its own.");
        println!("------------------------------------------------------------------------");
        println!(
            "{:<34} {:>5} {:>8} {:>7} {:>9} {:>9} {:>9} {:>9}",
            "Benchmark", "L:S", "FPS", "spread", "Lat avg", "P95 std", "P99 std", "Fill"
        );
        for s in &summary {
            let short: String = s.name.chars().take(32).collect();
            let fps = median(s.runs.iter().map(|r| r.fps));
            let spread = if fps > 0.0 {
                let lo = s.runs.iter().map(|r| r.fps).fold(f64::MAX, f64::min);
                let hi = s.runs.iter().map(|r| r.fps).fold(f64::MIN, f64::max);
                (hi - lo) / fps * 100.0
            } else {
                0.0
            };
            println!(
                "{:<34} {:>5} {:>8.1} {:>6.0}% {:>6.2} ms {:>6.2} ms {:>6.2} ms {:>6.2} ms",
                short,
                format!("{}:{}", s.layers, s.sources),
                fps,
                spread,
                median(s.runs.iter().map(|r| r.lat_avg)),
                median(s.runs.iter().map(|r| r.lat_p95)),
                median(s.runs.iter().map(|r| r.lat_p99)),
                median(s.runs.iter().map(|r| r.fill)),
            );
        }
        println!("========================================================================\n");
        print_real_media_verdicts(&summary);
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

#[cfg(test)]
mod tests {
    use super::*;
    use nexir::profiling::{FrameProfile, PipelineStage};
    use std::time::Duration;

    /// The CSV's rows must carry exactly the columns its header names.
    ///
    /// **This is the regression test for the `decode_ms` column task B added.** The
    /// file is read column-wise by something outside this process, so a header that
    /// names one more field than the rows carry shifts every later column onto the
    /// wrong stage — and the file still parses, still has the right row count, and
    /// still looks like a valid series. Task B's whole conclusion (the alternation is
    /// in `Decode`, not in `Upload`) came off this file, so a silent shift would have
    /// pointed the investigation at whichever stage happened to land in that column.
    #[test]
    fn every_frame_csv_row_has_the_columns_the_header_names() {
        let mut frames = Vec::new();
        for i in 0..3 {
            let mut f = FrameProfile::new(i, i as i64 * 1500);
            f.record_stage(PipelineStage::Decode, Duration::from_micros(40_000));
            f.record_stage(PipelineStage::Composite, Duration::from_micros(700));
            f.record_latency(Duration::from_micros(16_670));
            // GPU ticks on only some frames, so the `Option` columns are exercised
            // both ways: an unmeasured tick must be EMPTY rather than 0, and an empty
            // field still has to be counted as a field.
            if i == 1 {
                f.record_gpu(PipelineStage::GpuTransfer, Duration::from_micros(9_000));
            }
            frames.push(f);
        }
        let csv = frame_csv_text(&frames);
        let columns = frame_csv_columns_agree(&csv).expect("the CSV's shape must be consistent");

        // Named explicitly rather than derived from the header, so adding a column
        // without adding it to the writer fails here rather than shifting silently.
        let header = csv.lines().next().expect("the CSV has a header");
        assert_eq!(
            header,
            "frame,latency_ms,cpu_sum_ms,decode_ms,upload_ms,upload_prepare_ms,\
             upload_submit_ms,composite_ms,gpu_submit_ms,gpu_wait_ms,nvenc_ms,\
             gpu_graph_ms,gpu_transfer_ms"
        );
        assert_eq!(columns, 13);
        assert_eq!(csv.lines().count(), 1 + frames.len());

        // `decode_ms` is the 4th column (0-indexed 3) and must be the RECORDED stage,
        // not `cpu_sum - composite`: the two differ here by construction, and a
        // subtraction would put 40.0 in the column on a frame whose recorded decode is
        // 40.0 only by coincidence of this fixture.
        let row = csv.lines().nth(1).expect("one data row");
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.len(), columns);
        assert_eq!(fields[3], "40.0000", "decode_ms must be the recorded stage");
        // And an unmeasured GPU tick is empty rather than zero — gotcha 9's rule,
        // which only holds if the empty field is still a field.
        assert_eq!(fields[12], "", "an unmeasured GPU span must print empty");
    }

    /// A shifted CSV must be REFUSED rather than written.
    ///
    /// Falsifies the check above: without it, `every_frame_csv_row_has_the_columns…`
    /// would pass against a `frame_csv_columns_agree` that returned `Ok` for
    /// everything.
    #[test]
    fn a_row_with_the_wrong_column_count_is_rejected() {
        let shifted = "frame,latency_ms,decode_ms\n0,16.67,40.0,0.7\n";
        let err = frame_csv_columns_agree(shifted)
            .expect_err("4 fields under a 3-column header must be rejected");
        assert!(
            err.contains("4 field(s)") && err.contains("3"),
            "the error must name both counts so the mismatch is actionable: {err}"
        );
    }

    /// Four passes of one shader at different costs must not report as one figure.
    ///
    /// **This is the regression test for gotcha 27.** The per-node table's `per pass`
    /// column is a median POOLED over every bracket carrying that name, so on a row
    /// whose four layers hold four different pictures it is the median of a
    /// multi-modal sample — and `×/fr × per pass` then understates `per frame` by the
    /// spread. Measured 22-30% low on benchmark 7's real fixtures, which is what sent
    /// P2.4 after `Composite` for being "24% dearer than benchmark 5's" when the
    /// truth was that three of its four inputs were cheap and one was not.
    ///
    /// The fixture is the measured shape: three instances at ~0.27 ms and one at
    /// ~0.57 (`target/taskH_mixed_3x.txt`, three clean sources plus one noisy).
    #[test]
    fn a_node_whose_instances_differ_is_reported_per_instance() {
        // 5 frames, 4 instances each: three cheap, the last dear.
        let frames: Vec<Vec<(String, f64)>> = (0..5)
            .map(|_| {
                vec![
                    ("ColorCorrection".to_string(), 0.27),
                    ("ColorCorrection".to_string(), 0.27),
                    ("ColorCorrection".to_string(), 0.27),
                    ("ColorCorrection".to_string(), 0.57),
                    ("Composite".to_string(), 1.32),
                ]
            })
            .collect();
        let rows = aggregate_node_timings(&frames);
        let cc = rows
            .iter()
            .find(|r| r.name == "ColorCorrection")
            .expect("the aggregate must carry the node");

        assert_eq!(cc.per_frame, 4, "four brackets is four instances");
        assert_eq!(cc.instance_ms.len(), 4, "each instance gets its own median");
        // Instance order is EXECUTION order, so the dear pass stays identifiable.
        assert!(
            (cc.instance_ms[3] - 0.57).abs() < 1e-9,
            "the dear instance must stay in its own slot: {:?}",
            cc.instance_ms
        );
        // The frame total is the sum, and it is the figure that is right.
        assert!(
            (cc.frame_total_ms - 1.38).abs() < 1e-9,
            "per-frame must be the sum of the instances, got {}",
            cc.frame_total_ms
        );
        // And the pooled median understates it — the exact trap, asserted rather
        // than described.
        let implied = cc.per_frame as f64 * cc.median_ms;
        assert!(
            implied < cc.frame_total_ms - 0.1,
            "the pooled median × count ({implied:.3}) must be visibly below the \
             measured per-frame total ({:.3}), or this test is not exercising the \
             multi-modal case",
            cc.frame_total_ms
        );
        let spread = cc
            .instance_spread_percent()
            .expect("a 4-instance node has a spread");
        assert!(
            spread > NODE_INSTANCE_SPREAD_WARN,
            "a 0.27/0.57 ms split must trip the UNEVEN threshold, got {spread:.0}%"
        );

        // The control: a node whose instances agree must NOT be flagged, or the
        // warning fires on every row and stops meaning anything.
        let composite = rows
            .iter()
            .find(|r| r.name == "Composite")
            .expect("the single-instance node must be there too");
        assert_eq!(composite.instance_spread_percent(), None);
        assert!(
            (composite.frame_total_ms - composite.per_frame as f64 * composite.median_ms).abs()
                < 1e-9,
            "for an even node the pooled median × count IS the per-frame total"
        );
    }

    /// Instances that agree stay one figure, and the spread stays under the warning.
    ///
    /// Falsifies the test above: without this, `instance_spread_percent` could return
    /// something large for every node and the UNEVEN annotation would be noise.
    /// The numbers are benchmark 5's measured four-layer spread (0.301-0.301 ms per
    /// LUT pass across three repeats, `target/taskH_composite_3x.txt`).
    #[test]
    fn instances_that_cost_the_same_are_not_flagged() {
        let frames: Vec<Vec<(String, f64)>> = (0..5)
            .map(|_| {
                vec![
                    ("Lut3D".to_string(), 0.301),
                    ("Lut3D".to_string(), 0.302),
                    ("Lut3D".to_string(), 0.301),
                    ("Lut3D".to_string(), 0.300),
                ]
            })
            .collect();
        let rows = aggregate_node_timings(&frames);
        let lut = &rows[0];
        let spread = lut.instance_spread_percent().expect("four instances");
        assert!(
            spread <= NODE_INSTANCE_SPREAD_WARN,
            "a 0.300-0.302 ms spread is measurement noise, not unevenness: {spread:.1}%"
        );
        assert_eq!(lut.per_frame, 4);
        assert!((lut.frame_total_ms - 1.204).abs() < 1e-9);
    }
}
