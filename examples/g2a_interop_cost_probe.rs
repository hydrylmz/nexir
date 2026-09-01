// examples/g2a_interop_cost_probe.rs
//
// Task G2a — the one measurement that can kill Task G2 cheaply, taken BEFORE the
// ownership refactor rather than after it.
//
// `.hermes/plans/2026-08-31_g2-recosting.md` estimates that wiring
// `DecodeInteropTarget` into `IoLayer` removes the 10.00 ms `GPU transfer` row from
// the 4K frame, because a device→array `cuMemcpy2DAsync` runs at VRAM bandwidth
// rather than over PCIe. Every part of that estimate is quoted from a measured
// bench row EXCEPT one risk, and this probe is that risk:
//
//   `DecodeInteropTarget::copy_from_nvdec_frame` ends in `cuStreamSynchronize`
//   (src/interop/decode_interop.rs:153) — a blocking, whole-stream wait, once per
//   call. Benchmark 5's shape is FOUR distinct 4K sources per frame, so the
//   production path would perform four of those per frame, on a pipeline whose
//   entire advantage is that the CPU runs ahead of the GPU (CPU share 5.82 ms of a
//   17.85 ms frame). Four blocking syncs could cost more than the transfer they
//   save, and no existing test or benchmark reports them.
//
// So: one decoder and one interop target PER SOURCE, N sources, and a "frame" is
// all N decodes together — because one sync per frame is uninteresting and four
// might not be. The same N decodes on the CPU path are the baseline.
//
// WHAT THIS DOES NOT CLAIM. The sync cannot be bracketed from out here: it happens
// inside `Decoder::emit_frame`, which this probe calls rather than contains. So the
// figure below is the whole per-frame decode cost on each path, and the DIFFERENCE
// is the decode-side effect of the interop branch — copy and sync included, upload
// excluded, since neither pass hands anything to the graph. The 10.00 ms transfer
// row that G2 removes is measured by `bench.exe 5`, not here. Attributing the delta
// to the sync specifically would need a bracket inside `decode_interop.rs`.
//
// Every figure is timed with `Instant::now()` around the call it is attributed to.
// A missing capability is a printed skip with a reason, never a zero row (AGENTS.md
// gotcha 9), and each pass reports a median because the first frames after an open
// carry decoder initialisation and a keyframe.
//
// Run:
//     cargo build --example g2a_interop_cost_probe --release
//     cp target/debug/*.dll target/release/examples/      # cuda.dll beside the binary
//     ./target/release/examples/g2a_interop_cost_probe.exe [file.mp4 ...]
//
// With no arguments it uses the 4K fixtures `bench --media` writes, because the row
// being costed is the 4K one.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use nexir::interop::capability::InteropCapability;
use nexir::interop::cuda_context::CudaContext;
use nexir::interop::decode_interop::DecodeInteropTarget;
use nexir::io::decoder::Decoder;
use nexir::io::demuxer::Demuxer;
use nexir::render::device::GpuDevice;

/// Sources decoded per frame — benchmark 5's shape, which is the row Task G2 exists
/// to fix. The multiplicity is the whole point of the probe.
const SOURCES: usize = 4;

/// Frames per pass. Enough for a median that is not dominated by the open, short
/// enough that the 120-frame fixtures are not exhausted.
const FRAMES: usize = 50;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Median of a set of per-frame durations, or `None` when nothing was measured.
fn median(mut v: Vec<Duration>) -> Option<Duration> {
    if v.is_empty() {
        return None;
    }
    v.sort();
    Some(v[v.len() / 2])
}

/// One pass's per-frame cost, where a "frame" is all `SOURCES` decodes.
struct PassReading {
    /// Frames for which every source produced a picture.
    frames: usize,
    /// Median whole-frame cost, i.e. all sources together.
    median: Option<Duration>,
}

impl PassReading {
    fn print(&self, label: &str) {
        match self.median {
            Some(m) => println!(
                "    {label:<34} {:>7.2} ms/frame over {} source(s) (median of {})",
                ms(m),
                SOURCES,
                self.frames
            ),
            None => println!("    {label:<34} no frames decoded"),
        }
    }
}

/// One source: its own demuxer and decoder, so N of these are N independent decode
/// streams exactly as N distinct timeline sources would be.
struct Source {
    demuxer: Demuxer,
    decoder: Decoder,
}

fn open_sources(path: &Path) -> Result<(Vec<Source>, u32, u32), String> {
    let mut sources = Vec::with_capacity(SOURCES);
    let mut geometry = (0u32, 0u32);
    for _ in 0..SOURCES {
        let demuxer = Demuxer::open(path).map_err(|e| format!("Demuxer::open failed: {e:?}"))?;
        let stream = demuxer
            .video_stream
            .clone()
            .ok_or_else(|| "no video stream".to_string())?;
        geometry = (
            stream.width.ok_or("the stream has no width")?,
            stream.height.ok_or("the stream has no height")?,
        );
        // Hardware decode ENABLED, because that is what `IoLayer` does and the whole
        // question is what happens to an NVDEC surface afterwards.
        let decoder = Decoder::open(&stream, stream.codecpar, true)
            .map_err(|e| format!("Decoder::open failed: {e:?}"))?;
        sources.push(Source { demuxer, decoder });
    }
    Ok((sources, geometry.0, geometry.1))
}

/// What one `decode_into` call produced.
///
/// Three outcomes, not two, and the distinction is what makes the loops below
/// correct: frame-level threading holds ~10 frames back, so the first several calls
/// on every source return `Pending`. Treating that as end-of-input reports "no
/// frames decoded" on a perfectly good file — which is exactly what a two-state
/// return did here.
enum DecodeStep {
    /// A picture came out.
    Frame,
    /// The decoder wants more input; the file still has packets.
    Pending,
    /// The demuxer is out of packets.
    Eof,
}

/// Feed one packet and take at most one frame.
fn decode_one(
    src: &mut Source,
    dst: &mut [u8],
    interop: Option<(&CudaContext, &DecodeInteropTarget, &InteropCapability)>,
) -> Result<DecodeStep, String> {
    let Some(pkt) = src
        .demuxer
        .next_video_packet()
        .map_err(|e| format!("next_video_packet failed: {e:?}"))?
    else {
        return Ok(DecodeStep::Eof);
    };
    match src.decoder.decode_into(&pkt, dst, interop) {
        Ok(Some(_)) => Ok(DecodeStep::Frame),
        Ok(None) => Ok(DecodeStep::Pending),
        Err(e) => Err(format!("decode failed: {e:?}")),
    }
}

/// Feed packets to one source until it emits a picture, or the file ends.
///
/// Necessary rather than convenient: frame-level threading means one packet in does
/// not mean one frame out, so a loop that fed exactly one packet per source per
/// frame would drift further behind on every iteration and, during the first ~10
/// calls, never see a frame at all.
fn decode_until_frame(
    src: &mut Source,
    dst: &mut [u8],
    interop: Option<(&CudaContext, &DecodeInteropTarget, &InteropCapability)>,
) -> Result<bool, String> {
    loop {
        match decode_one(src, dst, interop)? {
            DecodeStep::Frame => return Ok(true),
            DecodeStep::Pending => continue,
            DecodeStep::Eof => return Ok(false),
        }
    }
}

/// N independent decoders on the CPU round-trip — what `io_layer.rs:175` does today.
fn time_cpu_frames(path: &Path) -> Result<PassReading, String> {
    let (mut sources, w, h) = open_sources(path)?;
    // Generously sized: an undersized destination is a silent truncation rather than
    // an error.
    let mut dsts: Vec<Vec<u8>> = (0..SOURCES)
        .map(|_| vec![0u8; w as usize * h as usize * 6])
        .collect();

    // Untimed priming: the first frame out of each decoder carries the open, the
    // keyframe and the thread-delay fill. Charging that to frame 0 would put a
    // one-off cost in a per-frame median.
    for (i, src) in sources.iter_mut().enumerate() {
        if !decode_until_frame(src, &mut dsts[i], None)? {
            return Err(format!("source {i} produced no frame at all"));
        }
    }

    let mut per_frame = Vec::with_capacity(FRAMES);
    'outer: while per_frame.len() < FRAMES {
        let t = Instant::now();
        for (i, src) in sources.iter_mut().enumerate() {
            // A frame is all SOURCES pictures: if one source runs dry the frame is
            // incomplete, and timing it would report a whole-frame cost for partial
            // work.
            if !decode_until_frame(src, &mut dsts[i], None)? {
                break 'outer;
            }
        }
        per_frame.push(t.elapsed());
    }

    Ok(PassReading {
        frames: per_frame.len(),
        median: median(per_frame),
    })
}

/// The same N decoders through `DecodeInteropTarget` — one target per source, since
/// the target's textures ARE that source's Y/UV planes and cannot be shared.
fn time_interop_frames(
    path: &Path,
    device: &GpuDevice,
    cuda_ctx: &std::sync::Arc<CudaContext>,
    capability: &InteropCapability,
) -> Result<PassReading, String> {
    let (mut sources, w, h) = open_sources(path)?;
    let mut targets = Vec::with_capacity(SOURCES);
    for i in 0..SOURCES {
        targets.push(
            DecodeInteropTarget::new(
                std::sync::Arc::clone(cuda_ctx),
                device,
                capability.transport,
                w,
                h,
            )
            .map_err(|e| format!("DecodeInteropTarget::new failed for source {i}: {e:?}"))?,
        );
    }

    // Deliberately one byte, and deliberately still passed: `decode_into` takes `dst`
    // by `&mut [u8]` whether or not it uses it, and the interop branch returns before
    // touching it. A 1-byte slice surviving the run is the evidence that the CPU
    // buffer is not in the path — if the branch were ever bypassed this would panic
    // on the slice bounds rather than quietly reporting a CPU round-trip as a GPU one.
    let mut unused = [0u8; 1];

    for (i, src) in sources.iter_mut().enumerate() {
        if !decode_until_frame(
            src,
            &mut unused,
            Some((cuda_ctx.as_ref(), &targets[i], capability)),
        )? {
            return Err(format!("source {i} produced no frame at all"));
        }
    }

    let mut per_frame = Vec::with_capacity(FRAMES);
    'outer: while per_frame.len() < FRAMES {
        let t = Instant::now();
        for (i, src) in sources.iter_mut().enumerate() {
            if !decode_until_frame(
                src,
                &mut unused,
                Some((cuda_ctx.as_ref(), &targets[i], capability)),
            )? {
                break 'outer;
            }
        }
        per_frame.push(t.elapsed());
    }

    Ok(PassReading {
        frames: per_frame.len(),
        median: median(per_frame),
    })
}

/// The 4K fixtures `bench --media` generates. 4K only: the `GPU transfer` row being
/// costed is the 4K one, and a 1080p answer would end up quoted for it.
fn default_fixtures() -> Vec<PathBuf> {
    let dir = nexir::bench_media::fixture_dir();
    nexir::bench_media::MEDIA_CLASSES
        .iter()
        .filter(|c| c.width >= 3840)
        .map(|c| dir.join(c.file_name()))
        .filter(|p| p.exists())
        .collect()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!(
        "\n=== Task G2a — what do {SOURCES} interop copies per frame cost? ===\n\n\
         `DecodeInteropTarget::copy_from_nvdec_frame` ends in a blocking\n\
         `cuStreamSynchronize`. Benchmark 5's shape is {SOURCES} distinct 4K sources per\n\
         frame, so the production path would do {SOURCES} of those per frame. This probe\n\
         reports whether that is cheaper than the CPU round-trip it replaces, at the\n\
         benchmark's own multiplicity, BEFORE the ownership refactor is written.\n"
    );

    let args: Vec<String> = std::env::args().skip(1).collect();
    let files: Vec<PathBuf> = if args.is_empty() {
        let found = default_fixtures();
        if found.is_empty() {
            println!(
                "SKIP: no 4K input files. Pass paths, or generate the bench fixtures:\n\n    \
                 ./target/release/bench.exe --media\n\n\
                 (which writes them to {})",
                nexir::bench_media::fixture_dir().display()
            );
            return;
        }
        println!(
            "No paths given; using the {} 4K bench fixture(s) in {}\n",
            found.len(),
            nexir::bench_media::fixture_dir().display()
        );
        found
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let device = match pollster::block_on(GpuDevice::new_headless()) {
        Ok(d) => d,
        Err(e) => {
            println!("SKIP: no GPU device ({e:?}) — nothing here can be measured.");
            return;
        }
    };
    let capability = InteropCapability::probe(&device);
    println!(
        "CUDA interop : {}",
        if capability.is_available() {
            format!("available (transport={:?})", capability.transport)
        } else {
            "UNAVAILABLE — the interop pass is skipped, so G2a is unanswered here".to_string()
        }
    );
    let cuda_ctx = if capability.is_available() {
        match CudaContext::new(&capability) {
            Ok(c) => Some(std::sync::Arc::new(c)),
            Err(e) => {
                println!("CUDA context : failed to create ({e:?}) — interop pass skipped");
                None
            }
        }
    } else {
        None
    };
    println!();

    for path in &files {
        println!("── {} ──", path.display());
        if !path.exists() {
            println!("    SKIP: no such file\n");
            continue;
        }

        let cpu = match time_cpu_frames(path) {
            Ok(r) => {
                r.print("CPU round-trip (None)");
                r.median
            }
            Err(e) => {
                println!("    CPU round-trip: FAILED — {e}");
                None
            }
        };
        let interop = match (&cuda_ctx, capability.is_available()) {
            (Some(ctx), true) => match time_interop_frames(path, &device, ctx, &capability) {
                Ok(r) => {
                    r.print("GPU interop (Some(..))");
                    r.median
                }
                Err(e) => {
                    println!("    GPU interop: FAILED — {e}");
                    None
                }
            },
            _ => {
                println!("    GPU interop (Some(..))             skipped: interop unavailable");
                None
            }
        };

        // The verdict, in the quantity the re-costing needs: is the interop path
        // cheaper per frame at this multiplicity, and by how much.
        if let (Some(c), Some(i)) = (cpu, interop) {
            let delta = ms(c) - ms(i);
            println!(
                "    → interop is {:.2} ms/frame {} at {SOURCES} source(s)",
                delta.abs(),
                if delta >= 0.0 { "CHEAPER" } else { "MORE EXPENSIVE" }
            );
            if delta < 0.0 {
                println!(
                    "      G2's estimate assumes the decode side does not get WORSE. It did,\n\
                      so the stream sync in copy_from_nvdec_frame needs replacing with an\n\
                      exported CUDA semaphore before G2 is worth starting."
                );
            }
        }
        println!();
    }

    println!(
        "Both passes decode the SAME file {SOURCES} times through {SOURCES} independent\n\
         decoders, which is benchmark 5's shape (4 layers over 4 distinct sources).\n\
         Neither pass uploads to the graph — the CPU pass stops at the staging buffer\n\
         and the interop pass at the Y/UV textures — so the delta above is the\n\
         DECODE-SIDE effect only, copy and sync included. The 10.00 ms `GPU transfer`\n\
         row G2 removes is measured by `bench.exe 5`, not here."
    );
}
