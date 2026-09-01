// examples/g1_decode_probe.rs
//
// Task G1 — establish the current state of GPU-native decode as a MEASUREMENT.
//
// The plan's question is not "is the interop code correct" (`tests::interop_correctness`
// already answers that) but three things the production path depends on and which no
// existing test or benchmark reports:
//
//   1. Is a hardware decoder selected at all on this machine, and which one?
//   2. What does the CPU round-trip actually cost per frame — the thing
//      `decode_into(&pkt, mapped, None)` in `src/io/io_layer.rs` forces?
//   3. Does the interop path work on a real H.264 file, and what does it cost?
//
// The answer decides G2's value, so it is measured on real media rather than argued
// from the code. Every number here is timed with `Instant::now()` around the call
// being attributed, and a missing capability prints a skip with a reason — never a
// zero row (AGENTS.md gotcha 9).
//
// Run:
//     cargo build --example g1_decode_probe --release
//     cp target/debug/*.dll target/release/     # cuda.dll beside the binary
//     RUST_LOG=info ./target/release/examples/g1_decode_probe.exe <file.mp4> [more.mp4 ...]
//
// With no arguments it looks for the fixtures `bench --media` writes (see
// `media_fixtures`), and explains how to make them if they are absent.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use nexir::interop::capability::InteropCapability;
use nexir::interop::cuda_context::CudaContext;
use nexir::interop::decode_interop::DecodeInteropTarget;
use nexir::io::decoder::Decoder;
use nexir::io::demuxer::Demuxer;
use nexir::render::device::GpuDevice;

/// How many frames each timing covers. Small on purpose: this probe answers
/// "which path, and roughly what does it cost", and a longer run would make the
/// fixture's own length the limiting factor rather than the decoder's.
const FRAMES: usize = 60;

/// Median of a set of per-frame durations.
///
/// Median rather than mean because the first frames after an open carry the
/// decoder's own initialisation and a keyframe, and a mean over 60 samples lets
/// those two dominate a figure meant to describe steady decoding.
fn median(mut v: Vec<Duration>) -> Option<Duration> {
    if v.is_empty() {
        return None;
    }
    v.sort();
    Some(v[v.len() / 2])
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// What one path reported for one file.
struct PathReading {
    frames: usize,
    median: Option<Duration>,
    total: Duration,
}

impl PathReading {
    fn print(&self, label: &str) {
        match self.median {
            Some(m) => println!(
                "    {label:<28} {:>7.2} ms/frame (median of {}), {:>6.2} ms total",
                ms(m),
                self.frames,
                ms(self.total)
            ),
            None => println!("    {label:<28} no frames decoded"),
        }
    }
}

/// Decode through the CPU staging path — `decode_into(.., None)`, exactly what
/// `IoLayer` does today.
fn time_cpu_path(path: &Path) -> Result<PathReading, String> {
    let mut demuxer = Demuxer::open(path).map_err(|e| format!("{e:?}"))?;
    let stream = demuxer
        .video_stream
        .clone()
        .ok_or_else(|| "no video stream".to_string())?;
    let w = stream.width.unwrap_or(1920) as usize;
    let h = stream.height.unwrap_or(1080) as usize;
    let mut decoder =
        Decoder::open(&stream, stream.codecpar, true).map_err(|e| format!("{e:?}"))?;

    // Generously sized: 4:4:4 16-bit would be 6 bytes/pixel, and an undersized
    // destination is a silent truncation rather than an error.
    let mut dst = vec![0u8; w * h * 6];
    let mut per_frame = Vec::with_capacity(FRAMES);
    let total_start = Instant::now();
    while per_frame.len() < FRAMES {
        let Some(pkt) = demuxer.next_video_packet().map_err(|e| format!("{e:?}"))? else {
            break;
        };
        let t = Instant::now();
        match decoder.decode_into(&pkt, &mut dst, None) {
            Ok(Some(_)) => per_frame.push(t.elapsed()),
            // A packet the decoder swallowed without emitting is real work but not
            // a frame; not timed, because dividing it into a per-frame figure
            // would report a cost for a frame that did not arrive.
            Ok(None) => {}
            Err(e) => return Err(format!("decode failed: {e:?}")),
        }
    }
    Ok(PathReading {
        frames: per_frame.len(),
        median: median(per_frame),
        total: total_start.elapsed(),
    })
}

/// Decode through `DecodeInteropTarget` — NVDEC's device memory copied straight
/// into shared Y/UV textures, no CPU staging buffer touched.
fn time_interop_path(
    path: &Path,
    device: &GpuDevice,
    cuda_ctx: &std::sync::Arc<CudaContext>,
    capability: &InteropCapability,
) -> Result<PathReading, String> {
    let mut demuxer = Demuxer::open(path).map_err(|e| format!("{e:?}"))?;
    let stream = demuxer
        .video_stream
        .clone()
        .ok_or_else(|| "no video stream".to_string())?;
    let w = stream.width.unwrap_or(1920);
    let h = stream.height.unwrap_or(1080);
    let mut decoder =
        Decoder::open(&stream, stream.codecpar, true).map_err(|e| format!("{e:?}"))?;

    let target = DecodeInteropTarget::new(
        std::sync::Arc::clone(cuda_ctx),
        device,
        capability.transport,
        w,
        h,
    )
    .map_err(|e| format!("DecodeInteropTarget::new failed: {e:?}"))?;

    // Deliberately tiny, and deliberately still passed: `decode_into` takes `dst`
    // by `&mut [u8]` whether or not it uses it, and the interop branch returns
    // before touching it. A 1-byte slice that survives 60 frames is itself the
    // evidence that the CPU buffer is not in the path — if the interop branch were
    // ever bypassed, this would panic on the slice bounds rather than quietly
    // reporting a CPU round-trip as a GPU one.
    let mut unused = [0u8; 1];
    let mut per_frame = Vec::with_capacity(FRAMES);
    let total_start = Instant::now();
    while per_frame.len() < FRAMES {
        let Some(pkt) = demuxer.next_video_packet().map_err(|e| format!("{e:?}"))? else {
            break;
        };
        let t = Instant::now();
        match decoder.decode_into(&pkt, &mut unused, Some((cuda_ctx, &target, capability))) {
            Ok(Some(_)) => per_frame.push(t.elapsed()),
            Ok(None) => {}
            Err(e) => return Err(format!("decode failed: {e:?}")),
        }
    }
    Ok(PathReading {
        frames: per_frame.len(),
        median: median(per_frame),
        total: total_start.elapsed(),
    })
}

/// The fixtures `bench --media` generates, if they are there.
fn default_fixtures() -> Vec<PathBuf> {
    let dir = nexir::bench_media::fixture_dir();
    nexir::bench_media::MEDIA_CLASSES
        .iter()
        .map(|c| dir.join(c.file_name()))
        .filter(|p| p.exists())
        .collect()
}

fn main() {
    // `env_logger` so `[decoder] Attached hardware decoder: ...` (src/io/decoder.rs:139)
    // reaches the terminal — that log line is half of what G1 asks for, and it is
    // emitted at info level.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("\n=== Task G1 — is NVDEC selected, and what does the CPU round-trip cost? ===\n");

    let args: Vec<String> = std::env::args().skip(1).collect();
    let files: Vec<PathBuf> = if args.is_empty() {
        let found = default_fixtures();
        if found.is_empty() {
            println!(
                "SKIP: no input files. Pass one or more media paths, or generate the\n\
                 bench fixtures first:\n\n    \
                 ./target/release/bench.exe --media\n\n\
                 (which writes them to {})",
                nexir::bench_media::fixture_dir().display()
            );
            return;
        }
        println!(
            "No paths given; using the {} bench fixture(s) in {}\n",
            found.len(),
            nexir::bench_media::fixture_dir().display()
        );
        found
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    // ── The interop side of the answer ────────────────────────────────────────
    // Probed once. Unavailable is outcome 3 in the plan: the phase is
    // unverifiable on this machine and must be reported as such rather than
    // implemented blind.
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
            "UNAVAILABLE — the interop column below will be skipped".to_string()
        }
    );
    let cuda_ctx = if capability.is_available() {
        match CudaContext::new(&capability) {
            Ok(c) => Some(std::sync::Arc::new(c)),
            Err(e) => {
                println!("CUDA context : failed to create ({e:?}) — interop column skipped");
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

        // The hardware-decoder log line is emitted by `Decoder::open` inside
        // `time_cpu_path`, so it appears above this block's numbers.
        match time_cpu_path(path) {
            Ok(r) => r.print("CPU round-trip (None)"),
            Err(e) => println!("    CPU round-trip: FAILED — {e}"),
        }
        match (&cuda_ctx, capability.is_available()) {
            (Some(ctx), true) => match time_interop_path(path, &device, ctx, &capability) {
                Ok(r) => r.print("GPU interop (Some(..))"),
                Err(e) => println!("    GPU interop: FAILED — {e}"),
            },
            _ => println!("    GPU interop (Some(..))       skipped: interop unavailable"),
        }
        println!();
    }

    println!(
        "Read the `[decoder] Attached hardware decoder:` line(s) above for which decoder\n\
         FFmpeg selected. `Cuda` means NVDEC is in play and the CPU figure above is the\n\
         cost `io_layer.rs:175`'s `None` argument forces on every playback and export\n\
         frame. `None` means the work is in src/io/ffi/hw_accel.rs first and G2 cannot be\n\
         evaluated until then."
    );
}
