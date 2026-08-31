// src/bench_media.rs
//
// Real-media fixtures, generated locally with the `ffmpeg` BINARY.
//
// WHY THIS IS A LIBRARY MODULE AND NOT PART OF THE BENCH BINARY. The plan's Task
// F1 says to reuse `src/tests/media_compat.rs`'s `ffmpeg_binary()` /
// `write_fixture()` rather than write a second copy — but those live inside a
// `#[cfg(test)] mod media_compat`, which a `[[bin]]` target cannot reach at all.
// So the shared parts moved HERE, where both the bench binary and the test module
// can call them, and `media_compat.rs` now calls [`ffmpeg_binary`] instead of
// keeping its own. One locator, one skip contract, one `NEXIR_FFMPEG` override.
//
// WHY FIXTURES ARE GENERATED RATHER THAN COMMITTED. A 4K60 H.264 fixture is tens
// of megabytes and would be a binary blob in the tree that nothing can diff. It is
// also the wrong kind of artefact: what the benchmark needs is *a* real
// inter-frame-coded file with grain and motion, not one specific file, and
// generating it makes the recipe reviewable where a checked-in MP4 would not be.
//
// A MISSING `ffmpeg` IS A PRINTED SKIP, NEVER A PASS — the same contract
// `media_compat.rs` documents, and `NEXIR_REQUIRE_MEDIA=1` turns the skip into an
// error for a machine that is supposed to have the tooling (mirroring
// `NEXIR_REQUIRE_COMPAT`).
//
// WHAT THESE FIXTURES ARE FOR. Every figure in the 4K60 audit was measured on
// eight vertical bars with a rigid horizontal pan — "the easiest motion a motion
// estimator can face". These classes exist to say how much of that generalises:
// grain defeats a decoder's own compression, real motion changes the bitrate
// per frame, and 4K30 vs 4K60 separates a per-frame cost from a per-second one.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether `NEXIR_REQUIRE_MEDIA` demands that the real-media work actually ran.
///
/// Mirrors `NEXIR_REQUIRE_COMPAT` in `media_compat.rs`: a skip is honest on a
/// machine without `ffmpeg`, and a lie on CI that is supposed to have it.
pub fn require() -> bool {
    std::env::var("NEXIR_REQUIRE_MEDIA")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

/// Locate an `ffmpeg` binary, or `None` with the reason left to the caller.
///
/// `ffmpeg` on PATH is the normal case; `NEXIR_FFMPEG` overrides it for a machine
/// where the binary is present but not on PATH (the DLLs this crate links against
/// say nothing about the CLI being installed).
///
/// This is the single locator for the whole tree — `src/tests/media_compat.rs`
/// calls it rather than keeping a copy.
pub fn ffmpeg_binary() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("NEXIR_FFMPEG") {
        let p = PathBuf::from(explicit);
        if p.exists() {
            return Some(p);
        }
    }
    // `-version` rather than `-h`: it exits 0, writes little, and needs no input
    // file.
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()
        .filter(|s| s.success())
        .map(|_| PathBuf::from("ffmpeg"))
}

/// Where generated fixtures live: a stable directory, NOT a per-process one.
///
/// Deliberately different from `media_compat.rs`'s `scratch_dir()`, which keys off
/// the process id because two concurrent `cargo test` runs must not share files.
/// Here the opposite is wanted: encoding a 4K60 clip takes tens of seconds, and a
/// benchmark that re-encoded its inputs on every invocation would spend most of a
/// run in x264. [`ensure_fixture`] treats an existing plausible file as done.
///
/// `NEXIR_MEDIA_DIR` overrides it, for a machine where the temp volume is small or
/// slow enough to distort the decode timings.
pub fn fixture_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NEXIR_MEDIA_DIR") {
        return PathBuf::from(dir);
    }
    std::env::temp_dir().join("nexir_media_fixtures")
}

/// One media class from the audit's P1.1 A–E list.
pub struct MediaClass {
    /// Short name, used for the file stem and every printed row.
    pub label: &'static str,
    /// Pixel dimensions.
    pub width: u32,
    pub height: u32,
    /// Frame rate, which for a fixed clip length also sets the frame count.
    pub fps: u32,
    /// Clip length in seconds.
    pub seconds: u32,
    /// The `ffmpeg` filter graph producing the source pixels, before `-c:v`.
    pub source_filter: &'static str,
    /// Encoder arguments.
    pub encoder_args: &'static [&'static str],
    /// Why this class exists — printed with its result so a skip is legible and a
    /// number has a reason.
    pub why: &'static str,
}

impl MediaClass {
    /// The file this class writes, inside [`fixture_dir`].
    pub fn file_name(&self) -> String {
        format!("{}.mp4", self.label)
    }

    /// Total frames the clip should contain.
    pub fn frame_count(&self) -> usize {
        (self.fps * self.seconds) as usize
    }
}

/// The classes the audit asks for (P1.1 A–E), minus the synthetic baseline, which
/// the existing six benchmarks already are.
///
/// `noise` over `testsrc2` rather than plain `testsrc2` for the high-detail row:
/// grain is the property that matters (it is incompressible, so it makes the
/// decoder's own output bandwidth realistic), and it is the one thing a synthetic
/// pattern is guaranteed not to have.
///
/// 2 seconds each: long enough that the pipeline-fill interval is a small share of
/// the run (gotcha 15), short enough that generating all four takes well under a
/// minute at `veryfast`.
pub const MEDIA_CLASSES: &[MediaClass] = &[
    MediaClass {
        label: "cam_1080p60",
        width: 1920,
        height: 1080,
        fps: 60,
        seconds: 2,
        // A moving pattern with detail at several scales — the closest `lavfi` gets
        // to handheld camera footage without a camera.
        source_filter: "testsrc2=size=1920x1080:rate=60",
        encoder_args: &["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
        why: "camera-like 1080p60 H.264 — the commonest real input",
    },
    MediaClass {
        label: "motion_1080p60",
        width: 1920,
        height: 1080,
        fps: 60,
        seconds: 2,
        // Whole-frame rotation: every macroblock moves and none of it is a
        // translation, which is the case the bench's rigid pan deliberately avoids.
        source_filter:
            "testsrc2=size=1920x1080:rate=60,rotate=a=t*0.6:c=black:ow=1920:oh=1080",
        encoder_args: &["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
        why: "high motion — non-translational, so P-frames cost real bits",
    },
    MediaClass {
        label: "grain_1080p60",
        width: 1920,
        height: 1080,
        fps: 60,
        seconds: 2,
        // `alls=30` is visible grain without becoming pure noise. Incompressible,
        // so the decoder emits a full-detail frame every time.
        source_filter: "testsrc2=size=1920x1080:rate=60,noise=alls=30:allf=t",
        encoder_args: &["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
        why: "high detail/grain — defeats compression, worst case for decode",
    },
    MediaClass {
        label: "cam_4k30",
        width: 3840,
        height: 2160,
        fps: 30,
        seconds: 2,
        source_filter: "testsrc2=size=3840x2160:rate=30",
        encoder_args: &["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"],
        why: "4K30 — separates per-frame cost from per-second cost",
    },
    MediaClass {
        label: "cam_4k60",
        width: 3840,
        height: 2160,
        fps: 60,
        seconds: 2,
        source_filter: "testsrc2=size=3840x2160:rate=60",
        encoder_args: &["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"],
        why: "4K60 — the target workload, on real coded frames",
    },
];

/// The INPUT half of a fixture's ffmpeg command line, up to and including the
/// source.
///
/// Extracted so the test below validates the arguments `ensure_fixture` really
/// passes rather than a re-derivation of them. That distinction is the whole
/// reason this function exists: the bug it guards against was not in
/// `source_filter` (which is well-formed on its own) but in how the duration was
/// attached to it — `format!("{source_filter}:duration={seconds}")` puts the
/// suffix on the LAST filter of a chain, and `rotate`/`noise` have no `duration`
/// option, so ffmpeg rejects the whole input with "Option not found". A test that
/// built its own command line would have kept passing while the two classes that
/// actually matter silently skipped.
///
/// Duration is therefore an input option (`-t`), which is where it belongs and
/// which works for any number of chained filters.
fn input_args(class: &MediaClass) -> Vec<String> {
    vec![
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-f".into(),
        "lavfi".into(),
        "-t".into(),
        class.seconds.to_string(),
        "-i".into(),
        class.source_filter.into(),
    ]
}

/// Generate `class`'s fixture if it is not already there, returning its path.
///
/// An existing file is reused when it is plausibly complete (see the size floor
/// below) — encoding 4K60 is slow enough that re-doing it per run would dominate
/// the benchmark it exists to feed.
///
/// `Err(reason)` means this `ffmpeg` build could not produce it, which is a
/// property of the HOST and therefore a per-class skip. That is different from a
/// file this crate then failed to decode, which is a property of the crate.
pub fn ensure_fixture(ffmpeg: &Path, class: &MediaClass) -> Result<PathBuf, String> {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let out = dir.join(class.file_name());

    // A plausibility floor rather than a mere existence check: an interrupted
    // encode leaves a short file behind, and silently reusing it would report a
    // decode failure against a crate that is fine. 64 KB is far below any real
    // 2-second clip and far above a truncated header.
    if let Ok(meta) = std::fs::metadata(&out) {
        if meta.len() > 64 * 1024 {
            return Ok(out);
        }
        let _ = std::fs::remove_file(&out);
    }

    // Duration is an ffmpeg INPUT option — see [`input_args`] for why, and for the
    // failure that taught it.
    let mut cmd = Command::new(ffmpeg);
    cmd.args(input_args(class))
        .arg("-y")
        .args(class.encoder_args)
        .arg(&out);

    let output = cmd
        .output()
        .map_err(|e| format!("could not run ffmpeg: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg could not encode it ({}): {}",
            output.status,
            stderr.lines().last().unwrap_or("(no stderr)").trim()
        ));
    }
    let size = std::fs::metadata(&out)
        .map_err(|e| format!("ffmpeg reported success but the file is unreadable: {e}"))?
        .len();
    if size < 64 * 1024 {
        return Err(format!("ffmpeg produced an implausible {size}-byte file"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every class must describe a distinct file, or two rows overwrite each other
    /// and the second silently reports the first's numbers.
    #[test]
    fn every_media_class_writes_its_own_file() {
        let mut names: Vec<String> = MEDIA_CLASSES.iter().map(|c| c.file_name()).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "two classes share a filename: {names:?}");
    }

    /// A class's declared geometry must match the filter that generates it.
    ///
    /// The two are written next to each other and read far apart: the bench prints
    /// `class.width`/`height` and the file carries whatever the filter said. A
    /// mismatch would have the report label a 1080p measurement as 4K, which is
    /// exactly the kind of unfalsifiable number gotcha 9 exists to prevent.
    #[test]
    fn declared_geometry_matches_the_filter() {
        for c in MEDIA_CLASSES {
            let expect = format!("size={}x{}", c.width, c.height);
            assert!(
                c.source_filter.contains(&expect),
                "{}: declares {}x{} but its filter says `{}`",
                c.label,
                c.width,
                c.height,
                c.source_filter
            );
            let rate = format!("rate={}", c.fps);
            assert!(
                c.source_filter.contains(&rate),
                "{}: declares {} fps but its filter says `{}`",
                c.label,
                c.fps,
                c.source_filter
            );
        }
    }

    /// Every class's filter graph must actually be accepted by this `ffmpeg`.
    ///
    /// The point is the RECIPE, not the encode, so it renders 0.1 s to `-f null`
    /// and never writes a file — a fraction of a second per class rather than the
    /// tens of seconds a real 4K60 encode costs.
    ///
    /// This test exists because the failure it catches is invisible from the
    /// benchmark's output unless you read the skip list: `duration=N` appended to a
    /// MULTI-filter chain lands on the last filter, which does not have that
    /// option, and ffmpeg rejects the whole input. Measured — `motion_1080p60` and
    /// `grain_1080p60` were skipped that way while the three single-filter classes
    /// ran, so the two classes carrying non-translational motion and grain (the
    /// only two properties the synthetic bars genuinely lack) were the two that
    /// never existed. A per-class skip that reads like a missing codec is exactly
    /// the shape of "the matrix silently shrank" that `media_compat.rs` warns
    /// about.
    #[test]
    fn every_class_filter_graph_is_accepted_by_ffmpeg() {
        let Some(ffmpeg) = ffmpeg_binary() else {
            assert!(
                !require(),
                "NEXIR_REQUIRE_MEDIA is set but there is no `ffmpeg` binary to \
                 validate the class recipes against"
            );
            eprintln!(
                "[bench_media] SKIP: no `ffmpeg` binary; no class recipe was validated."
            );
            return;
        };

        for c in MEDIA_CLASSES {
            // `input_args` — the SAME arguments `ensure_fixture` passes, not a
            // re-derivation. Then discard the output: the recipe is what is under
            // test, so `-f null` and 0.1 s of it.
            let mut args = input_args(c);
            // Override the duration in place rather than appending a second `-t`,
            // which ffmpeg accepts but resolves in a way that depends on order.
            if let Some(i) = args.iter().position(|a| a == "-t") {
                args[i + 1] = "0.1".into();
            }
            let out = Command::new(&ffmpeg)
                .args(&args)
                .args(["-f", "null", "-"])
                .output()
                .expect("could not run ffmpeg");
            assert!(
                out.status.success(),
                "{}: this ffmpeg rejects the input arguments the benchmark builds for \
                 it ({args:?}): {}",
                c.label,
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
    }

    /// The fixture directory must be stable across processes.
    ///
    /// Deliberately unlike `media_compat`'s per-pid scratch dir: if this ever
    /// picked up a process id, every bench invocation would re-encode 4K60 from
    /// scratch and most of the run would be x264 rather than nexir.
    #[test]
    fn the_fixture_directory_is_reused_not_per_process() {
        let dir = fixture_dir();
        let s = dir.to_string_lossy();
        assert!(
            !s.contains(&std::process::id().to_string()),
            "{s} is per-process; 4K fixtures would be re-encoded every run"
        );
        assert_eq!(dir, fixture_dir(), "the path must be deterministic");
    }
}
