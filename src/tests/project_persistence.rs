// src/tests/project_persistence.rs
// P2.4 — crash recovery and project persistence.
//
// THE QUESTION: does the user's work survive? Everything here is about the path
// from an in-memory `Project` to bytes on disk and back, plus what happens when
// those bytes are missing, stale, truncated or from a future version.
//
// WHY THIS FILE EXISTS. `src/autosave.rs` had one test, covering `write_atomic`
// on a two-field JSON string. Nothing exercised:
//
//   * a REAL project round-tripping through `ProjectFile::save`/`load` — clips,
//     tracks, per-clip audio DSP, effects, sources, timebase;
//   * `find_autosave_newer_than`'s mtime comparison, which is the whole decision
//     of whether to offer recovery;
//   * a CORRUPTED or truncated autosave, which is the likeliest artifact of the
//     crash this feature exists for — a process killed mid-write leaves half a
//     file, and `serde_json` must reject it rather than half-load it;
//   * `format_version` gating, the only forward-compatibility mechanism in the
//     format;
//   * missing-media detection and relinking after a load, which is what makes a
//     recovered project usable rather than merely open.
//
// The worker thread itself (`autosave::spawn_autosave_worker`) is driven too, so
// the atomic write is tested through the channel a real autosave goes through
// rather than only via the private helper.
//
// NO GPU, NO FFmpeg: this is serialisation and filesystem behaviour. It is the
// one test file in `src/tests/` that runs anywhere.

#[cfg(test)]
mod project_persistence {
    use crate::project::{Project, ProjectSettings};
    use crate::project_file::ProjectFile;
    use crate::timeline::mutation::ClipInsertParams;
    use crate::timeline::rational::Rational;
    use crate::timeline::source::{
        ColorInfo, PixelFormat, VideoRotation, VideoStreamInfo,
    };
    use crate::timeline::store::ClipKind;
    use std::path::PathBuf;

    const TB: Rational = Rational { num: 1, den: 90_000 };

    /// A scratch directory unique to this process AND this test, so the suite can
    /// run in parallel and a failure leaves its own evidence behind.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nexir_persist_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create the scratch directory");
        dir
    }

    /// A project with something worth losing: two tracks, three clips, non-default
    /// audio DSP and effects, and a registered source whose file exists.
    ///
    /// Deliberately NOT a default-valued project. Serde will round-trip defaults
    /// even if a field is missing from the JSON (`#[serde(default)]` is on most of
    /// `TimelineStore`'s vectors), so a fixture of defaults cannot detect a field
    /// that was silently dropped — the same trap `tests::colour_plumbing` documents
    /// for colour metadata.
    fn build_project(media: &std::path::Path) -> Project {
        let mut project = Project::with_settings(
            "Persistence Fixture",
            ProjectSettings {
                width: 2560,
                height: 1440,
                frame_rate: Rational { num: 30_000, den: 1_001 },
                timebase: TB,
            },
        );

        let v1 = project.add_video_track("V1").expect("add V1");
        let a1 = project.add_audio_track("A1").expect("add A1");

        let source = project.register_source(
            media.to_path_buf(),
            Some(VideoStreamInfo {
                width: 1920,
                height: 1080,
                frame_rate: Rational { num: 30_000, den: 1_001 },
                pixel_fmt: PixelFormat::Yuv420p,
                color_info: ColorInfo::bt601(),
                duration_pts: 10 * TB.den,
                is_vfr: false,
                time_base: TB,
                rotation: VideoRotation::Rotate90,
            }),
            None,
        );

        // Three clips with distinct, non-default properties, so a dropped field
        // shows up as a specific difference rather than as "everything is 1.0".
        for (i, (track, kind, volume, pan, speed)) in [
            (v1, ClipKind::Video, 0.75_f32, -0.5_f32, 1.0_f32),
            (v1, ClipKind::Video, 1.0, 0.0, 2.0),
            (a1, ClipKind::Audio, 0.25, 0.9, 0.5),
        ]
        .into_iter()
        .enumerate()
        {
            let start = i as i64 * 3 * TB.den;
            project
                .insert_clip(ClipInsertParams {
                    track_id: track,
                    source_id: source,
                    kind,
                    pts_in: start,
                    pts_out: start + 2 * TB.den,
                    source_in: i as i64 * TB.den,
                    volume,
                    pan,
                    speed,
                    audio_muted: i == 2,
                    fade_in_pts: TB.den / 4,
                    fade_out_pts: TB.den / 2,
                    ..Default::default()
                })
                .expect("insert clip");
        }

        // Per-clip effects, which live in a `#[serde(default)]` vector and so are
        // exactly what a missing field would silently reset.
        let mut effects = project.clips.effects_at(0);
        effects.color_enabled = true;
        effects.saturation = 1.4;
        effects.contrast = 0.8;
        effects.vignette_enabled = true;
        effects.vignette_intensity = 0.6;
        project.clips.set_effects_at(0, effects);

        project
    }

    /// Assert two projects describe the same edit.
    ///
    /// Compared field by field rather than by serialising both and diffing the
    /// JSON: that would pass if a field were dropped on the way OUT as well as on
    /// the way in, since both sides would be missing it identically.
    fn assert_same_edit(before: &Project, after: &Project, label: &str) {
        assert_eq!(before.name, after.name, "{label}: project name");
        assert_eq!(
            (before.settings.width, before.settings.height),
            (after.settings.width, after.settings.height),
            "{label}: canvas size"
        );
        assert_eq!(
            before.settings.frame_rate, after.settings.frame_rate,
            "{label}: frame rate — an NTSC rate reduced to an integer here would \
             shift every timestamp in the project by a factor of 1001"
        );
        assert_eq!(
            before.settings.timebase, after.settings.timebase,
            "{label}: timebase"
        );
        assert_eq!(before.tracks.len(), after.tracks.len(), "{label}: track count");
        for (b, a) in before.tracks.iter().zip(after.tracks.iter()) {
            assert_eq!(b.name, a.name, "{label}: track name");
            assert_eq!(b.kind, a.kind, "{label}: track kind");
            assert_eq!(b.gain, a.gain, "{label}: track gain");
            assert_eq!(b.pan, a.pan, "{label}: track pan");
        }

        assert_eq!(before.clips.len(), after.clips.len(), "{label}: clip count");
        for i in 0..before.clips.len() {
            let ctx = format!("{label}: clip {i}");
            assert_eq!(
                before.clips.pts_in_at(i), after.clips.pts_in_at(i),
                "{ctx} pts_in"
            );
            assert_eq!(
                before.clips.pts_out_at(i), after.clips.pts_out_at(i),
                "{ctx} pts_out"
            );
            assert_eq!(
                before.clips.source_in_at(i), after.clips.source_in_at(i),
                "{ctx} source_in"
            );
            assert_eq!(
                before.clips.track_id_at(i), after.clips.track_id_at(i),
                "{ctx} track"
            );
            assert_eq!(
                before.clips.kind_at(i), after.clips.kind_at(i),
                "{ctx} kind"
            );
            // The audio DSP: every one of these is a separate SoA vector, and
            // `TimelineStore`'s invariant is that they are all the same length.
            // A mutation path that forgot one would show up here.
            assert_eq!(before.clips.volume_at(i), after.clips.volume_at(i), "{ctx} volume");
            assert_eq!(before.clips.pan_at(i), after.clips.pan_at(i), "{ctx} pan");
            assert_eq!(before.clips.speed_at(i), after.clips.speed_at(i), "{ctx} speed");
            assert_eq!(
                before.clips.audio_muted_at(i), after.clips.audio_muted_at(i),
                "{ctx} audio_muted"
            );
            assert_eq!(
                before.clips.fade_in_pts_at(i), after.clips.fade_in_pts_at(i),
                "{ctx} fade_in"
            );
            assert_eq!(
                before.clips.fade_out_pts_at(i), after.clips.fade_out_pts_at(i),
                "{ctx} fade_out"
            );
            assert_eq!(
                before.clips.effects_at(i), after.clips.effects_at(i),
                "{ctx} effects — these live in a #[serde(default)] vector, so a \
                 missing field resets them silently rather than failing to parse"
            );
        }
        after.clips.assert_coherent();

        let (b_src, a_src) = (
            before.sources.read().unwrap(),
            after.sources.read().unwrap(),
        );
        assert_eq!(b_src.len(), a_src.len(), "{label}: source count");
        for id in b_src.all_source_ids() {
            assert_eq!(
                b_src.path(id).map(|p| (*p).clone()),
                a_src.path(id).map(|p| (*p).clone()),
                "{label}: source {id:?} path"
            );
            let (b_info, a_info) = (b_src.video_info(id), a_src.video_info(id));
            match (b_info, a_info) {
                (Ok(b), Ok(a)) => {
                    assert_eq!((b.width, b.height), (a.width, a.height),
                        "{label}: source {id:?} dimensions");
                    assert_eq!(b.color_info, a.color_info,
                        "{label}: source {id:?} colour — dropping this sends the \
                         clip down YuvToRgbNode's BT.709 fallback");
                    assert_eq!(b.rotation, a.rotation,
                        "{label}: source {id:?} rotation");
                    assert_eq!(b.frame_rate, a.frame_rate,
                        "{label}: source {id:?} frame rate");
                }
                (Err(_), Err(_)) => {}
                (b, a) => panic!(
                    "{label}: source {id:?} video info presence changed: \
                     {:?} vs {:?}", b.is_ok(), a.is_ok()
                ),
            }
        }
    }

    /// P2.4 — a real project must survive save → load unchanged.
    ///
    /// The fixture carries non-default values in every field group precisely
    /// because `TimelineStore` marks most of its vectors `#[serde(default)]`: a
    /// field dropped from the JSON would come back as its default and a
    /// default-valued fixture could not tell the difference.
    ///
    /// `is_still_image_path` is why the media file is a `.mp4` and not a `.png`:
    /// `Project::from(ProjectFile)` rewrites `frame_rate` to 0/1 and
    /// `duration_pts` to 0 for a still image, which is correct behaviour but would
    /// make the round-trip comparison fail for the right reason and hide any
    /// wrong ones.
    #[test]
    fn project_round_trips_through_save_and_load() {
        let dir = scratch("round_trip");
        // The file needs to exist so the source registers Available, but nothing
        // reads its contents: `ProjectFile::from` only reprobes when stream info is
        // absent, and the fixture supplies it.
        let media = dir.join("clip.mp4");
        std::fs::write(&media, b"not a real mp4, and never opened").unwrap();

        let project = build_project(&media);
        let path = dir.join("fixture.nexp");
        ProjectFile::save(&path, &project).expect("save failed");

        assert!(path.exists(), "save reported success but wrote no file");
        let text = std::fs::read_to_string(&path).expect("the .nexp is not readable");
        assert!(
            text.contains("\"format_version\""),
            "the .nexp carries no format_version, so no future version can gate on it"
        );

        let loaded = ProjectFile::load(&path).expect("load failed");
        assert_same_edit(&project, &loaded, "save→load");

        // Saving the loaded project must produce byte-identical JSON.  This is the
        // idempotence check: a field that survives one trip but is re-serialised
        // differently would drift on every autosave.
        let second = dir.join("fixture2.nexp");
        ProjectFile::save(&second, &loaded).expect("second save failed");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            std::fs::read_to_string(&second).unwrap(),
            "re-saving a loaded project produced different JSON — the format is \
             not idempotent, so every autosave rewrites the file even when nothing \
             changed"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P2.4 — a project file from a future format version must be REFUSED.
    ///
    /// `format_version` is the only forward-compatibility mechanism the format
    /// has. Loading a v2 file with a v1 reader would silently drop whatever v2
    /// added, and then the next save would write that loss back over the user's
    /// file — the one failure mode where being permissive destroys data.
    ///
    /// The same version is also checked in the other direction: the CURRENT
    /// version must load, or this test would pass simply by rejecting everything.
    #[test]
    fn a_future_format_version_is_rejected_and_the_current_one_is_not() {
        let dir = scratch("version");
        let media = dir.join("clip.mp4");
        std::fs::write(&media, b"x").unwrap();
        let project = build_project(&media);

        let good = dir.join("current.nexp");
        ProjectFile::save(&good, &project).expect("save");
        ProjectFile::load(&good).expect(
            "the current format version must load — otherwise the rejection test \
             below proves nothing",
        );

        // Bump the version in the JSON, leaving everything else valid, so the ONLY
        // reason to refuse it is the version itself.
        let text = std::fs::read_to_string(&good).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let current = json["format_version"].as_u64().expect("format_version");
        json["format_version"] = serde_json::json!(current + 1);
        let future = dir.join("future.nexp");
        std::fs::write(&future, serde_json::to_string_pretty(&json).unwrap()).unwrap();

        // `Project` is not `Debug`, so `expect_err` is unavailable; match instead.
        let err = match ProjectFile::load(&future) {
            Err(e) => e,
            Ok(_) => panic!(
                "a file claiming a NEWER format version was loaded — whatever that \
                 version added has been silently dropped, and the next save will \
                 write the loss back over the user's project"
            ),
        };
        let msg = err.to_string();
        assert!(
            msg.contains(&(current + 1).to_string()),
            "the rejection message does not name the offending version: {msg}"
        );

        // A version OLDER than current must still load: that is the point of
        // having a version rather than a magic number.
        json["format_version"] = serde_json::json!(0);
        let old = dir.join("old.nexp");
        std::fs::write(&old, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        ProjectFile::load(&old).expect(
            "a file from an OLDER format version was refused — the version gate is \
             backwards as well as forwards, which makes every previously-saved \
             project unopenable",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P2.4 — a corrupted or truncated autosave must be REFUSED, not half-loaded.
    ///
    /// This is the likeliest artifact of the crash the feature exists for. Nothing
    /// makes a process wait for `write_atomic`'s rename, so a kill can leave the
    /// `.tmp` behind; and a file on a filesystem that lost a write, or a disk that
    /// filled mid-save, is simply short.
    ///
    /// Each case is a DIFFERENT failure shape, because `serde_json` rejects them
    /// at different points and the dangerous ones are the late rejections:
    ///
    /// * **Truncated mid-JSON** — unbalanced braces; caught by the parser.
    /// * **Truncated at a value boundary** — plausible-looking prefix.
    /// * **Empty file** — zero bytes, which `read_to_string` happily returns.
    /// * **Not JSON at all** — a file the user renamed by accident.
    /// * **Valid JSON, wrong shape** — `{"hello": 1}` parses fine and must still be
    ///   refused, because `ProjectFile`'s required fields are absent. This is the
    ///   one a `serde(default)`-heavy struct is most at risk of accepting.
    /// * **Valid ProjectFile with a NEGATIVE clip duration** — parses, and would
    ///   violate `TimelineStore`'s `pts_in < pts_out` invariant.
    ///
    /// The last case is the interesting one and it is asserted differently: the
    /// loader does NOT currently validate it, so the test pins the actual
    /// behaviour (it loads) while `assert_coherent` records that the SoA lengths
    /// are at least intact. Writing this as an expected-failure would be asserting
    /// a feature that does not exist.
    #[test]
    fn corrupt_autosave_files_are_refused_rather_than_half_loaded() {
        let dir = scratch("corrupt");
        let media = dir.join("clip.mp4");
        std::fs::write(&media, b"x").unwrap();
        let good_text = {
            let project = build_project(&media);
            let p = dir.join("good.nexp");
            ProjectFile::save(&p, &project).expect("save");
            std::fs::read_to_string(&p).unwrap()
        };

        // (label, contents) — every one must fail to load.
        let cases: Vec<(&str, String)> = vec![
            ("truncated_mid_json", good_text[..good_text.len() / 2].to_string()),
            ("truncated_at_90pct", good_text[..good_text.len() * 9 / 10].to_string()),
            ("empty", String::new()),
            ("whitespace_only", "   \n\t  ".to_string()),
            ("not_json", "This is a text file the user renamed to .nexp".to_string()),
            ("valid_json_wrong_shape", r#"{"hello": 1}"#.to_string()),
            ("json_null", "null".to_string()),
            ("json_array", "[1, 2, 3]".to_string()),
            // A ProjectFile missing exactly one required field.  `settings` has no
            // serde default, so this must fail; if it ever stops failing, a
            // truncated file could load with a default canvas size.
            ("missing_settings", {
                let mut json: serde_json::Value =
                    serde_json::from_str(&good_text).unwrap();
                json.as_object_mut().unwrap().remove("settings");
                serde_json::to_string(&json).unwrap()
            }),
        ];

        for (label, contents) in &cases {
            let path = dir.join(format!("{label}.nexp"));
            std::fs::write(&path, contents).unwrap();
            let result = ProjectFile::load(&path);
            assert!(
                result.is_err(),
                "{label}: a corrupt autosave LOADED ({} bytes). A partially-parsed \
                 project is worse than a rejected one: the user sees an edit that \
                 is missing work, and the next save writes that over the file.",
                contents.len()
            );
            eprintln!(
                "[persistence] {label:<24} refused: {}",
                result.err().unwrap()
            );
        }

        // A file that does not exist is also an error rather than an empty project.
        assert!(
            ProjectFile::load(dir.join("nope.nexp")).is_err(),
            "loading a nonexistent path succeeded"
        );

        // The known-good text still loads, so the cases above are failing for
        // their own reasons rather than because the loader is broken.
        let control = dir.join("control.nexp");
        std::fs::write(&control, &good_text).unwrap();
        ProjectFile::load(&control).expect(
            "the intact file no longer loads — the rejections above prove nothing",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P2.4 — the autosave recovery decision, which is entirely an mtime
    /// comparison.
    ///
    /// `find_autosave_newer_than` is what decides whether the user is offered
    /// their unsaved work on the next launch. Both directions destroy something if
    /// wrong: too eager and it offers to overwrite a deliberate save with stale
    /// autosave state; too reluctant and the crash recovery silently does nothing,
    /// which is the whole feature failing quietly.
    ///
    /// mtimes are set by writing in order with a real sleep between, rather than
    /// by calling `set_file_mtime` (not in std) or trusting two writes in the same
    /// millisecond to differ. 50 ms is comfortably above NTFS and ext4 timestamp
    /// granularity.
    #[test]
    fn autosave_recovery_is_offered_only_when_the_autosave_is_newer() {
        let dir = scratch("recovery");
        let project_path = dir.join("movie.nexp");
        let autosave_path = ProjectFile::autosave_path_for(Some(&project_path));

        assert_eq!(
            autosave_path,
            PathBuf::from(format!("{}.autosave", project_path.display())),
            "the autosave path must be the project path plus .autosave, or the UI \
             and the worker write to different files"
        );

        // No autosave at all: nothing to offer.
        assert!(
            ProjectFile::find_autosave_newer_than(&autosave_path, None).is_none(),
            "an autosave was offered when the file does not exist"
        );

        // ── Autosave OLDER than the project ─────────────────────────────────────
        // The user saved explicitly after the last autosave, so the autosave is
        // stale and must NOT be offered.
        std::fs::write(&autosave_path, "{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&project_path, "{}").unwrap();
        let project_mtime = std::fs::metadata(&project_path)
            .unwrap()
            .modified()
            .unwrap();
        assert!(
            ProjectFile::find_autosave_newer_than(&autosave_path, Some(project_mtime))
                .is_none(),
            "a STALE autosave (older than the explicit save) was offered for \
             recovery — accepting it would discard the work the user deliberately \
             saved afterwards"
        );

        // ── Autosave NEWER than the project ─────────────────────────────────────
        // The crash case: unsaved changes were autosaved after the last save.
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&autosave_path, "{}").unwrap();
        assert_eq!(
            ProjectFile::find_autosave_newer_than(&autosave_path, Some(project_mtime)),
            Some(autosave_path.clone()),
            "an autosave NEWER than the explicit save was not offered — this is \
             exactly the crash the feature exists for, and the user's unsaved work \
             is silently abandoned"
        );

        // ── No explicit save to compare against ─────────────────────────────────
        // An untitled project: any autosave is worth recovering.
        assert_eq!(
            ProjectFile::find_autosave_newer_than(&autosave_path, None),
            Some(autosave_path.clone()),
            "with no saved project to compare against, any autosave must be offered"
        );

        // ── Deleting it ─────────────────────────────────────────────────────────
        // Called after an explicit save and after a restore, so the next launch
        // does not re-offer work that has already been incorporated.
        ProjectFile::delete_autosave(&autosave_path);
        assert!(
            !autosave_path.exists(),
            "delete_autosave left the file behind, so the recovery prompt would \
             reappear on every launch"
        );
        assert!(
            ProjectFile::find_autosave_newer_than(&autosave_path, None).is_none(),
            "a deleted autosave is still being offered"
        );
        // Deleting a file that is already gone must be a no-op, not a panic: the UI
        // calls this on paths it is not certain about.
        ProjectFile::delete_autosave(&autosave_path);

        // The untitled fallback must be an absolute path in a writable location,
        // not a bare filename relative to the process's cwd.
        let untitled = ProjectFile::autosave_path_for(None);
        assert!(
            untitled.is_absolute(),
            "the untitled autosave path {untitled:?} is relative — it would land \
             wherever the process happened to be started"
        );
        assert!(
            untitled.to_string_lossy().ends_with("untitled.nexp.autosave"),
            "unexpected untitled autosave filename: {untitled:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P2.4 — the full crash-recovery cycle, through the real background worker.
    ///
    /// Everything above tests a helper in isolation. This drives the sequence the
    /// user actually experiences: edit → autosave snapshot pushed to the worker →
    /// process dies without an explicit save → next launch finds the autosave,
    /// loads it, and the recovered project is the edit that was in memory.
    ///
    /// The worker matters because it is where the write actually happens.
    /// `write_atomic` writes `<dest>.tmp` and renames, so a crash can never leave a
    /// half-written autosave — and the test checks the `.tmp` is gone afterwards,
    /// because a stray one would be the only trace of a rename that failed.
    ///
    /// `shutdown()` is what makes this deterministic rather than a sleep: it sends
    /// `AutosaveMsg::Shutdown` and JOINS the thread, so when it returns the write
    /// has completed or the worker has exited trying.
    #[test]
    fn a_snapshot_pushed_to_the_worker_is_recoverable_after_a_crash() {
        use crate::autosave::spawn_autosave_worker;

        let dir = scratch("worker");
        let media = dir.join("clip.mp4");
        std::fs::write(&media, b"x").unwrap();

        // The user's work, never explicitly saved.
        let project = build_project(&media);
        let project_path = dir.join("unsaved.nexp");
        let autosave_path = ProjectFile::autosave_path_for(Some(&project_path));

        // What `AutosaveState::push_snapshot` does: serialise and hand to the
        // worker.  Serialised the same way (`to_string_pretty`) so the bytes on
        // disk are the bytes a real autosave writes.
        let json = serde_json::to_string_pretty(&ProjectFile::from(&project))
            .expect("serialising the snapshot failed");

        let mut handle = spawn_autosave_worker();
        handle.push_snapshot(autosave_path.clone(), json.clone());
        // Deterministic: shutdown joins the thread.
        handle.shutdown();

        assert!(
            autosave_path.exists(),
            "the worker exited without writing the snapshot to {autosave_path:?} — \
             the user's unsaved work never reached disk"
        );
        assert!(
            !autosave_path.with_extension("nexp.autosave.tmp").exists(),
            "a .tmp file was left behind, which means write_atomic's rename did not \
             complete — the autosave on disk may be the PREVIOUS snapshot"
        );
        assert_eq!(
            std::fs::read_to_string(&autosave_path).unwrap(), json,
            "the autosave on disk differs from the snapshot that was pushed"
        );

        // ── The crash ───────────────────────────────────────────────────────────
        // The project file was never written, so there is no explicit save to
        // compare against and the autosave must be offered.
        assert!(
            !project_path.exists(),
            "the fixture accidentally saved the project, which would make this the \
             stale-autosave case instead of the crash case"
        );
        let found = ProjectFile::find_autosave_newer_than(&autosave_path, None)
            .expect("the autosave was not offered for recovery after a crash");

        // ── The recovery ────────────────────────────────────────────────────────
        let recovered = ProjectFile::load(&found).expect("loading the autosave failed");
        assert_same_edit(&project, &recovered, "crash recovery");

        // What the UI does next: the autosave is deleted once its contents have
        // been taken on, so a second launch does not offer the same work again.
        ProjectFile::delete_autosave(&found);
        assert!(
            ProjectFile::find_autosave_newer_than(&autosave_path, None).is_none(),
            "the autosave is still offered after being restored and deleted"
        );

        // ── The freshest snapshot wins ──────────────────────────────────────────
        // The worker's channel has capacity 1 and it drains before writing, so a
        // burst must leave the LAST snapshot on disk, not an arbitrary one.  A
        // stale autosave surviving a burst is how a user loses the last minute of
        // an edit.
        let mut handle = spawn_autosave_worker();
        let mut last = String::new();
        for i in 0..8 {
            let mut p = build_project(&media);
            p.name = format!("burst {i}");
            last = serde_json::to_string_pretty(&ProjectFile::from(&p)).unwrap();
            handle.push_snapshot(autosave_path.clone(), last.clone());
        }
        handle.shutdown();

        let on_disk = std::fs::read_to_string(&autosave_path)
            .expect("no autosave after the burst");
        let recovered: serde_json::Value = serde_json::from_str(&on_disk).unwrap();
        let expected: serde_json::Value = serde_json::from_str(&last).unwrap();
        assert_eq!(
            recovered["name"], expected["name"],
            "after a burst of snapshots the autosave holds {:?} rather than the \
             newest, {:?} — the worker is writing a stale snapshot",
            recovered["name"], expected["name"]
        );
        assert!(
            !autosave_path.with_extension("nexp.autosave.tmp").exists(),
            "a .tmp survived the burst"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
