// ui/src/history_tests.rs
//
// P2.7 — undo/redo, which is the one part of the editor UX checklist that is pure
// logic rather than a dialog.
//
// WHY THIS IS TESTABLE AND THE REST OF P2.7 IS NOT. The audit's UX list is media
// import, drag and drop, timeline placement, trim, split, delete, move, undo,
// redo, preview, export. Most of those are already covered where the work
// actually happens — `nexir::timeline::mutation` has 75 tests over split/trim/
// move/ripple/roll/slip/slide, `tests::render_integration` covers preview and
// `tests::export_validation` covers export end to end. What was left untested was
// `HistoryState`, and it needed no window: it is two stacks of `Project`
// snapshots. The genuinely unverifiable remainder is the five `rfd` native file
// dialogs (open, save-as, import, relink), which block on a human and panic
// without a display.
//
// WHAT THESE TESTS ARE FOR. Undo is the feature users trust most and notice least
// until it misbehaves, and its failures are asymmetric: a missed `record` does not
// error, it silently reverts the WRONG edit. So the assertions are about the
// history model's invariants rather than about any single operation:
//
//   * an undone edit is exactly reversed, and redo restores it exactly;
//   * a new edit after an undo invalidates the redo branch (linear history);
//   * the stack is bounded and bounded cheaply;
//   * `change_token` moves on every mutation, because autosave keys off it and a
//     token that fails to move means the user's change is never written;
//   * a settings change is undoable — the bug this file was written to pin.
//
// No GPU, no FFmpeg, no display.

use crate::history::HistoryState;
use nexir::project::{Project, ProjectSettings};
use nexir::timeline::ids::{SourceId, TrackId};
use nexir::timeline::mutation::ClipInsertParams;
use nexir::timeline::rational::Rational;
use nexir::timeline::store::ClipKind;

const TB: Rational = Rational { num: 1, den: 90_000 };

/// A project with one video track and one clip, which is the smallest state in
/// which every history operation is meaningful.
///
/// Returns the track and the registered `SourceId` because the id types have
/// crate-private fields — the `ui` crate cannot fabricate a `SourceId(0)`, which
/// is the right constraint: ids come from the registry that owns them.
fn project_with_one_clip() -> (Project, TrackId, SourceId) {
    let mut project = Project::with_settings(
        "History Fixture",
        ProjectSettings {
            width: 1920,
            height: 1080,
            frame_rate: Rational { num: 30, den: 1 },
            timebase: TB,
        },
    );
    let track = project.add_video_track("V1").expect("add track");
    // A source is registered but never read: `SourceRegistry` is deliberately
    // outside the snapshot (see `ProjectSnapshot`), so nothing here depends on it.
    let source = project.register_source(std::path::PathBuf::from("fixture.mp4"), None, None);
    project
        .insert_clip(ClipInsertParams {
            track_id: track,
            source_id: source,
            kind: ClipKind::Video,
            pts_in: 0,
            pts_out: 3 * TB.den,
            ..Default::default()
        })
        .expect("insert clip");
    (project, track, source)
}

/// Append a clip starting after everything currently on the timeline.
fn append_clip(project: &mut Project, track: TrackId, source: SourceId) {
    let start = (0..project.clips.len())
        .map(|i| project.clips.pts_out_at(i))
        .max()
        .unwrap_or(0);
    project
        .insert_clip(ClipInsertParams {
            track_id: track,
            source_id: source,
            kind: ClipKind::Video,
            pts_in: start,
            pts_out: start + TB.den,
            ..Default::default()
        })
        .expect("append clip");
}

/// One clip's comparable state: time bounds, track, and the audio DSP fields.
///
/// Named because it is compared in every test here and the tuple is wide enough
/// that an unnamed one obscures which column is which.
type ClipShape = (i64, i64, i64, usize, f32, f32, f32, bool);

/// The timeline reduced to a comparable shape: one entry per clip.
///
/// Compared instead of the `TimelineStore` itself, which is not `PartialEq`.
/// Includes the audio DSP fields as well as the time bounds, because those live in
/// separate SoA vectors and a snapshot/restore that missed one would otherwise
/// look correct.
fn timeline_shape(project: &Project) -> Vec<ClipShape> {
    (0..project.clips.len())
        .map(|i| {
            (
                project.clips.pts_in_at(i),
                project.clips.pts_out_at(i),
                project.clips.source_in_at(i),
                project.clips.track_id_at(i).index(),
                project.clips.volume_at(i),
                project.clips.pan_at(i),
                project.clips.speed_at(i),
                project.clips.audio_muted_at(i),
            )
        })
        .collect()
}

/// An edit must be reversed exactly by undo, and reapplied exactly by redo.
///
/// The round trip is asserted in both directions and over three separate edits, so
/// a snapshot that captured a shallow copy — or restored the stacks in the wrong
/// order — cannot pass by getting one step right.
#[test]
fn undo_reverses_an_edit_and_redo_reapplies_it() {
    let (mut project, track, source) = project_with_one_clip();
    let mut history = HistoryState::default();

    let states = {
        let mut states = vec![timeline_shape(&project)];
        for _ in 0..3 {
            history.record(&project);
            append_clip(&mut project, track, source);
            states.push(timeline_shape(&project));
        }
        states
    };

    assert_eq!(project.clips.len(), 4, "the fixture did not build three edits");
    assert!(history.can_undo(), "three recorded edits but nothing to undo");
    assert!(!history.can_redo(), "redo is available before any undo");

    // Walk all the way back.
    for step in (0..3).rev() {
        assert!(history.undo(&mut project), "undo {step} refused");
        assert_eq!(
            timeline_shape(&project), states[step],
            "after undoing to step {step} the timeline does not match the state \
             recorded at that point"
        );
    }
    assert!(
        !history.can_undo(),
        "undo is still available after unwinding every recorded edit — the stack \
         holds a snapshot nothing recorded"
    );
    assert!(
        !history.undo(&mut project),
        "undo on an empty stack returned true"
    );

    // And all the way forward.
    for (step, want) in states.iter().enumerate().skip(1) {
        assert!(history.redo(&mut project), "redo {step} refused");
        assert_eq!(
            timeline_shape(&project), *want,
            "after redoing to step {step} the timeline does not match the original"
        );
    }
    assert!(
        !history.redo(&mut project),
        "redo on an empty stack returned true"
    );
    assert_eq!(
        timeline_shape(&project), states[3],
        "a full undo/redo round trip did not return the timeline to where it started"
    );
}

/// A new edit after an undo must discard the redo branch.
///
/// This is linear-history semantics, and the alternative is worse than useless: a
/// redo that reapplied a snapshot taken before a divergent edit would resurrect
/// clips the user has since deleted, on top of work they have since done.
#[test]
fn a_new_edit_after_undo_invalidates_redo() {
    let (mut project, track, source) = project_with_one_clip();
    let mut history = HistoryState::default();

    history.record(&project);
    append_clip(&mut project, track, source);
    let two_clips = timeline_shape(&project);

    assert!(history.undo(&mut project));
    assert!(
        history.can_redo(),
        "redo is not available immediately after an undo"
    );

    // Diverge.
    history.record(&project);
    append_clip(&mut project, track, source);
    let divergent = timeline_shape(&project);

    assert!(
        !history.can_redo(),
        "the redo branch survived a divergent edit — redoing now would reapply a \
         snapshot from a history that no longer exists"
    );
    assert!(!history.redo(&mut project), "redo ran on an invalidated branch");
    assert_eq!(
        timeline_shape(&project), divergent,
        "a refused redo still modified the project"
    );

    // The divergent edit is itself undoable, and lands on the same one-clip state
    // the original edit started from — not on `two_clips`.
    assert!(history.undo(&mut project));
    assert_ne!(
        timeline_shape(&project), two_clips,
        "undoing the divergent edit restored the abandoned branch"
    );
}

/// A settings change must be undoable, and must not take the timeline with it.
///
/// P2.7 — this is the bug the file was written for. The top bar's combo boxes
/// mutate `project.settings` in place while drawing, so nothing could `record()`
/// beforehand and the change was not recorded at all. The consequence was not a
/// no-op: the user changed the frame rate, pressed Ctrl+Z expecting it back, and
/// instead had an EARLIER clip edit reverted while the new rate stayed. Undo
/// appearing to corrupt the timeline is about the worst failure this feature has.
///
/// Both halves are asserted, because the fix is only correct if it is narrow:
/// undo restores the previous settings AND leaves clips exactly as they are, since
/// a settings edit never touched them.
#[test]
fn a_settings_change_is_undoable_without_reverting_the_timeline() {
    let (mut project, track, source) = project_with_one_clip();
    let mut history = HistoryState::default();

    // An earlier, unrelated edit — the one that used to be reverted by mistake.
    history.record(&project);
    append_clip(&mut project, track, source);
    let shape_before_settings = timeline_shape(&project);
    let original = project.settings.clone();

    // What the top bar does: mutate in place, then report the previous value.
    // 30000/1001 specifically: an NTSC rate is where a frame duration that drops
    // `frame_rate.den` is wrong by a factor of 1001 (see the P1.8 work), so a
    // settings field that silently fails to restore is at its most damaging here.
    project.settings.frame_rate = Rational { num: 30_000, den: 1_001 };
    project.settings.width = 3840;
    project.settings.height = 2160;
    history.record_settings_change(&project, original.clone());

    let changed = project.settings.clone();
    assert_ne!(changed, original, "the fixture did not change the settings");

    assert!(
        history.undo(&mut project),
        "a recorded settings change is not undoable"
    );
    assert_eq!(
        project.settings, original,
        "undo did not restore the previous settings — the user pressed Ctrl+Z after \
         a frame-rate change and the rate stayed put"
    );
    assert_eq!(
        timeline_shape(&project), shape_before_settings,
        "undoing a SETTINGS change also reverted the timeline — it reached back \
         past the settings edit and undid an unrelated clip edit, which is the \
         original bug in its most destructive form"
    );

    // Redo must reapply the settings and still leave the timeline alone.
    assert!(history.redo(&mut project), "a settings change is not redoable");
    assert_eq!(
        project.settings, changed,
        "redo did not reapply the settings change"
    );
    assert_eq!(
        timeline_shape(&project), shape_before_settings,
        "redoing a settings change modified the timeline"
    );

    // And the earlier clip edit is still undoable underneath it, in the right
    // order: the settings snapshot must not have displaced it.
    assert!(history.undo(&mut project), "settings undo unavailable");
    assert!(history.undo(&mut project), "the earlier clip edit was lost");
    assert_eq!(
        project.clips.len(), 1,
        "unwinding to the bottom of the stack did not reach the one-clip state"
    );
}

/// The undo stack is bounded, and the oldest entries are the ones dropped.
///
/// A history that grows without limit is a leak measured in whole timelines: each
/// snapshot clones every SoA vector in `TimelineStore`. The cap must trim from the
/// FRONT, so the most recent MAX_HISTORY edits are the ones kept.
///
/// `MAX_HISTORY` is private, so the bound is discovered rather than asserted
/// against a literal — a test that hardcoded 100 would silently stop checking
/// anything if the constant changed.
#[test]
fn the_undo_stack_is_bounded_and_drops_the_oldest_entries() {
    let (mut project, track, source) = project_with_one_clip();
    let mut history = HistoryState::default();

    // Comfortably past any plausible cap.
    const EDITS: usize = 400;
    for _ in 0..EDITS {
        history.record(&project);
        append_clip(&mut project, track, source);
    }
    let clips_at_top = project.clips.len();

    // Unwind everything and count how far back the history actually reached.
    let mut undos = 0usize;
    while history.undo(&mut project) {
        undos += 1;
        assert!(
            undos <= EDITS,
            "undo ran more times than there were edits — the stacks are cycling"
        );
    }

    assert!(
        undos < EDITS,
        "all {EDITS} edits were retained, so the undo stack is unbounded: each \
         entry clones the whole timeline, and a long session would grow without \
         limit"
    );
    assert!(
        undos >= 50,
        "only {undos} edits were retained, which is too shallow to be a usable \
         undo history"
    );

    // The entries kept must be the NEWEST ones: unwinding `undos` steps from the
    // top leaves exactly `undos` fewer clips, which is only true if the dropped
    // snapshots came off the front.
    assert_eq!(
        project.clips.len(), clips_at_top - undos,
        "after {undos} undos the timeline is not {undos} edits behind the top — \
         the cap dropped entries from the wrong end, so undo jumps to an ancient \
         state instead of stepping back one edit"
    );

    eprintln!("[history] undo depth retained: {undos} of {EDITS} edits");
}

/// `change_token` must move on every mutation, in both directions.
///
/// Autosave is driven entirely by this: `AutosaveState::tick` compares it against
/// the value it last serialised and writes only when they differ. A token that
/// fails to move after an operation means that operation is never autosaved —
/// which surfaces as work lost in a crash, with nothing to point at.
///
/// Undo and redo are included deliberately: they are the easy ones to forget,
/// because they feel like "going back" rather than like changes.
#[test]
fn change_token_moves_on_every_mutation() {
    let (mut project, track, source) = project_with_one_clip();
    let mut history = HistoryState::default();

    let mut seen = vec![history.change_token()];
    let expect_moved = |history: &HistoryState, seen: &mut Vec<u64>, what: &str| {
        let token = history.change_token();
        assert_ne!(
            token,
            *seen.last().unwrap(),
            "change_token did not move after {what} — autosave compares this value \
             and would never write that change to disk"
        );
        seen.push(token);
    };

    history.record(&project);
    append_clip(&mut project, track, source);
    expect_moved(&history, &mut seen, "record");

    history.record(&project);
    append_clip(&mut project, track, source);
    expect_moved(&history, &mut seen, "a second record");

    let previous = project.settings.clone();
    project.settings.width = 1280;
    history.record_settings_change(&project, previous);
    expect_moved(&history, &mut seen, "record_settings_change");

    assert!(history.undo(&mut project));
    expect_moved(&history, &mut seen, "undo");

    assert!(history.redo(&mut project));
    expect_moved(&history, &mut seen, "redo");

    // A refused undo changed nothing, so the token must NOT move: a token that
    // ticks on a no-op makes autosave rewrite the file on every keypress at the
    // bottom of the stack.
    while history.undo(&mut project) {}
    let idle = history.change_token();
    assert!(!history.undo(&mut project));
    assert_eq!(
        history.change_token(), idle,
        "change_token moved on a REFUSED undo — autosave would rewrite the project \
         every time the user pressed Ctrl+Z with nothing left to undo"
    );

    // Every token observed was distinct, so nothing reused a value.
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(), seen.len(),
        "change_token repeated a value across distinct mutations: {seen:?}"
    );
}
