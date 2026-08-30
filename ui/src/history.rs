// ui/src/history.rs
//
// Undo/redo for the editor shell.

use nexir::project::{Project, ProjectSettings};
use nexir::timeline::store::TimelineStore;
use nexir::timeline::track::TrackList;

const MAX_HISTORY: usize = 100;

/// One point in the edit history.
///
/// WHAT IS AND IS NOT CAPTURED, because both are deliberate:
///
/// * `tracks` + `clips` — the edit itself. Cloned wholesale; `TimelineStore` is
///   structure-of-arrays, so a clone is a handful of `Vec` allocations rather
///   than a per-clip walk.
/// * `settings` — canvas size, frame rate, timebase. Captured since P2.7: a
///   frame-rate change that undo could not reverse left the user pressing Ctrl+Z
///   and silently losing an unrelated clip edit instead (see
///   [`HistoryState::record_settings_change`]).
/// * `sources` — deliberately NOT captured. `Project::sources` is an
///   `Arc<RwLock<SourceRegistry>>` **shared with `IoLayer`**, which holds open
///   demuxers and decoders keyed by `SourceId`, and with the still-image cache
///   keyed by path. Restoring an older registry behind their backs would leave
///   those keyed on sources that no longer exist. Media import is therefore not
///   an undoable operation, which is also what most NLEs do — the media pool is a
///   library, not part of the edit.
#[derive(Clone)]
struct ProjectSnapshot {
    tracks:   TrackList,
    clips:    TimelineStore,
    settings: ProjectSettings,
}

impl ProjectSnapshot {
    fn capture(project: &Project) -> Self {
        Self {
            tracks:   project.tracks.clone(),
            clips:    project.clips.clone(),
            settings: project.settings.clone(),
        }
    }

    fn restore(self, project: &mut Project) {
        project.tracks = self.tracks;
        project.clips = self.clips;
        project.settings = self.settings;
    }
}

#[derive(Default)]
pub struct HistoryState {
    undo_stack: Vec<ProjectSnapshot>,
    redo_stack: Vec<ProjectSnapshot>,
    /// Monotonically-increasing counter incremented on every `record()` or
    /// `undo()` / `redo()` call.  The autosave ticker compares this against
    /// the value it last snapshotted to detect whether the project has changed
    /// since the last autosave write, without needing a deep project comparison.
    change_token: u64,
}

impl HistoryState {
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Current change token.  The autosave ticker stores this and compares on
    /// each frame; a different value means the project has changed.
    pub fn change_token(&self) -> u64 {
        self.change_token
    }

    /// Capture the project as it is NOW, before a mutation is applied.
    ///
    /// Every call site follows the same order — record, then mutate — so the
    /// snapshot on the stack is the state undo must return to.
    pub fn record(&mut self, project: &Project) {
        self.push_undo(ProjectSnapshot::capture(project));
    }

    /// Record an already-applied change to `project.settings`.
    ///
    /// P2.7 — settings are edited by the top bar's combo boxes, which mutate
    /// `project.settings` in place during `draw`, so there is no moment at which a
    /// caller can `record()` beforehand. Before this existed the change was not
    /// recorded at all, and the next Ctrl+Z undid whatever the user had done
    /// EARLIER while leaving the new frame rate in place — the surprising and
    /// destructive shape, because the user believes they reverted the setting.
    ///
    /// `previous` is the settings value from before the frame's UI ran; the clips
    /// and tracks are taken as they are now, since the settings edit did not touch
    /// them. Undo therefore restores the old canvas/rate and leaves the edit alone.
    pub fn record_settings_change(&mut self, project: &Project, previous: ProjectSettings) {
        let mut snapshot = ProjectSnapshot::capture(project);
        snapshot.settings = previous;
        self.push_undo(snapshot);
    }

    /// Push onto the undo stack, invalidate redo, and bump the change token.
    ///
    /// The cap is enforced with `VecDeque`-like semantics via `drain`, not
    /// `remove(0)`: at MAX_HISTORY that was an O(n) memmove of 100 full timeline
    /// clones on every single edit past the cap.
    fn push_undo(&mut self, snapshot: ProjectSnapshot) {
        self.undo_stack.push(snapshot);
        if self.undo_stack.len() > MAX_HISTORY {
            let excess = self.undo_stack.len() - MAX_HISTORY;
            self.undo_stack.drain(..excess);
        }
        // A new edit makes the redo branch unreachable — standard linear-history
        // behaviour, and the reason `redo_stack` is cleared rather than kept.
        self.redo_stack.clear();
        self.change_token = self.change_token.wrapping_add(1);
    }

    pub fn undo(&mut self, project: &mut Project) -> bool {
        let Some(snapshot) = self.undo_stack.pop() else {
            return false;
        };
        self.redo_stack.push(ProjectSnapshot::capture(project));
        snapshot.restore(project);
        self.change_token = self.change_token.wrapping_add(1);
        true
    }

    pub fn redo(&mut self, project: &mut Project) -> bool {
        let Some(snapshot) = self.redo_stack.pop() else {
            return false;
        };
        self.undo_stack.push(ProjectSnapshot::capture(project));
        snapshot.restore(project);
        self.change_token = self.change_token.wrapping_add(1);
        true
    }
}
