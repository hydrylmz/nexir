use nexir::project::Project;
use nexir::timeline::store::TimelineStore;
use nexir::timeline::track::TrackList;

const MAX_HISTORY: usize = 100;

#[derive(Clone)]
struct ProjectSnapshot {
    tracks: TrackList,
    clips: TimelineStore,
}

impl ProjectSnapshot {
    fn capture(project: &Project) -> Self {
        Self {
            tracks: project.tracks.clone(),
            clips: project.clips.clone(),
        }
    }

    fn restore(self, project: &mut Project) {
        project.tracks = self.tracks;
        project.clips = self.clips;
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

    pub fn record(&mut self, project: &Project) {
        self.undo_stack.push(ProjectSnapshot::capture(project));
        if self.undo_stack.len() > MAX_HISTORY {
            self.undo_stack.remove(0);
        }
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
