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
}

impl HistoryState {
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn record(&mut self, project: &Project) {
        self.undo_stack.push(ProjectSnapshot::capture(project));
        if self.undo_stack.len() > MAX_HISTORY {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    pub fn undo(&mut self, project: &mut Project) -> bool {
        let Some(snapshot) = self.undo_stack.pop() else {
            return false;
        };
        self.redo_stack.push(ProjectSnapshot::capture(project));
        snapshot.restore(project);
        true
    }

    pub fn redo(&mut self, project: &mut Project) -> bool {
        let Some(snapshot) = self.redo_stack.pop() else {
            return false;
        };
        self.undo_stack.push(ProjectSnapshot::capture(project));
        snapshot.restore(project);
        true
    }
}
