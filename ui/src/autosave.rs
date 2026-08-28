// ui/src/autosave.rs
//
// UI-side autosave coordinator.
//
// `AutosaveState` is stored on `NexirApp` and ticked once per frame from
// `update()`.  It compares the HistoryState change-token against the value
// it last serialized and — if the project has changed AND 30 seconds have
// elapsed since the last write — pushes a new snapshot to the background
// autosave worker.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use nexir::autosave::{spawn_autosave_worker, AutosaveHandle};
use nexir::project::Project;
use nexir::project_file::ProjectFile;

/// How long to wait after the most-recent change before writing the autosave.
const AUTOSAVE_DEBOUNCE: Duration = Duration::from_secs(30);

/// Information about a stale autosave file found at startup (or project open
/// time).  The UI shows a recovery modal until the user dismisses it.
#[derive(Clone, Debug)]
pub struct RecoveryInfo {
    /// Absolute path of the autosave file to restore from.
    pub autosave_path: PathBuf,
    /// Human-readable project name read from inside the autosave file
    /// (used for display in the modal — avoids requiring a full project load).
    pub project_name: String,
    /// Modification time of the autosave file, for display.
    pub modified: Option<std::time::SystemTime>,
}

pub struct AutosaveState {
    handle: AutosaveHandle,
    /// The `change_token` value at the time of the last serialized snapshot.
    last_saved_token: u64,
    /// When the most-recent unsaved change was first detected.
    dirty_since: Option<Instant>,
    /// The last autosave path we wrote to (used for `delete_autosave`).
    pub last_autosave_path: Option<PathBuf>,
}

impl AutosaveState {
    pub fn new() -> Self {
        Self {
            handle: spawn_autosave_worker(),
            last_saved_token: 0,
            dirty_since: None,
            last_autosave_path: None,
        }
    }

    // ─────────────────────────────────────────────
    // Per-frame tick
    // ─────────────────────────────────────────────

    /// Call this once per frame from `NexirApp::update()`.
    ///
    /// `change_token` comes from `HistoryState::change_token()`.
    /// `current_project_path` is `NexirApp::current_project_path`.
    pub fn tick(
        &mut self,
        project: &Project,
        change_token: u64,
        current_project_path: Option<&std::path::Path>,
    ) {
        let now = Instant::now();

        if change_token != self.last_saved_token {
            // Project changed since last autosave.
            if self.dirty_since.is_none() {
                self.dirty_since = Some(now);
            }
        }

        // Fire the autosave if we have been dirty for at least DEBOUNCE seconds.
        let should_save = self
            .dirty_since
            .map(|t| now.duration_since(t) >= AUTOSAVE_DEBOUNCE)
            .unwrap_or(false);

        if should_save {
            self.push_snapshot(project, current_project_path, change_token);
            self.dirty_since = None;
        }
    }

    /// Force an immediate snapshot push without the debounce delay.
    /// Useful on application close / new-project to guarantee the latest
    /// state is saved before we lose it.
    pub fn flush(
        &mut self,
        project: &Project,
        change_token: u64,
        current_project_path: Option<&std::path::Path>,
    ) {
        if change_token != self.last_saved_token {
            self.push_snapshot(project, current_project_path, change_token);
            self.dirty_since = None;
        }
    }

    /// Delete the last-written autosave file (called after an explicit save).
    pub fn delete_last_autosave(&self) {
        if let Some(ref path) = self.last_autosave_path {
            ProjectFile::delete_autosave(path);
        }
    }

    // ─────────────────────────────────────────────
    // Recovery check
    // ─────────────────────────────────────────────

    /// Check whether a stale autosave exists for the given project path.
    ///
    /// - `project_path = None` → check the untitled autosave location.
    /// - `project_path = Some(p)` → check `p.nexp.autosave` and compare
    ///   timestamps with `p`.
    ///
    /// Returns `Some(RecoveryInfo)` if the autosave is newer than the saved
    /// project file (or there is no saved file), `None` otherwise.
    pub fn check_for_recovery(project_path: Option<&std::path::Path>) -> Option<RecoveryInfo> {
        let autosave_path = ProjectFile::autosave_path_for(project_path);

        let reference_mtime = project_path.and_then(|p| {
            std::fs::metadata(p).ok()?.modified().ok()
        });

        let found = ProjectFile::find_autosave_newer_than(&autosave_path, reference_mtime)?;

        // Peek at the project name without a full deserialize (fast path).
        let project_name = peek_project_name_from(&found).unwrap_or_else(|| "Untitled".into());
        let modified = std::fs::metadata(&found).ok().and_then(|m| m.modified().ok());

        Some(RecoveryInfo {
            autosave_path: found,
            project_name,
            modified,
        })
    }

    // ─────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────

    fn push_snapshot(
        &mut self,
        project: &Project,
        current_project_path: Option<&std::path::Path>,
        change_token: u64,
    ) {
        let path = ProjectFile::autosave_path_for(current_project_path);
        let pf = ProjectFile::from(project);
        match serde_json::to_string_pretty(&pf) {
            Ok(json) => {
                self.handle.push_snapshot(path.clone(), json);
                self.last_saved_token = change_token;
                self.last_autosave_path = Some(path);
            }
            Err(e) => {
                log::error!("[autosave] serialization failed: {}", e);
            }
        }
    }
}

/// Read just the `name` field from the autosave JSON without fully
/// deserializing the whole project structure.
pub fn peek_project_name_from(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    // Simple approach: `serde_json::from_str` into a `serde_json::Value`
    // and extract `name`.
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("name")?.as_str().map(|s| s.to_owned())
}
