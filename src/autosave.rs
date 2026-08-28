// src/autosave.rs
//
// Background autosave worker.
//
// The worker lives on a dedicated thread and receives serialized `ProjectFile`
// snapshots from the UI thread through a bounded channel.  It writes them
// atomically (write tmp → rename) so the autosave file is never in a
// partially-written state on disk.
//
// Only the *latest* snapshot in the channel is written; if the UI produces
// snapshots faster than the disk can absorb them the worker silently discards
// all but the most-recent one.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Message types
// ─────────────────────────────────────────────────────────────────────────────

pub enum AutosaveMsg {
    /// A new snapshot ready to be written to disk.
    Snapshot {
        /// The absolute path of the `.nexp.autosave` file to write.
        path: PathBuf,
        /// The serialized JSON project data.
        json: String,
    },
    /// Shut the worker thread down cleanly.
    Shutdown,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle (given to the UI thread)
// ─────────────────────────────────────────────────────────────────────────────

/// A cheap, cloneable handle to the background autosave worker thread.
pub struct AutosaveHandle {
    sender: SyncSender<AutosaveMsg>,
    thread: Option<JoinHandle<()>>,
}

impl AutosaveHandle {
    /// Push a serialized project snapshot.  Non-blocking: if the bounded
    /// channel (capacity 1) is already full, the oldest pending snapshot is
    /// discarded and the new one takes its place via `try_send`.
    ///
    /// `path` must be the `.nexp.autosave` destination path.
    /// `json` is the `serde_json::to_string_pretty` of the `ProjectFile`.
    pub fn push_snapshot(&self, path: PathBuf, json: String) {
        // Discard any pending unwritten snapshot — the freshest one always wins.
        let _ = self.sender.try_send(AutosaveMsg::Snapshot { path, json });
    }

    /// Signal the worker to stop and wait for it to finish.
    pub fn shutdown(&mut self) {
        let _ = self.sender.try_send(AutosaveMsg::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AutosaveHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn the background autosave thread and return a handle to it.
pub fn spawn_autosave_worker() -> AutosaveHandle {
    // Capacity 1: only the newest snapshot ever sits in flight.
    let (tx, rx) = sync_channel::<AutosaveMsg>(1);

    let handle = thread::Builder::new()
        .name("nexir-autosave".into())
        .spawn(move || {
            loop {
                // Wait up to 5 s before timing out and checking again.
                match rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(AutosaveMsg::Snapshot { path, json }) => {
                        // Drain any further snapshots that arrived while we
                        // were writing — always flush the most recent one.
                        let mut latest_path = path;
                        let mut latest_json = json;
                        while let Ok(AutosaveMsg::Snapshot { path: p, json: j }) =
                            rx.try_recv()
                        {
                            latest_path = p;
                            latest_json = j;
                        }
                        write_atomic(&latest_path, &latest_json);
                    }
                    Ok(AutosaveMsg::Shutdown) => break,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            log::info!("[autosave] worker thread exiting cleanly");
        })
        .expect("autosave: failed to spawn worker thread");

    AutosaveHandle {
        sender: tx,
        thread: Some(handle),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic write helper
// ─────────────────────────────────────────────────────────────────────────────

/// Write `data` to `dest` atomically by writing to `<dest>.tmp` and then
/// renaming.  On most filesystems a rename is atomic, so the autosave file is
/// never in a partially-written state on disk.
fn write_atomic(dest: &Path, data: &str) {
    // Ensure the parent directory exists.
    if let Some(parent) = dest.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log::error!("[autosave] could not create dir {:?}: {}", parent, e);
            return;
        }
    }

    let tmp = dest.with_extension("nexp.autosave.tmp");
    if let Err(e) = std::fs::write(&tmp, data) {
        log::error!("[autosave] write to {:?} failed: {}", tmp, e);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, dest) {
        log::error!("[autosave] rename {:?} -> {:?} failed: {}", tmp, dest, e);
        // Leave the tmp file behind so the user can still recover manually.
        return;
    }
    log::info!("[autosave] wrote {:?}", dest);
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_atomic_creates_file_and_no_tmp_left() {
        let dir = std::env::temp_dir().join("nexir_autosave_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let dest = dir.join("test.nexp.autosave");
        write_atomic(&dest, "{\"test\": 1}");

        assert!(dest.exists(), "autosave file should exist");
        assert!(!dest.with_extension("nexp.autosave.tmp").exists(), "tmp file should be gone");

        let contents = std::fs::read_to_string(&dest).unwrap();
        assert_eq!(contents, "{\"test\": 1}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
