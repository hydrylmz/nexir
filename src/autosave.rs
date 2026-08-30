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
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Message types
// ─────────────────────────────────────────────────────────────────────────────

/// One snapshot waiting to be written.
#[derive(Clone)]
struct Snapshot {
    path: PathBuf,
    json: String,
}

/// The single slot holding the newest snapshot not yet on disk.
///
/// P2.4 — the newest snapshot has to live somewhere the producer can OVERWRITE.
/// It used to be sent through the bounded channel, and `try_send` on a full
/// channel returns `Err(Full)` and drops the message it was given — i.e. the
/// NEWEST one, keeping the stale one. The doc comment claimed the opposite
/// ("the oldest pending snapshot is discarded and the new one takes its place"),
/// and the code had said that for long enough to be believed. Measured with a
/// burst of eight snapshots: the file on disk held snapshot 2, not snapshot 7.
///
/// That is silent data loss in the direction that matters — a crash recovers an
/// older edit than the one the autosave timer captured.
type SnapshotSlot = Arc<Mutex<Option<Snapshot>>>;

/// Wakes the worker. Carries no data: the payload is in the [`SnapshotSlot`], so
/// a full channel costs nothing but a redundant wake-up.
enum AutosaveMsg {
    /// A new snapshot is in the slot.
    Wake,
    /// Shut the worker thread down cleanly.
    Shutdown,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle (given to the UI thread)
// ─────────────────────────────────────────────────────────────────────────────

/// A cheap, cloneable handle to the background autosave worker thread.
pub struct AutosaveHandle {
    /// `None` after [`AutosaveHandle::shutdown`] has taken it.
    ///
    /// Held in an `Option` so shutdown can DROP it before joining: dropping the
    /// last sender makes the worker's `recv_timeout` return `Disconnected`, which
    /// is what actually ends its loop. See `shutdown` for why `try_send` alone is
    /// not enough.
    sender: Option<SyncSender<AutosaveMsg>>,
    /// Where the newest unwritten snapshot lives; shared with the worker.
    slot:   SnapshotSlot,
    thread: Option<JoinHandle<()>>,
}

impl AutosaveHandle {
    /// Push a serialized project snapshot.
    ///
    /// Non-blocking, and the NEWEST snapshot always wins: it is written into the
    /// shared slot, replacing any snapshot not yet on disk, and the worker is then
    /// woken. A full wake channel is fine — the worker is already awake and will
    /// find the new contents when it next reads the slot.
    ///
    /// P2.4 — the old implementation put the snapshot IN the channel and relied on
    /// `try_send` to displace the pending one. It does the opposite: `try_send` on
    /// a full channel returns the message back as `Err(Full)`, which the `let _ =`
    /// threw away, so a burst kept the OLDEST snapshot and discarded every later
    /// one. Measured: eight pushes left snapshot 2 on disk.
    ///
    /// `path` must be the `.nexp.autosave` destination path.
    /// `json` is the `serde_json::to_string_pretty` of the `ProjectFile`.
    pub fn push_snapshot(&self, path: PathBuf, json: String) {
        // Overwrite rather than queue: an unwritten snapshot is strictly stale.
        // A poisoned mutex is recovered from rather than propagated — losing an
        // autosave because a previous panic poisoned a lock is the worst possible
        // trade.
        {
            let mut slot = self
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            *slot = Some(Snapshot { path, json });
        }
        if let Some(tx) = self.sender.as_ref() {
            let _ = tx.try_send(AutosaveMsg::Wake);
        }
    }

    /// Signal the worker to stop and wait for it to finish.
    ///
    /// P2.4 — this used to be `try_send(Shutdown)` followed by `join()`, which
    /// **hangs forever** whenever a snapshot is still sitting in the channel. The
    /// channel's capacity is 1, so `try_send` fails and is discarded by the `let _
    /// =`; the worker then writes the snapshot it already had, loops, and sits in
    /// `recv_timeout(5s)` returning `Timeout` and `continue` for ever, because no
    /// Shutdown was ever delivered and the sender is still alive. `join()` never
    /// returns.
    ///
    /// That is reachable from `Drop`, so closing the app immediately after an
    /// autosave hung the process instead of exiting — and it is deterministic, not
    /// a race: `tests::project_persistence` hit it on the first try by pushing a
    /// snapshot and shutting down straight away.
    ///
    /// The fix is to DROP the sender before joining. `recv_timeout` then returns
    /// `Disconnected` and the worker breaks out of its loop, whether or not the
    /// Shutdown message got through. The `try_send` is kept so a worker that is
    /// waiting with room in the channel stops promptly rather than after its
    /// current 5-second timeout.
    pub fn shutdown(&mut self) {
        if let Some(tx) = self.sender.as_ref() {
            let _ = tx.try_send(AutosaveMsg::Shutdown);
        }
        // Closing the channel is what guarantees the worker exits.
        self.sender = None;
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
    // Capacity 1 is plenty: the channel carries only wake-ups now, and a wake that
    // cannot be delivered means one is already pending.
    let (tx, rx) = sync_channel::<AutosaveMsg>(1);
    let slot: SnapshotSlot = Arc::new(Mutex::new(None));
    let worker_slot = Arc::clone(&slot);

    let handle = thread::Builder::new()
        .name("nexir-autosave".into())
        .spawn(move || {
            /// Write whatever is in the slot, taking it so it is written once.
            ///
            /// Taking (rather than peeking) is what makes the burst case correct:
            /// anything pushed WHILE this write is in progress lands in the empty
            /// slot and is picked up by the next iteration.
            fn drain(slot: &SnapshotSlot) {
                let pending = slot
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                if let Some(s) = pending {
                    write_atomic(&s.path, &s.json);
                }
            }

            loop {
                // Wait up to 5 s before timing out and checking again.
                match rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(AutosaveMsg::Wake) => drain(&worker_slot),
                    Ok(AutosaveMsg::Shutdown) => break,
                    // The timeout is not just a liveness check: a wake-up that
                    // could not be delivered because the channel was full leaves a
                    // snapshot in the slot, and this is what eventually writes it.
                    Err(RecvTimeoutError::Timeout) => drain(&worker_slot),
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            // The handle is gone, so nothing more can be pushed — but something may
            // have been pushed between the last wake and now, and that is the
            // snapshot a shutdown-on-quit exists to preserve.
            drain(&worker_slot);
            log::info!("[autosave] worker thread exiting cleanly");
        })
        .expect("autosave: failed to spawn worker thread");

    AutosaveHandle {
        sender: Some(tx),
        slot,
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
