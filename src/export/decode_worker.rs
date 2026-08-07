// src/export/decode_worker.rs
//
// Dedicated pre-decode pipeline for the export path.
//
// Problem: `decode_blocking` (FFmpeg H.264 SW decode of 1080p) takes ~30–60ms per
// frame per clip.  When called synchronously on the dispatch/render thread it
// serialises decode → GPU render → readback, yielding ≈2 fps.
//
// Solution: spawn one thread per active clip source that decodes frames sequentially
// in export order, depositing results into the shared FrameCache exactly LOOKAHEAD
// frames ahead of the render cursor.  The render loop then *only* hits the cache
// (cache hit ≈ 0ms), so GPU work can proceed without ever blocking on I/O.
//
// The worker communicates through a lightweight `Condvar`-based barrier:
//   - The render thread waits on the barrier for frame N if it isn't cached yet.
//   - The decode worker signals the barrier each time it inserts frame N.

use std::sync::{Arc, Condvar, Mutex};
use std::collections::HashMap;
use crate::io::io_layer::IoLayer;
use crate::io::frame_cache::FrameCache;
use crate::timeline::ids::SourceId;
use crate::timeline::rational::Rational;

/// How many frames the decode worker tries to stay ahead of the render cursor.
pub const DECODE_LOOKAHEAD: usize = 16;

/// Shared state between the decode worker and the render thread.
pub struct DecodeSync {
    /// The index of the next frame the render loop is about to consume.
    /// The decode worker reads this to know how far ahead to decode.
    pub render_cursor: Mutex<usize>,
    pub condvar:       Condvar,
}

impl DecodeSync {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            render_cursor: Mutex::new(0),
            condvar:       Condvar::new(),
        })
    }

    /// Called by the render thread before reading frame `frame_idx`.
    /// Advances the cursor so the decode worker knows it can decode further ahead.
    pub fn advance_cursor(&self, frame_idx: usize) {
        let mut cur = self.render_cursor.lock().unwrap();
        if frame_idx + 1 > *cur {
            *cur = frame_idx + 1;
        }
        self.condvar.notify_all();
    }
}

/// One decode worker per (source_id, pts_sequence) pair.
pub struct ExportDecodeWorker {
    io:          Arc<IoLayer>,
    cache:       Arc<FrameCache>,
    source_id:   SourceId,
    /// Ordered list of (frame_idx, quantized_source_pts) to decode.
    work_items:  Vec<(usize, i64)>,
    sync:        Arc<DecodeSync>,
}

impl ExportDecodeWorker {
    pub fn new(
        io:         Arc<IoLayer>,
        cache:      Arc<FrameCache>,
        source_id:  SourceId,
        work_items: Vec<(usize, i64)>,
        sync:       Arc<DecodeSync>,
    ) -> Self {
        Self { io, cache, source_id, work_items, sync }
    }

    pub fn run(self) {
        let Self { io, cache, source_id, work_items, sync } = self;

        for (frame_idx, pts) in work_items {
            // Wait until the render cursor is close enough that decoding this
            // frame makes sense (don't run too far ahead — wastes cache slots).
            let mut cur = sync.render_cursor.lock().unwrap();
            while frame_idx >= *cur + DECODE_LOOKAHEAD {
                cur = sync.condvar.wait_timeout(cur, std::time::Duration::from_millis(10)).unwrap().0;
            }
            drop(cur);

            // Skip if already in cache (e.g., duplicate PTS across clips)
            if cache.get(source_id, pts).is_some() {
                continue;
            }

            // Decode this frame into the cache.  `decode_blocking` is safe to
            // call from this thread since each worker owns its own demuxer/decoder
            // handle via the per-source DashMap entries in IoLayer.
            let _ = io.decode_blocking(source_id, pts);
        }
    }
}

/// Builds work item lists for all sources active during the export, grouped by source.
///
/// `frames`: iterator over (frame_index, source_id, quantized_pts) for every
/// frame in the export — the caller builds this from FrameScheduler state.
pub fn build_work_items(
    frames: &[(usize, SourceId, i64)],
) -> HashMap<SourceId, Vec<(usize, i64)>> {
    let mut map: HashMap<SourceId, Vec<(usize, i64)>> = HashMap::new();
    for &(fi, sid, pts) in frames {
        map.entry(sid).or_default().push((fi, pts));
    }
    map
}

/// Spawns one decode worker thread per source and returns the join handles.
pub fn spawn_decode_workers(
    io:      Arc<IoLayer>,
    cache:   Arc<FrameCache>,
    items:   HashMap<SourceId, Vec<(usize, i64)>>,
    sync:    Arc<DecodeSync>,
) -> Vec<std::thread::JoinHandle<()>> {
    items.into_iter().map(|(sid, work)| {
        let io    = Arc::clone(&io);
        let cache = Arc::clone(&cache);
        let sync  = Arc::clone(&sync);
        std::thread::Builder::new()
            .name(format!("ve-decode-src{}", sid.index()))
            .spawn(move || {
                ExportDecodeWorker::new(io, cache, sid, work, sync).run();
            })
            .expect("failed to spawn decode worker")
    }).collect()
}
