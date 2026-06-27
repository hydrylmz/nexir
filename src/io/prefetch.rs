// src/io/prefetch.rs

use std::collections::HashSet;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use crate::timeline::ids::SourceId;
use crate::io::io_layer::IoLayer;
use crate::io::frame_cache::FrameCache;

pub const PREFETCH_DEPTH: usize = 8;

pub struct PrefetchRequest {
    pub source_id: SourceId,
    pub pts:       i64,
    pub priority:  u8,
}

pub struct PrefetchWorker {
    rx:       std::sync::mpsc::Receiver<PrefetchRequest>,
    io_layer: Arc<IoLayer>,
    cache:    Arc<FrameCache>,
    shutdown: Arc<AtomicBool>,
}

impl PrefetchWorker {
    pub fn new(
        rx:       std::sync::mpsc::Receiver<PrefetchRequest>,
        io_layer: Arc<IoLayer>,
        cache:    Arc<FrameCache>,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        Self { rx, io_layer, cache, shutdown }
    }

    pub fn run(self) {
        let mut in_flight: HashSet<(SourceId, i64)> = HashSet::new();

        while !self.shutdown.load(Ordering::Relaxed) {
            let mut pending: Vec<PrefetchRequest> = Vec::new();

            loop {
                match self.rx.try_recv() {
                    Ok(req) => pending.push(req),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
                }
            }

            if !pending.is_empty() {
                pending.sort_unstable_by_key(|r| std::cmp::Reverse(r.priority)); // highest priority (0) at end so pop() gets it
                pending.dedup_by_key(|r| (r.source_id, r.pts));

                while let Some(req) = pending.pop() {
                    if self.shutdown.load(Ordering::Relaxed) {
                        return;
                    }
                    if self.cache.get(req.source_id, req.pts).is_some() {
                        continue;
                    }
                    if in_flight.contains(&(req.source_id, req.pts)) {
                        continue;
                    }

                    in_flight.insert((req.source_id, req.pts));
                    self.io_layer.decode_blocking(req.source_id, req.pts);
                    in_flight.remove(&(req.source_id, req.pts));
                }
            } else {
                std::thread::sleep(Duration::from_micros(500));
            }
        }
    }
}

pub fn spawn_prefetch_worker(
    worker: PrefetchWorker,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("ve-prefetch".into())
        .spawn(move || worker.run())
        .expect("prefetch thread spawn failed")
}
