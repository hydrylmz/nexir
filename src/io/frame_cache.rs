// src/io/frame_cache.rs

use std::sync::{Arc, Mutex};
use std::num::NonZeroUsize;
use dashmap::DashMap;
use lru::LruCache;
use crate::timeline::ids::SourceId;
use crate::io::slot_pool::{FrameSlotId, FrameSlotPool};

type CacheKey = (SourceId, i64); // (source, pts in project timebase)

pub struct FrameCache {
    /// DashMap for O(1) concurrent reads from multiple threads.
    index: DashMap<CacheKey, (FrameSlotId, bool)>,
    /// LRU tracker — mutex-protected because eviction order requires serialisation.
    lru:   Mutex<LruCache<CacheKey, ()>>,
    /// Pool reference for releasing evicted slots.
    pool:  Arc<FrameSlotPool>,
}

impl FrameCache {
    /// Create a cache of `capacity` frames.
    pub fn new(pool: Arc<FrameSlotPool>, capacity: usize) -> Self {
        Self {
            index: DashMap::new(),
            lru: Mutex::new(LruCache::new(NonZeroUsize::new(capacity).unwrap())),
            pool,
        }
    }

    /// Look up a cached frame. Returns the (FrameSlotId, is_nv12) if present.
    /// Does NOT promote the key in the LRU — use `touch()` for that.
    pub fn get(&self, source_id: SourceId, pts: i64) -> Option<(FrameSlotId, bool)> {
        self.index.get(&(source_id, pts)).map(|r| *r)
    }

    /// Promote a key to most-recently-used and return its slot.
    pub fn touch(&self, source_id: SourceId, pts: i64) -> Option<(FrameSlotId, bool)> {
        let slot = self.get(source_id, pts)?;
        self.lru.lock().unwrap().promote(&(source_id, pts));
        Some(slot)
    }

    /// Insert a newly decoded frame into the cache.
    /// Evicts the least-recently-used entry if the cache is full.
    pub fn insert(&self, source_id: SourceId, pts: i64, slot_id: FrameSlotId, is_nv12: bool) {
        // Step 1 — Try to insert into LRU (evict if full).
        let mut lru = self.lru.lock().unwrap();
        if let Some((evicted_key, _)) = lru.push((source_id, pts), ()) {
            if let Some((_, (evicted_slot, _))) = self.index.remove(&evicted_key) {
                self.pool.release(evicted_slot);
            }
        }

        // Step 2 — Insert into the index.
        self.index.insert((source_id, pts), (slot_id, is_nv12));
    }

    pub fn evict_one(&self) -> bool {
        let mut lru = self.lru.lock().unwrap();
        if let Some((evicted_key, _)) = lru.pop_lru() {
            if let Some((_, slot_info)) = self.index.remove(&evicted_key) {
                self.pool.release(slot_info.0);
                return true;
            }
        }
        false
    }

    /// Forcibly evict all frames for a given source.
    pub fn evict_source(&self, source_id: SourceId) {
        let mut lru = self.lru.lock().unwrap();
        // We must iterate the index to find all pts for this source
        let mut to_remove = Vec::new();
        for kv in self.index.iter() {
            if kv.key().0 == source_id {
                to_remove.push(*kv.key());
            }
        }

        for key in to_remove {
            if let Some((_, slot_info)) = self.index.remove(&key) {
                // slot_info is (FrameSlotId, bool)
                self.pool.release(slot_info.0);
            }
            // Ignore if missing from LRU, but try to remove it
            let _ = lru.pop(&key);
        }
    }

    /// Number of cached frames.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Evict all cached frames, releasing their slots back to the pool.
    pub fn clear_all(&self) {
        let mut lru = self.lru.lock().unwrap();
        lru.clear();
        let keys: Vec<CacheKey> = self.index.iter().map(|kv| *kv.key()).collect();
        for key in keys {
            if let Some((_, slot_info)) = self.index.remove(&key) {
                self.pool.release(slot_info.0);
            }
        }
    }
}
