// src/io/io_layer.rs

use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use crate::timeline::ids::SourceId;
use crate::timeline::rational::Rational;
use crate::io::demuxer::Demuxer;
use crate::io::decoder::Decoder;
use crate::io::slot_pool::FrameSlotPool;
use crate::io::frame_cache::FrameCache;
use crate::io::prefetch::PrefetchRequest;
use crate::timeline::source::SourceRegistry;

pub struct IoLayer {
    pub device:  Arc<wgpu::Device>,
    pub pool:        Arc<FrameSlotPool>,
    pub cache:   Arc<FrameCache>,
    pub source_reg:  Arc<std::sync::RwLock<SourceRegistry>>,
    demuxers:    DashMap<SourceId, Arc<Mutex<Demuxer>>>,
    decoders:    DashMap<SourceId, Arc<Mutex<Decoder>>>,
    prefetch_tx: std::sync::mpsc::SyncSender<PrefetchRequest>,
    project_tb:  Rational,
    /// Last successfully decoded pts per source (for sequential playback optimisation).
    last_decoded_pts: DashMap<SourceId, i64>,
}

impl IoLayer {
    pub fn new(
        device:      Arc<wgpu::Device>,
        pool:        Arc<FrameSlotPool>,
        cache:       Arc<FrameCache>,
        source_reg:  Arc<std::sync::RwLock<SourceRegistry>>,
        prefetch_tx: std::sync::mpsc::SyncSender<PrefetchRequest>,
        project_tb:  Rational,
    ) -> Self {
        Self {
            device,
            pool,
            cache,
            source_reg,
            demuxers: DashMap::new(),
            decoders: DashMap::new(),
            prefetch_tx,
            project_tb,
            last_decoded_pts: DashMap::new(),
        }
    }

    /// Clear all stale decoder/demuxer/cache state.
    /// Must be called when the project's SourceRegistry changes (new/open project).
    pub fn reset_for_new_project(&self, new_source_reg: &std::sync::Arc<std::sync::RwLock<SourceRegistry>>) {
        // Replace the contents of our source_reg with the new one
        let new_reg = new_source_reg.read().unwrap();
        let mut our_reg = self.source_reg.write().unwrap();
        *our_reg = new_reg.clone();
        drop(our_reg);
        drop(new_reg);

        // Clear stale demuxers and decoders (they reference old source files)
        self.demuxers.clear();
        self.decoders.clear();
        self.last_decoded_pts.clear();
        self.cache.clear_all();
    }

    pub fn get_or_decode(
        &self,
        source_id: SourceId,
        pts:       i64,
    ) -> Option<crate::io::frame_cache::CachedFrame> {
        if let Some(slot) = self.cache.touch(source_id, pts) {
            return Some(slot);
        }

        let _ = self.prefetch_tx.try_send(PrefetchRequest {
            source_id,
            pts,
            priority: 0,
        });

        None
    }

    pub fn decode_blocking(
        &self,
        source_id: SourceId,
        pts:       i64,
    ) -> Option<crate::io::frame_cache::CachedFrame> {
        if let Some(slot) = self.cache.touch(source_id, pts) {
            return Some(slot);
        }

        let is_still_image = self
            .source_reg
            .read()
            .unwrap()
            .path(source_id)
            .map(|path| crate::timeline::source::is_still_image_path(path.as_ref()))
            .unwrap_or(false);
        if is_still_image {
            let resolved_id = self.source_reg.read().unwrap().resolve_proxy(source_id);
            self.demuxers.remove(&resolved_id);
            self.decoders.remove(&resolved_id);
            self.last_decoded_pts.remove(&source_id);
        }

        let demuxer_arc = self.get_or_open_demuxer(source_id)?;
        let decoder_arc = self.get_or_open_decoder(source_id, &demuxer_arc)?;

        // Check if we can decode forward without seeking.
        // Only seek if target is before last decoded position or too far ahead (>5s in 90kHz ticks).
        let prev_pts = self.last_decoded_pts.get(&source_id).map(|v| *v).unwrap_or(i64::MIN);
        // 5 seconds in project timebase (90000 ticks/second)
        let five_seconds_pts = 5 * self.project_tb.den;
        let need_seek = is_still_image
            || prev_pts == i64::MIN
            || pts < prev_pts
            || (pts - prev_pts) > five_seconds_pts;

        let target_stream_pts = {
            if need_seek {
                if is_still_image {
                    let stream_pts = {
                        let demuxer = demuxer_arc.lock().unwrap();
                        let stream_tb = demuxer.video_stream.as_ref()?.time_base;
                        self.project_tb.rescale_pts(0, stream_tb)
                    };
                    let mut decoder = decoder_arc.lock().unwrap();
                    decoder.flush();
                    stream_pts
                } else {
                    let mut demuxer = demuxer_arc.lock().unwrap();
                    let stream_pts = demuxer.seek(pts, self.project_tb).ok()?;
                    let mut decoder = decoder_arc.lock().unwrap();
                    decoder.seek_to(&mut demuxer, stream_pts).ok()?
                }
            } else {
                // Sequential path: translate project PTS to stream PTS without seeking
                let demux = demuxer_arc.lock().unwrap();
                let stream_tb = demux.video_stream.as_ref()?.time_base;
                drop(demux);
                self.project_tb.rescale_pts(pts, stream_tb)
            }
        };

        // Acquire slot AFTER early returns to avoid leaking on seek/open failures
        let required = self.source_reg.read().unwrap().frame_size_bytes(source_id).ok()?;
        let mut slot = self.pool.acquire(required);
        while slot.is_none() {
            if !self.cache.evict_one() {
                break;
            }
            slot = self.pool.acquire(required);
        }
        let slot = slot?;

        // Metadata of the frame that actually landed in the slot.  Defaults to
        // 8-bit BT.709 only as a placeholder; `decoded_anything` gates whether it
        // is ever used, so an undecoded slot never reaches the cache.
        let mut final_meta = crate::timeline::source::DecodedFrameMeta::default();
        let mut decoded_anything = false;
        let _decoded_pts = {
            let mut dec = decoder_arc.lock().unwrap();
            let mut dem = demuxer_arc.lock().unwrap();
            let mut found_pts = target_stream_pts;

            self.pool.with_buffer_mut(slot, |mapped| {
                // Read forward up to 600 packets to find the target frame
                for _ in 0..600 {
                    let pkt = match dem.next_video_packet().ok().flatten() {
                        Some(p) => p,
                        None    => break,
                    };
                    let pkt_pts = pkt.pts;
                    if let Some(frame) = dec.decode_into(&pkt, mapped, None).ok().flatten() {
                        let eff_pts = if frame.pts == 0 { pkt_pts } else { frame.pts };
                        if eff_pts >= target_stream_pts {
                            found_pts = eff_pts;
                            final_meta = frame.meta;
                            decoded_anything = true;
                            break;
                        }
                    }
                }
            });

            found_pts
        };

        if !decoded_anything {
            self.pool.release(slot);
            if let Some(prev) = self.last_decoded_pts.get(&source_id).map(|v| *v) {
                if let Some(cached) = self.cache.touch(source_id, prev) {
                    return Some(cached);
                }
            }
            return None;
        }

        // Update sequential tracking using requested pts
        self.last_decoded_pts.insert(source_id, pts);

        self.cache.insert(source_id, pts, slot, final_meta);
        Some((slot, final_meta))
    }

    fn get_or_open_demuxer(
        &self,
        source_id: SourceId,
    ) -> Option<Arc<Mutex<Demuxer>>> {
        let resolved_id = self.source_reg.read().unwrap().resolve_proxy(source_id);
        if let Some(d) = self.demuxers.get(&resolved_id) {
            return Some(d.clone());
        }

        let path = self.source_reg.read().unwrap().path(resolved_id)?;
        let demuxer = Demuxer::open(&path).ok()?;
        let arc = Arc::new(Mutex::new(demuxer));
        self.demuxers.insert(resolved_id, arc.clone());
        Some(arc)
    }

    fn get_or_open_decoder(
        &self,
        source_id: SourceId,
        demuxer:   &Arc<Mutex<Demuxer>>,
    ) -> Option<Arc<Mutex<Decoder>>> {
        let resolved_id = self.source_reg.read().unwrap().resolve_proxy(source_id);
        if let Some(d) = self.decoders.get(&resolved_id) {
            return Some(d.clone());
        }

        let demux_lock = demuxer.lock().unwrap();
        let stream_info = demux_lock.video_stream.as_ref()?;
        let is_still_image = self
            .source_reg
            .read()
            .unwrap()
            .path(resolved_id)
            .map(|path| crate::timeline::source::is_still_image_path(path.as_ref()))
            .unwrap_or(false);
        let decoder = if is_still_image {
            Decoder::open_sw(stream_info, stream_info.codecpar).ok()?
        } else {
            Decoder::open(stream_info, stream_info.codecpar, true).ok()?
        };
        let arc = Arc::new(Mutex::new(decoder));
        self.decoders.insert(resolved_id, arc.clone());
        Some(arc)
    }

    /// Enqueue prefetch requests for every registered source at each of the
    /// given PTS values.  Call this at the start of an export segment so the
    /// prefetch worker can decode ahead while the GPU renders the first frame.
    pub fn prime_export_prefetch(&self, pts_list: &[i64]) {
        let source_ids: Vec<SourceId> = {
            let reg = self.source_reg.read().unwrap();
            reg.all_source_ids()
        };
        for &pts in pts_list {
            for &sid in &source_ids {
                let _ = self.prefetch_tx.try_send(PrefetchRequest {
                    source_id: sid,
                    pts,
                    priority: 0,
                });
            }
        }
    }
}
