// src/io/io_layer.rs

use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use crate::render::device::GpuDevice;
use crate::timeline::ids::SourceId;
use crate::timeline::rational::Rational;
use crate::io::demuxer::Demuxer;
use crate::io::decoder::Decoder;
use crate::io::slot_pool::{FrameSlotPool, FrameSlotId};
use crate::io::frame_cache::FrameCache;
use crate::io::prefetch::PrefetchRequest;
use crate::timeline::source::SourceRegistry;

pub struct IoLayer {
    pub device:  Arc<wgpu::Device>,
    pool:        Arc<FrameSlotPool>,
    pub cache:   Arc<FrameCache>,
    source_reg:  Arc<std::sync::RwLock<SourceRegistry>>,
    demuxers:    DashMap<SourceId, Arc<Mutex<Demuxer>>>,
    decoders:    DashMap<SourceId, Arc<Mutex<Decoder>>>,
    prefetch_tx: std::sync::mpsc::SyncSender<PrefetchRequest>,
    project_tb:  Rational,
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
        }
    }

    pub fn get_or_decode(
        &self,
        source_id: SourceId,
        pts:       i64,
    ) -> Option<FrameSlotId> {
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
    ) -> Option<FrameSlotId> {
        if let Some(slot) = self.cache.get(source_id, pts) {
            return Some(slot);
        }

        let required = self.source_reg.read().unwrap().frame_size_bytes(source_id).ok()?;
        let slot = self.pool.acquire(required)?;

        let demuxer_arc = self.get_or_open_demuxer(source_id)?;
        let decoder_arc = self.get_or_open_decoder(source_id, &demuxer_arc)?;

        let actual_pts = {
            let mut demuxer = demuxer_arc.lock().unwrap();
            let stream_pts = demuxer.seek(pts, self.project_tb).ok()?;
            let mut decoder = decoder_arc.lock().unwrap();
            decoder.seek_to(&mut demuxer, stream_pts).ok()?
        };

        {
            let mut dec = decoder_arc.lock().unwrap();
            let mut dem = demuxer_arc.lock().unwrap();

            self.pool.with_buffer_mut(slot, |mapped| {
                loop {
                    let pkt = match dem.next_video_packet().ok().flatten() {
                        Some(p) => p,
                        None => break,
                    };
                    if let Some((frame_pts, _is_nv12, _w, _h)) = dec.decode_into(&pkt, mapped, None).ok().flatten() {
                        if frame_pts >= actual_pts {
                            break;
                        }
                    }
                }
            });
        } // mapped range dropped here

        // GPU upload is handled in the YuvUploadNode during render graph execution.
        // We just return the slot ID. The slot pool maps CPU memory.
        self.cache.insert(source_id, pts, slot);
        Some(slot)
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
        let decoder = Decoder::open(stream_info, stream_info.codecpar, true).ok()?;
        let arc = Arc::new(Mutex::new(decoder));
        self.decoders.insert(resolved_id, arc.clone());
        Some(arc)
    }
}
