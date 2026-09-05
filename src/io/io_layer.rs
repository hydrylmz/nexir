// src/io/io_layer.rs

use std::sync::{Arc, Mutex};
use dashmap::DashMap;
use crate::timeline::ids::SourceId;
use crate::timeline::rational::Rational;
use crate::io::demuxer::Demuxer;
use crate::io::decoder::Decoder;
use crate::io::slot_pool::FrameSlotPool;
use crate::io::frame_cache::FrameCache;
use crate::io::interop_decode::{InteropDecode, InteropDecodeTargets, InteropFrame};
use crate::io::prefetch::PrefetchRequest;
use crate::timeline::source::SourceRegistry;

pub struct IoLayer {
    pub device:  Arc<wgpu::Device>,
    pub pool:        Arc<FrameSlotPool>,
    pub cache:   Arc<FrameCache>,
    pub source_reg:  Arc<std::sync::RwLock<SourceRegistry>>,
    /// Per-source NVDEC decode targets — G2c.
    ///
    /// Always present, never optional: [`InteropDecodeTargets`] answers
    /// `Unavailable` for every source on a host without CUDA and allocates nothing,
    /// so an `Option` here would only move the same check to every call site. The
    /// CPU path below is unchanged and is what runs whenever this declines.
    interop:     Arc<InteropDecodeTargets>,
    demuxers:    DashMap<SourceId, Arc<Mutex<Demuxer>>>,
    decoders:    DashMap<SourceId, Arc<Mutex<Decoder>>>,
    prefetch_tx: std::sync::mpsc::SyncSender<PrefetchRequest>,
    project_tb:  Rational,
    /// Last successfully decoded pts per source (for sequential playback optimisation).
    last_decoded_pts: DashMap<SourceId, i64>,
}

impl IoLayer {
    /// Build the layer.
    ///
    /// `interop` is the per-source GPU decode registry (G2c). Pass
    /// [`InteropDecodeTargets::disabled`] for a layer that must stay on the CPU
    /// upload path — the export layer, whose decode workers fill the CPU
    /// `FrameCache` ahead of the render cursor, and tests.
    pub fn new(
        device:      Arc<wgpu::Device>,
        pool:        Arc<FrameSlotPool>,
        cache:       Arc<FrameCache>,
        source_reg:  Arc<std::sync::RwLock<SourceRegistry>>,
        prefetch_tx: std::sync::mpsc::SyncSender<PrefetchRequest>,
        project_tb:  Rational,
        interop:     Arc<InteropDecodeTargets>,
    ) -> Self {
        Self {
            device,
            pool,
            cache,
            source_reg,
            interop,
            demuxers: DashMap::new(),
            decoders: DashMap::new(),
            prefetch_tx,
            project_tb,
            last_decoded_pts: DashMap::new(),
        }
    }

    /// The GPU decode targets, for callers that build graphs from what they hold.
    pub fn interop(&self) -> &Arc<InteropDecodeTargets> {
        &self.interop
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
        // And the GPU targets: a `SourceId` indexes the registry that was just
        // replaced, so a kept target would hand a new source the previous
        // occupant's textures — at the previous occupant's dimensions.
        self.interop.clear();
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

    /// Decode `pts` for `source_id` straight into that source's own GPU textures.
    ///
    /// **This is the G2c path, and it is the whole point of the task**: NVDEC's
    /// output surface is copied device→array into the textures the render graph
    /// binds, so the frame never crosses PCIe. What it replaces is a
    /// `av_hwframe_transfer_data` to host memory followed by a
    /// `YuvUploadNode` upload back — the 10.00 ms `GPU transfer` row on a 17.85 ms
    /// 4K frame.
    ///
    /// Returns `None` for every reason the CPU path must be used instead: no CUDA,
    /// no NVDEC on this source, a >8-bit source, a failed copy. A caller that gets
    /// `None` calls [`Self::decode_blocking`] and behaves exactly as before, which
    /// is why this can land behind a capability check without touching the
    /// fallback.
    ///
    /// **One frame in flight per source**, by construction — see
    /// `io::interop_decode`'s module header. A repeat request for the resident pts
    /// costs nothing; a request for a different one re-decodes into the same
    /// textures, so a caller must finish using a frame (i.e. submit the graph)
    /// before asking for the next.
    pub fn decode_interop(&self, source_id: SourceId, pts: i64) -> Option<InteropFrame> {
        if !self.interop.is_available() {
            return None;
        }
        // Cheap answer first: no locks beyond the target map, no decoder.
        if let Some(frame) = self.interop.held_frame(source_id, pts) {
            return Some(frame);
        }
        if self.interop.rejection_reason(source_id).is_some() {
            return None;
        }

        // A still image has no NVDEC decoder (`get_or_open_decoder` opens it with
        // `open_sw`), and its render path bypasses YUV entirely.
        let (is_still_image, info) = {
            let reg = self.source_reg.read().unwrap();
            let resolved = reg.resolve_proxy(source_id);
            let still = reg
                .path(resolved)
                .map(|path| crate::timeline::source::is_still_image_path(path.as_ref()))
                .unwrap_or(false);
            (still, reg.video_info(resolved).ok().cloned())
        };
        if is_still_image {
            return None;
        }
        let info = info?;

        let demuxer_arc = self.get_or_open_demuxer(source_id)?;
        let decoder_arc = self.get_or_open_decoder(source_id, &demuxer_arc)?;

        // Same sequential-vs-seek decision as `decode_blocking`, and for the same
        // reason: seeking per frame during playback would throw away the decoder's
        // reference frames every time.
        let prev_pts = self.last_decoded_pts.get(&source_id).map(|v| *v).unwrap_or(i64::MIN);
        let five_seconds_pts = 5 * self.project_tb.den;
        let need_seek =
            prev_pts == i64::MIN || pts < prev_pts || (pts - prev_pts) > five_seconds_pts;

        let target_stream_pts = if need_seek {
            let mut demuxer = demuxer_arc.lock().unwrap();
            let stream_pts = demuxer.seek(pts, self.project_tb).ok()?;
            let mut decoder = decoder_arc.lock().unwrap();
            // The textures still hold the pre-seek frame; forget it, or the next
            // request for this pts could be served as a `Cached` hit showing the
            // wrong picture.
            self.interop.invalidate(source_id);
            // FLUSH, NOT `seek_to` — see `decode_blocking`'s copy of this branch
            // for what `seek_to` costs. Same bug on this arm, and it was invisible
            // to `bench --interop`'s pixel check for a second reason: `Δpx 0/255`
            // compares the two arms against each other and BOTH were shifted by
            // one frame, so a cross-arm comparison cannot see a defect the arms
            // share.
            decoder.flush();
            stream_pts
        } else {
            let demux = demuxer_arc.lock().unwrap();
            let stream_tb = demux.video_stream.as_ref()?.time_base;
            drop(demux);
            self.project_tb.rescale_pts(pts, stream_tb)
        };

        let mut dec = decoder_arc.lock().unwrap();
        let mut dem = demuxer_arc.lock().unwrap();

        // Read forward for the target frame, exactly as the CPU path does — the
        // 600-packet bound is what stops a corrupt file spinning here.
        //
        // Every frame passed through on the way lands in the SAME textures and is
        // overwritten by the next, which is why `mark_held` is called only for the
        // one that is kept: marking a discarded frame resident would serve it as a
        // hit on the following request, showing the wrong picture with no error.
        for _ in 0..600 {
            let pkt = match dem.next_video_packet().ok().flatten() {
                Some(p) => p,
                None => break,
            };
            let pkt_pts = pkt.pts;
            match self.interop.decode_into_target(source_id, &info, &mut dec, &pkt) {
                InteropDecode::Decoded(frame) => {
                    // Same rule as the CPU path: a frame whose own pts is 0 has
                    // probably not had one set, so fall back to the packet's.
                    let eff_pts = if frame.pts == 0 { pkt_pts } else { frame.pts };
                    if eff_pts >= target_stream_pts {
                        self.last_decoded_pts.insert(source_id, pts);
                        self.interop.mark_held(source_id, pts, &frame);
                        return Some(frame);
                    }
                }
                InteropDecode::Pending => continue,
                InteropDecode::Unavailable(reason) => {
                    log::debug!(
                        "[interop] source {} falls back to the CPU path: {reason}",
                        source_id.index()
                    );
                    return None;
                }
            }
        }
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
                // FLUSH AND LET THE READ-FORWARD LOOP FIND IT — never `seek_to`.
                //
                // **`Decoder::seek_to` DISCARDS the frame it lands on**, and that is
                // the whole remaining G3 failure. Its discard loop decodes until
                // `frame_pts >= target`, unrefs that frame and returns its pts; the
                // loop below then searches for `eff_pts >= target_stream_pts`
                // starting from the NEXT frame. So a cold request for frame N is
                // answered with frame N+1, **cached under N's key**, and every
                // sequential request after it is shifted by one — until the tail,
                // where there is no N+1 left and the `!decoded_anything` arm serves
                // the previous frame out of `FrameCache`.
                //
                // Measured with `examples/g3_drain_probe.rs` (phases 4 and 5 exist
                // to separate this from the drain): replicating this function
                // WITHOUT the cold seek serves 60/60 frames of a 60-frame file, all
                // fresh, the last two out of the drain. Replicating it WITH the seek
                // serves 59 and prints `got=Some(512)` for a request whose target is
                // 0, `Some(29184)` for a target of 28672, and `None` for the last —
                // i.e. the wrong picture on all 59 and nothing at all on the 60th.
                //
                // **The shift was the more serious half and it was invisible.** A
                // one-frame offset in the picture is not something any assertion in
                // the tree looked at: the frame is cached under the pts that was
                // asked for, so `cache.get(source, pts)` succeeds and the G3 test's
                // own `fresh` count was 59/60 rather than 0/60. Only the single
                // missing frame at the end showed up, which is why this read as "one
                // last stale frame" rather than as "every frame after a seek is the
                // wrong one".
                //
                // The read-forward loop below already does exactly what `seek_to`'s
                // discard loop does — decode from the keyframe until a frame at or
                // past the target — except that it KEEPS that frame, in the slot
                // acquired for it. So the seek branch only has to put the decoder in
                // a state where the loop can run: `flush()` (which also clears the
                // draining flag and any EAGAIN backlog from before the seek) plus the
                // rescaled target. That is what the still-image branch has always
                // done; the two are now the same shape for the same reason.
                //
                // What changes with it: the post-seek scan is now bounded by the same
                // 600 packets as any other read-forward, where `seek_to`'s loop ran
                // to EOF. A file whose GOP exceeds 600 packets therefore reports a
                // stale frame instead of grinding through it — the corrupt-file guard
                // doing its job, and `hit_eof` keeps it from draining (see the drain
                // block below and `the_600_packet_bound_does_not_leave_the_decoder_draining`).
                let mut demuxer = demuxer_arc.lock().unwrap();
                let stream_pts = if is_still_image {
                    let stream_tb = demuxer.video_stream.as_ref()?.time_base;
                    self.project_tb.rescale_pts(0, stream_tb)
                } else {
                    demuxer.seek(pts, self.project_tb).ok()?
                };
                let mut decoder = decoder_arc.lock().unwrap();
                decoder.flush();
                stream_pts
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
                let mut hit_eof = false;
                for _ in 0..600 {
                    let pkt = match dem.next_video_packet().ok().flatten() {
                        Some(p) => p,
                        None    => { hit_eof = true; break }
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

                // ── G3: DRAIN AT EOF ─────────────────────────────────────────
                //
                // **Without this the last frames of every clip are unreachable**,
                // and the symptom is not an error. `avcodec_set_thread_count(ctx, 0)`
                // gives FFmpeg one thread per core and each holds a frame back, so
                // the send/receive loop above simply stops producing before the
                // file ends — measured: frame 57 of a 60-frame `cam_4k30`. What the
                // user sees is the playhead moving over the last second while the
                // picture freezes, because the `!decoded_anything` arm below hands
                // back the previous frame out of `FrameCache`. That disguise is why
                // this went unnoticed; `bench --interop` found it only because a
                // one-frame interop target has no cache to fall back on.
                //
                // Draining at end of stream is what every media pipeline does. It
                // writes into the SAME slot already acquired above, so nothing about
                // the cache or the slot pool changes.
                //
                // Only when the demuxer actually ran dry: `hit_eof` distinguishes
                // that from the 600-packet bound, which is the corrupt-file guard
                // and must not put the decoder into draining mode (it refuses input
                // afterwards, so every later frame on this source would disappear
                // until a seek flushed it).
                if decoded_anything || !hit_eof {
                    return;
                }
                loop {
                    match dec.drain_into(mapped) {
                        Ok(Some(frame)) => {
                            // No packet to fall back on here, so the frame's own pts
                            // is all there is — a drained frame always carries one.
                            if frame.pts >= target_stream_pts {
                                found_pts = frame.pts;
                                final_meta = frame.meta;
                                decoded_anything = true;
                                break;
                            }
                        }
                        // Fully drained: the decoder holds nothing more, so the
                        // request is genuinely past the end of the stream.
                        Ok(None) => break,
                        Err(e) => {
                            log::debug!("[io] drain_into at EOF failed: {e:?}");
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
