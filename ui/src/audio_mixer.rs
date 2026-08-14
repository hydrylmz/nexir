// ui/src/audio_mixer.rs
//
// Multi-track audio mixing layer.
// Each active audio clip gets its own per-clip AudioRingBuffer.
// A dedicated mixer thread reads from all per-clip rings, sums them,
// and writes the result into the master AudioRingBuffer that CPAL reads from.
//
// This module is currently unused — multi-track mixing is scaffolded here
// for future integration. The primary audio path remains in app.rs.

#![allow(dead_code)]

use nexir::audio::ring_buffer::AudioRingBuffer;
use nexir::sync::master_clock::MasterClock;
use nexir::timeline::ids::ClipId;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone)]
pub struct AudioClipInfo {
    pub clip_id: ClipId,
    pub path: PathBuf,
    pub volume: f32,
    pub pan: f32,
    pub muted: bool,
    pub speed: f32,
    pub pitch: f32,
    pub source_pts: i64,   // source file pts to start decoding from
    pub timeline_pts: i64, // timeline pts where this clip starts
    pub source_in_pts: i64,
    pub source_out_pts: i64,
}

struct PerClipDecoder {
    shutdown: Arc<AtomicBool>,
    ring: Arc<AudioRingBuffer>,
}

pub struct AudioMixer {
    active: Arc<Mutex<Vec<AudioClipInfo>>>,
    seek_tx: Arc<Mutex<Option<i64>>>,
    shutdown: Arc<AtomicBool>,
    master_ring: Arc<AudioRingBuffer>,
    master_clock: Arc<MasterClock>,
}

impl AudioMixer {
    pub fn new(master_ring: Arc<AudioRingBuffer>, master_clock: Arc<MasterClock>) -> Self {
        Self {
            active: Arc::new(Mutex::new(Vec::new())),
            seek_tx: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            master_ring,
            master_clock,
        }
    }

    pub fn set_active(&self, clips: Vec<AudioClipInfo>) {
        *self.active.lock().unwrap() = clips;
    }

    pub fn seek(&self, timeline_pts: i64) {
        *self.seek_tx.lock().unwrap() = Some(timeline_pts);
    }

    /// Spawn the background mixer thread.
    pub fn start(&self) {
        let active = Arc::clone(&self.active);
        let seek_tx = Arc::clone(&self.seek_tx);
        let shutdown = Arc::clone(&self.shutdown);
        let master_ring = Arc::clone(&self.master_ring);
        let master_clock = Arc::clone(&self.master_clock);

        std::thread::spawn(move || {
            let mut decoders: HashMap<ClipId, PerClipDecoder> = HashMap::new();

            while !shutdown.load(Ordering::Relaxed) {
                // ── Handle seek ───────────────────────────────────────────
                let seek_target = seek_tx.lock().unwrap().take();
                if let Some(timeline_pts) = seek_target {
                    for dec in decoders.values() {
                        dec.shutdown.store(true, Ordering::Relaxed);
                    }
                    decoders.clear();
                    master_ring.clear();
                    master_clock.seek(timeline_pts);
                }

                // ── Sync active decoders ──────────────────────────────────
                let clips = active.lock().unwrap().clone();

                // Remove decoders for clips that are no longer active
                decoders.retain(|id, dec| {
                    let alive = clips.iter().any(|c| c.clip_id == *id);
                    if !alive {
                        dec.shutdown.store(true, Ordering::Relaxed);
                    }
                    alive
                });

                // Spawn decoders for new active clips
                for clip in &clips {
                    if decoders.contains_key(&clip.clip_id) {
                        continue;
                    }

                    let clip_shutdown = Arc::new(AtomicBool::new(false));
                    let clip_seek =
                        Arc::new(Mutex::new(Some((clip.source_pts, clip.timeline_pts))));
                    let clip_ring = AudioRingBuffer::new(1 << 16); // 64 k samples

                    let path_clone = clip.path.clone();
                    let ring_clone = Arc::clone(&clip_ring);
                    let clock_clone = Arc::clone(&master_clock);
                    let sd_clone = Arc::clone(&clip_shutdown);
                    let seek_clone = Arc::clone(&clip_seek);
                    let project_tb = nexir::timeline::rational::Rational {
                        num: 1,
                        den: 90_000,
                    };
                    let volume = clip.volume;
                    let pan = clip.pan;
                    let muted = clip.muted;
                    let speed = clip.speed;
                    let pitch = clip.pitch;
                    let src_in = clip.source_in_pts;
                    let src_out = clip.source_out_pts;

                    std::thread::spawn(move || {
                        if let Ok(dec) = nexir::audio::audio_decoder::AudioDecoder::new(
                            &path_clone,
                            ring_clone,
                            clock_clone,
                            project_tb,
                            sd_clone,
                            seek_clone,
                            volume,
                            pan,
                            muted,
                            speed,
                            pitch,
                            src_in,
                            src_out,
                        ) {
                            dec.run();
                        }
                    });

                    decoders.insert(
                        clip.clip_id,
                        PerClipDecoder {
                            shutdown: clip_shutdown,
                            ring: clip_ring,
                        },
                    );
                }

                // ── Mix all per-clip rings into the master ring ───────────
                const CHUNK: usize = 512;
                if master_ring.available_write() < CHUNK * 2 {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    continue;
                }

                let mut mixed = [0.0f32; CHUNK * 2];
                let mut any_data = false;

                for dec in decoders.values() {
                    if dec.ring.available_read() >= CHUNK * 2 {
                        let mut tmp = [0.0f32; CHUNK * 2];
                        let n = dec.ring.read(&mut tmp);
                        if n > 0 {
                            any_data = true;
                            for i in 0..n {
                                mixed[i] += tmp[i];
                            }
                        }
                    }
                }

                // Clamp and write (silence if nothing active, to keep CPAL fed)
                for v in &mut mixed {
                    *v = v.clamp(-1.0, 1.0);
                }
                master_ring.write(&mixed);

                if !any_data {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }

            // Cleanup
            for dec in decoders.values() {
                dec.shutdown.store(true, Ordering::Relaxed);
            }
        });
    }
}

impl Drop for AudioMixer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}
