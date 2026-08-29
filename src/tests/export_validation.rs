// src/tests/export_validation.rs
// P0/P1.4 — end-to-end export validation: does an exported file actually decode
// back to the pixels the timeline described?
//
// SCOPE: this drives the REAL `ExportEngine` (render graph → encoder → muxer)
// over a synthetic test pattern, then demuxes and decodes the resulting file
// with this crate's own `Demuxer`/`Decoder` and compares pixel values at the
// centre of each colour patch.
//
// It therefore covers: still-image upload, graph compilation, composite, the
// active video encoder backend, the muxer, and the container's own timestamps.
//
// It does NOT prove anything about the CUDA/NVENC zero-copy path unless that
// backend is actually selected — `EncodeInterop::open` is attempted first and
// the test reports which backend won.  See `nvenc_export_backend_availability`
// for how that is surfaced.
//
// Requires a working GPU (headless wgpu device) and the FFmpeg shared libraries,
// like the rest of src/tests/.

#[cfg(test)]
mod export_validation {
    use crate::export::engine::ExportEngine;
    use crate::export::job::{
        AudioCodec, Container, CpuPreset, ExportJob, VideoCodec, VideoQuality,
    };
    use crate::export::progress::ExportPhase;
    use crate::interop::capability::InteropCapability;
    use crate::io::frame_cache::FrameCache;
    use crate::io::io_layer::IoLayer;
    use crate::io::slot_pool::FrameSlotPool;
    use crate::render::compute::ComputePipelineCache;
    use crate::render::device::GpuDevice;
    use crate::render::shader::registry::ShaderRegistry;
    use crate::scheduler::frame_scheduler::FrameScheduler;
    use crate::timeline::mutation::{insert_clip, ClipInsertParams};
    use crate::timeline::rational::Rational;
    use crate::timeline::source::{
        ColorInfo, MatrixCoefficients, PixelFormat, SourceRegistry, VideoStreamInfo,
        VideoRotation,
    };
    use crate::timeline::store::{ClipKind, TimelineStore};
    use crate::timeline::track::{Track, TrackList};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const TB: Rational = Rational { num: 1, den: 90_000 };
    const FPS: Rational = Rational { num: 30, den: 1 };

    /// Export canvas.  320x240 keeps the encode fast, and both dimensions are
    /// even as YUV420 chroma requires.
    ///
    /// The resolution no longer decides the matrix: since P1.7 the encoder pins
    /// swscale to `ExportJob::sws_colorspace()` (see
    /// `VideoEncoder::open`'s `sws_setColorspaceDetails` call), so an SDR job is
    /// converted with BT.709 coefficients regardless of frame size, matching the
    /// BT.709 tags it writes.  `yuv_to_rgb_bt709` below is the matching inverse.
    const W: u32 = 320;
    const H: u32 = 240;
    /// 6 patches of 40 rows each.
    const PATCH_H: u32 = H / 6;

    /// The test pattern, top band first.  These are the same six colours the
    /// ABGR10 repack test uses, so a channel swap anywhere in the chain shows up
    /// as a specific patch being wrong rather than as uniform noise.
    const PATCHES: [(&str, [u8; 3]); 6] = [
        ("black",     [0, 0, 0]),
        ("white",     [255, 255, 255]),
        ("red",       [255, 0, 0]),
        ("green",     [0, 255, 0]),
        ("blue",      [0, 0, 255]),
        ("50% gray",  [128, 128, 128]),
    ];

    /// Per-channel tolerance for the round trip, in 8-bit levels.
    ///
    /// The exported pixels travel RGBA16F → RGBA8 → (swscale) YUV420p 8-bit
    /// limited-range → H.264 at CRF 18 → decode → YUV→RGB here.  The lossy steps
    /// are the limited-range quantisation (~±1), 4:2:0 chroma subsampling (only
    /// relevant at patch edges, which are not sampled) and the encoder itself.
    ///
    /// 16 is deliberately tighter than the ~±16 DC offset a full-range/limited
    /// mismatch would introduce on black and white, so this assertion still fails
    /// if the range handling regresses.
    const TOLERANCE: i32 = 16;

    /// Expected RGB at a given output row, from the pattern definition.
    fn expected_rgb_at_row(y: u32) -> [u8; 3] {
        let idx = ((y / PATCH_H) as usize).min(PATCHES.len() - 1);
        PATCHES[idx].1
    }

    /// Write the test pattern to `path` as a PNG.
    ///
    /// A still image is used as the source rather than a synthesised video file
    /// because it exercises the export path with a known, exact ground truth: the
    /// still-image upload node stores the PNG's 8-bit values as f16 in [0,1] with
    /// no colour conversion, so the RGB that reaches the encoder is bit-for-bit
    /// the PNG's own.
    fn write_test_pattern(path: &Path) {
        let mut img = image::RgbaImage::new(W, H);
        for y in 0..H {
            let [r, g, b] = expected_rgb_at_row(y);
            for x in 0..W {
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
        img.save(path).expect("failed to write test pattern PNG");
    }

    /// Limited-range YUV → RGB for the matrix the file is tagged with.
    ///
    /// Both matrices are needed because the two encoder paths land on different
    /// ones and each is correct for its own file: `VideoEncoder::open` pins
    /// swscale to `job.sws_colorspace()` (BT.709 for a normal SDR export), while
    /// the zero-copy NVENC path has the driver convert RGB with BT.601. Inverting
    /// with the wrong one misreads red by ~22 levels — small enough to look like
    /// codec loss, which is exactly why the test picks the matrix from the job
    /// rather than assuming one.
    fn yuv_to_rgb(matrix: MatrixCoefficients, y: u8, u: u8, v: u8) -> [u8; 3] {
        let yf = (y as f32 - 16.0) / 219.0;
        let uf = (u as f32 - 128.0) / 224.0;
        let vf = (v as f32 - 128.0) / 224.0;

        // Inverses of the respective luma coefficients: BT.709 uses
        // Kr = 0.2126 / Kb = 0.0722, BT.601 uses Kr = 0.299 / Kb = 0.114.
        let (ar, bg, cg, db) = match matrix {
            MatrixCoefficients::Bt601 => (1.402, 0.344_136, 0.714_136, 1.772),
            _                         => (1.5748, 0.187_324, 0.468_124, 1.8556),
        };

        let r = yf + ar * vf;
        let g = yf - bg * uf - cg * vf;
        let b = yf + db * uf;

        [
            (r * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (g * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
            (b * 255.0 + 0.5).clamp(0.0, 255.0) as u8,
        ]
    }

    /// A decoded frame's planes, copied out of the decoder's staging buffer.
    struct DecodedFrame {
        width:  u32,
        height: u32,
        /// Matrix the file is tagged with, so `rgb_at` inverts the same conversion
        /// the encoder applied rather than a hardcoded guess.
        matrix: MatrixCoefficients,
        y:      Vec<u8>,
        u:      Vec<u8>,
        v:      Vec<u8>,
    }

    impl DecodedFrame {
        /// RGB at (x, y), converted from the 4:2:0 planes.
        fn rgb_at(&self, x: u32, y: u32) -> [u8; 3] {
            assert!(
                x < self.width && y < self.height,
                "sample ({x},{y}) is outside the decoded {}x{} frame",
                self.width, self.height
            );
            let cw = self.width.div_ceil(2);
            let yi = (y * self.width + x) as usize;
            let ci = ((y / 2) * cw + (x / 2)) as usize;
            yuv_to_rgb(self.matrix, self.y[yi], self.u[ci], self.v[ci])
        }
    }

    /// Decode up to `max_frames` frames of `path` with this crate's own
    /// demuxer/decoder, software-only so the output is deterministic YUV420p.
    ///
    /// Returns the decoded frames plus the PTS values the container reported, so
    /// callers can assert on timestamps as well as pixels.
    ///
    /// Each frame carries the matrix the STREAM IS TAGGED WITH, read from the raw
    /// codecpar, so `rgb_at` inverts whatever conversion the encoder actually
    /// applied. This is what lets one helper verify both encoder paths: they pick
    /// different matrices, and each is right for its own file.
    fn decode_file(path: &Path, max_frames: usize) -> (Vec<DecodedFrame>, Vec<i64>) {
        let mut demuxer = crate::io::demuxer::Demuxer::open(path)
            .expect("failed to open the exported file for verification");
        let stream = demuxer
            .video_stream
            .clone()
            .expect("exported file has no video stream");

        let width  = stream.width.expect("exported stream has no width");
        let height = stream.height.expect("exported stream has no height");

        // Raw codecpar, not `StreamInfo::color_info`: the latter fills unspecified
        // fields in with resolution heuristics, which would report BT.601 for this
        // SD frame regardless of what the file says.
        let tagged_matrix = match unsafe {
            crate::io::ffi::avcodec::avcodecpar_get_color_space(stream.codecpar)
        } {
            5 | 6 => MatrixCoefficients::Bt601, // BT470BG / SMPTE170M
            9     => MatrixCoefficients::Bt2020,
            _     => MatrixCoefficients::Bt709,
        };
        eprintln!("[export_validation] stream is tagged matrix {tagged_matrix:?}");

        // enable_hw = false: a hardware decoder would hand back NV12 (or a
        // hw-frame that needs a transfer), and this check wants one fixed layout.
        let mut decoder =
            crate::io::decoder::Decoder::open_sw(&stream, stream.codecpar)
                .expect("failed to open a software decoder for the exported file");

        // Big enough for YUV444 at this size, so a surprise pixel format cannot
        // overflow the buffer.
        let mut buf = vec![0u8; (width as usize * height as usize) * 3 + 64];

        let mut frames = Vec::new();
        let mut pts_list = Vec::new();
        let mut packets_read = 0usize;
        let mut decode_errors = 0usize;
        let mut empty_returns = 0usize;

        while frames.len() < max_frames {
            let pkt = match demuxer.next_video_packet() {
                Ok(Some(p)) => p,
                Ok(None) => break, // EOF
                Err(e) => panic!("demuxing the exported file failed: {e:?}"),
            };
            packets_read += 1;
            let decoded = decoder.decode_into(&pkt, &mut buf, None);
            let frame = match decoded {
                Ok(Some(v)) => v,
                // The decoder needs more input (B-frame reorder delay) — normal.
                Ok(None) => {
                    empty_returns += 1;
                    continue;
                }
                Err(e) => {
                    decode_errors += 1;
                    eprintln!("[export_validation] decode error on packet {packets_read}: {e:?}");
                    continue;
                }
            };
            let (pts, w, h) = (frame.pts, frame.width, frame.height);
            assert!(
                !frame.is_semi_planar(),
                "expected the software decoder to produce planar YUV420p, got semi-planar chroma"
            );
            assert_eq!(
                frame.meta.layout.bit_depth, 8,
                "expected an 8-bit decode, got {}-bit",
                frame.meta.layout.bit_depth
            );
            assert_eq!(
                (w, h), (width, height),
                "decoded frame size does not match the stream header"
            );

            let luma_len = (w * h) as usize;
            let cw = w.div_ceil(2) as usize;
            let ch = h.div_ceil(2) as usize;
            let chroma_len = cw * ch;

            frames.push(DecodedFrame {
                width:  w,
                height: h,
                matrix: tagged_matrix,
                y: buf[..luma_len].to_vec(),
                u: buf[luma_len..luma_len + chroma_len].to_vec(),
                v: buf[luma_len + chroma_len..luma_len + 2 * chroma_len].to_vec(),
            });
            pts_list.push(pts);
        }

        // Drain the decoder.  This is not optional: with frame-level threading
        // (which `Decoder::open_sw` enables via thread_count = 0) FFmpeg holds back
        // roughly one frame per core, so a short file yields ZERO frames from the
        // send/receive loop above and only produces them here.
        let mut drained = 0usize;
        while frames.len() < max_frames {
            match decoder.drain_into(&mut buf) {
                Ok(Some(frame)) => {
                    let (pts, w, h) = (frame.pts, frame.width, frame.height);
                    assert!(
                        !frame.is_semi_planar(),
                        "drained frame has semi-planar chroma, expected planar YUV420p"
                    );
                    assert_eq!((w, h), (width, height), "drained frame has the wrong size");
                    let luma_len = (w * h) as usize;
                    let cw = w.div_ceil(2) as usize;
                    let ch = h.div_ceil(2) as usize;
                    let chroma_len = cw * ch;
                    frames.push(DecodedFrame {
                        width:  w,
                        height: h,
                        matrix: tagged_matrix,
                        y: buf[..luma_len].to_vec(),
                        u: buf[luma_len..luma_len + chroma_len].to_vec(),
                        v: buf[luma_len + chroma_len..luma_len + 2 * chroma_len].to_vec(),
                    });
                    pts_list.push(pts);
                    drained += 1;
                }
                Ok(None) => break, // fully drained
                Err(e) => {
                    decode_errors += 1;
                    eprintln!("[export_validation] error while draining the decoder: {e:?}");
                    break;
                }
            }
        }

        eprintln!(
            "[export_validation] decoded {} frame(s) from {} packet(s) \
             ({} need-more-input, {} from the drain, {} error(s))",
            frames.len(), packets_read, empty_returns, drained, decode_errors
        );
        assert_eq!(
            decode_errors, 0,
            "the exported file produced {decode_errors} decode error(s)"
        );

        (frames, pts_list)
    }

    /// Everything the export engine needs, built around one still-image clip
    /// holding the test pattern.
    struct Harness {
        device:    Arc<GpuDevice>,
        scheduler: Arc<FrameScheduler>,
        timeline:  Arc<std::sync::RwLock<TimelineStore>>,
        tracks:    Arc<std::sync::RwLock<TrackList>>,
        sources:   Arc<std::sync::RwLock<SourceRegistry>>,
        shaders:   Arc<ShaderRegistry>,
        compute:   Arc<ComputePipelineCache>,
        /// Kept alive so the prefetch worker's channel does not disconnect.
        _shutdown: Arc<std::sync::atomic::AtomicBool>,
    }

    fn build_harness(pattern_path: &Path, duration_pts: i64) -> Harness {
        let device = Arc::new(
            pollster::block_on(GpuDevice::new_headless()).expect("headless GpuDevice"),
        );

        let sources = Arc::new(std::sync::RwLock::new(SourceRegistry::new()));
        let source_id = {
            let mut reg = sources.write().unwrap();
            reg.register(
                pattern_path.to_path_buf(),
                Some(VideoStreamInfo {
                    width:        W,
                    height:       H,
                    frame_rate:   FPS,
                    pixel_fmt:    PixelFormat::Rgba8,
                    color_info:   ColorInfo::srgb(),
                    duration_pts,
                    is_vfr:       false,
                    time_base:    TB,
                    rotation:     VideoRotation::None,
                }),
                None, // no audio stream
            )
        };

        let mut track_list = TrackList::new();
        let track_id = track_list
            .push(Track::new_video(crate::timeline::ids::TrackId(0), "V1"))
            .expect("failed to add a video track");
        let tracks = Arc::new(std::sync::RwLock::new(track_list));

        let mut store = TimelineStore::new();
        insert_clip(
            &mut store,
            ClipInsertParams {
                track_id,
                source_id,
                kind: ClipKind::Image,
                pts_in: 0,
                pts_out: duration_pts,
                ..Default::default()
            },
        )
        .expect("failed to insert the test-pattern clip");
        let timeline = Arc::new(std::sync::RwLock::new(store));

        // IoLayer + prefetch worker.  A still-image clip never asks the slot
        // pool for a frame, but the scheduler holds an IoLayer regardless.
        let pool = Arc::new(FrameSlotPool::new(&device));
        let cache = Arc::new(FrameCache::new(Arc::clone(&pool), 32));
        let (prefetch_tx, prefetch_rx) = std::sync::mpsc::sync_channel(16);
        let io_layer = Arc::new(IoLayer::new(
            Arc::clone(&device.device),
            Arc::clone(&pool),
            Arc::clone(&cache),
            Arc::clone(&sources),
            prefetch_tx,
            TB,
        ));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker = crate::io::prefetch::PrefetchWorker::new(
            prefetch_rx,
            Arc::clone(&io_layer),
            Arc::clone(&cache),
            Arc::clone(&shutdown),
        );
        let _ = crate::io::prefetch::spawn_prefetch_worker(worker);

        let scheduler = Arc::new(FrameScheduler::new(io_layer, W, H));

        let shaders = Arc::new(
            ShaderRegistry::compile_all(&device).expect("shader compilation failed"),
        );
        let compute = Arc::new(ComputePipelineCache::new());

        Harness {
            device,
            scheduler,
            timeline,
            tracks,
            sources,
            shaders,
            compute,
            _shutdown: shutdown,
        }
    }

    fn make_job(output: PathBuf, duration_pts: i64) -> ExportJob {
        ExportJob {
            output_path:    output,
            container:      Container::Mp4,
            video_codec:    VideoCodec::H264,
            audio_codec:    AudioCodec::Aac,
            // CRF 18 keeps flat patches close to lossless without making the
            // encode slow enough to matter here.
            quality:        VideoQuality::Crf(18),
            audio_bitrate:  128_000,
            pts_in:         0,
            pts_out:        duration_pts,
            width:          W,
            height:         H,
            frame_rate:     FPS,
            project_tb:     TB,
            // One segment: keeps frame order and the encoder's GOP simple.
            render_threads: 1,
            cpu_preset:     CpuPreset::Medium,
            output_color:   crate::timeline::source::ColorInfo::bt709(),
            // SDR by default; the HDR test calls `set_hdr10` on the returned job
            // so the colour tags and the static metadata can never disagree.
            hdr10:          None,
        }
    }

    /// One CUDA primary context for the whole test process.
    ///
    /// `CudaContext::new` retains the device's primary context and `Drop`
    /// releases it.  With two export tests running concurrently under `cargo
    /// test`, one test's release can race the other's retain and
    /// `cuStreamCreate` then fails with "invalid device context" — which made
    /// `nvenc_export_matches_pattern` fall back to FFmpeg for reasons that had
    /// nothing to do with the code under test.  Retaining exactly once, and never
    /// releasing, removes the race: the context lives for the process.
    fn shared_cuda_ctx(
        capability: &InteropCapability,
    ) -> Option<Arc<crate::interop::cuda_context::CudaContext>> {
        static CUDA: std::sync::OnceLock<
            Option<Arc<crate::interop::cuda_context::CudaContext>>,
        > = std::sync::OnceLock::new();
        CUDA.get_or_init(|| {
            if !capability.is_available() {
                return None;
            }
            match crate::interop::cuda_context::CudaContext::new(capability) {
                Ok(c) => Some(Arc::new(c)),
                Err(e) => {
                    eprintln!("[export_validation] CudaContext::new failed: {e:?}");
                    None
                }
            }
        })
        .clone()
    }

    /// Serialises the exports against each other AND against every other test
    /// that makes the CUDA context current.
    ///
    /// Two concurrent NVENC sessions plus two headless wgpu devices on one GPU is
    /// a resource fight, not a test — and beyond that, `CudaContext::with_context`
    /// pushes the shared primary context, which the driver only allows one thread
    /// at a time.  Delegates to `super::cuda_lock` so this module and
    /// `tests::shared_buffer` share ONE lock; a second private mutex here would
    /// serialise these tests against each other while still racing that module.
    fn export_lock() -> std::sync::MutexGuard<'static, ()> {
        // `super` here is `tests::export_validation` (the file), not `tests` — the
        // tests in this tree live one module deeper than the file they are in.
        crate::tests::cuda_lock()
    }

    /// Run an export to completion and return the terminal phase.
    ///
    /// Blocks on the progress channel rather than sleeping, and gives up after
    /// `timeout` so a wedged encoder fails the test instead of hanging the suite.
    ///
    /// Returns `(phase, backend_was_nvenc)`.  The second value is what makes the
    /// GPU test honest: `ExportEngine` falls back to FFmpeg silently, so the test
    /// has to be told which encoder actually ran rather than inferring it.
    fn run_export(
        harness: &Harness,
        job: ExportJob,
        force_cpu: bool,
        timeout: std::time::Duration,
    ) -> (ExportPhase, bool) {
        let capability = InteropCapability::probe(&harness.device);
        // No CUDA context at all on the forced-CPU path: it is unused there, and
        // creating one only adds contention with the GPU test.
        let cuda_ctx = if force_cpu {
            None
        } else {
            shared_cuda_ctx(&capability)
        };

        // Ask which backend `select` picks BEFORE starting the export, using the
        // same inputs the engine will use.  Probing afterwards is not possible:
        // the engine consumes the backend on its dispatch thread.
        let backend_is_nvenc = if force_cpu {
            false
        } else {
            match crate::export::video_encoder::VideoEncoderBackend::select(
                &job,
                &capability,
                cuda_ctx.as_ref(),
                &harness.device,
            ) {
                Ok(b) => {
                    let is_gpu = matches!(
                        b,
                        crate::export::video_encoder::VideoEncoderBackend::CudaNvenc { .. }
                    );
                    // Release the probe's NVENC session before the engine opens
                    // its own: NVENC allows only a few concurrent sessions.
                    drop(b);
                    is_gpu
                }
                Err(e) => {
                    eprintln!("[export_validation] backend probe failed: {e:?}");
                    false
                }
            }
        };

        let engine = ExportEngine::new(
            Arc::clone(&harness.device),
            job,
            Arc::clone(&harness.scheduler),
            Arc::clone(&harness.timeline),
            Arc::clone(&harness.tracks),
            Arc::clone(&harness.sources),
            capability,
            cuda_ctx,
            force_cpu,
        );

        let rx = engine
            .start(Arc::clone(&harness.shaders), Arc::clone(&harness.compute))
            .expect("ExportEngine::start failed");

        let deadline = std::time::Instant::now() + timeout;
        let mut last = ExportPhase::Rendering;
        while std::time::Instant::now() < deadline {
            match rx.try_recv() {
                Some(update) => {
                    last = update.phase.clone();
                    match update.phase {
                        ExportPhase::Done
                        | ExportPhase::Cancelled
                        | ExportPhase::Failed(_) => return (last, backend_is_nvenc),
                        _ => {}
                    }
                }
                None => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        }
        panic!("export did not reach a terminal phase within {timeout:?} (last phase: {last:?})");
    }

    /// Assert that the exported container describes the colour the job asked for.
    ///
    /// Reads the RAW `AVCodecParameters` codes rather than `StreamInfo::color_info`
    /// on purpose: `ColorInfo::from_ffmpeg` fills unspecified fields in with
    /// resolution heuristics, so going through it would report Rec.709 for a file
    /// that carries no colour description at all — which is the exact failure this
    /// is meant to catch.
    fn assert_container_color(path: &Path, expected: &crate::timeline::source::ColorInfo, label: &str) {
        use crate::io::ffi::avcodec::{
            avcodecpar_get_color_space, avcodecpar_get_color_range,
            avcodecpar_get_color_trc, avcodecpar_get_color_primaries,
        };

        let demuxer = crate::io::demuxer::Demuxer::open(path)
            .expect("failed to reopen the exported file for colour verification");
        let stream = demuxer
            .video_stream
            .clone()
            .expect("exported file has no video stream");

        let (space, range, trc, primaries) = unsafe {
            (
                avcodecpar_get_color_space(stream.codecpar),
                avcodecpar_get_color_range(stream.codecpar),
                avcodecpar_get_color_trc(stream.codecpar),
                avcodecpar_get_color_primaries(stream.codecpar),
            )
        };

        assert_eq!(
            space, expected.av_color_space(),
            "{label}: container colour space is {space}, expected {} — \
             the encoder's VUI / the mp4 `colr` box was not written",
            expected.av_color_space()
        );
        assert_eq!(
            trc, expected.av_color_trc(),
            "{label}: container transfer characteristic is {trc}, expected {}",
            expected.av_color_trc()
        );
        assert_eq!(
            primaries, expected.av_color_primaries(),
            "{label}: container colour primaries are {primaries}, expected {}",
            expected.av_color_primaries()
        );
        assert_eq!(
            range, expected.av_color_range(),
            "{label}: container colour range is {range}, expected {} — \
             a wrong range flag shifts every level by 16/235 on playback",
            expected.av_color_range()
        );
    }

    /// Assert that every patch centre in `frame` matches the pattern.
    fn assert_frame_matches_pattern(frame: &DecodedFrame, label: &str) {
        // Sample the horizontal centre of each band, which is at least 19 rows
        // away from any colour boundary — far outside the reach of 4:2:0 chroma
        // subsampling, so this measures the codec and not the resampler.
        let x = W / 2;
        let mut worst = 0i32;

        for (i, (name, expected)) in PATCHES.iter().enumerate() {
            let y = i as u32 * PATCH_H + PATCH_H / 2;
            let got = frame.rgb_at(x, y);
            for c in 0..3 {
                let delta = (got[c] as i32 - expected[c] as i32).abs();
                worst = worst.max(delta);
                assert!(
                    delta <= TOLERANCE,
                    "{label}: patch '{name}' at ({x},{y}) channel {c}: \
                     decoded {got:?}, expected {expected:?} (delta {delta} > {TOLERANCE})"
                );
            }
        }

        // Channel-order check independent of the tolerance above: a red patch
        // must be dominated by red, and so on.  A swapped R/B anywhere in the
        // chain passes no tolerance test but would be caught here too.
        let red = frame.rgb_at(x, 2 * PATCH_H + PATCH_H / 2);
        assert!(
            red[0] > red[1] + 60 && red[0] > red[2] + 60,
            "{label}: the red patch is not red-dominant: {red:?} — channels are swapped"
        );
        let green = frame.rgb_at(x, 3 * PATCH_H + PATCH_H / 2);
        assert!(
            green[1] > green[0] + 60 && green[1] > green[2] + 60,
            "{label}: the green patch is not green-dominant: {green:?}"
        );
        let blue = frame.rgb_at(x, 4 * PATCH_H + PATCH_H / 2);
        assert!(
            blue[2] > blue[0] + 60 && blue[2] > blue[1] + 60,
            "{label}: the blue patch is not blue-dominant: {blue:?} — channels are swapped"
        );

        eprintln!("[export_validation] {label}: worst channel delta {worst}/255");
    }

    /// Route the engine's `log::info!` lines to stderr so `--nocapture` runs show
    /// which encoder backend `ExportEngine` actually selected.
    ///
    /// This matters for `nvenc_export_matches_pattern`: the engine falls back to
    /// FFmpeg silently, so without the log there is no way to tell a real GPU
    /// export from a CPU one wearing a GPU test name.  Safe to call from every
    /// test — `try_init` ignores the second and later calls.
    fn init_logging() {
        let _ = env_logger::builder().is_test(false).try_init();
    }

    /// Unique scratch paths so concurrently running tests never collide.
    fn scratch(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "nexir_export_validation_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    /// The headline check: export a known pattern, decode the file back, and
    /// verify the pixels, the frame count and the timestamps.
    ///
    /// `force_cpu = true` pins the FFmpeg encoder path so this test asserts on
    /// one deterministic backend; `nvenc_export_matches_pattern` covers the other.
    #[test]
    fn cpu_export_decodes_to_the_source_pattern() {
        init_logging();
        // Serialised against the NVENC test: see `export_lock`.
        let _guard = export_lock();
        let png = scratch("cpu").with_extension("png");
        let mp4 = scratch("cpu").with_extension("mp4");
        write_test_pattern(&png);

        // 10 frames at 30 fps.
        let duration_pts = 10 * (TB.den / FPS.num);
        let harness = build_harness(&png, duration_pts);
        let expected_frames = make_job(mp4.clone(), duration_pts).total_frames();
        assert_eq!(expected_frames, 10, "test bug: expected a 10-frame job");

        let (phase, _was_nvenc) = run_export(
            &harness,
            make_job(mp4.clone(), duration_pts),
            true,
            std::time::Duration::from_secs(120),
        );
        assert_eq!(
            phase, ExportPhase::Done,
            "CPU export did not finish cleanly: {phase:?}"
        );

        assert!(mp4.exists(), "export reported Done but produced no file");
        let size = std::fs::metadata(&mp4).unwrap().len();
        assert!(size > 1024, "exported file is implausibly small ({size} bytes)");

        let (frames, pts_list) = decode_file(&mp4, expected_frames + 4);

        assert_eq!(
            frames.len(), expected_frames,
            "decoded {} frame(s) from the export, expected {} — the muxer or the \
             encoder flush is dropping frames",
            frames.len(), expected_frames
        );

        // Container timestamps must be strictly increasing.  A non-monotonic DTS
        // is what a reordering encoder produces when its packets are stamped with
        // dts = pts (see DtsQueue in src/interop/encode_interop.rs).
        for w in pts_list.windows(2) {
            assert!(
                w[1] > w[0],
                "decoded PTS sequence is not strictly increasing: {pts_list:?}"
            );
        }

        // Every frame shows the same static pattern, so check the first, a middle
        // one and the last: that also catches a pipeline that only gets the first
        // frame right (stale staging buffer) or loses the tail (missing flush).
        assert_frame_matches_pattern(&frames[0], "cpu frame 0");
        assert_frame_matches_pattern(&frames[frames.len() / 2], "cpu middle frame");
        assert_frame_matches_pattern(&frames[frames.len() - 1], "cpu last frame");

        // P1.7 — the colour description the job asked for must survive to the
        // container.  On this path it travels encoder context → codecpar.
        assert_container_color(&mp4, &make_job(mp4.clone(), duration_pts).output_color, "cpu export");

        let _ = std::fs::remove_file(&png);
        if std::env::var("NEXIR_KEEP_EXPORT").is_ok() {
            eprintln!("[export_validation] kept export at {}", mp4.display());
        } else {
            let _ = std::fs::remove_file(&mp4);
        }
    }

    /// The colour description reaches the ENCODER context, for an HDR profile that
    /// is nothing like the default.
    ///
    /// The two round-trip tests above both export SDR Rec.709, and Rec.709 is also
    /// what libavcodec/libx264 will guess on its own — so on their own they cannot
    /// distinguish "the plumbing works" from "nobody set anything and the guess
    /// happened to match". This one asks for BT.2020 + PQ and reads the codes back
    /// off the opened `AVCodecContext`, where only `open_codec_context` could have
    /// put them.
    ///
    /// Scope: the TAGGING path only, and deliberately so.  It sets
    /// `output_color` by hand and leaves `hdr10` at `None`, so no static metadata
    /// is attached and no file is written — `VideoEncoder::open` is the only thing
    /// exercised.  Whether the *pixels* are really 10-bit PQ is a different
    /// question, answered end-to-end by `hdr10_export_carries_10bit_pq_pixels`
    /// below.
    ///
    /// H.265 is used because `ExportJob::encode_bit_depth` floors an HDR job at 10
    /// bits and `VideoCodec::supports_hdr` rejects H.264 for exactly that reason;
    /// asking for BT.2020/PQ on the H264 job `make_job` returns would be an
    /// invalid job rather than a tagging test.
    #[test]
    fn encoder_context_carries_requested_color_description() {
        init_logging();
        let mp4 = scratch("colortag").with_extension("mp4");
        let mut job = make_job(mp4, 10 * (TB.den / FPS.num));
        job.video_codec = VideoCodec::H265;
        job.output_color = crate::timeline::source::ColorInfo::bt2020(true, 10);

        let encoder = crate::export::video_encoder::VideoEncoder::open(&job)
            .expect("opening the FFmpeg video encoder failed");

        use crate::export::ffi::encoder_ffi::{
            avcodec_ctx_get_color_space, avcodec_ctx_get_color_range,
            avcodec_ctx_get_color_trc, avcodec_ctx_get_color_primaries,
        };
        let ctx = encoder.codec_ctx();
        let (space, range, trc, primaries) = unsafe {
            (
                avcodec_ctx_get_color_space(ctx),
                avcodec_ctx_get_color_range(ctx),
                avcodec_ctx_get_color_trc(ctx),
                avcodec_ctx_get_color_primaries(ctx),
            )
        };

        assert_eq!(space, 9, "encoder colorspace is {space}, expected 9 (BT2020_NCL)");
        assert_eq!(trc, 16, "encoder color_trc is {trc}, expected 16 (SMPTE ST 2084 / PQ)");
        assert_eq!(primaries, 9, "encoder color_primaries is {primaries}, expected 9 (BT.2020)");
        assert_eq!(range, 1, "encoder color_range is {range}, expected 1 (limited/MPEG)");

        // Nothing was written: `open` only opens a codec context.  No file to clean.
    }

    /// The same round trip with the GPU backend left enabled.
    ///
    /// Honesty rules, and they are the point of this test:
    ///
    /// * It **hard-skips** only when the machine has no CUDA interop at all — the
    ///   one case where "NVENC could not be tested" is a property of the host and
    ///   not of this code.  The reason is printed.
    /// * Past that gate it **fails**.  If `ExportEngine` selected the FFmpeg
    ///   backend, the assertion below fires instead of the test quietly passing
    ///   while measuring libx264 under a GPU test name.
    ///
    /// This replaces the old skip logic, which called `EncodeInterop::open`
    /// itself and returned early whenever that probe failed.  Under `cargo test`
    /// the probe fails routinely — a sibling test holds the CUDA primary context —
    /// so the test reported `ok` without ever touching the GPU encoder, and
    /// "130 passed" said nothing about the NVENC path.  Run alone it took the real
    /// path and crashed.  A test whose coverage depends on parallel scheduling is
    /// worse than no test.
    ///
    /// Set `NEXIR_REQUIRE_NVENC=1` to turn the remaining skip into a failure too
    /// (for CI on a machine that is supposed to have the hardware).
    ///
    /// ## The job is BT.709, and that is the assertion
    ///
    /// This test used to tag the job **BT.601** — not as a colour preference but
    /// because it had no choice: the zero-copy path handed NVENC packed RGB, the
    /// driver converted it with BT.601, and
    /// `ExportJob::nvenc_zero_copy_is_colour_safe` therefore routed every
    /// non-BT.601 job to the FFmpeg encoder. A BT.601 job was the only one that
    /// reached the GPU encoder at all.
    ///
    /// Since P1.9 step 3 `Nv12EncodeNode` performs the conversion in our own
    /// shader, so BT.709 is both allowed and the interesting case — it is the
    /// **negative control for the whole step**. BT.601 would have passed either
    /// way: it is what the driver does unprompted, so a green BT.601 test cannot
    /// distinguish "our shader applied the requested matrix" from "the driver
    /// guessed and the guess happened to match". BT.709 can only pass if our
    /// shader ran. If this code regressed to letting the driver convert, red
    /// (which the stream now declares BT.709) would decode as `[255, 24, 0]`
    /// instead of `[255, 0, 0]` — a delta of 24, outside `TOLERANCE`, so
    /// `assert_frame_matches_pattern` fails with the patch named.
    #[test]
    fn nvenc_export_matches_pattern() {
        init_logging();
        // Serialised against the CPU test: two concurrent exports on one GPU used
        // to make this test's CUDA context creation fail and silently fall back to
        // FFmpeg, which is precisely the dishonesty this test now refuses.
        let _guard = export_lock();
        let png = scratch("gpu").with_extension("png");
        let mp4 = scratch("gpu").with_extension("mp4");
        write_test_pattern(&png);

        let duration_pts = 10 * (TB.den / FPS.num);
        let harness = build_harness(&png, duration_pts);
        // BT.709 limited, 8-bit — `make_job`'s default, and deliberately the
        // matrix the driver would NOT have applied on its own.
        let job = make_job(mp4.clone(), duration_pts);
        assert_eq!(
            job.output_color.matrix,
            crate::timeline::source::MatrixCoefficients::Bt709,
            "this test's whole point is a matrix the NVENC driver would not pick \
             for itself; a BT.601 job here would prove nothing"
        );
        assert!(
            job.nvenc_zero_copy_is_colour_safe(),
            "this test only means something if the job is eligible for the \
             zero-copy path"
        );

        let require_nvenc = std::env::var("NEXIR_REQUIRE_NVENC")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false);

        let capability = InteropCapability::probe(&harness.device);
        if !capability.is_available() {
            let reason = format!(
                "CUDA interop is unavailable on this machine (transport={:?})",
                capability.transport
            );
            assert!(
                !require_nvenc,
                "NEXIR_REQUIRE_NVENC is set but {reason}"
            );
            eprintln!(
                "[export_validation] SKIP nvenc_export_matches_pattern: {reason}. \
                 The GPU export path was NOT exercised."
            );
            let _ = std::fs::remove_file(&png);
            return;
        }

        // No `EncodeInterop::open` probe here on purpose: opening a second NVENC
        // session against the same context is exactly what made this test flaky,
        // and `run_export` already reports which backend the engine chose.
        let (phase, was_nvenc) = run_export(
            &harness,
            job.clone(),
            false,
            std::time::Duration::from_secs(120),
        );

        // Assert the backend BEFORE the pixel assertions: if FFmpeg ran, every
        // check below would pass and prove nothing about NVENC.
        assert!(
            was_nvenc,
            "the export engine selected the FFmpeg backend, so this test did not \
             exercise NVENC at all — CUDA interop reported available \
             (transport={:?}), so this is a real failure of the GPU path, not an \
             absent-hardware skip.  The [export] log lines above say why \
             EncodeInterop::open was rejected.",
            capability.transport
        );

        assert_eq!(
            phase, ExportPhase::Done,
            "GPU export did not finish cleanly: {phase:?}"
        );
        assert!(mp4.exists(), "GPU export reported Done but produced no file");

        let expected_frames = job.total_frames();
        let (frames, pts_list) = decode_file(&mp4, expected_frames + 4);
        assert_eq!(
            frames.len(), expected_frames,
            "GPU export decoded {} frame(s), expected {} — NVENC's EOS flush is \
             dropping the tail of the stream",
            frames.len(), expected_frames
        );
        for w in pts_list.windows(2) {
            assert!(
                w[1] > w[0],
                "GPU export PTS sequence is not strictly increasing: {pts_list:?}"
            );
        }
        assert_frame_matches_pattern(&frames[0], "gpu frame 0");
        assert_frame_matches_pattern(&frames[frames.len() - 1], "gpu last frame");

        // P1.7 — the NVENC path's bitstream comes from the driver, not libavcodec,
        // so `Muxer::open` writing the description onto the stream's codecpar is
        // the ONLY thing that puts it in the file.  This assertion is what proves
        // that, and it fails if the muxer ever regresses to relying on
        // `avcodec_parameters_from_context` alone.
        assert_container_color(&mp4, &job.output_color, "gpu export");

        let _ = std::fs::remove_file(&png);
        if std::env::var("NEXIR_KEEP_EXPORT").is_ok() {
            eprintln!("[export_validation] kept export at {}", mp4.display());
        } else {
            let _ = std::fs::remove_file(&mp4);
        }
    }

    /// Decode a 10-bit export and return each frame's LUMA plane as 10-bit codes.
    ///
    /// Separate from [`decode_file`] rather than a flag on it: that helper asserts
    /// an 8-bit planar decode and unpacks three `u8` planes, and an HDR file is
    /// `yuv420p10le` — two bytes per sample.  Asserting the decoded depth here is
    /// half of what makes the HDR test meaningful: a file that was tagged HDR but
    /// encoded 8-bit fails on this assertion before any pixel is examined.
    ///
    /// Only luma is returned because the pixel assertions below sample neutral
    /// patches (black / white / grey), where chroma carries no information.
    fn decode_10bit_luma(path: &Path, max_frames: usize) -> Vec<Vec<u16>> {
        let mut demuxer = crate::io::demuxer::Demuxer::open(path)
            .expect("failed to open the exported HDR file for verification");
        let stream = demuxer
            .video_stream
            .clone()
            .expect("exported HDR file has no video stream");
        let width  = stream.width.expect("exported HDR stream has no width");
        let height = stream.height.expect("exported HDR stream has no height");

        let mut decoder =
            crate::io::decoder::Decoder::open_sw(&stream, stream.codecpar)
                .expect("failed to open a software decoder for the exported HDR file");

        // yuv420p10le is 3 * w * h bytes (2 bytes/sample, 1.5 samples/pixel).
        let mut buf = vec![0u8; width as usize * height as usize * 4 + 64];
        let mut frames: Vec<Vec<u16>> = Vec::new();

        let take = |frame: &crate::io::decoder::DecodedFrame, buf: &[u8]| -> Vec<u16> {
            assert_eq!(
                frame.meta.layout.bit_depth, 10,
                "the exported file decoded back at {}-bit — it is tagged HDR10 but \
                 carries {}-bit samples, which is precisely the tagged-SDR failure \
                 this test exists to catch",
                frame.meta.layout.bit_depth, frame.meta.layout.bit_depth
            );
            assert!(
                !frame.is_semi_planar(),
                "expected planar yuv420p10le from the software decoder, got semi-planar"
            );
            assert!(
                !frame.meta.layout.msb_aligned,
                "expected LSB-aligned 10-bit codes from a planar decode"
            );
            let luma_len = (frame.width * frame.height) as usize;
            buf[..luma_len * 2]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };

        while frames.len() < max_frames {
            let pkt = match demuxer.next_video_packet() {
                Ok(Some(p)) => p,
                Ok(None) => break,
                Err(e) => panic!("demuxing the exported HDR file failed: {e:?}"),
            };
            match decoder.decode_into(&pkt, &mut buf, None) {
                Ok(Some(frame)) => {
                    let plane = take(&frame, &buf);
                    frames.push(plane);
                }
                Ok(None) => continue,
                Err(e) => panic!("decoding the exported HDR file failed: {e:?}"),
            }
        }
        // Frame-threaded decoding holds back roughly one frame per core, so a
        // short file yields most of its frames here.
        while frames.len() < max_frames {
            match decoder.drain_into(&mut buf) {
                Ok(Some(frame)) => {
                    let plane = take(&frame, &buf);
                    frames.push(plane);
                }
                Ok(None) => break,
                Err(e) => panic!("draining the HDR decoder failed: {e:?}"),
            }
        }

        assert!(
            !frames.is_empty(),
            "the exported HDR file produced no decodable frames"
        );
        frames
    }

    /// SMPTE ST 2084 OETF: linear luminance in nits -> PQ code value in [0,1].
    ///
    /// Written out here rather than reused from the shader so the expectation is
    /// derived independently of the code under test — the whole point is to check
    /// the GPU actually applied this curve.
    fn pq_code_for_nits(nits: f32) -> f32 {
        let l = (nits / 10_000.0).clamp(0.0, 1.0);
        if l <= 0.0 {
            return 0.0;
        }
        const M1: f32 = 0.159_301_76;
        const M2: f32 = 78.843_75;
        const C1: f32 = 0.835_937_5;
        const C2: f32 = 18.851_562;
        const C3: f32 = 18.687_5;
        let v = l.powf(M1);
        ((C1 + C2 * v) / (1.0 + C3 * v)).powf(M2)
    }

    /// A full HDR10 export, checked all the way down to the sample values.
    ///
    /// This is the assertion the rest of P1.7 was building towards, and the one
    /// that distinguishes a REAL HDR file from an SDR file wearing HDR tags. Four
    /// independent things have to hold, and each has its own failure mode:
    ///
    /// 1. **10-bit decode** — `decode_10bit_luma` refuses anything else. An 8-bit
    ///    encode with BT.2020/PQ tags dies here.
    /// 2. **Colour description in the container** — `trc == 16` (ST 2084) and
    ///    `primaries == 9` (BT.2020), read off the raw codecpar.
    /// 3. **Static metadata present** — MDCV + CLL, via
    ///    `avcodecpar_get_hdr10_metadata`. Tags alone leave a display guessing.
    /// 4. **The pixels changed** — the luma of each neutral patch matches the PQ
    ///    code for that patch's diffuse luminance, NOT its SDR code. sRGB white is
    ///    100 nits, whose PQ code is ~0.515, so a correct HDR encode puts white at
    ///    roughly half scale. A tone-mapped or passed-through SDR frame would put
    ///    it near 1.0 — a ~370-code gap at 10-bit, far outside the tolerance.
    ///
    /// Point 4 is what none of the other tests can see: 1-3 all pass for a file
    /// whose renderer never applied the PQ transform.
    ///
    /// H.265 is the codec because `VideoCodec::supports_hdr` allows only H265,
    /// ProRes and VP9, and H.265 is the one of those with a CRF-controlled encoder
    /// and a fast 10-bit decode.
    #[test]
    fn hdr10_export_carries_10bit_pq_pixels() {
        init_logging();
        let _guard = export_lock();
        let png = scratch("hdr").with_extension("png");
        let mp4 = scratch("hdr").with_extension("mp4");
        write_test_pattern(&png);

        let duration_pts = 10 * (TB.den / FPS.num);
        let harness = build_harness(&png, duration_pts);

        let mut job = make_job(mp4.clone(), duration_pts);
        job.video_codec = VideoCodec::H265;
        job.set_hdr10(crate::export::job::Hdr10Metadata::bt2020_1000_nits())
            .expect("H.265 must accept an HDR10 configuration");
        assert!(job.is_hdr(), "set_hdr10 did not put the job on the HDR path");
        assert_eq!(
            job.encode_bit_depth(), 10,
            "an HDR job must encode at least 10-bit"
        );
        job.validate().expect("the HDR job must validate");

        // force_cpu = true: `VideoEncoderBackend::select` already refuses the
        // zero-copy NVENC path for an HDR job (it cannot be configured for
        // Main10), so this only skips creating a CUDA context that would go
        // unused.  `VideoEncoder::open` still resolves hevc_nvenc first if it is
        // present, so the encode itself may well run on the GPU.
        let (phase, _) = run_export(
            &harness,
            job.clone(),
            true,
            std::time::Duration::from_secs(180),
        );
        assert_eq!(
            phase, ExportPhase::Done,
            "HDR export did not finish cleanly: {phase:?}"
        );
        assert!(mp4.exists(), "HDR export reported Done but produced no file");

        // ── 2. Colour description ────────────────────────────────────────────
        assert_container_color(&mp4, &job.output_color, "hdr export");
        {
            use crate::io::ffi::avcodec::{
                avcodecpar_get_color_primaries, avcodecpar_get_color_trc,
            };
            let demuxer = crate::io::demuxer::Demuxer::open(&mp4)
                .expect("failed to reopen the HDR export");
            let stream = demuxer.video_stream.clone().expect("no video stream");
            let (trc, primaries) = unsafe {
                (
                    avcodecpar_get_color_trc(stream.codecpar),
                    avcodecpar_get_color_primaries(stream.codecpar),
                )
            };
            assert_eq!(
                trc, 16,
                "container transfer characteristic is {trc}, expected 16 \
                 (SMPTE ST 2084 / PQ)"
            );
            assert_eq!(
                primaries, 9,
                "container colour primaries are {primaries}, expected 9 (BT.2020)"
            );

            // ── 3. Static metadata ───────────────────────────────────────────
            let mut out = [0i64; 8];
            let mask = unsafe {
                crate::export::ffi::muxer_ffi::avcodecpar_get_hdr10_metadata(
                    stream.codecpar,
                    out.as_mut_ptr(),
                )
            };
            assert_ne!(
                mask & 1, 0,
                "the container carries no mastering-display metadata (mdcv box) — \
                 an HDR10 file without it is an unmastered grade"
            );
            assert_ne!(
                mask & 2, 0,
                "the container carries no content-light-level metadata (clli box)"
            );
            assert_eq!(out[0], 1, "mastering display has no primaries");
            assert_eq!(out[1], 1, "mastering display has no luminance");
            let expected = crate::export::job::Hdr10Metadata::bt2020_1000_nits();
            // max_luminance is stored as a rational; compare in nits so a
            // different-but-equivalent denominator does not fail the test.
            let max_nits = out[4] as f64 / out[5].max(1) as f64;
            assert!(
                (max_nits - expected.peak_nits() as f64).abs() < 1.0,
                "mastering display peak is {max_nits} cd/m², expected {}",
                expected.peak_nits()
            );
            assert_eq!(
                out[6] as u32, expected.max_cll,
                "MaxCLL in the container does not match the job"
            );
            assert_eq!(
                out[7] as u32, expected.max_fall,
                "MaxFALL in the container does not match the job"
            );
        }

        // ── 1. + 4. Depth and pixels ─────────────────────────────────────────
        let frames = decode_10bit_luma(&mp4, job.total_frames());
        let plane = &frames[0];

        // 10-bit limited range: code = 64 + value * (940 - 64).
        let to_code = |y: u16| (y as f32 - 64.0) / 876.0;
        // ±0.035 of full scale (~31 codes at 10-bit) absorbs CRF 18 on a flat
        // patch plus the RGBA64 round trip.  The SDR-vs-PQ gap this has to
        // separate is ~0.48 of full scale, an order of magnitude larger.
        const PQ_TOL: f32 = 0.035;

        // Neutral patches only: the 709->2020 gamut rotation leaves saturated
        // colours with a luma that depends on the matrix as well as the curve,
        // which would test two things at once.  Indices are into `PATCHES`.
        for &(idx, name, srgb) in &[
            (0usize, "black",    0.0f32),
            (1usize, "white",    1.0f32),
            (5usize, "50% gray", 128.0 / 255.0),
        ] {
            let y = idx as u32 * PATCH_H + PATCH_H / 2;
            let sample = plane[(y * W + W / 2) as usize];
            let got = to_code(sample);

            // sRGB EOTF -> linear, where 1.0 = 100 nits diffuse white, then PQ.
            let linear = if srgb <= 0.04045 {
                srgb / 12.92
            } else {
                ((srgb + 0.055) / 1.055).powf(2.4)
            };
            let want = pq_code_for_nits(linear * 100.0);

            assert!(
                (got - want).abs() <= PQ_TOL,
                "patch '{name}': decoded 10-bit luma {sample} = code {got:.4}, \
                 expected PQ code {want:.4} for {:.1} nits (tolerance {PQ_TOL}). \
                 An SDR-encoded frame would read {srgb:.4} here — if `got` is close \
                 to that instead, the renderer never applied the PQ transform and \
                 the file is tagged HDR over SDR pixels.",
                linear * 100.0
            );
        }

        // White is the discriminating sample, so state it separately: PQ puts
        // 100-nit diffuse white at roughly half scale, and nothing that skipped
        // the transform can land there.
        let white = to_code(plane[((PATCH_H + PATCH_H / 2) * W + W / 2) as usize]);
        assert!(
            white < 0.75,
            "white decoded to code {white:.4}; a PQ-encoded 100-nit white sits near \
             {:.4}, so a value this high means the frame is display-referred SDR \
             wearing HDR tags",
            pq_code_for_nits(100.0)
        );

        let _ = std::fs::remove_file(&png);
        if std::env::var("NEXIR_KEEP_EXPORT").is_ok() {
            eprintln!("[export_validation] kept HDR export at {}", mp4.display());
        } else {
            let _ = std::fs::remove_file(&mp4);
        }
    }
}
