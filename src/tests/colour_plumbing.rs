// src/tests/colour_plumbing.rs
// P1.6 — does a decoded frame's colour metadata actually REACH the shader?
//
// `colour::yuv`'s unit tests prove the maths: given a `ColorInfo`, the derived
// matrix and range constants are right.  `tests::nv12_encode` proves the encode
// shader reads its push constants.  Neither proves the PLUMBING — that the
// metadata FFmpeg reported for a real decoded frame is the metadata
// `YuvToRgbNode` converts with.  Every link in that chain was previously
// untested:
//
//     AVFrame colour fields
//       → Decoder::read_frame_color              (src/io/decoder.rs)
//       → DecodedFrame::meta                     (src/timeline/source.rs)
//       → FrameCache slot metadata               (src/io/frame_cache.rs)
//       → ClipRenderEntry::frame_meta            (src/render/frame_state.rs)
//       → YuvToRgbNode::new_with_layout          (src/render/nodes/yuv_to_rgb.rs)
//       → YuvConversion push constants           (src/colour/yuv.rs)
//
// A break anywhere in it is invisible in the common case, because BT.709 limited
// range is both the overwhelmingly common input AND every fallback in the chain:
// `ColorInfo::default()`, `ColorInfo::from_ffmpeg`'s heuristic for an HD frame,
// and `luma_coefficients`' arm for `Unknown`.  So the tests here deliberately use
// a **BT.601** source: it is the one common tagging that no default lands on, and
// reading it as BT.709 is a ~20-level error on saturated colour — visible, but
// small enough to be mistaken for codec loss.
//
// SCOPE, and what each test is worth:
//
//  1. `decoded_frame_metadata_reaches_the_node` — the wiring claim itself, with
//     no GPU work: decode a real file, hand the frame's own metadata to the node
//     exactly as `ExportRenderer::ensure_graph` does, and read back the matrix
//     the node resolved.
//  2. `node_decodes_the_pattern_using_the_signalled_matrix` — the same path with
//     the shader actually running, checked against the pixels the source was
//     built from, plus a mis-tagged control proving the check has teeth.
//  3. `the_two_matrices_are_not_interchangeable` — pins that the source files
//     really do differ, so test 2 cannot pass by both matrices agreeing.
//
// The source files are written by THIS CRATE's own encoder (`VideoEncoder` +
// `Muxer`), not by an external ffmpeg binary: the tag and the samples then come
// from the code under test's own colour handling, there is no CLI dependency, and
// `VideoEncoder::open` pins swscale to `job.sws_colorspace()` so the samples
// genuinely carry the matrix the file is tagged with.
//
// Requires a working GPU (headless wgpu device) and the FFmpeg shared libraries,
// like the rest of src/tests/.  Requires no CUDA and no NVENC session.

#[cfg(test)]
mod colour_plumbing {
    use crate::colour::yuv::YuvConversion;
    use crate::export::job::{
        AudioCodec, Container, CpuPreset, ExportJob, VideoCodec, VideoQuality,
    };
    use crate::export::renderer::RawFrame;
    use crate::io::ffi::avutil::AVRational;
    use crate::render::compute::ComputePipelineCache;
    use crate::render::context::RenderContext;
    use crate::render::device::GpuDevice;
    use crate::render::frame_state::FrameState;
    use crate::render::graph::{RenderGraphCompiler, RenderNode};
    use crate::render::nodes::yuv_to_rgb::YuvToRgbNode;
    use crate::render::nodes::yuv_upload::YuvUploadNode;
    use crate::render::resource::{ResourceBuilder, ResourceId, TextureAccess};
    use crate::render::shader::registry::ShaderRegistry;
    use crate::timeline::rational::Rational;
    use crate::timeline::source::{
        ColorInfo, ColorRange, DecodedFrameMeta, MatrixCoefficients,
    };
    use half::f16;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const TB: Rational = Rational { num: 1, den: 90_000 };
    const FPS: Rational = Rational { num: 30, den: 1 };

    /// 320x240: small enough to encode in well under a second, both dimensions
    /// even as 4:2:0 requires.
    ///
    /// The size is NOT load-bearing for the matrix here, and that is the point:
    /// both source files carry an explicit `colorspace` tag, so
    /// `ColorInfo::effective_matrix`'s resolution heuristic (< 1280x720 → BT.601)
    /// never runs.  A test that relied on the heuristic would pass even if the
    /// file's own tag never reached the node, which is exactly the bug being
    /// checked for.
    const W: u32 = 320;
    const H: u32 = 240;

    /// Six flat bands, top first.  Saturated primaries are what make a matrix
    /// error visible: black, white and grey decode identically under BT.601 and
    /// BT.709, so a pattern of greys could not detect this bug at all.
    const PATCHES: [(&str, [u8; 3]); 6] = [
        ("black",    [0, 0, 0]),
        ("white",    [255, 255, 255]),
        ("red",      [255, 0, 0]),
        ("green",    [0, 255, 0]),
        ("blue",     [0, 0, 255]),
        ("50% gray", [128, 128, 128]),
    ];

    fn patch_height() -> u32 {
        H / PATCHES.len() as u32
    }

    /// Expected RGB at output row `y`.
    fn expected_rgb_at_row(y: u32) -> [u8; 3] {
        let idx = ((y / patch_height()) as usize).min(PATCHES.len() - 1);
        PATCHES[idx].1
    }

    /// Per-channel tolerance, in 8-bit levels, for the whole round trip
    /// PNG-equivalent RGB → RGBA16F → swscale → H.264 → decode → GPU YUV→RGB.
    ///
    /// The lossy steps are limited-range quantisation (~±1) and the encoder at
    /// CRF 1 (near-lossless on flat bands).  4:2:0 subsampling does not
    /// contribute: every sample is taken at a band's centre, twenty rows from any
    /// colour boundary.
    ///
    /// 8 is far tighter than the ~20-level error a BT.601/BT.709 mix-up produces
    /// on red and green, which is the whole reason for the number: the assertion
    /// has to fail when the matrix is wrong.  `the_two_matrices_are_not_interchangeable`
    /// measures that error rather than trusting this comment.
    const TOLERANCE: i32 = 8;

    /// The test pattern as the RGBA16Float bytes the export renderer would hand
    /// the encoder: four f16 channels per pixel, values in `[0, 1]`.
    fn pattern_rgba16f() -> Vec<u8> {
        let mut out = Vec::with_capacity((W * H) as usize * 8);
        for y in 0..H {
            let [r, g, b] = expected_rgb_at_row(y);
            for _ in 0..W {
                for c in [r, g, b] {
                    out.extend_from_slice(&f16::from_f32(c as f32 / 255.0).to_le_bytes());
                }
                out.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
            }
        }
        out
    }

    /// A job describing a source file tagged with `matrix`.
    ///
    /// CRF 1 rather than the 18 `export_validation` uses: this file is a test
    /// FIXTURE, and its own codec loss is noise added to what the tests below
    /// measure, so it is worth the extra bytes to keep it near-lossless.
    fn source_job(path: PathBuf, matrix: MatrixCoefficients, frames: i64) -> ExportJob {
        let mut color = ColorInfo::bt709();
        color.matrix = matrix;
        color.range = ColorRange::Limited;

        ExportJob {
            output_path:    path,
            container:      Container::Mp4,
            video_codec:    VideoCodec::H264,
            audio_codec:    AudioCodec::Aac,
            quality:        VideoQuality::Crf(1),
            audio_bitrate:  128_000,
            pts_in:         0,
            pts_out:        frames * (TB.den / FPS.num),
            width:          W,
            height:         H,
            frame_rate:     FPS,
            project_tb:     TB,
            render_threads: 1,
            cpu_preset:     CpuPreset::Medium,
            output_color:   color,
            hdr10:          None,
        }
    }

    /// Write a short H.264 file of the test pattern, tagged AND converted with
    /// `matrix`.
    ///
    /// Uses the crate's own `VideoEncoder`/`Muxer`, which is what makes the
    /// fixture trustworthy: `VideoEncoder::open` calls
    /// `sws_setColorspaceDetails(job.sws_colorspace())`, so the YUV samples in the
    /// file are converted with the same matrix the VUI is tagged with.  A file
    /// whose tag and samples disagreed would make every assertion below
    /// meaningless.
    ///
    /// An audio stream is opened and left empty, exactly as an export with no
    /// audio clips does — `Muxer::open` requires an audio encoder.
    fn write_source_video(path: &Path, matrix: MatrixCoefficients, frames: i64) {
        use crate::export::audio_encoder::AudioMuxEncoder;
        use crate::export::muxer::Muxer;
        use crate::export::video_encoder::{VideoEncoder, VideoEncoderBackend};

        let job = source_job(path.to_path_buf(), matrix, frames);
        let mut backend = VideoEncoderBackend::FfmpegEncoder(
            VideoEncoder::open(&job).expect("failed to open the fixture video encoder"),
        );
        let audio = AudioMuxEncoder::open(&job).expect("failed to open the fixture audio encoder");

        let video_tb = AVRational { num: FPS.den as i32, den: FPS.num as i32 };
        let audio_tb = AVRational { num: 1, den: 48_000 };
        let muxer = Muxer::open(&job, &backend, &audio, video_tb, audio_tb)
            .expect("failed to open the fixture muxer");

        let data = pattern_rgba16f();
        for frame_index in 0..frames as usize {
            let raw = RawFrame {
                frame_index,
                pts: frame_index as i64,
                data: data.clone(),
            };
            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                muxer
                    .write_packet(pkt, true)
                    .expect("fixture muxer rejected a video packet");
            };
            backend
                .encode_frame(&raw, &mut sink)
                .expect("fixture encode_frame failed");
        }

        let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
            muxer
                .write_packet(pkt, true)
                .expect("fixture muxer rejected a flushed packet");
        };
        backend.flush(&mut sink).expect("fixture flush failed");
        muxer.finalise_sync().expect("fixture muxer finalise failed");

        let size = std::fs::metadata(path)
            .expect("fixture file was not created")
            .len();
        assert!(
            size > 1024,
            "fixture {path:?} is implausibly small ({size} bytes) — the encoder \
             produced no bitstream"
        );
    }

    /// One decoded frame of a fixture: the planar bytes plus the metadata the
    /// DECODER reported for them.
    struct Decoded {
        /// Exactly what `Decoder::decode_into` wrote — handed to
        /// `YuvUploadNode::upload_frame` unmodified, as the production upload path
        /// does with the slot-pool buffer.
        buf:    Vec<u8>,
        meta:   DecodedFrameMeta,
        width:  u32,
        height: u32,
    }

    impl Decoded {
        /// The luma code at (x, y), for the CPU-side cross-check.
        fn luma(&self, x: u32, y: u32) -> u8 {
            self.buf[(y * self.width + x) as usize]
        }

        /// `(Cb, Cr)` for the chroma block covering (x, y).
        fn chroma(&self, x: u32, y: u32) -> (u8, u8) {
            let luma_len = (self.width * self.height) as usize;
            let cw = self.width.div_ceil(2) as usize;
            let ch = self.height.div_ceil(2) as usize;
            let ci = (y as usize / 2) * cw + (x as usize / 2);
            (self.buf[luma_len + ci], self.buf[luma_len + cw * ch + ci])
        }
    }

    /// Decode the first frame of `path` with this crate's own demuxer/decoder.
    ///
    /// Software-only (`open_sw`), for the same reason `export_validation` does it:
    /// a hardware decoder would hand back NV12 or a hardware surface, and these
    /// tests want one known planar layout so the CPU cross-check can index the
    /// planes directly.  The metadata path being tested is identical either way —
    /// `read_frame_color` reads the AVFrame after any hardware transfer.
    ///
    /// The drain loop is not optional: `open_sw` enables frame-level threading, so
    /// a short file yields zero frames from the send/receive loop alone.
    fn decode_first_frame(path: &Path) -> Decoded {
        let mut demuxer = crate::io::demuxer::Demuxer::open(path)
            .expect("failed to open the fixture for decoding");
        let stream = demuxer
            .video_stream
            .clone()
            .expect("fixture has no video stream");
        let mut decoder = crate::io::decoder::Decoder::open_sw(&stream, stream.codecpar)
            .expect("failed to open a software decoder for the fixture");

        // Room for 4:4:4 at this size, so a surprise pixel format cannot overflow.
        let mut buf = vec![0u8; (W as usize * H as usize) * 3 + 64];

        let mut decoded: Option<Decoded> = None;
        while decoded.is_none() {
            match demuxer.next_video_packet() {
                Ok(Some(pkt)) => {
                    if let Ok(Some(frame)) = decoder.decode_into(&pkt, &mut buf, None) {
                        decoded = Some(Decoded {
                            buf:    buf.clone(),
                            meta:   frame.meta,
                            width:  frame.width,
                            height: frame.height,
                        });
                    }
                }
                Ok(None) => break, // EOF — fall through to the drain
                Err(e) => panic!("demuxing the fixture failed: {e:?}"),
            }
        }

        if decoded.is_none() {
            // One drained frame is enough; `drain_into` returning `Ok(None)` means
            // the decoder is empty, and an error there is a real failure rather
            // than something to retry.
            match decoder.drain_into(&mut buf) {
                Ok(Some(frame)) => {
                    decoded = Some(Decoded {
                        buf:    buf.clone(),
                        meta:   frame.meta,
                        width:  frame.width,
                        height: frame.height,
                    });
                }
                Ok(None) => {}
                Err(e) => panic!("draining the fixture decoder failed: {e:?}"),
            }
        }

        let decoded = decoded.expect("the fixture produced no decoded frame");
        assert_eq!(
            (decoded.width, decoded.height), (W, H),
            "decoded frame size does not match the fixture"
        );
        assert!(
            !decoded.meta.layout.semi_planar,
            "expected planar YUV420P from the software decoder, got semi-planar chroma"
        );
        assert_eq!(
            decoded.meta.layout.bit_depth, 8,
            "expected an 8-bit decode, got {}-bit",
            decoded.meta.layout.bit_depth
        );
        decoded
    }

    /// Copies one texture into a mappable buffer at the end of the graph.
    ///
    /// `bytes_per_row` is `W * 8` for Rgba16Float, which at W = 320 is 2560 — a
    /// multiple of the 256-byte alignment `copy_texture_to_buffer` requires, so no
    /// row padding has to be unpicked on the way out.
    struct ReadbackNode {
        src: ResourceId,
        buf: Arc<wgpu::Buffer>,
    }

    impl RenderNode for ReadbackNode {
        fn name(&self) -> &str {
            "ColourPlumbingReadback"
        }

        fn declare_resources(&self, builder: &mut ResourceBuilder) {
            builder.read(self.src, TextureAccess::CopySrc);
        }

        fn record(
            &self,
            encoder: &mut wgpu::CommandEncoder,
            ctx: &RenderContext,
            _frame: &FrameState,
        ) {
            let src = ctx.get(self.src);
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture:   src.texture,
                    mip_level: 0,
                    origin:    wgpu::Origin3d::ZERO,
                    aspect:    wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &self.buf,
                    layout: wgpu::ImageDataLayout {
                        offset:         0,
                        bytes_per_row:  Some(W * 8),
                        rows_per_image: Some(H),
                    },
                },
                wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            );
        }
    }

    /// Run `YuvUploadNode → YuvToRgbNode` over a decoded frame and return the RGB
    /// the shader produced, quantised to 8-bit for comparison.
    ///
    /// `color` is passed in rather than taken from `decoded.meta` so a test can
    /// deliberately supply the WRONG description and measure what that costs —
    /// which is what gives the correct-metadata assertions their teeth.  The
    /// production callers (`ExportRenderer::ensure_graph`, `NexirApp::
    /// compile_export_graph`) always pass `clip.frame_meta.color`; the node
    /// construction here mirrors them argument for argument.
    fn render_yuv_to_rgb(
        device:  &GpuDevice,
        shaders: &ShaderRegistry,
        compute: &ComputePipelineCache,
        decoded: &Decoded,
        color:   ColorInfo,
    ) -> Vec<[u8; 3]> {
        let layout = decoded.meta.layout;

        let mut compiler = RenderGraphCompiler::new();
        let mut id_counter = 2u32; // 0 = FINAL_COLOR, 1 = SCREEN
        let y_id = ResourceId::next(&mut id_counter);
        let uv_id = ResourceId::next(&mut id_counter);
        let rgba_id = ResourceId::next(&mut id_counter);

        let upload = YuvUploadNode::new_with_layout(
            device, 0, decoded.width, decoded.height, y_id, uv_id, layout,
        );
        upload.upload_frame(&decoded.buf, layout.semi_planar, decoded.width, decoded.height);
        compiler.add_node(Box::new(upload));

        compiler.add_node(Box::new(YuvToRgbNode::new_with_layout(
            device,
            shaders,
            compute,
            y_id,
            uv_id,
            rgba_id,
            decoded.width,
            decoded.height,
            color,
            layout.semi_planar,
        )));

        let out = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colour_plumbing_readback"),
            size:  (W * H * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        compiler.add_node(Box::new(ReadbackNode { src: rgba_id, buf: Arc::clone(&out) }));

        let graph = compiler.compile(W, H).expect("YUV→RGB graph failed to compile");
        let frame = FrameState::test_empty(W, H);
        let mut encoder = device.begin_frame();
        graph.execute(&mut encoder, device, &frame);
        let submission = device.submit(encoder);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range().to_vec();
        out.unmap();

        mapped
            .chunks_exact(8)
            .map(|px| {
                let mut rgb = [0u8; 3];
                for c in 0..3 {
                    let v = f16::from_le_bytes([px[c * 2], px[c * 2 + 1]]).to_f32();
                    rgb[c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
                rgb
            })
            .collect()
    }

    /// Unique scratch paths so concurrent tests never collide.
    fn scratch(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "nexir_colour_plumbing_{}_{}_{}.mp4",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    // ── 4. The same claim, through the GPU decode path ────────────────────────
    //
    // G2c/G2d. The interop path bypasses `YuvUploadNode` entirely: NVDEC writes
    // into textures the graph imports. That is EXACTLY the kind of bypass gotcha 11
    // warns about — the upload path is where colour metadata is read on the CPU
    // path, and a path that skips it is a path that can drop it. So the BT.601
    // fixtures above have to survive it, with the same mis-tagged control, because
    // BT.709 fixtures would pass even with the metadata discarded entirely.

    /// The stream info the registry would hold for a fixture, from the container.
    ///
    /// Built from the DEMUXER rather than hardcoded, because
    /// `InteropDecodeTargets::ensure_target` reads `pixel_fmt` and the dimensions
    /// off it to decide whether the source may use the interop path at all — a
    /// hardcoded `Nv12` would make this test pass for a fixture the production gate
    /// would reject.
    fn stream_info_for(path: &Path) -> (crate::timeline::source::VideoStreamInfo, u32, u32) {
        let demuxer = crate::io::demuxer::Demuxer::open(path).expect("Demuxer::open");
        let stream = demuxer.video_stream.clone().expect("no video stream");
        let w = stream.width.expect("no width");
        let h = stream.height.expect("no height");
        (
            crate::timeline::source::VideoStreamInfo {
                width: w,
                height: h,
                frame_rate: FPS,
                // 8-bit 4:2:0, which is what `write_source_video` encodes.
                pixel_fmt: crate::timeline::source::PixelFormat::Yuv420p,
                color_info: stream.color_info,
                duration_pts: stream.duration,
                is_vfr: stream.is_vfr,
                time_base: stream.time_base,
                rotation: Default::default(),
            },
            w,
            h,
        )
    }

    /// Decode the first frame of `path` STRAIGHT INTO GPU TEXTURES, through the
    /// production registry.
    ///
    /// Returns `None` for every host reason the interop path is unavailable (no
    /// CUDA, no NVDEC for this codec, an allocation failure) so the caller can
    /// print a skip rather than fail — gotcha 9's rule applied to a test.
    ///
    /// Driven through [`crate::io::interop_decode::InteropDecodeTargets`] rather
    /// than calling `DecodeInteropTarget` directly: the thing under test is the
    /// wiring, and the registry is what production calls. Its read-forward
    /// contract is honoured here too — `Pending` means feed another packet, which
    /// matters because frame-level threading holds several frames back.
    ///
    /// **The registry is RETURNED, not dropped, and that is load-bearing.** It owns
    /// the `DecodeInteropTarget`, whose `SharedTexture`s own the CUDA side of the
    /// allocation — dropping it runs `cuMipmappedArrayDestroy` +
    /// `cuDestroyExternalMemory` while the returned `ImportedTexture`s (which hold
    /// only the wgpu `Arc`) are still about to be sampled. The wgpu texture survives
    /// that, so nothing here fails; what happened instead was **three unrelated
    /// NVENC tests in `tests::export_validation` starting to report "the export
    /// engine selected the FFmpeg backend"**, because tearing down a live CUDA
    /// import left the driver in a state where `EncodeInterop::open` was refused.
    /// The registry outliving every frame it produced is a production invariant too:
    /// `IoLayer` holds it for the session.
    ///
    /// Built with `with_context` and the process-wide context, NOT with `new`: see
    /// [`shared_cuda_ctx`].
    fn decode_first_frame_interop(
        device: &Arc<GpuDevice>,
        path: &Path,
    ) -> Option<(
        crate::io::interop_decode::InteropDecodeTargets,
        crate::io::interop_decode::InteropFrame,
    )> {
        use crate::interop::capability::InteropCapability;
        use crate::io::interop_decode::{InteropDecode, InteropDecodeTargets};

        let capability = InteropCapability::probe(device);
        let cuda = crate::tests::shared_cuda_ctx(&capability);
        let targets =
            InteropDecodeTargets::with_context(Arc::clone(device), capability, cuda);
        if !targets.is_available() {
            eprintln!("[colour_plumbing] SKIP: CUDA interop is unavailable on this host");
            return None;
        }

        let (info, _, _) = stream_info_for(path);
        let mut demuxer = crate::io::demuxer::Demuxer::open(path).expect("Demuxer::open");
        let stream = demuxer.video_stream.clone().expect("no video stream");
        // Hardware decode ENABLED — this is the whole point, and `Decoder::open`
        // falls back to software silently, which `ensure_target` then rejects.
        let mut decoder = crate::io::decoder::Decoder::open(&stream, stream.codecpar, true)
            .expect("Decoder::open");

        let source_id = crate::timeline::ids::SourceId::new(0);
        for _ in 0..600 {
            let Some(pkt) = demuxer.next_video_packet().ok().flatten() else {
                break;
            };
            match targets.decode_into_target(source_id, &info, &mut decoder, &pkt) {
                InteropDecode::Decoded(frame) => return Some((targets, frame)),
                InteropDecode::Pending => continue,
                InteropDecode::Unavailable(reason) => {
                    eprintln!("[colour_plumbing] SKIP: the interop path declined: {reason}");
                    return None;
                }
            }
        }
        eprintln!("[colour_plumbing] SKIP: no frame came out of the interop path");
        None
    }

    /// Run `YuvToRgbNode` over IMPORTED planes and return the RGB it produced.
    ///
    /// The G2d shape, and the three decisions that make it one decision:
    ///
    ///  * **No `YuvUploadNode` in the graph at all.** Leaving one in and not
    ///    feeding it does not import anything — `declare_resources` *creates* both
    ///    planes, so the graph would allocate two pooled textures nobody writes and
    ///    the clip would render black while NVDEC wrote into textures the graph
    ///    never looked at. No error, no failing assertion in the pool's counters.
    ///  * **`with_imported_planes(true)`**, so the node declares those ids with
    ///    `import` rather than `read`. Without it the compile fails with
    ///    `MissingProducer`, which is the good outcome — it is why that error had to
    ///    keep its meaning (gotcha 18).
    ///  * **The ids come from `FrameScheduler`**, the same function that binds
    ///    them, because nothing else makes the reader and the binder agree.
    ///
    /// `color` is passed in so a caller can supply deliberately wrong metadata and
    /// measure what that costs — the control that gives the pixel assertions teeth.
    fn render_interop_yuv_to_rgb(
        device: &GpuDevice,
        shaders: &ShaderRegistry,
        compute: &ComputePipelineCache,
        frame_in: &crate::io::interop_decode::InteropFrame,
        color: ColorInfo,
    ) -> (Vec<[u8; 3]>, Vec<String>, crate::render::resource::PoolStats) {
        use crate::render::resource::ImportedResources;
        use crate::scheduler::frame_scheduler::FrameScheduler;

        let layout = frame_in.meta.layout;
        let y_id = FrameScheduler::interop_y_id(0);
        let uv_id = FrameScheduler::interop_uv_id(0);
        let mut id_counter = FrameScheduler::interop_id_counter_start(1);
        let rgba_id = ResourceId::next(&mut id_counter);

        let mut compiler = RenderGraphCompiler::new();
        compiler.add_node(Box::new(
            YuvToRgbNode::new_with_layout(
                device,
                shaders,
                compute,
                y_id,
                uv_id,
                rgba_id,
                frame_in.width,
                frame_in.height,
                color,
                layout.semi_planar,
            )
            .with_imported_planes(true),
        ));

        let out = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colour_plumbing_interop_readback"),
            size: (W * H * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        compiler.add_node(Box::new(ReadbackNode { src: rgba_id, buf: Arc::clone(&out) }));

        let graph = compiler
            .compile(W, H)
            .expect("an imported plane is its own producer, so this must compile");
        let names: Vec<String> = graph
            .execution_order()
            .iter()
            .map(|&i| graph.node_name(i).to_string())
            .collect();

        let mut frame = FrameState::test_empty(W, H);
        let mut binding = ImportedResources::new();
        binding.bind(y_id, frame_in.planes.y.clone());
        binding.bind(uv_id, frame_in.planes.uv.clone());
        frame.imported = binding;

        let mut encoder = device.begin_frame();
        graph.execute(&mut encoder, device, &frame);
        let submission = device.submit(encoder);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(submission));

        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range().to_vec();
        out.unmap();

        let pixels = mapped
            .chunks_exact(8)
            .map(|px| {
                let mut rgb = [0u8; 3];
                for c in 0..3 {
                    let v = f16::from_le_bytes([px[c * 2], px[c * 2 + 1]]).to_f32();
                    rgb[c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
                rgb
            })
            .collect();
        (pixels, names, graph.pool_stats())
    }

    /// Read the luma plane back and report the largest code in it.
    ///
    /// **Not a convenience — a guard against a false PASS and a false FAIL.** The
    /// interop copy can silently write nothing whenever the CUDA calls in
    /// `CudaContext::with_context` run against the wrong context: the bind is
    /// checked and logged now (`cuCtxSetCurrent`), but a failed bind still falls
    /// through and runs the closure, and `cuMemcpy2DAsync` then reports success
    /// having copied nothing. This is what the old `cuCtxPushCurrent` did on every
    /// call after the first `Decoder::open` — push needs a floating context, and
    /// `AV_CUDA_USE_PRIMARY_CONTEXT` means FFmpeg holds this one (gotcha 4). The
    /// result is an all-zero Y plane, which the shader then converts to black.
    ///
    /// Distinguishing that from "the shader got the matrix wrong" is the whole
    /// point: the first is a host/serialisation condition this test cannot control
    /// and must skip on with a reason, the second is the failure it exists to catch.
    /// Asserting on pixels without checking this makes a green suite that exercised
    /// nothing look identical to a broken conversion.
    fn peak_luma(device: &GpuDevice, frame: &crate::io::interop_decode::InteropFrame) -> u8 {
        let bytes_per_row = (frame.width + 255) & !255;
        let buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interop_luma_probe"),
            size: (bytes_per_row * frame.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.begin_frame();
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: frame.planes.y.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
            },
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let peak = slice.get_mapped_range().iter().copied().max().unwrap_or(0);
        buf.unmap();
        peak
    }

    /// A BT.601 source decoded STRAIGHT INTO GPU TEXTURES must still decode to the
    /// pixels it was built from — and reading it as BT.709 must still not.
    ///
    /// G2d's colour acceptance criterion. The interop path never touches
    /// `YuvUploadNode`, so it is a path where the frame's colour metadata could be
    /// dropped silently: `Decoder::emit_frame`'s interop arm reads `read_frame_color`
    /// from the AVFrame and pins the layout to NV12/P010, and this is what proves
    /// the result reaches `YuvToRgbNode`'s push constants.
    ///
    /// **BT.601, for the reason this whole file is BT.601** (gotcha 11): BT.709
    /// limited is `ColorInfo::default()`, `from_ffmpeg`'s HD heuristic and
    /// `luma_coefficients`' `Unknown` arm, so a BT.709 fixture would pass even if
    /// the metadata never arrived. The mis-tagged control at the end is what stops a
    /// shader that hardcoded one matrix from passing too.
    ///
    /// Also asserts the graph SHAPE, because the pixels alone cannot distinguish
    /// "imported correctly" from "an upload node quietly allocated two textures and
    /// someone fed them": `YuvUpload` must be absent from the execution order, and
    /// the pool must have allocated exactly the one RGBA output.
    ///
    /// Holds `tests::cuda_lock()` for its whole body (gotcha 4): `CudaContext`
    /// wraps the device's primary context, so every instance in the process is one
    /// `CUcontext` on one stream, and the CUDA work inside two tests' closures must
    /// not interleave. (Binding it is safe concurrently — `with_context` is
    /// `cuCtxSetCurrent`, not `cuCtxPushCurrent`.)
    #[test]
    fn interop_decoded_frames_carry_their_matrix_to_the_shader() {
        let _cuda = crate::tests::cuda_lock();

        let device = Arc::new(
            pollster::block_on(GpuDevice::new_headless())
                .expect("failed to create headless GpuDevice"),
        );
        let shaders = ShaderRegistry::compile_all(&device).expect("shader compilation failed");
        let compute = ComputePipelineCache::new();

        let path = scratch("interop_bt601");
        write_source_video(&path, MatrixCoefficients::Bt601, 8);

        let Some((_targets, frame)) = decode_first_frame_interop(&device, &path) else {
            let _ = std::fs::remove_file(&path);
            return; // a printed skip with a reason, never a false pass
        };
        // `_targets` is held for the whole test on purpose — see
        // `decode_first_frame_interop`'s doc comment. Dropping it destroys the CUDA
        // import of textures this test is still sampling.

        // NVDEC emits NV12, and the layout must say so — a planar description of
        // interleaved chroma is a green/magenta picture, not a subtle shift.
        assert!(
            frame.meta.layout.semi_planar,
            "NVDEC's output is NV12; the interop arm reported planar chroma, so the \
             shader will de-interleave nothing"
        );
        assert_eq!(
            frame.meta.layout.bit_depth, 8,
            "the fixture is 8-bit; a P010 layout here means the 10-bit gate let an \
             8-bit source through"
        );
        assert_eq!(
            (frame.width, frame.height), (W, H),
            "the decoded frame is not the fixture's size"
        );

        // THE CLAIM: the file's own matrix survived the bypass.
        assert_eq!(
            frame.meta.color.matrix, MatrixCoefficients::Bt601,
            "the interop path reported matrix {:?} for a file tagged BT.601 — the \
             colour metadata is dropped when `YuvUploadNode` is bypassed, which is \
             exactly the failure AGENTS.md gotcha 11 describes",
            frame.meta.color.matrix
        );
        assert_eq!(
            frame.meta.color.effective_range(), ColorRange::Limited,
            "expected limited range, got {:?}",
            frame.meta.color.effective_range()
        );

        // Did the copy actually land? An all-zero luma plane means the CUDA calls in
        // the closure ran against the wrong context — reported as success, having
        // copied nothing (gotcha 4). That is a host condition, not a colour bug, and
        // the pixel assertions below cannot tell the two apart, so skip on it rather
        // than failing for the wrong reason. `write_source_video`'s pattern includes a
        // white band, so a landed copy has luma near 235 (limited range).
        let peak = peak_luma(&device, &frame);
        if peak < 200 {
            eprintln!(
                "[colour_plumbing] SKIP: the interop luma plane peaks at {peak}, so the \
                 device→array copy wrote nothing (the CUDA context could not be made \
                 current — gotcha 4). Nothing about the colour path can be concluded \
                 from this run."
            );
            let _ = std::fs::remove_file(&path);
            return;
        }

        let (correct, names, stats) =
            render_interop_yuv_to_rgb(&device, &shaders, &compute, &frame, frame.meta.color);

        // ── The graph SHAPE, which the pixels cannot check ────────────────────
        assert!(
            !names.iter().any(|n| n == "YuvUpload"),
            "an interop clip's upload node must be ABSENT, not merely unfed: it \
             creates both planes, so leaving it in allocates two pooled textures \
             nobody writes and renders black. Order was {names:?}"
        );
        assert_eq!(
            stats.pooled, 1,
            "only the RGBA output may come from the pool; {} textures were pooled, \
             so the graph allocated something for an imported plane",
            stats.pooled
        );
        assert_eq!(
            stats.misses, 1,
            "exactly one allocation (the RGBA output) is expected; {} were made",
            stats.misses
        );

        // ── The pixels ────────────────────────────────────────────────────────
        let x = W / 2;
        let ph = patch_height();
        let mut worst = 0i32;
        for (i, (name, want)) in PATCHES.iter().enumerate() {
            let y = i as u32 * ph + ph / 2;
            let got = correct[(y * W + x) as usize];
            for c in 0..3 {
                let delta = (got[c] as i32 - want[c] as i32).abs();
                worst = worst.max(delta);
                assert!(
                    delta <= TOLERANCE,
                    "interop patch '{name}' at ({x},{y}) channel {c}: shader produced \
                     {got:?}, expected {want:?} (delta {delta} > {TOLERANCE})"
                );
            }
        }

        // ── The control: deliberately wrong metadata ──────────────────────────
        let mut mistagged = frame.meta.color;
        mistagged.matrix = MatrixCoefficients::Bt709;
        let (wrong, _, _) =
            render_interop_yuv_to_rgb(&device, &shaders, &compute, &frame, mistagged);
        let mut worst_wrong = 0i32;
        for (i, (_, want)) in PATCHES.iter().enumerate() {
            let y = i as u32 * ph + ph / 2;
            let got = wrong[(y * W + x) as usize];
            for c in 0..3 {
                worst_wrong = worst_wrong.max((got[c] as i32 - want[c] as i32).abs());
            }
        }
        assert!(
            worst_wrong > TOLERANCE,
            "reading the interop frame as BT.709 changed the output by at most \
             {worst_wrong} levels, inside the {TOLERANCE}-level tolerance — the \
             assertions above cannot detect a matrix error on this path either"
        );

        eprintln!(
            "[colour_plumbing] interop BT.601: worst delta with the signalled \
             matrix {worst}/255, worst delta when mis-read as BT.709 {worst_wrong}/255; \
             graph was {names:?}"
        );

        let _ = std::fs::remove_file(&path);
    }

    // ── 1. The wiring claim, without the GPU ──────────────────────────────────

    /// The metadata FFmpeg reported for a real decoded frame must be the metadata
    /// the node converts with.
    ///
    /// `YuvToRgbNode::conversion()` exposes the resolved `YuvConversion`, so this
    /// reads the actual push-constant block the shader would receive and checks
    /// the Cr→R coefficient against the published value for the matrix the FILE
    /// carries — 1.402 for BT.601, 1.5748 for BT.709.
    ///
    /// The BT.601 case is the one that matters.  Every fallback in the chain
    /// resolves to BT.709 (`ColorInfo::default`, `from_ffmpeg`'s heuristics,
    /// `luma_coefficients`' `Unknown` arm), so a break that dropped the frame's
    /// metadata entirely would still produce a BT.709 conversion and the BT.709
    /// half of this test would pass regardless.
    #[test]
    fn decoded_frame_metadata_reaches_the_node() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");
        let shaders = ShaderRegistry::compile_all(&device).expect("shader compilation failed");
        let compute = ComputePipelineCache::new();

        // (label, tagged matrix, expected Cr→R coefficient)
        let cases: [(&str, MatrixCoefficients, f32); 2] = [
            ("bt601", MatrixCoefficients::Bt601, 1.402),
            ("bt709", MatrixCoefficients::Bt709, 1.5748),
        ];

        for (label, matrix, want_cr_r) in cases {
            let path = scratch(label);
            write_source_video(&path, matrix, 4);
            let decoded = decode_first_frame(&path);

            // The decoder must have read the file's own tag rather than guessing.
            assert_eq!(
                decoded.meta.color.matrix, matrix,
                "{label}: the decoder reported matrix {:?} for a file tagged \
                 {matrix:?} — the colour metadata is lost before it reaches the \
                 render graph at all",
                decoded.meta.color.matrix
            );
            assert_eq!(
                decoded.meta.color.effective_range(), ColorRange::Limited,
                "{label}: expected limited range, got {:?}",
                decoded.meta.color.effective_range()
            );

            // Constructed exactly as `ExportRenderer::ensure_graph` does: colour
            // and layout both from the DECODED frame.
            let node = YuvToRgbNode::new_with_layout(
                &device,
                &shaders,
                &compute,
                ResourceId(2),
                ResourceId(3),
                ResourceId(4),
                decoded.width,
                decoded.height,
                decoded.meta.color,
                decoded.meta.layout.semi_planar,
            );
            let got = node.conversion();

            assert!(
                (got.row_r[2] - want_cr_r).abs() < 1e-3,
                "{label}: the node resolved Cr→R = {} but the file is tagged \
                 {matrix:?}, whose coefficient is {want_cr_r} — the frame's \
                 metadata did not reach YuvToRgbNode",
                got.row_r[2]
            );

            // And the whole block matches what the metadata implies, not just the
            // one coefficient: range, bit depth and sample scaling too.
            let want = YuvConversion::new(
                decoded.meta.color,
                decoded.width,
                decoded.height,
                decoded.meta.layout.semi_planar,
            );
            assert_eq!(
                bytemuck::bytes_of(got), bytemuck::bytes_of(&want),
                "{label}: the node's push-constant block differs from the one the \
                 decoded metadata implies"
            );
            assert!(
                (got.luma_offset - 16.0 / 255.0).abs() < 1e-4
                    && (got.luma_scale - 255.0 / 219.0).abs() < 1e-3,
                "{label}: limited-range luma constants are wrong: offset {}, \
                 scale {}", got.luma_offset, got.luma_scale
            );
            assert!(
                (got.sample_scale - 1.0).abs() < 1e-6,
                "{label}: 8-bit data must need no sample rescaling, got {}",
                got.sample_scale
            );

            let _ = std::fs::remove_file(&path);
        }
    }

    // ── 2. The same path with the shader running ───────────────────────────────

    /// A BT.601 source must decode back to the pixels it was built from — and
    /// reading it as BT.709 must not.
    ///
    /// The first half is the end-to-end pixel claim: metadata → push constants →
    /// shader → RGB.  The second half is what makes it meaningful.  Without the
    /// mis-tagged control, a shader that ignored its push constants entirely and
    /// hardcoded one matrix could still pass, because one of the two matrices
    /// would be right by accident.
    #[test]
    fn node_decodes_the_pattern_using_the_signalled_matrix() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");
        let shaders = ShaderRegistry::compile_all(&device).expect("shader compilation failed");
        let compute = ComputePipelineCache::new();

        let path = scratch("pixels_bt601");
        write_source_video(&path, MatrixCoefficients::Bt601, 4);
        let decoded = decode_first_frame(&path);
        assert_eq!(
            decoded.meta.color.matrix, MatrixCoefficients::Bt601,
            "fixture is not tagged BT.601"
        );

        let x = W / 2;
        let ph = patch_height();

        // ── The honest path: the frame's own metadata ─────────────────────────
        let correct = render_yuv_to_rgb(&device, &shaders, &compute, &decoded, decoded.meta.color);
        let mut worst = 0i32;
        for (i, (name, want)) in PATCHES.iter().enumerate() {
            let y = i as u32 * ph + ph / 2;
            let got = correct[(y * W + x) as usize];
            for c in 0..3 {
                let delta = (got[c] as i32 - want[c] as i32).abs();
                worst = worst.max(delta);
                assert!(
                    delta <= TOLERANCE,
                    "patch '{name}' at ({x},{y}) channel {c}: shader produced \
                     {got:?}, expected {want:?} (delta {delta} > {TOLERANCE})"
                );
            }
        }

        // Channel order, independent of the tolerance above: a swapped R/B would
        // fail the loop, but this names the failure.
        let red = correct[((2 * ph + ph / 2) * W + x) as usize];
        assert!(
            red[0] > red[1] + 60 && red[0] > red[2] + 60,
            "the red band is not red-dominant: {red:?} — channels are swapped"
        );
        let blue = correct[((4 * ph + ph / 2) * W + x) as usize];
        assert!(
            blue[2] > blue[0] + 60 && blue[2] > blue[1] + 60,
            "the blue band is not blue-dominant: {blue:?} — channels are swapped"
        );

        // ── The CPU reference, on the same bytes ──────────────────────────────
        // `YuvConversion::apply` is the documented CPU twin of the shader, so this
        // separates "the metadata is wrong" from "the shader disagrees with the
        // maths we unit-tested".
        let conversion = YuvConversion::new(
            decoded.meta.color, decoded.width, decoded.height,
            decoded.meta.layout.semi_planar,
        );
        for (i, (name, _)) in PATCHES.iter().enumerate() {
            let y = i as u32 * ph + ph / 2;
            let (cb, cr) = decoded.chroma(x, y);
            let cpu = conversion.apply(
                decoded.luma(x, y) as f32 / 255.0,
                cb as f32 / 255.0,
                cr as f32 / 255.0,
            );
            let got = correct[(y * W + x) as usize];
            for c in 0..3 {
                let cpu_u8 = (cpu[c] * 255.0 + 0.5).clamp(0.0, 255.0) as i32;
                assert!(
                    (got[c] as i32 - cpu_u8).abs() <= 2,
                    "patch '{name}' channel {c}: shader {} vs CPU reference \
                     {cpu_u8} — the GPU and `YuvConversion::apply` disagree on \
                     the same bytes",
                    got[c]
                );
            }
        }

        // ── The control: deliberately wrong metadata ──────────────────────────
        let mut mistagged = decoded.meta.color;
        mistagged.matrix = MatrixCoefficients::Bt709;
        let wrong = render_yuv_to_rgb(&device, &shaders, &compute, &decoded, mistagged);

        let mut worst_wrong = 0i32;
        for (i, (_, want)) in PATCHES.iter().enumerate() {
            let y = i as u32 * ph + ph / 2;
            let got = wrong[(y * W + x) as usize];
            for c in 0..3 {
                worst_wrong = worst_wrong.max((got[c] as i32 - want[c] as i32).abs());
            }
        }
        assert!(
            worst_wrong > TOLERANCE,
            "reading a BT.601 frame as BT.709 changed the output by at most \
             {worst_wrong} levels, which is inside the {TOLERANCE}-level \
             tolerance — the pixel assertions above cannot detect a matrix error \
             and are worthless as written"
        );

        eprintln!(
            "[colour_plumbing] BT.601 source: worst delta with the signalled \
             matrix {worst}/255, worst delta when mis-read as BT.709 \
             {worst_wrong}/255"
        );

        let _ = std::fs::remove_file(&path);
    }

    // ── 3. The fixtures really do differ ──────────────────────────────────────

    /// The two source files must carry measurably different samples.
    ///
    /// Everything above rests on it: if `sws_setColorspaceDetails` silently failed
    /// and both fixtures were converted with the same matrix, test 2's control
    /// would be comparing a file against itself and test 1 would be checking a tag
    /// nothing acted on.
    #[test]
    fn the_two_matrices_are_not_interchangeable() {
        let bt601_path = scratch("differ_bt601");
        let bt709_path = scratch("differ_bt709");
        write_source_video(&bt601_path, MatrixCoefficients::Bt601, 4);
        write_source_video(&bt709_path, MatrixCoefficients::Bt709, 4);

        let bt601 = decode_first_frame(&bt601_path);
        let bt709 = decode_first_frame(&bt709_path);

        let x = W / 2;
        let ph = patch_height();

        // Red and green move the most between the two matrices; grey does not move
        // at all, which is asserted below so this test cannot pass for the wrong
        // reason (e.g. two files that differ only in noise).
        let mut max_colour_delta = 0i32;
        for band in [2usize, 3] {
            let y = band as u32 * ph + ph / 2;
            let d = (bt601.luma(x, y) as i32 - bt709.luma(x, y) as i32).abs();
            max_colour_delta = max_colour_delta.max(d);
        }
        assert!(
            max_colour_delta > 10,
            "the BT.601 and BT.709 fixtures differ by at most {max_colour_delta} \
             luma levels on saturated colour — swscale converted both with the \
             same matrix, so the tag and the samples disagree in one of them and \
             every assertion in this file is testing nothing"
        );

        let grey_y = 5 * ph + ph / 2;
        let grey_delta =
            (bt601.luma(x, grey_y) as i32 - bt709.luma(x, grey_y) as i32).abs();
        assert!(
            grey_delta <= 2,
            "the fixtures differ by {grey_delta} luma levels on the grey band, \
             where the matrix has no effect — they differ for some reason other \
             than the colour matrix"
        );

        eprintln!(
            "[colour_plumbing] fixtures differ by {max_colour_delta} luma levels \
             on saturated colour, {grey_delta} on grey"
        );

        let _ = std::fs::remove_file(&bt601_path);
        let _ = std::fs::remove_file(&bt709_path);
    }
}
