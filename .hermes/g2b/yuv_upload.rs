// src/tests/yuv_upload.rs
//
// P0.1 — the CPU→GPU upload path.
//
// WHAT THESE TESTS DEFEND.  `YuvUploadNode::upload_frame` has two paths into the
// same staging buffers: a general one that repacks rows into a padded scratch
// buffer, and a contiguous one that hands the decoder's bytes straight to
// `queue.write_buffer` because the source rows already sit at the stride the
// staging buffer expects.
//
// The contiguous path is a claim about arithmetic — `round_up_256(width * bpp)
// == width * bpp` — and an off-by-one in that condition does NOT produce an
// error. It produces a sheared or vertically-collapsed picture, silently, the
// same way a disagreeing NV12 pitch does (AGENTS.md gotcha 6). So the central
// test here does not check that the fast path runs; it uploads the same pixels
// both ways and requires the resulting GPU textures to be byte-identical, and
// to match the source.

#[cfg(test)]
mod yuv_upload {
    use crate::render::device::GpuDevice;
    use crate::render::nodes::yuv_upload::YuvUploadNode;
    use crate::render::resource::ResourceId;
    use crate::timeline::source::FrameLayout;

    fn round_up_256(n: u32) -> u32 {
        (n + 255) & !255
    }

    /// A distinguishable NV12 frame: every row and column contributes, so a row
    /// mix-up, a collapsed plane or a stride slip all change the bytes.
    ///
    /// Flat or repeating data would let a broken upload pass, which is why this
    /// is not a solid grey frame.
    fn nv12_pattern(w: u32, h: u32) -> Vec<u8> {
        let y_size = (w * h) as usize;
        let mut buf = vec![0u8; y_size + y_size / 2];
        for y in 0..h {
            for x in 0..w {
                // 16..235 limited-range luma, varying in BOTH axes.
                buf[(y * w + x) as usize] = 16 + ((y * 7 + x * 3) % 219) as u8;
            }
        }
        for cy in 0..h / 2 {
            for cx in 0..w / 2 {
                let base = y_size + (cy * w + cx * 2) as usize;
                buf[base] = 16 + ((cy * 5 + cx * 11) % 219) as u8;
                buf[base + 1] = 16 + ((cy * 13 + cx * 3) % 219) as u8;
            }
        }
        buf
    }

    /// Read the luma plane the node most recently uploaded, cropped to
    /// `frame_w × frame_h`.
    ///
    /// Goes through `record()` rather than inspecting the staging buffer, so what
    /// is compared is the texture the shaders will sample: the staging buffer can
    /// be correct while the copy extent is wrong, and only this catches that.
    ///
    /// Deliberately does NOT upload — the caller uploads, so a node allocated
    /// wider than its frame can be tested without this helper silently
    /// re-uploading at the wrong size.
    fn read_luma(
        device: &GpuDevice,
        node: &YuvUploadNode,
        frame_w: u32,
        frame_h: u32,
    ) -> Vec<u8> {
        use crate::render::context::RenderContext;
        use crate::render::frame_state::FrameState;
        use crate::render::graph::RenderNode;

        // The graph would supply these; build them by hand so `record` can be
        // exercised without compiling a whole graph.
        let y_tex = device.create_texture(
            Some("test_y"),
            node.width,
            node.height,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        );
        let uv_tex = device.create_texture(
            Some("test_uv"),
            node.width / 2,
            node.height / 2,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        );
        let y_view = y_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = uv_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut resources: Vec<
            Option<(wgpu::Texture, wgpu::TextureView, crate::render::resource::ViewId)>,
        > = (0..4).map(|_| None).collect();
        resources[node.out_y.0 as usize] =
            Some((y_tex, y_view, crate::render::resource::ViewId::new()));
        resources[node.out_uv.0 as usize] =
            Some((uv_tex, uv_view, crate::render::resource::ViewId::new()));
        let ctx = RenderContext::new(resources);

        let frame = FrameState {
            pts: 0,
            canvas_width: frame_w,
            canvas_height: frame_h,
            clips: vec![],
            test_textures: vec![],
            imported: Default::default(),
        };

        // Texture-to-buffer readback needs a 256-aligned row stride of its own.
        let stride = round_up_256(frame_w);
        let readback = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_readback"),
            size: stride as u64 * frame_h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.begin_frame();
        node.record(&mut enc, &ctx, &frame);
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: ctx.get(node.out_y).texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(frame_h),
                },
            },
            wgpu::Extent3d {
                width: frame_w,
                height: frame_h,
                depth_or_array_layers: 1,
            },
        );
        let sid = device.submit(enc);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        rx.recv().unwrap().unwrap();

        // Strip the readback padding so the comparison is over pixels.
        let view = slice.get_mapped_range();
        let mut out = Vec::with_capacity((frame_w * frame_h) as usize);
        for row in 0..frame_h as usize {
            let start = row * stride as usize;
            out.extend_from_slice(&view[start..start + frame_w as usize]);
        }
        drop(view);
        readback.unmap();
        out
    }

    /// Which real resolutions can take the contiguous path, measured rather than
    /// assumed.
    ///
    /// This corrects an assumption worth recording: 3840 IS 256-aligned, but
    /// **1920 is not** — 8-bit luma at 1080p pads to 2048, so 1080p keeps
    /// repacking rows and only 4K gets the copy-free path. The scratch-buffer
    /// reuse in `upload_frame` is what helps 1080p.
    ///
    /// If this arithmetic ever changes, the guard in `upload_frame` must change
    /// with it — hence a test rather than a comment.
    #[test]
    fn which_widths_can_skip_the_repack() {
        // 8-bit luma: byte width == pixel width.
        assert_eq!(round_up_256(3840), 3840, "4K luma needs no padding");
        assert_eq!(round_up_256(1280), 1280, "720p luma needs no padding");
        assert_eq!(
            round_up_256(1920),
            2048,
            "1080p luma DOES pad — the contiguous path does not apply at 1080p"
        );
        // 16-bit samples double the byte width; 1920 then becomes aligned.
        assert_eq!(round_up_256(1920 * 2), 1920 * 2);
        // And a genuinely odd width pads too.
        assert_ne!(round_up_256(1922), 1922);
    }

    /// The two upload paths must place identical bytes in the texture, and those
    /// bytes must be the source's.
    ///
    /// Both nodes are 512 wide (aligned, so the contiguous path is available).
    /// The second is allocated WIDER than the frame it is fed, which fails the
    /// "frame fills the allocation" condition and forces the general repack.
    /// Comparing the two catches a stride slip in either.
    #[test]
    fn contiguous_and_general_paths_agree_byte_for_byte() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        let (w, h) = (512u32, 64u32);
        let src = nv12_pattern(w, h);

        // ── Contiguous path: allocation == frame, stride == row width ──────────
        let fast = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );
        let cost = fast.upload_frame(&src, true, w, h);
        assert!(
            cost.contiguous,
            "a 512-wide full-size NV12 frame must take the contiguous path"
        );
        assert_eq!(
            cost.prepare,
            std::time::Duration::ZERO,
            "the contiguous path must not spend any time repacking rows"
        );
        assert!(cost.bytes > 0, "bytes handed to the queue must be counted");
        let fast_luma = read_luma(&device, &fast, w, h);

        // ── General path: same pixels, wider allocation ────────────────────────
        let general = YuvUploadNode::new_with_layout(
            &device,
            0,
            w + 256,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );
        let gcost = general.upload_frame(&src, true, w, h);
        assert!(
            !gcost.contiguous,
            "a frame narrower than the allocation must take the general path"
        );
        let general_luma = read_luma(&device, &general, w, h);

        assert_eq!(
            fast_luma.len(),
            general_luma.len(),
            "both readbacks cover the same {w}x{h} region"
        );
        for row in 0..h as usize {
            let lo = row * w as usize;
            let hi = lo + w as usize;
            assert_eq!(
                &fast_luma[lo..hi],
                &general_luma[lo..hi],
                "row {row} differs between the contiguous and general upload paths \
                 — the stride guard is wrong"
            );
            // And against the source, so two identically-broken paths cannot pass.
            assert_eq!(
                &fast_luma[lo..hi],
                &src[lo..hi],
                "row {row} does not match the source frame"
            );
        }
    }

    /// A frame whose row width is not 256-aligned must still upload correctly.
    ///
    /// This is the case the contiguous path must decline. Taking it anyway would
    /// write `width` bytes per row into a buffer expecting `round_up_256(width)`,
    /// so every row after the first would land at the wrong offset — a shear with
    /// no error anywhere.
    #[test]
    fn unaligned_width_uploads_correctly_via_the_general_path() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        // 300 rounds up to 512, so the destination stride exceeds the row width.
        let (w, h) = (300u32, 32u32);
        let src = nv12_pattern(w, h);
        let node = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );

        let cost = node.upload_frame(&src, true, w, h);
        assert!(
            !cost.contiguous,
            "width 300 pads to 512, so the contiguous path must be declined"
        );

        let luma = read_luma(&device, &node, w, h);
        for row in 0..h as usize {
            let lo = row * w as usize;
            let hi = lo + w as usize;
            assert_eq!(
                &luma[lo..hi],
                &src[lo..hi],
                "row {row} of an unaligned-width frame is wrong — this is the shear \
                 the stride guard exists to prevent"
            );
        }
    }

    /// Repeated uploads must not grow the heap.
    ///
    /// The regression this pins: the general path used to `vec![0u8; …]` a fresh
    /// padded buffer per plane per call — 8.3 MB per 4K layer, allocated, zeroed
    /// and filled every frame. The scratch buffers are now allocated once, so a
    /// second upload of the same size costs a memcpy and nothing else.
    ///
    /// Asserted as a timing ratio rather than an allocation count because Rust
    /// has no stable allocation hook; the first call is excluded so this measures
    /// steady state, and the bound is loose enough not to be flaky.
    #[test]
    fn repeated_uploads_reuse_scratch_rather_than_reallocating() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        // Unaligned width so the general (repacking) path is the one measured.
        let (w, h) = (1000u32, 512u32);
        let src = nv12_pattern(w, h);
        let node = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );

        // Warm up: first call touches the scratch pages for the first time.
        let first = node.upload_frame(&src, true, w, h);
        assert!(!first.contiguous);

        let mut total = std::time::Duration::ZERO;
        const N: u32 = 20;
        for _ in 0..N {
            total += node.upload_frame(&src, true, w, h).prepare;
        }
        let per_call = total / N;

        // A 1000x512 NV12 frame is ~768 KB of repacking. Allocating and zeroing
        // 768 KB per call would dominate; a memcpy of it should not approach a
        // millisecond on any machine that can run this suite.
        assert!(
            per_call < std::time::Duration::from_millis(2),
            "steady-state repack took {per_call:?} per call for a 1000x512 frame — \
             that is allocation cost, not memcpy cost"
        );
        eprintln!("steady-state repack: {per_call:?}/frame for 1000x512 NV12");
    }

    /// The `write_texture` path must land the same pixels as the staging path.
    ///
    /// The two mechanisms are genuinely different — `write_buffer` +
    /// `copy_buffer_to_texture` through a persistent buffer at the 256-aligned
    /// stride, versus `queue.write_texture` straight to the texture at the SOURCE
    /// stride, with wgpu re-striding internally. Which is faster is a measurement
    /// (Task I3), but neither is allowed to be wrong, and a stride mix-up in
    /// `write_texture_planes` is the same silent shear the staging guard protects
    /// against: `bytes_per_row` there is the unpadded source stride, and passing
    /// the padded one would offset every row.
    ///
    /// 1000 is deliberately NOT 256-aligned, so this also covers the case where
    /// wgpu must re-stride and the staging path must repack — the two arrive at
    /// the same texture by different routes.
    #[test]
    fn write_texture_path_matches_the_staging_path() {
        use crate::render::nodes::yuv_upload::UploadPath;

        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        for (w, h) in [(512u32, 64u32), (1000, 32)] {
            let src = std::sync::Arc::new(nv12_pattern(w, h));

            let staging = YuvUploadNode::new_with_layout(
                &device,
                0,
                w,
                h,
                ResourceId(2),
                ResourceId(3),
                FrameLayout::NV12,
            );
            staging.upload_frame(&src, true, w, h);
            let staging_luma = read_luma(&device, &staging, w, h);

            let mut wt = YuvUploadNode::new_with_layout(
                &device,
                0,
                w,
                h,
                ResourceId(2),
                ResourceId(3),
                FrameLayout::NV12,
            );
            wt.set_upload_path(UploadPath::WriteTexture);
            let cost = wt.upload_frame_shared(std::sync::Arc::clone(&src), true, w, h);
            assert_eq!(
                cost.prepare,
                std::time::Duration::ZERO,
                "the write_texture path performs no CPU repack of its own"
            );
            assert_eq!(
                cost.bytes,
                (w as u64 * h as u64) + (w as u64 * (h / 2) as u64),
                "bytes must count the unpadded planes actually handed to wgpu"
            );
            let wt_luma = read_luma(&device, &wt, w, h);

            for row in 0..h as usize {
                let lo = row * w as usize;
                let hi = lo + w as usize;
                assert_eq!(
                    &wt_luma[lo..hi],
                    &src[lo..hi],
                    "{w}x{h} row {row}: write_texture did not match the source"
                );
                assert_eq!(
                    &wt_luma[lo..hi],
                    &staging_luma[lo..hi],
                    "{w}x{h} row {row}: the two upload paths disagree"
                );
            }
        }
    }

    /// `upload_frame_shared` on the staging path must behave exactly like
    /// `upload_frame`, so a caller can hand over an `Arc` unconditionally and let
    /// the node decide.
    ///
    /// Without this, the shared entry point is a second code path that only the
    /// `write_texture` configuration ever exercises, and the default build would
    /// stop covering it.
    #[test]
    fn shared_upload_on_the_staging_path_is_the_ordinary_upload() {
        use crate::render::nodes::yuv_upload::UploadPath;

        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        let (w, h) = (512u32, 64u32);
        let src = std::sync::Arc::new(nv12_pattern(w, h));
        let mut node = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );
        node.set_upload_path(UploadPath::StagingBuffer);

        let cost = node.upload_frame_shared(std::sync::Arc::clone(&src), true, w, h);
        assert!(
            cost.contiguous,
            "a 512-wide full-size frame still takes the contiguous staging path"
        );
        let luma = read_luma(&device, &node, w, h);
        assert_eq!(&luma[..w as usize], &src[..w as usize]);
    }

    /// `Auto` must pick the path the measurement says wins for that node.
    ///
    /// The measurement (recorded on `UploadPath`) splits exactly on whether the
    /// staging path would have to repack: `write_texture` won every 1080p row, where
    /// 1920 pads to 2048, and lost the 4K row, where 3840 does not pad. So the rule
    /// is "repack ⇒ write_texture", and this test states it in terms of real
    /// resolutions so a future edit cannot quietly invert it.
    #[test]
    fn auto_picks_write_texture_exactly_when_staging_would_repack() {
        use crate::render::nodes::yuv_upload::UploadPath;

        // 1080p 8-bit luma: 1920 → 2048, staging repacks, so write_texture wins.
        assert_eq!(
            UploadPath::Auto.resolve(1920, round_up_256(1920)),
            UploadPath::WriteTexture,
            "1080p pads, so Auto must avoid the CPU repack"
        );
        // 4K 8-bit luma: already aligned, staging is a plain memcpy and wins.
        assert_eq!(
            UploadPath::Auto.resolve(3840, round_up_256(3840)),
            UploadPath::StagingBuffer,
            "4K needs no repack, and staging measured faster there"
        );
        // An explicit choice is never overridden — that is what makes the
        // NEXIR_UPLOAD_PATH comparison runs valid.
        assert_eq!(
            UploadPath::StagingBuffer.resolve(1920, 2048),
            UploadPath::StagingBuffer
        );
        assert_eq!(
            UploadPath::WriteTexture.resolve(3840, 3840),
            UploadPath::WriteTexture
        );
    }

    /// A node configured for `write_texture` must still render correctly when a
    /// caller uses the borrowed-slice API.
    ///
    /// THE BUG THIS PINS. `record` originally branched on `self.upload_path`, so a
    /// `WriteTexture` node returned early and never recorded the
    /// `copy_buffer_to_texture` — while `upload_frame` had faithfully filled the
    /// staging buffer. The result is a BLACK frame with no error, no warning and no
    /// failing assertion anywhere else in the suite, because every other test
    /// exercises one API or the other but never a node that serves both. The UI
    /// (`ui/src/app.rs`) and the export renderer both take the borrowed path, so on
    /// a 1080p timeline — where `Auto` selects `WriteTexture` — this would have been
    /// every preview frame.
    #[test]
    fn a_write_texture_node_still_serves_the_borrowed_slice_api() {
        use crate::render::nodes::yuv_upload::UploadPath;

        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        let (w, h) = (512u32, 64u32);
        let src = nv12_pattern(w, h);
        let mut node = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );
        node.set_upload_path(UploadPath::WriteTexture);

        // The borrowed API, on a node configured for the deferred path.
        node.upload_frame(&src, true, w, h);
        let luma = read_luma(&device, &node, w, h);
        for row in 0..h as usize {
            let lo = row * w as usize;
            assert_eq!(
                &luma[lo..lo + w as usize],
                &src[lo..lo + w as usize],
                "row {row} is wrong — a write_texture-configured node dropped a \
                 staging upload"
            );
        }

        // And planar chroma, which `upload_frame_shared` must also route to staging
        // rather than handing to `write_texture_planes`.
        let mut planar = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::YUV420P8,
        );
        planar.set_upload_path(UploadPath::WriteTexture);
        planar.upload_frame_shared(std::sync::Arc::new(src.clone()), false, w, h);
        let planar_luma = read_luma(&device, &planar, w, h);
        assert_eq!(
            &planar_luma[..w as usize],
            &src[..w as usize],
            "planar input must fall back to the staging path and still upload"
        );
    }

    /// A later borrowed upload must supersede an earlier deferred one.
    ///
    /// The mirror of the test above: with a stale `pending_frame` left in place,
    /// `record` would re-upload the OLD pixels and silently discard the new ones —
    /// a one-frame-stale picture, which looks like a sync bug rather than an upload
    /// bug and would be hunted in the wrong module.
    #[test]
    fn a_borrowed_upload_supersedes_a_pending_deferred_one() {
        use crate::render::nodes::yuv_upload::UploadPath;

        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };

        let (w, h) = (512u32, 64u32);
        let first = std::sync::Arc::new(nv12_pattern(w, h));
        // A visibly different second frame: every luma sample inverted.
        let second: Vec<u8> = first.iter().map(|b| 255 - b).collect();

        let mut node = YuvUploadNode::new_with_layout(
            &device,
            0,
            w,
            h,
            ResourceId(2),
            ResourceId(3),
            FrameLayout::NV12,
        );
        node.set_upload_path(UploadPath::WriteTexture);

        node.upload_frame_shared(std::sync::Arc::clone(&first), true, w, h);
        node.upload_frame(&second, true, w, h);

        let luma = read_luma(&device, &node, w, h);
        assert_eq!(
            &luma[..w as usize],
            &second[..w as usize],
            "the most recent upload must be the one that reaches the texture"
        );
        assert_ne!(
            &luma[..w as usize],
            &first[..w as usize],
            "fixture is wrong: the two frames must differ for this to test anything"
        );
    }
}
