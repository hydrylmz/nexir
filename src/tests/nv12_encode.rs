// src/tests/nv12_encode.rs
// P1.9 (step 1) — RGB→NV12 encode shader correctness.
//
// SCOPE: this file validates ONE link of the new zero-copy export chain:
//
//     WGPU RGBA16Float texture  --[Nv12EncodeNode]-->  pitched NV12 in a buffer
//
// It proves the colour conversion, the 4:2:0 chroma siting, the two-plane byte
// layout, and the pitch handling, bit-exactly against the independent CPU
// reference in `colour::yuv::RgbToYuv` plus hardcoded codes that came from a
// standalone C probe the hardware itself agreed with.
//
// It deliberately proves NOTHING about the links that follow: the D3D12 shared
// buffer allocation, `cuExternalMemoryGetMappedBuffer`, NVENC registration as
// NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, or whether an exported file decodes
// to the expected pixels.  Those are the next steps.  A green run here must NOT
// be reported as "the NV12 zero-copy path is correct".
//
// Structured to mirror `src/tests/abgr10_repack.rs`, which does the same job for
// the packed-RGB shader this one replaces.
//
// Requires a working GPU (headless wgpu device), like the other tests in
// src/tests/.  Requires no CUDA, no NVENC session, and no export job.

#[cfg(test)]
mod nv12_encode {
    use crate::colour::yuv::RgbToYuv;
    use crate::interop::nv12_encode::Nv12EncodeNode;
    use crate::render::device::GpuDevice;
    use crate::timeline::source::{
        ColorInfo, ColorPrimaries, ColorRange, MatrixCoefficients, TransferFunction,
    };
    use half::f16;

    fn info(matrix: MatrixCoefficients, range: ColorRange) -> ColorInfo {
        ColorInfo {
            transfer_fn: TransferFunction::Bt709,
            range,
            matrix,
            primaries: ColorPrimaries::Bt709,
            bit_depth: 8,
        }
    }

    /// An NV12 frame read back from the GPU, with the row padding still in place
    /// so the tests can assert on what the padding contains.
    struct Nv12Readback {
        bytes:  Vec<u8>,
        width:  u32,
        height: u32,
        pitch:  u32,
    }

    impl Nv12Readback {
        fn luma(&self, x: u32, y: u32) -> u8 {
            self.bytes[(y * self.pitch + x) as usize]
        }

        /// `(Cb, Cr)` at a chroma-plane coordinate (half resolution).
        fn chroma(&self, cx: u32, cy: u32) -> (u8, u8) {
            let base = (self.pitch * self.height + cy * self.pitch + cx * 2) as usize;
            (self.bytes[base], self.bytes[base + 1])
        }

        fn chroma_dims(&self) -> (u32, u32) {
            (self.width.div_ceil(2), self.height.div_ceil(2))
        }
    }

    /// Run `Nv12EncodeNode` over a `width` x `height` RGBA16Float image built from
    /// `pixels` (row-major, one (r,g,b) triple per pixel) and return the NV12
    /// bytes.
    ///
    /// Constructing the node is itself part of the assertion: it proves both
    /// compute pipelines, the `texture_storage_2d<rgba16float, read>` binding, the
    /// storage-buffer binding and the 96-byte push-constant block compile and
    /// validate on this adapter under wgpu 0.19.  In production that only happens
    /// partway into an export, where failure reaches the user as a panic.
    fn run_encode(
        device: &GpuDevice,
        width:  u32,
        height: u32,
        pitch:  u32,
        color:  ColorInfo,
        pixels: &[(f32, f32, f32)],
    ) -> Nv12Readback {
        assert_eq!(
            pixels.len(),
            (width * height) as usize,
            "test bug: pixel count does not match {width}x{height}"
        );

        let in_texture = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nv12_encode test input"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut in_bytes = Vec::<u8>::with_capacity(pixels.len() * 8);
        for &(r, g, b) in pixels {
            for component in [r, g, b, 1.0f32] {
                in_bytes.extend_from_slice(&f16::from_f32(component).to_le_bytes());
            }
        }

        device.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &in_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &in_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        let size = Nv12EncodeNode::buffer_size(height, pitch);
        let nv12 = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_encode test output"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nv12_encode test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let node = Nv12EncodeNode::new(device, color, width, height);
        let in_view = in_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nv12_encode test"),
            });
        node.record(&mut encoder, device, &in_view, &nv12, pitch);
        encoder.copy_buffer_to_buffer(&nv12, 0, &readback, 0, size);
        device.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let bytes = slice.get_mapped_range().to_vec();
        readback.unmap();

        Nv12Readback { bytes, width, height, pitch }
    }

    /// The headline check: every pixel of a flat colour must equal what the CPU
    /// reference produces, and the hardcoded codes must match the values a
    /// standalone C probe fed to NVENC and read back out of the decoded file
    /// (`nvchk/nv12_probe.c`).
    ///
    /// The hardcoded half matters: it catches the case where `RgbToYuv` and the
    /// shader are wrong in the same direction, which a pure
    /// reference-vs-shader comparison cannot.
    #[test]
    fn encode_matches_the_cpu_reference_and_known_codes() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        // (label, rgb, expected Y/U/V) — the same four bars the C probe used.
        let cases: [(&str, (f32, f32, f32), [u8; 3]); 4] = [
            ("red",   (1.0, 0.0, 0.0), [63,  102, 240]),
            ("green", (0.0, 1.0, 0.0), [173, 42,  26]),
            ("blue",  (0.0, 0.0, 1.0), [32,  240, 118]),
            ("white", (1.0, 1.0, 1.0), [235, 128, 128]),
        ];

        let color = info(MatrixCoefficients::Bt709, ColorRange::Limited);
        let reference = RgbToYuv::new(color, 8, 8);

        for (label, rgb, want) in cases {
            // A flat 8x8 frame: every luma pixel and every chroma pair must carry
            // the same codes, so a shader that mis-sites chroma or drops a plane
            // fails here rather than needing a gradient to show it.
            let pixels = vec![rgb; 64];
            let got = run_encode(&device, 8, 8, 8, color, &pixels);

            let cpu = reference.apply_u8(rgb.0, rgb.1, rgb.2);
            assert_eq!(
                cpu, want,
                "{label}: the CPU reference disagrees with the hardware-verified \
                 codes before the GPU is even involved — got {cpu:?}, expected {want:?}"
            );

            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(
                        got.luma(x, y), want[0],
                        "{label}: luma at ({x},{y}) is {}, expected {}",
                        got.luma(x, y), want[0]
                    );
                }
            }
            let (cw, ch) = got.chroma_dims();
            for cy in 0..ch {
                for cx in 0..cw {
                    let (cb, cr) = got.chroma(cx, cy);
                    assert_eq!(
                        (cb, cr), (want[1], want[2]),
                        "{label}: chroma at ({cx},{cy}) is ({cb},{cr}), expected \
                         ({},{})", want[1], want[2]
                    );
                }
            }
        }
    }

    /// Every matrix and range combination must agree with the CPU reference.
    ///
    /// This is what proves the shader reads the push-constant block correctly: if
    /// it ignored the matrix rows and hardcoded BT.709, the BT.601 and BT.2020
    /// cases would fail.  If it ignored the range scalars, the full-range cases
    /// would.
    #[test]
    fn every_matrix_and_range_matches_the_reference() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        // A colour with three distinct channels, so a transposed matrix row shows
        // up as a wrong value rather than a coincidental match.
        let rgb = (0.75f32, 0.25, 0.5);

        for matrix in [
            MatrixCoefficients::Bt601,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Bt2020,
        ] {
            for range in [ColorRange::Limited, ColorRange::Full] {
                let color = info(matrix, range);
                let pixels = vec![rgb; 16];
                let got = run_encode(&device, 4, 4, 4, color, &pixels);
                let want = RgbToYuv::new(color, 4, 4).apply_u8(rgb.0, rgb.1, rgb.2);

                assert_eq!(
                    got.luma(0, 0), want[0],
                    "{matrix:?}/{range:?}: luma {} != reference {}",
                    got.luma(0, 0), want[0]
                );
                let (cb, cr) = got.chroma(0, 0);
                assert_eq!(
                    (cb, cr), (want[1], want[2]),
                    "{matrix:?}/{range:?}: chroma ({cb},{cr}) != reference ({},{})",
                    want[1], want[2]
                );
            }
        }
    }

    /// Different matrices must produce measurably different bytes.
    ///
    /// Without this, a shader that ignored the push constants entirely could pass
    /// `every_matrix_and_range_matches_the_reference` if the reference were also
    /// broken in the same way. This asserts the output actually moves.
    #[test]
    fn matrix_choice_changes_the_encoded_bytes() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        let rgb = (0.8f32, 0.3, 0.15);
        let pixels = vec![rgb; 16];

        let luma_for = |matrix| {
            run_encode(&device, 4, 4, 4, info(matrix, ColorRange::Limited), &pixels)
                .luma(0, 0)
        };
        let bt601  = luma_for(MatrixCoefficients::Bt601);
        let bt709  = luma_for(MatrixCoefficients::Bt709);
        let bt2020 = luma_for(MatrixCoefficients::Bt2020);

        assert!(
            bt601.abs_diff(bt709) > 2,
            "BT.601 luma {bt601} and BT.709 luma {bt709} must differ — the shader \
             is ignoring the matrix in its push constants"
        );
        assert!(
            bt709.abs_diff(bt2020) > 2,
            "BT.709 luma {bt709} and BT.2020 luma {bt2020} must differ"
        );
    }

    /// Chroma must be the box average of its 2x2 luma block, not a point sample.
    ///
    /// A half-white/half-black checkerboard makes the two indistinguishable in
    /// luma but far apart in chroma: point-sampling the top-left pixel of each
    /// block gives a different answer from averaging all four.
    #[test]
    fn chroma_is_the_average_of_its_2x2_block() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        // One 2x2 block: red, green / blue, white.  Its average is a mid grey-ish
        // colour whose chroma differs from any of the four corners.
        let pixels = vec![
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 1.0, 1.0),
        ];
        let color = info(MatrixCoefficients::Bt709, ColorRange::Limited);
        let got = run_encode(&device, 2, 2, 4, color, &pixels);

        let reference = RgbToYuv::new(color, 2, 2);
        let avg = (
            (1.0 + 0.0 + 0.0 + 1.0) / 4.0,
            (0.0 + 1.0 + 0.0 + 1.0) / 4.0,
            (0.0 + 0.0 + 1.0 + 1.0) / 4.0,
        );
        let want = reference.apply_u8(avg.0, avg.1, avg.2);
        let (cb, cr) = got.chroma(0, 0);
        assert_eq!(
            (cb, cr), (want[1], want[2]),
            "chroma ({cb},{cr}) is not the block average ({},{}) — the shader is \
             point-sampling instead of averaging",
            want[1], want[2]
        );

        // And a point sample of the top-left pixel would give something else, so
        // the assertion above has teeth.
        let point = reference.apply_u8(1.0, 0.0, 0.0);
        assert_ne!(
            (want[1], want[2]), (point[1], point[2]),
            "test bug: this pattern cannot distinguish averaging from point sampling"
        );

        // Luma is per-pixel and must keep all four distinct values.
        for (i, &(r, g, b)) in pixels.iter().enumerate() {
            let (x, y) = (i as u32 % 2, i as u32 / 2);
            let want_y = reference.apply_u8(r, g, b)[0];
            assert_eq!(
                got.luma(x, y), want_y,
                "luma at ({x},{y}) is {}, expected {want_y} — luma must stay full \
                 resolution", got.luma(x, y)
            );
        }
    }

    /// A pitch wider than the frame must not shift the image or corrupt the
    /// chroma plane's position.
    ///
    /// This is the layout property the C probe verified against the driver
    /// (pitch=320 with width=256): every row starts at `y * pitch`, and chroma
    /// starts at `pitch * height`. If the shader used `width` anywhere instead,
    /// rows would shear and the chroma plane would land inside the luma plane.
    ///
    /// Width 6 with pitch 16 is chosen so the row's last word STRADDLES the frame
    /// edge (word 1 covers pixels 4..7, of which 6 and 7 are outside), while words
    /// 2 and 3 lie entirely outside. That distinction is the padding contract
    /// asserted at the end.
    #[test]
    fn wider_pitch_keeps_the_layout_and_leaves_padding_alone() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        const W: u32 = 6;
        const H: u32 = 4;
        const PITCH: u32 = 16; // deliberately wider than the 8-byte minimum

        // A vertical gradient, so a row written at the wrong offset produces a
        // visibly wrong value rather than matching its neighbour.
        let mut pixels = Vec::new();
        for y in 0..H {
            for _ in 0..W {
                let v = y as f32 / (H - 1) as f32;
                pixels.push((v, v, v));
            }
        }

        let color = info(MatrixCoefficients::Bt709, ColorRange::Limited);
        let got = run_encode(&device, W, H, PITCH, color, &pixels);
        let reference = RgbToYuv::new(color, W, H);

        for y in 0..H {
            let v = y as f32 / (H - 1) as f32;
            let want = reference.apply_u8(v, v, v)[0];
            for x in 0..W {
                assert_eq!(
                    got.luma(x, y), want,
                    "luma at ({x},{y}) with pitch {PITCH} is {}, expected {want} — \
                     rows are not being written at y*pitch",
                    got.luma(x, y)
                );
            }
        }

        // The chroma plane must be exactly at pitch*height and carry neutral
        // chroma for this greyscale input; finding luma-looking values here would
        // mean the plane offset was computed from width.
        let (cw, ch) = got.chroma_dims();
        for cy in 0..ch {
            for cx in 0..cw {
                let (cb, cr) = got.chroma(cx, cy);
                assert_eq!(
                    (cb, cr), (128, 128),
                    "chroma at ({cx},{cy}) is ({cb},{cr}), expected neutral (128,128) \
                     for greyscale input — the chroma plane is not at pitch*height"
                );
            }
        }

        // The padding contract, which is a real property of the shader and worth
        // pinning because it bounds how much of the buffer is written:
        //
        //  * Bytes in the LAST STRADDLING WORD (here x = 6, 7) get the replicated
        //    edge pixel. They have to be written — the word also holds real pixels
        //    4 and 5, and a partial word cannot be stored.
        //  * Bytes in words ENTIRELY past the width (here x = 8..16) are never
        //    written at all, because the dispatch guard drops those invocations.
        //    They stay at whatever the buffer held, which for a fresh wgpu buffer
        //    is zero.
        //
        // Both are fine for NVENC: it reads `width` pixels per row and ignores the
        // rest of the stride. What would NOT be fine is writing past the row into
        // the next one, which the luma assertions above rule out.
        for y in 0..H {
            let edge = got.luma(W - 1, y);
            for x in W..8 {
                assert_eq!(
                    got.luma(x, y), edge,
                    "row {y} byte {x} is in the straddling word and must hold the \
                     replicated edge pixel {edge}, got {}", got.luma(x, y)
                );
            }
            for x in 8..PITCH {
                assert_eq!(
                    got.luma(x, y), 0,
                    "row {y} byte {x} lies in a word entirely past the frame and \
                     must never be written, got {}", got.luma(x, y)
                );
            }
        }
    }

    /// Dimensions that are not multiples of the dispatch granularity must still
    /// come out right.
    ///
    /// 5x3 exercises three separate edge cases at once: the luma dispatch is 4
    /// pixels wide per invocation (so the last invocation is 3/4 out of bounds),
    /// the workgroup is 8x8 (so most threads are outside the image), and an odd
    /// width and height mean the chroma plane's last column and row cover only
    /// one real pixel instead of four.
    #[test]
    fn odd_dimensions_are_encoded_completely() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        const W: u32 = 5;
        const H: u32 = 3;
        let pitch = Nv12EncodeNode::min_pitch(W); // 8

        let rgb = (0.25f32, 0.5, 0.75);
        let pixels = vec![rgb; (W * H) as usize];
        let color = info(MatrixCoefficients::Bt709, ColorRange::Limited);

        let got = run_encode(&device, W, H, pitch, color, &pixels);
        let want = RgbToYuv::new(color, W, H).apply_u8(rgb.0, rgb.1, rgb.2);

        for y in 0..H {
            for x in 0..W {
                assert_eq!(
                    got.luma(x, y), want[0],
                    "luma at ({x},{y}) of {W}x{H}: got {}, expected {} — the \
                     unaligned dispatch dropped or corrupted a pixel",
                    got.luma(x, y), want[0]
                );
            }
        }

        // ceil(5/2) x ceil(3/2) = 3x2 chroma samples; the last column and row are
        // the clamped ones. A flat colour makes every one of them equal.
        let (cw, ch) = got.chroma_dims();
        assert_eq!((cw, ch), (3, 2), "chroma dims for {W}x{H} must round up");
        for cy in 0..ch {
            for cx in 0..cw {
                let (cb, cr) = got.chroma(cx, cy);
                assert_eq!(
                    (cb, cr), (want[1], want[2]),
                    "chroma at ({cx},{cy}) of {W}x{H}: got ({cb},{cr}), expected \
                     ({},{}) — the clamped edge block is wrong",
                    want[1], want[2]
                );
            }
        }
    }

    /// Out-of-range input must saturate on the gamut boundary, not wrap.
    ///
    /// The clamp lives in RGB, before the matrix, so super-white encodes to legal
    /// white (235) rather than reaching 255 and occupying codes limited range
    /// reserves. This is the GPU-side counterpart of
    /// `colour::yuv::tests::out_of_range_input_is_clamped_not_wrapped`.
    #[test]
    fn out_of_range_input_saturates_in_the_shader_too() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        let color = info(MatrixCoefficients::Bt709, ColorRange::Limited);
        // f16 holds these exactly; 4.0 and -2.0 are well outside [0,1].
        let pixels = vec![(4.0f32, 4.0, 4.0); 16];
        let bright = run_encode(&device, 4, 4, 4, color, &pixels);
        assert_eq!(
            bright.luma(0, 0), 235,
            "super-white must clamp to legal white 235, got {}",
            bright.luma(0, 0)
        );
        assert_eq!(
            bright.chroma(0, 0), (128, 128),
            "clamped white must stay neutral, got {:?}", bright.chroma(0, 0)
        );

        let pixels = vec![(-2.0f32, -2.0, -2.0); 16];
        let dark = run_encode(&device, 4, 4, 4, color, &pixels);
        assert_eq!(
            dark.luma(0, 0), 16,
            "sub-black must clamp to legal black 16, got {}", dark.luma(0, 0)
        );

        // Out of range in one channel only: must match legal pure red exactly,
        // proving the clamp is applied in RGB and not to Y/Cb/Cr independently
        // (which would shift the hue).
        let over_red = run_encode(&device, 4, 4, 4, color, &[(3.0f32, 0.0, 0.0); 16]);
        let pure_red = run_encode(&device, 4, 4, 4, color, &[(1.0f32, 0.0, 0.0); 16]);
        assert_eq!(
            (over_red.luma(0, 0), over_red.chroma(0, 0)),
            (pure_red.luma(0, 0), pure_red.chroma(0, 0)),
            "over-bright red must encode identically to pure red — the clamp is \
             happening after the matrix, which shifts hue"
        );
    }
}
