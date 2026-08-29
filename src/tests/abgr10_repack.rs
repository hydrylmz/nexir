// src/tests/abgr10_repack.rs
// P1.4 (execution-order item 1) — ABGR10 repack shader correctness.
//
// SCOPE: this file validates ONE link of the zero-copy export chain:
//
//     WGPU RGBA16Float texture  --[Abgr10RepackNode]-->  WGPU R32Uint texture
//
// It proves the packing arithmetic and channel positions the shader produces,
// bit-exactly, against an independent CPU reference plus hardcoded sentinels.
//
// It deliberately proves NOTHING about the links that follow: CUDA external
// memory import, CUDA array format, NVENC's interpretation of
// NV_ENC_BUFFER_FORMAT_ABGR10, input pitch semantics, or whether an exported
// file decodes to the expected pixels.  Those are separate validation steps.
// A green run here must NOT be reported as "the zero-copy path is correct".
//
// Requires a working GPU (headless wgpu device), like the other tests in
// src/tests/.  Requires no CUDA, no NVENC session, and no export job.

#[cfg(test)]
mod abgr10_repack {
    use crate::interop::encode_interop::Abgr10RepackNode;
    use crate::render::device::GpuDevice;
    use crate::render::resource::ResourceId;
    use half::f16;

    /// wgpu's COPY_BYTES_PER_ROW_ALIGNMENT: `copy_texture_to_buffer` requires
    /// `bytes_per_row` to be a multiple of 256, so readback rows are padded.
    const COPY_ALIGN: u32 = 256;

    /// Independent CPU reference for the packing the shader performs:
    ///
    ///   packed = (a2 << 30) | (b10 << 20) | (g10 << 10) | r10
    ///   channel10 = round(clamp(v, 0.0, 1.0) * 1023.0)
    ///   a2 = 0b11 (always fully opaque)
    ///
    /// Written from the ABGR10 layout contract documented at
    /// src/interop/encode_interop.rs:15, NOT by transcribing the WGSL, so that
    /// it can disagree with the shader.  The hardcoded sentinels in
    /// `repack_packs_known_colours_bit_exact` guard against this reference and
    /// the shader being wrong in the same direction.
    fn expected_pack(r: f32, g: f32, b: f32) -> u32 {
        let q = |v: f32| -> u32 { (v.clamp(0.0, 1.0) * 1023.0 + 0.5) as u32 };
        (3u32 << 30) | (q(b) << 20) | (q(g) << 10) | q(r)
    }

    /// Run `Abgr10RepackNode` over a `width` x `height` RGBA16Float image built
    /// from `pixels` (row-major, one (r,g,b) triple per pixel) and return the
    /// packed R32Uint words, row-major, unpadded.
    ///
    /// Constructing the node is itself part of the assertion: it proves the
    /// `texture_storage_2d<rgba16float, read>` binding and the compute pipeline
    /// compile and validate on this adapter under wgpu 0.19.  In production that
    /// only happens partway into an export, where failure reaches the user as a
    /// panic.
    fn run_repack(
        device: &GpuDevice,
        width: u32,
        height: u32,
        pixels: &[(f32, f32, f32)],
    ) -> Vec<u32> {
        assert_eq!(
            pixels.len(),
            (width * height) as usize,
            "test bug: pixel count does not match {width}x{height}"
        );

        // --- Input: RGBA16Float, read-only storage + copy destination. ---
        let in_texture = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("abgr10_repack test input"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Pack the source pixels as f16 RGBA (alpha 1.0; the shader ignores it
        // and hardcodes a2 = 0b11, which the assertions verify).
        let mut in_bytes = Vec::<u8>::with_capacity(pixels.len() * 8);
        for &(r, g, b) in pixels {
            for component in [r, g, b, 1.0f32] {
                in_bytes.extend_from_slice(&f16::from_f32(component).to_le_bytes());
            }
        }

        // queue.write_texture has no bytes_per_row alignment requirement.
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
                bytes_per_row: Some(width * 8), // 4 channels x 2 bytes
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // --- Output: R32Uint, matching the production texture at
        // src/interop/encode_interop.rs:544 (STORAGE_BINDING | COPY_SRC). ---
        let out_texture = device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("abgr10_repack test output"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // The node's ResourceIds are bookkeeping only — `record` takes views
        // directly — so any distinct pair is valid here.
        let node = Abgr10RepackNode::new(device, ResourceId::FINAL_COLOR, ResourceId(1000));

        let in_view = in_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let out_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Padded readback buffer: each row rounded up to COPY_ALIGN bytes.
        let unpadded_bpr = width * 4;
        let padded_bpr = unpadded_bpr.div_ceil(COPY_ALIGN) * COPY_ALIGN;
        let readback = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("abgr10_repack test readback"),
            size: (padded_bpr * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("abgr10_repack test") },
        );
        node.record(&mut encoder, device, &in_view, &out_view, width, height);
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &out_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        device.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range();

        // Strip row padding.
        let mut out = Vec::<u32>::with_capacity((width * height) as usize);
        for y in 0..height {
            let row_start = (y * padded_bpr) as usize;
            for x in 0..width {
                let off = row_start + (x * 4) as usize;
                out.push(u32::from_le_bytes([
                    mapped[off],
                    mapped[off + 1],
                    mapped[off + 2],
                    mapped[off + 3],
                ]));
            }
        }
        drop(mapped);
        readback.unmap();
        out
    }

    /// Validates R/G/B/A channel positions, 10-bit quantization, clamping, and
    /// the WGPU R32Uint representation against known colours.
    ///
    /// All input components are values exactly representable in f16 (0, 1, 0.5,
    /// 0.25, 0.75) so the expected 10-bit results are unambiguous and this test
    /// cannot flake on rounding.
    #[test]
    fn repack_packs_known_colours_bit_exact() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        // (label, r, g, b)
        let cases: [(&str, f32, f32, f32); 8] = [
            ("black",           0.0,  0.0,  0.0),
            ("white",           1.0,  1.0,  1.0),
            ("red",             1.0,  0.0,  0.0),
            ("green",           0.0,  1.0,  0.0),
            ("blue",            0.0,  0.0,  1.0),
            ("50% gray",        0.5,  0.5,  0.5),
            ("out-of-range",    2.0, -1.0,  0.5), // must clamp to 1023 / 0 / 512
            ("asymmetric",     0.25, 0.75,  1.0),
        ];
        let pixels: Vec<(f32, f32, f32)> =
            cases.iter().map(|&(_, r, g, b)| (r, g, b)).collect();

        let width = cases.len() as u32;
        let got = run_repack(&device, width, 1, &pixels);

        // Independent-reference comparison.
        for (i, &(label, r, g, b)) in cases.iter().enumerate() {
            let expected = expected_pack(r, g, b);
            assert_eq!(
                got[i], expected,
                "pixel {i} ({label}): repack produced 0x{:08X}, expected 0x{:08X} \
                 (r10={} g10={} b10={} a2=3)",
                got[i],
                expected,
                expected & 0x3FF,
                (expected >> 10) & 0x3FF,
                (expected >> 20) & 0x3FF,
            );
        }

        // Hardcoded sentinels, derived by hand from the ABGR10 bit layout.
        // These catch the case where `expected_pack` and the shader are wrong
        // in the same way (e.g. both swapping R and B).
        assert_eq!(
            got[0], 0xC000_0000,
            "black must pack to a2=3, b=g=r=0; got 0x{:08X}", got[0]
        );
        assert_eq!(
            got[1], 0xFFFF_FFFF,
            "white must pack to all-ones (a2=3, b=g=r=1023); got 0x{:08X}", got[1]
        );
        assert_eq!(
            got[2], 0xC000_03FF,
            "red must place 1023 in the LOW 10 bits; got 0x{:08X}", got[2]
        );
        assert_eq!(
            got[3], 0xC00F_FC00,
            "green must place 1023 in bits 10..19; got 0x{:08X}", got[3]
        );
        assert_eq!(
            got[4], 0xFFF0_0000,
            "blue must place 1023 in bits 20..29; got 0x{:08X}", got[4]
        );
        // a2=3 (0xC000_0000) | b=512 (0x2000_0000) | g=512 (0x0008_0000) | r=512 (0x200)
        assert_eq!(
            got[5], 0xE008_0200,
            "50% gray must quantize to 512 in every channel; got 0x{:08X}", got[5]
        );

        // Alpha must be 0b11 on every pixel, including the clamped one.
        for (i, &word) in got.iter().enumerate() {
            assert_eq!(
                word >> 30, 3,
                "pixel {i}: 2-bit alpha must be 0b11 (opaque), got {}", word >> 30
            );
        }
    }

    /// The dispatch is 8x8 workgroups (`width.div_ceil(8)`), so a 5x3 image
    /// launches threads outside the image.  This verifies the
    /// `gid >= dims` early-return guard at src/interop/encode_interop.rs:33
    /// and that every in-bounds pixel is still written.
    ///
    /// This is the cheap half of P1.5's "widths not aligned to common GPU pitch
    /// boundaries"; it says nothing about NVENC's pitch handling.
    #[test]
    fn repack_handles_unaligned_dimensions() {
        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");

        const W: u32 = 5;
        const H: u32 = 3;

        // A colour with all three channels distinct, so a transposed or
        // mis-strided write shows up as a wrong value rather than a match.
        let colour = (0.25f32, 0.5f32, 0.75f32);
        let pixels = vec![colour; (W * H) as usize];

        let got = run_repack(&device, W, H, &pixels);
        let expected = expected_pack(colour.0, colour.1, colour.2);

        assert_eq!(got.len(), (W * H) as usize, "wrong number of words read back");
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                assert_eq!(
                    got[i], expected,
                    "pixel ({x},{y}) at {W}x{H}: got 0x{:08X}, expected 0x{:08X} \
                     — unaligned dispatch dropped or corrupted a pixel",
                    got[i], expected
                );
            }
        }
    }
}
