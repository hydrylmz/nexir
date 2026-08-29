// src/tests/shared_buffer.rs
// P1.9 (step 2) — SharedBuffer: one D3D12 allocation, two APIs.
//
// SCOPE: this file validates ONE link of the NV12 zero-copy export chain:
//
//     wgpu::Buffer  ==  the same bytes as  ==  CUdeviceptr
//
// It proves that `SharedBuffer::new` produces a single allocation both APIs
// address, in both directions (write via wgpu → read via CUDA, and the reverse),
// and that `Nv12EncodeNode` writing into that buffer lands byte-identical bytes
// where CUDA sees them.
//
// It deliberately proves NOTHING about the links that follow: NVENC registration
// as NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, `NV_ENC_PIC_PARAMS::inputPitch`,
// or whether an exported file decodes to the expected pixels.  Those are the next
// steps.  A green run here must NOT be reported as "the NV12 zero-copy path is
// correct".
//
// HARDWARE GATE.  Unlike `nv12_encode.rs`, these tests need CUDA, not just a GPU.
// The gate is a property of the HOST (`InteropCapability::probe`), checked once;
// past it every test asserts and fails rather than skipping — a self-probe that
// opened a second CUDA context would itself create the order dependence the skill
// warns about.  Set `REQUIRE_CUDA=1` to turn the host gate itself into a failure
// on a machine that is supposed to have the hardware.
//
// The aliasing assertion is the whole point of the file and is worth stating
// plainly: `assert_eq!(cuda_bytes, wgpu_bytes)` on data BOTH sides merely read
// proves nothing — two separate allocations holding the same pattern pass it.
// Every test here writes through one API and reads through the OTHER.

#[cfg(test)]
mod shared_buffer {
    use crate::colour::yuv::RgbToYuv;
    use crate::interop::capability::{InteropCapability, InteropTransport};
    use crate::interop::cuda_context::CudaContext;
    use crate::interop::external_buffer::SharedBuffer;
    use crate::interop::nv12_encode::Nv12EncodeNode;
    use crate::render::device::GpuDevice;
    use crate::timeline::source::{
        ColorInfo, ColorPrimaries, ColorRange, MatrixCoefficients, TransferFunction,
    };
    use half::f16;
    use std::sync::Arc;

    /// Everything the tests need, or a reason the host cannot provide it.
    ///
    /// Carries the process-wide CUDA lock (see [`crate::tests::cuda_lock`]).
    /// `_cuda_guard` is declared LAST so it is dropped last: fields drop in
    /// declaration order, and releasing the lock before the `Arc<CudaContext>` and
    /// the device are gone would let the next test push the context while this
    /// one's teardown is still calling into the driver.
    struct Harness {
        device:   GpuDevice,
        cuda_ctx: Arc<CudaContext>,
        transport: InteropTransport,
        _cuda_guard: std::sync::MutexGuard<'static, ()>,
    }

    /// One CUDA primary context for the whole test binary.
    ///
    /// Retained once and never released, for the reason `export_validation.rs`
    /// documents: tests run in parallel threads, and a context that dies when one
    /// test finishes makes an unrelated test fail for reasons that have nothing to
    /// do with the code under test.
    fn shared_cuda_ctx(capability: &InteropCapability) -> Option<Arc<CudaContext>> {
        static CUDA: std::sync::OnceLock<Option<Arc<CudaContext>>> = std::sync::OnceLock::new();
        CUDA.get_or_init(|| match CudaContext::new(capability) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                eprintln!("[shared_buffer] CudaContext::new failed: {e:?}");
                None
            }
        })
        .clone()
    }

    /// Build the harness, or explain why this host cannot run these tests.
    ///
    /// Returns `None` only for a HOST property: no NVIDIA GPU, no CUDA driver, no
    /// D3D12/Vulkan interop transport.  `REQUIRE_CUDA=1` turns that into a panic
    /// so a machine that should have the hardware fails loudly instead of
    /// reporting a green suite that exercised nothing.
    fn harness() -> Option<Harness> {
        // Taken FIRST, before the wgpu device exists: two headless devices plus two
        // CUDA imports on one GPU is the resource fight this lock exists to avoid,
        // and `CudaContext::with_context` cannot be pushed on two threads at once.
        let cuda_guard = crate::tests::cuda_lock();

        let device = pollster::block_on(GpuDevice::new_headless())
            .expect("failed to create headless GpuDevice");
        let capability = InteropCapability::probe(&device);

        let bail = |reason: String| -> Option<Harness> {
            if std::env::var("REQUIRE_CUDA").is_ok() {
                panic!("REQUIRE_CUDA is set but {reason}");
            }
            eprintln!(
                "[shared_buffer] SKIPPING — {reason}.  The CUDA external-memory buffer path \
                 was NOT exercised by this run."
            );
            None
        };

        if !capability.is_available() {
            return bail(format!(
                "CUDA interop is unavailable (transport={:?}, driver='{}')",
                capability.transport, capability.driver_version
            ));
        }
        if capability.transport != InteropTransport::D3D12Win32Handle {
            // SharedBuffer's Vulkan arm returns an error by design (GpuDevice does
            // not enable VK_KHR_external_memory_fd), so on a Vulkan host there is
            // nothing to test rather than something failing.
            return bail(format!(
                "transport is {:?}; SharedBuffer only implements D3D12Win32Handle",
                capability.transport
            ));
        }
        let cuda_ctx = match shared_cuda_ctx(&capability) {
            Some(c) => c,
            None => return bail("a CUDA context could not be created".into()),
        };

        Some(Harness { device, cuda_ctx, transport: capability.transport, _cuda_guard: cuda_guard })
    }

    fn info(matrix: MatrixCoefficients, range: ColorRange) -> ColorInfo {
        ColorInfo {
            transfer_fn: TransferFunction::Bt709,
            range,
            matrix,
            primaries: ColorPrimaries::Bt709,
            bit_depth: 8,
        }
    }

    /// Read a `wgpu::Buffer` back through wgpu, via a staging copy.
    ///
    /// The shared buffer itself cannot be MAP_READ: it lives on a
    /// `HEAP_TYPE_DEFAULT` heap, which is exactly what makes it shareable, so the
    /// wgpu-side read has to go through a copy the way production readback does.
    fn read_via_wgpu(device: &GpuDevice, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared_buffer test staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shared_buffer test readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        device.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        let bytes = slice.get_mapped_range().to_vec();
        staging.unmap();
        bytes
    }

    /// Write a `wgpu::Buffer` through wgpu and make sure the write has actually
    /// reached the GPU before returning.
    ///
    /// `queue.write_buffer` does NOT write anything by itself: wgpu 0.19 copies
    /// the data into a staging buffer held in the device's `pending_writes`, and
    /// those are flushed by the next `queue.submit` — `poll(Maintain::Wait)` waits
    /// for submitted work and does not flush them.  So `write_buffer` followed by
    /// `poll` reads back zeroes, which is exactly what these tests saw first:
    /// "61200/61440 bytes differ, first at offset 0 (got 0x00)" on a buffer that
    /// really was shared.  The empty submit is what makes the write happen.
    fn write_via_wgpu(device: &GpuDevice, buffer: &wgpu::Buffer, bytes: &[u8]) {
        device.queue.write_buffer(buffer, 0, bytes);
        device.queue.submit(std::iter::empty());
        device.device.poll(wgpu::Maintain::Wait);
    }

    /// `Result::expect_err` needs `T: Debug`, and `SharedBuffer` deliberately has
    /// no `Debug` (printing a raw device pointer and an NT handle in a log is not
    /// useful and invites treating them as stable identifiers).  So unwrap the
    /// error side by hand.
    fn expect_err(result: Result<SharedBuffer, crate::interop::cuda_context::CudaError>, what: &str)
        -> crate::interop::cuda_context::CudaError
    {
        match result {
            Ok(_)  => panic!("{what}"),
            Err(e) => e,
        }
    }

    /// A byte pattern with no short period, so a mapping that is off by a few
    /// bytes or wrapped at a row boundary cannot pass by coincidence.
    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| ((i as u32).wrapping_mul(31).wrapping_add(seed as u32) & 0xFF) as u8)
            .collect()
    }

    /// Report the first mismatch rather than just the count: "differ at 40960" is
    /// a plane offset and names the bug, where "8192 bytes differ" does not.
    fn assert_bytes_eq(got: &[u8], want: &[u8], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length {} != {}", got.len(), want.len());
        if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
            let differing = (0..got.len()).filter(|&i| got[i] != want[i]).count();
            panic!(
                "{what}: {differing}/{} bytes differ; first at offset {i} \
                 (got 0x{:02X}, expected 0x{:02X})",
                got.len(), got[i], want[i]
            );
        }
    }

    /// 61440 bytes = the 256x128 NV12 frame the C probes used, so a failure here
    /// can be compared directly against `nvchk/extbuf_probe.c buf` output.
    const PROBE_SIZE: u64 = 61440;

    /// Allocation succeeds and reports the geometry it was asked for.
    ///
    /// Deliberately the smallest possible test: `SharedBuffer::new` is three
    /// fallible native calls deep (CreateCommittedResource, CreateSharedHandle,
    /// cuImportExternalMemory, cuExternalMemoryGetMappedBuffer) and in production
    /// the first of them only runs partway into an export.
    #[test]
    fn allocates_and_maps_a_shared_buffer() {
        let Some(h) = harness() else { return };

        let shared = SharedBuffer::new(
            Arc::clone(&h.cuda_ctx),
            &h.device,
            h.transport,
            "shared_buffer test alloc",
            PROBE_SIZE,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .expect("SharedBuffer::new failed on a host that reported D3D12 CUDA interop");

        assert_eq!(
            shared.buffer.size(), PROBE_SIZE,
            "the wgpu buffer must report the LOGICAL size, not D3D12's padded \
             allocation — anything larger lets a shader write bytes CUDA never mapped"
        );
        assert_eq!(shared.external.size, PROBE_SIZE, "the CUDA mapping covers the logical size");
        assert_ne!(
            shared.external.device_ptr(), 0,
            "cuExternalMemoryGetMappedBuffer returned a null device pointer"
        );
    }

    /// The aliasing proof, wgpu → CUDA: bytes written by a wgpu queue write are
    /// visible through the CUDA pointer.
    ///
    /// `queue.write_buffer` is the simplest wgpu-side writer, which keeps this
    /// test about the SHARING rather than about a compute shader.
    #[test]
    fn bytes_written_through_wgpu_are_visible_through_cuda() {
        let Some(h) = harness() else { return };

        let shared = SharedBuffer::new(
            Arc::clone(&h.cuda_ctx),
            &h.device,
            h.transport,
            "shared_buffer wgpu->cuda",
            PROBE_SIZE,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .expect("SharedBuffer::new failed");

        let want = pattern(PROBE_SIZE as usize, 7);
        // No cross-API fence exists, so the wgpu work must be KNOWN complete
        // before CUDA reads. Without this the test is a race that usually passes.
        write_via_wgpu(&h.device, &shared.buffer, &want);

        let got = shared
            .external
            .read_to_host()
            .expect("cuMemcpyDtoH from the shared buffer failed");
        assert_bytes_eq(&got, &want, "CUDA's view of bytes wgpu wrote");
    }

    /// The aliasing proof, CUDA → wgpu: the other direction, which rules out the
    /// (unlikely but untested) case of a driver satisfying reads from a copy.
    #[test]
    fn bytes_written_through_cuda_are_visible_through_wgpu() {
        let Some(h) = harness() else { return };

        let shared = SharedBuffer::new(
            Arc::clone(&h.cuda_ctx),
            &h.device,
            h.transport,
            "shared_buffer cuda->wgpu",
            PROBE_SIZE,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
        .expect("SharedBuffer::new failed");

        let want = pattern(PROBE_SIZE as usize, 199);
        shared
            .external
            .write_from_host(&want)
            .expect("cuMemcpyHtoD into the shared buffer failed");

        let got = read_via_wgpu(&h.device, &shared.buffer, PROBE_SIZE);
        assert_bytes_eq(&got, &want, "wgpu's view of bytes CUDA wrote");
    }

    /// A negative control for the two tests above.
    ///
    /// Two SEPARATE shared buffers must NOT alias: if they did, the aliasing tests
    /// would pass for a trivial reason (every allocation returning the same
    /// memory) and prove nothing about sharing.  Writing a different pattern into
    /// each and reading both back through CUDA must return each buffer's own.
    #[test]
    fn two_shared_buffers_do_not_alias_each_other() {
        let Some(h) = harness() else { return };

        let make = |label: &'static str| {
            SharedBuffer::new(
                Arc::clone(&h.cuda_ctx),
                &h.device,
                h.transport,
                label,
                PROBE_SIZE,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
            .expect("SharedBuffer::new failed")
        };
        let a = make("shared_buffer distinct a");
        let b = make("shared_buffer distinct b");

        assert_ne!(
            a.external.device_ptr(), b.external.device_ptr(),
            "two live shared buffers were mapped to the same device pointer"
        );

        let pa = pattern(PROBE_SIZE as usize, 1);
        let pb = pattern(PROBE_SIZE as usize, 2);
        write_via_wgpu(&h.device, &a.buffer, &pa);
        write_via_wgpu(&h.device, &b.buffer, &pb);

        assert_bytes_eq(&a.external.read_to_host().unwrap(), &pa, "buffer a's own bytes");
        assert_bytes_eq(&b.external.read_to_host().unwrap(), &pb, "buffer b's own bytes");
    }

    /// The real thing: `Nv12EncodeNode` writes NV12 into a shared buffer, and CUDA
    /// reads back exactly what the shader produced.
    ///
    /// This is the step-1-to-step-2 join.  The expected codes are the same
    /// hardware-verified ones `tests::nv12_encode` uses — computed by a standalone
    /// C probe and confirmed by decoding NVENC's own output — so the assertion
    /// does not merely compare the pipeline against itself.
    #[test]
    fn nv12_encode_writes_into_a_shared_buffer() {
        let Some(h) = harness() else { return };

        const W: u32 = 256;
        const H: u32 = 128;
        // A pitch deliberately WIDER than the frame, and not a multiple of it:
        // pitch == width hides every stride bug, and `pitch * height` and
        // `width * height` are indistinguishable there.  This is the same 320 the
        // C probe used against NVENC.
        const PITCH: u32 = 320;
        let size = Nv12EncodeNode::buffer_size(H, PITCH);
        assert_eq!(size, PROBE_SIZE, "test bug: 320x128x1.5 should be {PROBE_SIZE}");

        let color = info(MatrixCoefficients::Bt709, ColorRange::Limited);

        // Four vertical bars, matching nvchk/nv12_probe.c's pattern.
        let bars: [(f32, f32, f32); 4] =
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)];
        let expected: [[u8; 3]; 4] =
            [[63, 102, 240], [173, 42, 26], [32, 240, 118], [235, 128, 128]];

        let reference = RgbToYuv::new(color, W, H);
        for (i, rgb) in bars.iter().enumerate() {
            assert_eq!(
                reference.apply_u8(rgb.0, rgb.1, rgb.2), expected[i],
                "bar {i}: the CPU reference disagrees with the hardware-verified codes"
            );
        }

        let in_texture = h.device.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shared_buffer nv12 input"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let mut in_bytes = Vec::<u8>::with_capacity((W * H * 8) as usize);
        for _y in 0..H {
            for x in 0..W {
                let (r, g, b) = bars[(x * 4 / W) as usize];
                for c in [r, g, b, 1.0f32] {
                    in_bytes.extend_from_slice(&f16::from_f32(c).to_le_bytes());
                }
            }
        }
        h.device.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &in_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &in_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(W * 8),
                rows_per_image: Some(H),
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );

        let shared = SharedBuffer::new(
            Arc::clone(&h.cuda_ctx),
            &h.device,
            h.transport,
            "shared_buffer nv12 output",
            size,
            // Exactly what the export path needs: the compute pass writes it, and
            // nothing copies it — NVENC reads it through CUDA instead.
            wgpu::BufferUsages::STORAGE,
        )
        .expect("SharedBuffer::new failed");

        // Fill through CUDA first, with a value no bar produces.  Anything the
        // shader fails to write therefore shows up as 0xAA rather than as a
        // plausible zero, which is what let a mis-sited chroma plane look like
        // "black" in the C probes.
        shared
            .external
            .write_from_host(&vec![0xAAu8; size as usize])
            .expect("pre-filling the shared buffer through CUDA failed");

        let node = Nv12EncodeNode::new(&h.device, color, W, H);
        let in_view = in_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = h
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shared_buffer nv12 encode"),
            });
        node.record(&mut encoder, &h.device, &in_view, &shared.buffer, PITCH);
        h.device.queue.submit(Some(encoder.finish()));
        h.device.device.poll(wgpu::Maintain::Wait);

        let nv12 = shared
            .external
            .read_to_host()
            .expect("reading the NV12 buffer back through CUDA failed");

        // Luma: sample the centre of each bar, away from the boundaries where the
        // shader's own 2x2 chroma averaging legitimately blends neighbours.
        for (i, want) in expected.iter().enumerate() {
            let x = i as u32 * (W / 4) + (W / 8);
            let y = H / 2;
            let got = nv12[(y * PITCH + x) as usize];
            assert_eq!(
                got, want[0],
                "bar {i} luma at ({x},{y}) read through CUDA is {got}, expected {} — \
                 either the shader wrote elsewhere or the two APIs are not sharing \
                 one allocation",
                want[0]
            );
        }

        // Chroma, at pitch * height: the offset NVENC reads, and the one the C
        // probe verified with this exact pitch.
        let chroma_base = (PITCH * H) as usize;
        for (i, want) in expected.iter().enumerate() {
            let cx = (i as u32 * (W / 4) + (W / 8)) / 2;
            let cy = (H / 2) / 2;
            let off = chroma_base + (cy * PITCH + cx * 2) as usize;
            let (cb, cr) = (nv12[off], nv12[off + 1]);
            assert_eq!(
                (cb, cr), (want[1], want[2]),
                "bar {i} chroma at ({cx},{cy}) read through CUDA is ({cb},{cr}), \
                 expected ({},{}) — a chroma plane at width*height instead of \
                 pitch*height lands {} bytes early",
                want[1], want[2], (PITCH - W) * H
            );
        }

        // The inter-row padding must still hold the 0xAA pre-fill: the shader is
        // allowed to write the alignment bytes inside a straddling word, but never
        // past the row.  W is a multiple of 4 here, so no word straddles the edge
        // and every padding byte should be untouched.
        for y in 0..H {
            for x in W..PITCH {
                let off = (y * PITCH + x) as usize;
                assert_eq!(
                    nv12[off], 0xAA,
                    "luma row {y} padding byte {x} was overwritten with 0x{:02X} — \
                     the shader is writing across the row stride",
                    nv12[off]
                );
            }
        }
    }

    /// Dropping a `SharedBuffer` must release CUDA's mapping and import and close
    /// the NT handle, in that order, without faulting.
    ///
    /// The ordering is the point.  `cuMemFree` on a mapped external buffer and
    /// `cuDestroyExternalMemory` are only valid while the CUDA context lives, and
    /// the NT handle must outlive both.  Getting it wrong is an access violation
    /// during unwinding, not an error return — the failure mode the skill's
    /// "poison the handle" section exists for.  Allocating and dropping in a loop
    /// makes a leak or a double-free show up as a failure to allocate.
    #[test]
    fn repeated_allocation_and_drop_stays_clean() {
        let Some(h) = harness() else { return };

        let mut seen = Vec::new();
        for i in 0..8 {
            let shared = SharedBuffer::new(
                Arc::clone(&h.cuda_ctx),
                &h.device,
                h.transport,
                "shared_buffer churn",
                PROBE_SIZE,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "SharedBuffer::new failed on iteration {i} after {} successful \
                     allocate/drop cycles: {e:?} — teardown is leaking",
                    seen.len()
                )
            });

            // Touch it through both APIs each time, so a cycle that silently
            // produced an unusable mapping fails here rather than later.
            let want = pattern(PROBE_SIZE as usize, i as u8);
            write_via_wgpu(&h.device, &shared.buffer, &want);
            assert_bytes_eq(
                &shared.external.read_to_host().unwrap(),
                &want,
                &format!("iteration {i} round trip"),
            );

            seen.push(shared.external.device_ptr());
            // Explicit drop, so the release path runs inside the loop rather than
            // all at once at the end.
            drop(shared);
        }
        assert_eq!(seen.len(), 8, "not every iteration allocated");
    }

    /// A zero-byte request is rejected rather than producing a buffer nothing can
    /// use, and the error says so.
    #[test]
    fn zero_sized_request_is_rejected() {
        let Some(h) = harness() else { return };

        let err = expect_err(
            SharedBuffer::new(
                Arc::clone(&h.cuda_ctx),
                &h.device,
                h.transport,
                "shared_buffer zero",
                0,
                wgpu::BufferUsages::STORAGE,
            ),
            "a zero-byte shared buffer was allocated",
        );
        let text = format!("{err:?}");
        assert!(
            text.contains("zero-byte"),
            "the error should name the problem, got {text}"
        );
    }

    /// The Vulkan arm reports that it is unimplemented instead of half-attempting
    /// an export and failing at `cuImportExternalMemory` with "invalid argument".
    ///
    /// Runs on every host, including this Windows/D3D12 one: the arm is selected
    /// by the `transport` argument, not by the platform.
    #[test]
    fn vulkan_transport_reports_unimplemented() {
        let Some(h) = harness() else { return };

        for (transport, expect) in [
            (InteropTransport::VulkanOpaqueFd, "VK_KHR_external_memory_fd"),
            (InteropTransport::None, "No interop transport"),
        ] {
            let err = expect_err(
                SharedBuffer::new(
                    Arc::clone(&h.cuda_ctx),
                    &h.device,
                    transport,
                    "shared_buffer unsupported transport",
                    PROBE_SIZE,
                    wgpu::BufferUsages::STORAGE,
                ),
                "an unsupported transport produced a buffer",
            );
            let text = format!("{err:?}");
            assert!(
                text.contains(expect),
                "{transport:?} should explain itself with '{expect}', got {text}"
            );
        }
    }
}
