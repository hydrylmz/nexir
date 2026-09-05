// examples/g2e_interop_target_probe.rs
//
// Task G2e — why `bench --interop` reports `live targets 0`.
//
// `InteropDecodeTargets::ensure_target` collapses every allocation failure into one
// `&'static str` ("allocating the shared Y/UV texture pair failed") and puts the
// real `CudaError` behind `log::warn!`. `src/bin/bench.rs` installs no logger — it
// is a `[[bin]]`, so it cannot use the `env_logger` dev-dependency at all — so the
// cause of a fallback is invisible from the profile that reports it.
//
// This probe calls the same constructors in the same order, out of `cargo run
// --example`, and prints each error verbatim:
//
//   1. `InteropCapability::probe`             — transport, ordinal, driver
//   2. `CudaContext::new`                     — the primary-context retain, including
//                                               the BLOCKING_SYNC flag dance FFmpeg's
//                                               AV_CUDA_USE_PRIMARY_CONTEXT checks
//   3. `SharedTexture::new` per plane         — D3D12 shared alloc + CUDA import,
//                                               reported separately because a D3D12
//                                               failure and a CUDA import failure
//                                               have nothing in common
//   4. `DecodeInteropTarget::new`             — what `ensure_target` actually calls
//
// Both plane geometries every class needs are attempted (1080p and 4K), because a
// failure that is really about allocation SIZE looks like a failure about flags
// until the two sizes disagree.
//
// Run:
//     cargo build --example g2e_interop_target_probe --release
//     cp target/debug/*.dll target/release/examples/    # cuda.dll beside the binary
//     ./target/release/examples/g2e_interop_target_probe.exe

use std::sync::Arc;

use nexir::interop::capability::InteropCapability;
use nexir::interop::cuda_context::CudaContext;
use nexir::interop::decode_interop::DecodeInteropTarget;
use nexir::interop::external_texture::SharedTexture;
use nexir::render::device::GpuDevice;

fn main() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(false)
        .try_init();

    println!("=== G2e interop target probe ===\n");

    let device = match pollster::block_on(GpuDevice::new_headless()) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            println!("SKIP: no headless GPU on this machine: {e:?}");
            return;
        }
    };
    let info = device.adapter.get_info();
    println!("adapter   : {} ({:?}), vendor 0x{:X}", info.name, info.backend, info.vendor);

    let capability = InteropCapability::probe(&device);
    println!(
        "capability: transport={:?}, ordinal={}, driver={}",
        capability.transport,
        capability.cuda_device_ordinal,
        if capability.driver_version.is_empty() {
            "n/a"
        } else {
            &capability.driver_version
        }
    );
    if !capability.is_available() {
        println!("\nSKIP: no interop transport, so nothing below can be attempted.");
        return;
    }

    let cuda = match CudaContext::new(&capability) {
        Ok(c) => {
            println!("CudaContext::new: ok (ctx={:?})", c.raw_context());
            Arc::new(c)
        }
        Err(e) => {
            println!("CudaContext::new FAILED: {e:?}");
            return;
        }
    };

    const PLANE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
        .union(wgpu::TextureUsages::COPY_DST)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::STORAGE_BINDING);

    for (w, h) in [(1920u32, 1080u32), (3840, 2160)] {
        println!("\n--- {w}x{h} ---");

        // The two planes separately, so the report names which one failed and with
        // what — `DecodeInteropTarget::new` returns only the first error.
        match SharedTexture::new(
            Arc::clone(&cuda),
            &device,
            capability.transport,
            "probe Y plane",
            w,
            h,
            wgpu::TextureFormat::R8Unorm,
            PLANE_USAGE,
        ) {
            Ok(_) => println!("  Y  plane R8Unorm  {w}x{h}: ok"),
            Err(e) => println!("  Y  plane R8Unorm  {w}x{h}: FAILED {e:?}"),
        }
        match SharedTexture::new(
            Arc::clone(&cuda),
            &device,
            capability.transport,
            "probe UV plane",
            w / 2,
            h / 2,
            wgpu::TextureFormat::Rg8Unorm,
            PLANE_USAGE,
        ) {
            Ok(_) => println!("  UV plane Rg8Unorm {}x{}: ok", w / 2, h / 2),
            Err(e) => println!("  UV plane Rg8Unorm {}x{}: FAILED {e:?}", w / 2, h / 2),
        }

        match DecodeInteropTarget::new(Arc::clone(&cuda), &device, capability.transport, w, h) {
            Ok(t) => println!(
                "  DecodeInteropTarget::new: ok, dims {:?}, view ids {:?}/{:?}",
                t.dimensions(),
                t.y_import().view_id(),
                t.uv_import().view_id()
            ),
            Err(e) => println!("  DecodeInteropTarget::new: FAILED {e:?}"),
        }
    }

    // Two targets alive at once, which is what a multi-source timeline holds and
    // what a per-pass registry rebuild does across repeats.
    println!("\n--- two 4K targets at once ---");
    let a = DecodeInteropTarget::new(Arc::clone(&cuda), &device, capability.transport, 3840, 2160);
    let b = DecodeInteropTarget::new(Arc::clone(&cuda), &device, capability.transport, 3840, 2160);
    match &a {
        Ok(_) => println!("  first : ok"),
        Err(e) => println!("  first : FAILED {e:?}"),
    }
    match &b {
        Ok(_) => println!("  second: ok"),
        Err(e) => println!("  second: FAILED {e:?}"),
    }
    drop(a);
    drop(b);

    // ── The ORDER `bench --interop` uses ────────────────────────────────────────
    //
    // This is the part the checks above cannot see. In the benchmark the sequence is
    // `CudaContext::new` → `Decoder::open` (which is what calls
    // `av_hwdevice_ctx_create`, now with `AV_CUDA_USE_PRIMARY_CONTEXT`) →
    // `ensure_target` → `DecodeInteropTarget::new`. An allocation that succeeds on
    // its own and fails after FFmpeg has touched the same primary context is a
    // different bug from one that never worked, and only the ordered attempt can
    // tell them apart.
    let fixtures: Vec<std::path::PathBuf> = {
        let dir = nexir::bench_media::fixture_dir();
        nexir::bench_media::MEDIA_CLASSES
            .iter()
            .map(|c| dir.join(c.file_name()))
            .filter(|p| p.exists())
            .collect()
    };
    let Some(input) = fixtures.first() else {
        println!(
            "\nSKIP the ordered attempt: no bench fixtures in {}. Generate them with\n\
             ./target/release/bench.exe --media",
            nexir::bench_media::fixture_dir().display()
        );
        return;
    };

    println!("\n--- after Decoder::open (the benchmark's order) — {} ---", input.display());
    let demuxer = match nexir::io::demuxer::Demuxer::open(input) {
        Ok(d) => d,
        Err(e) => {
            println!("  Demuxer::open FAILED: {e:?}");
            return;
        }
    };
    let stream = match demuxer.video_stream.clone() {
        Some(s) => s,
        None => {
            println!("  the fixture has no video stream");
            return;
        }
    };
    let (w, h) = (stream.width.unwrap_or(0), stream.height.unwrap_or(0));
    let decoder = match nexir::io::decoder::Decoder::open(&stream, stream.codecpar, true) {
        Ok(d) => d,
        Err(e) => {
            println!("  Decoder::open FAILED: {e:?}");
            return;
        }
    };
    println!("  Decoder::open: ok, hw_type={:?}", decoder.hw_type());

    // What the driver actually thinks is going on, before anything else is tried.
    // Printed rather than inferred: "the push failed" is consistent with the context
    // being current on this thread, on another thread, or with the primary context
    // having been reconfigured, and those have different fixes.
    unsafe {
        use nexir::interop::ffi::cuda_driver as cu;
        let mut cur: cu::CUcontext = std::ptr::null_mut();
        let r = cu::cuCtxGetCurrent(&mut cur);
        println!(
            "  cuCtxGetCurrent -> {} ({:?}); our ctx is {:?}; same={}",
            r,
            cur,
            cuda.raw_context(),
            cur == cuda.raw_context()
        );
        let mut flags = 0u32;
        let mut active = 0i32;
        let r = cu::cuDevicePrimaryCtxGetState(capability.cuda_device_ordinal, &mut flags, &mut active);
        println!("  cuDevicePrimaryCtxGetState -> {r}, flags=0x{flags:x}, active={active}");

        // A bare push/pop pair, reported on its own, so the failure is attributed to
        // the driver call rather than to whatever `with_context` wraps it in.
        let push = cu::cuCtxPushCurrent(cuda.raw_context());
        println!("  cuCtxPushCurrent(ours) -> {push} ({})", cu::cu_err_to_string(push));
        if push == cu::CUDA_SUCCESS {
            let mut popped: cu::CUcontext = std::ptr::null_mut();
            let pop = cu::cuCtxPopCurrent(&mut popped);
            println!("  cuCtxPopCurrent -> {pop}, got {popped:?}");
        }

        // Is our HANDLE stale, or is the context current on another thread? Retaining
        // again answers it: the primary context has one handle per device, so a
        // different pointer means the context we hold was destroyed and remade.
        let mut again: cu::CUcontext = std::ptr::null_mut();
        let r = cu::cuDevicePrimaryCtxRetain(&mut again, capability.cuda_device_ordinal);
        println!(
            "  cuDevicePrimaryCtxRetain again -> {r}, handle {:?}, same as ours={}",
            again,
            again == cuda.raw_context()
        );
        if r == cu::CUDA_SUCCESS {
            let push2 = cu::cuCtxPushCurrent(again);
            println!(
                "  cuCtxPushCurrent(freshly retained) -> {push2} ({})",
                cu::cu_err_to_string(push2)
            );
            if push2 == cu::CUDA_SUCCESS {
                let mut popped: cu::CUcontext = std::ptr::null_mut();
                cu::cuCtxPopCurrent(&mut popped);
            }
            cu::cuDevicePrimaryCtxRelease(capability.cuda_device_ordinal);
        }

        // `cuCtxSetCurrent` instead of a push. Since CUDA 4.0 a context may be
        // current to several threads at once, and set has no floating requirement —
        // so if this returns 0 where the push returned 201, the fix is the API
        // choice rather than any kind of ordering.
        let set = cu::cuCtxSetCurrent(cuda.raw_context());
        println!("  cuCtxSetCurrent(ours) -> {set} ({})", cu::cu_err_to_string(set));
        if set == cu::CUDA_SUCCESS {
            let mut cur2: cu::CUcontext = std::ptr::null_mut();
            cu::cuCtxGetCurrent(&mut cur2);
            println!("    then cuCtxGetCurrent -> {cur2:?} (ours={})", cur2 == cuda.raw_context());
            let sync = cu::cuCtxSynchronize();
            println!("    cuCtxSynchronize -> {sync} ({})", cu::cu_err_to_string(sync));
            // Unbind again so the allocation below exercises the real code path from
            // the same starting state the benchmark has.
            cu::cuCtxSetCurrent(std::ptr::null_mut());
        }
    }

    match DecodeInteropTarget::new(Arc::clone(&cuda), &device, capability.transport, w, h) {
        Ok(t) => println!("  DecodeInteropTarget::new after the decoder: ok, dims {:?}", t.dimensions()),
        Err(e) => println!("  DecodeInteropTarget::new after the decoder: FAILED {e:?}"),
    }

    println!("\n=== done ===");
}
