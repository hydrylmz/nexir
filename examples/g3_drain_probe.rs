// examples/g3_drain_probe.rs
//
// G3 — what does the decoder actually hand back at EOF?
//
// `the_last_frames_of_a_clip_are_reachable_through_the_io_layer` fails with the
// last 3 frames of a 60-frame file still served stale, even though `decode_blocking`
// now drains. This probe replicates the sequence outside `IoLayer` and prints every
// pts involved, so the failure can be attributed to one of: the demuxer never
// reporting EOF, `drain_into` returning nothing, or the drained frames' pts not
// comparing the way the caller assumes.
//
//   cargo run -p nexir --example g3_drain_probe -- <file.mp4>

use nexir::io::decoder::Decoder;
use nexir::io::demuxer::Demuxer;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: g3_drain_probe <file.mp4>");
    let mut demuxer = Demuxer::open(std::path::Path::new(&path)).expect("Demuxer::open");
    let stream = demuxer.video_stream.clone().expect("no video stream");
    println!(
        "stream: {:?}x{:?}, time_base {:?}, frame_rate {:?}, duration {}",
        stream.width, stream.height, stream.time_base, stream.frame_rate, stream.duration
    );

    let mut decoder =
        Decoder::open(&stream, stream.codecpar, true).expect("Decoder::open");
    println!("decoder hw_type: {:?}", decoder.hw_type());

    let mut buf = vec![0u8; 320 * 240 * 6 + 4096];

    // Phase 1: read forward until the demuxer runs dry, exactly as the read-forward
    // loop in `decode_blocking` does.
    let mut sent = 0usize;
    let mut got: Vec<i64> = Vec::new();
    loop {
        let pkt = match demuxer.next_video_packet() {
            Ok(Some(p)) => p,
            Ok(None) => {
                println!("demuxer EOF after {sent} packet(s)");
                break;
            }
            Err(e) => {
                println!("demux error: {e:?}");
                break;
            }
        };
        sent += 1;
        match decoder.decode_into(&pkt, &mut buf, None) {
            Ok(Some(f)) => got.push(f.pts),
            Ok(None) => {}
            Err(e) => println!("decode error on packet {sent}: {e:?}"),
        }
    }
    println!(
        "decode_into produced {} frame(s); last 6 pts: {:?}",
        got.len(),
        &got[got.len().saturating_sub(6)..]
    );

    // Phase 2: the drain, printing every frame it yields.
    let mut drained: Vec<i64> = Vec::new();
    loop {
        match decoder.drain_into(&mut buf) {
            Ok(Some(f)) => drained.push(f.pts),
            Ok(None) => {
                println!("drain_into: fully drained after {} frame(s)", drained.len());
                break;
            }
            Err(e) => {
                println!("drain_into error after {} frame(s): {e:?}", drained.len());
                break;
            }
        }
    }
    println!("drained pts: {drained:?}");
    println!(
        "TOTAL reachable: {} of a file the container says is {} tick(s) long",
        got.len() + drained.len(),
        stream.duration
    );

    // Phase 3: the same file through `IoLayer`, one frame at a time in order, which
    // is what the engine does — and the only place the interaction between the
    // read-forward loop, the 600-packet bound and the drain is visible.
    println!("\n--- through IoLayer, sequentially ---");
    let device = std::sync::Arc::new(
        pollster::block_on(nexir::render::device::GpuDevice::new_headless())
            .expect("headless GpuDevice"),
    );
    let fps = stream.frame_rate.unwrap();
    let tb = nexir::timeline::rational::Rational::TIMEBASE_90K;
    let total = (got.len() + drained.len()) as usize;
    let duration_pts = nexir::timeline::rational::frame_to_pts(total as i64, fps, tb);

    let sources = std::sync::Arc::new(std::sync::RwLock::new(
        nexir::timeline::source::SourceRegistry::new(),
    ));
    let sid = sources.write().unwrap().register(
        std::path::PathBuf::from(&path),
        Some(nexir::timeline::source::VideoStreamInfo {
            width: stream.width.unwrap(),
            height: stream.height.unwrap(),
            frame_rate: fps,
            pixel_fmt: nexir::timeline::source::PixelFormat::Yuv420p,
            color_info: stream.color_info,
            duration_pts,
            is_vfr: false,
            time_base: tb,
            rotation: Default::default(),
        }),
        None,
    );
    let pool = std::sync::Arc::new(nexir::io::slot_pool::FrameSlotPool::new(&device));
    let cache = std::sync::Arc::new(nexir::io::frame_cache::FrameCache::new(
        std::sync::Arc::clone(&pool),
        32,
    ));
    let (tx, _rx) = std::sync::mpsc::sync_channel(16);
    let io = nexir::io::io_layer::IoLayer::new(
        std::sync::Arc::clone(&device.device),
        pool,
        std::sync::Arc::clone(&cache),
        sources,
        tx,
        tb,
        std::sync::Arc::new(nexir::io::interop_decode::InteropDecodeTargets::disabled(
            std::sync::Arc::clone(&device),
        )),
    );

    for i in 0..total {
        let pts = nexir::timeline::rational::frame_to_pts(i as i64, fps, tb);
        let served = io.decode_blocking(sid, pts).is_some();
        let fresh = cache.get(sid, pts).is_some();
        // Only the interesting part: the tail, plus anything stale earlier.
        if !fresh || i + 5 >= total {
            println!("frame {i:>3} pts {pts:>7}: served={served} fresh={fresh}");
        }
    }

    // Phase 4: `decode_blocking`'s own algorithm, replicated on a THIRD
    // demuxer/decoder pair with every decision printed.
    //
    // Phase 3 says which frame is stale; only this says why. It reproduces the
    // read-forward loop, the `hit_eof` gate and the drain loop line for line, so
    // the packet count, the pts each call yields and the comparison that
    // discards a frame are all visible per request.
    println!("\n--- decode_blocking, replicated, per request ---");
    let mut dem2 = Demuxer::open(std::path::Path::new(&path)).expect("Demuxer::open");
    let stream2 = dem2.video_stream.clone().expect("no video stream");
    let stream_tb = stream2.time_base;
    let mut dec2 = Decoder::open(&stream2, stream2.codecpar, true).expect("Decoder::open");
    let mut buf2 = vec![0u8; 320 * 240 * 6 + 4096];

    for i in 0..total {
        let project_pts = nexir::timeline::rational::frame_to_pts(i as i64, fps, tb);
        let target = tb.rescale_pts(project_pts, stream_tb);

        let mut packets = 0usize;
        let mut hit_eof = false;
        let mut decoded_anything = false;
        let mut got: Option<i64> = None;
        for _ in 0..600 {
            let pkt = match dem2.next_video_packet().ok().flatten() {
                Some(p) => p,
                None => {
                    hit_eof = true;
                    break;
                }
            };
            packets += 1;
            let pkt_pts = pkt.pts;
            if let Some(frame) = dec2.decode_into(&pkt, &mut buf2, None).ok().flatten() {
                let eff = if frame.pts == 0 { pkt_pts } else { frame.pts };
                if eff >= target {
                    got = Some(eff);
                    decoded_anything = true;
                    break;
                }
            }
        }

        let mut drained: Vec<(i64, bool)> = Vec::new();
        if !decoded_anything && hit_eof {
            loop {
                match dec2.drain_into(&mut buf2) {
                    Ok(Some(frame)) => {
                        let keep = frame.pts >= target;
                        drained.push((frame.pts, keep));
                        if keep {
                            got = Some(frame.pts);
                            decoded_anything = true;
                            break;
                        }
                    }
                    Ok(None) => {
                        println!("frame {i:>3} target {target:>6}: drain returned None (fully drained)");
                        break;
                    }
                    Err(e) => {
                        println!("frame {i:>3} target {target:>6}: drain error {e:?}");
                        break;
                    }
                }
            }
        }

        if hit_eof || !drained.is_empty() || !decoded_anything || i + 5 >= total {
            println!(
                "frame {i:>3} target {target:>6}: {packets} pkt(s), eof={hit_eof}, \
                 got={got:?}, decoded={decoded_anything}, drained={drained:?}"
            );
        }
    }

    // Phase 5: the same replication, but WITH the cold seek `decode_blocking`
    // performs on the first request (`prev_pts == i64::MIN` → `need_seek`).
    //
    // This is the only difference between phase 4, which serves all 60 frames, and
    // phase 3, which serves 59 — so if the tail is short by one it is the seek, not
    // the drain. Prints the pts ACTUALLY obtained per request, which is also the
    // only view an off-by-one picture is visible in: the cache is keyed by the
    // REQUESTED pts either way.
    println!("\n--- decode_blocking, replicated WITH the cold seek ---");
    let mut dem3 = Demuxer::open(std::path::Path::new(&path)).expect("Demuxer::open");
    let stream3 = dem3.video_stream.clone().expect("no video stream");
    let mut dec3 = Decoder::open(&stream3, stream3.codecpar, true).expect("Decoder::open");
    let mut buf3 = vec![0u8; 320 * 240 * 6 + 4096];

    let seek_target = dem3.seek(0, tb).expect("Demuxer::seek");
    let landed = dec3.seek_to(&mut dem3, seek_target).expect("seek_to");
    println!("seek_to({seek_target}) landed on pts {landed} — and DISCARDED that frame");

    let mut stale = 0usize;
    for i in 0..total {
        let project_pts = nexir::timeline::rational::frame_to_pts(i as i64, fps, tb);
        let target = tb.rescale_pts(project_pts, stream_tb);

        let mut hit_eof = false;
        let mut got: Option<i64> = None;
        for _ in 0..600 {
            let pkt = match dem3.next_video_packet().ok().flatten() {
                Some(p) => p,
                None => {
                    hit_eof = true;
                    break;
                }
            };
            let pkt_pts = pkt.pts;
            if let Some(frame) = dec3.decode_into(&pkt, &mut buf3, None).ok().flatten() {
                let eff = if frame.pts == 0 { pkt_pts } else { frame.pts };
                if eff >= target {
                    got = Some(eff);
                    break;
                }
            }
        }
        if got.is_none() && hit_eof {
            loop {
                match dec3.drain_into(&mut buf3) {
                    Ok(Some(frame)) => {
                        if frame.pts >= target {
                            got = Some(frame.pts);
                            break;
                        }
                    }
                    Ok(None) | Err(_) => break,
                }
            }
        }
        if got.is_none() {
            stale += 1;
        }
        if i < 3 || i + 4 >= total || got.is_none() {
            println!(
                "frame {i:>3} target {target:>6}: got={got:?}{}",
                match got {
                    Some(p) if p != target => format!("  ← WRONG PICTURE, wanted {target}"),
                    Some(_) => String::new(),
                    None => "  ← STALE".to_string(),
                }
            );
        }
    }
    println!("with the cold seek: {stale} of {total} request(s) had nothing to serve");
}
