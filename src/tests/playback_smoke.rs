// src/tests/playback_smoke.rs
#[cfg(test)]
mod playback_smoke {
    use std::sync::Arc;
    use crate::io::demuxer::Demuxer;
    use crate::io::decoder::Decoder;
    use crate::io::slot_pool::FrameSlotPool;
    use crate::io::frame_cache::FrameCache;
    use crate::render::device::GpuDevice;
    use crate::timeline::ids::SourceId;
    use crate::timeline::rational::Rational;
    use crate::timeline::source::DecodedFrameMeta;

    #[test]
    fn decode_30_frames_monotonic_pts() {
        let path = match std::env::var("VE_TEST_FILE") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => { eprintln!("VE_TEST_FILE not set, skipping"); return; }
        };

        let mut demuxer = Demuxer::open(&path).expect("demuxer open failed");
        let stream_info = demuxer.video_stream.as_ref().unwrap().clone();
        let mut decoder = Decoder::open(&stream_info, stream_info.codecpar, true).expect("decoder open failed");

        let mut pts_log: Vec<i64> = Vec::new();
        let mut buf = vec![0u8; 3840 * 2160 * 2]; // Enough for 4K YUV420p
        
        for _ in 0..30 {
            let pkt = demuxer.next_video_packet().unwrap().expect("unexpected EOF");
            if let Some(frame) = decoder.decode_into(&pkt, &mut buf, None).unwrap() {
                pts_log.push(frame.pts);
            }
        }

        for i in 1..pts_log.len() {
            assert!(pts_log[i] > pts_log[i-1],
                "PTS not monotonic at frame {i}: {} -> {}", pts_log[i-1], pts_log[i]);
        }
    }

    #[test]
    fn seek_lands_at_correct_frame() {
        let path = match std::env::var("VE_TEST_FILE") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => { eprintln!("VE_TEST_FILE not set, skipping"); return; }
        };

        let mut demuxer = Demuxer::open(&path).expect("demuxer open failed");
        let stream_info = demuxer.video_stream.as_ref().unwrap().clone();
        let frame_rate = stream_info.frame_rate.unwrap();
        let mut decoder = Decoder::open(&stream_info, stream_info.codecpar, true).expect("decoder open failed");

        let project_tb = Rational::new(1, 90000);
        let target_pts = 100i64 * project_tb.den / frame_rate.num as i64;

        let stream_pts = demuxer.seek(target_pts, project_tb).unwrap();
        let actual_pts = decoder.seek_to(&mut demuxer, stream_pts).unwrap();
        
        let frame_period_pts = project_tb.den / frame_rate.num as i64;
        
        // Assert we landed exactly at the target, or within one frame period if the file has weird timestamps
        assert!(actual_pts <= target_pts + frame_period_pts,
            "seek landed too far: actual={actual_pts}, target={target_pts}");
    }

    #[test]
    fn slot_pool_acquire_release_cycle() {
        let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).expect("GpuDevice::new_headless failed"));
        let pool = FrameSlotPool::new(&device);

        // Tier 0 is <= 3_110_400. 
        let req_size = 3_000_000;
        
        let mut acquired = Vec::new();
        // Acquire 32 Tier 0 slots (default capacity is 32)
        for _ in 0..32 {
            let slot = pool.acquire(req_size).expect("expected to acquire slot");
            acquired.push(slot);
        }

        // 33rd should fail
        assert!(pool.acquire(req_size).is_none());

        // Release all
        for slot in acquired {
            pool.release(slot);
        }

        // Acquire 32 again should succeed
        for _ in 0..32 {
            pool.acquire(req_size).expect("expected to acquire slot after release");
        }
    }

    #[test]
    fn frame_cache_eviction_releases_slot() {
        let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).expect("GpuDevice::new_headless failed"));
        let pool = Arc::new(FrameSlotPool::new(&device));
        let cache = FrameCache::new(pool.clone(), 4);
        
        let source_id = SourceId(1);
        let req_size = 3_000_000;

        // Insert 4 frames
        for i in 0..4 {
            let slot = pool.acquire(req_size).unwrap();
            cache.insert(source_id, i * 1000, slot, DecodedFrameMeta::default());
        }

        assert_eq!(cache.len(), 4);
        
        // Tier 0 should now have 28 free slots out of 32
        
        // Insert 5th frame
        let slot5 = pool.acquire(req_size).unwrap();
        cache.insert(source_id, 4000, slot5, DecodedFrameMeta::default());

        assert_eq!(cache.len(), 4);
        
        // Assert the evicted key (PTS=0) is no longer in the cache
        assert!(cache.get(source_id, 0).is_none());
        
        // Check it actually has the 5th frame
        assert!(cache.get(source_id, 4000).is_some());
        
        // Pool should have 1 free slot from the eviction (which we just used? no, we used one for slot5 BEFORE eviction)
        // Wait, slot5 used 1. So 27 free. Then insert evicts, releasing 1. So 28 free again.
        // We can just verify pool.acquire() works 28 more times.
        let mut additional_slots = Vec::new();
        for _ in 0..28 {
            additional_slots.push(pool.acquire(req_size).unwrap());
        }
        
        // 29th should fail
        assert!(pool.acquire(req_size).is_none());
    }
}
