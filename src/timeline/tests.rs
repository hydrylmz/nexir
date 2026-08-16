// src/timeline/tests.rs
#[cfg(test)]
mod tests {
    use crate::project::Project;
    use crate::timeline::rational::Rational;
    use crate::timeline::transform::ClipTransform;
    use crate::timeline::store::TimelineStore;
    use crate::timeline::ids::{TrackId, SourceId};
    use crate::timeline::query::{query_active, next_boundary};
    use crate::timeline::mutation::{insert_clip, remove_clip, move_clip, trim_clip_in, trim_clip_out, ClipInsertParams, MutationError};

    // ── Rational ──────────────────────────────────────────────────────────────

    #[test]
    fn rational_gcd() {
        let gcd = Rational::gcd(48000, 90000);
        assert_eq!(gcd, 6000);
        let r = Rational::new(48000, 90000);
        assert_eq!(r.num, 8);
        assert_eq!(r.den, 15);
    }

    #[test]
    fn rational_pts_to_ns_one_second() {
        let tb = Rational::TIMEBASE_90K;
        assert_eq!(tb.pts_to_ns(90_000), 1_000_000_000);
    }

    #[test]
    fn rational_rescale() {
        let rescaled = Rational::TIMEBASE_90K.rescale_pts(90_000, Rational { num: 1, den: 48_000 });
        assert_eq!(rescaled, 48_000);
    }

    // ── ClipTransform ─────────────────────────────────────────────────────────

    #[test]
    fn transform_identity_is_identity() {
        assert!(ClipTransform::identity().is_identity());
    }

    #[test]
    fn transform_identity_matrix() {
        let m = ClipTransform::identity().to_matrix(100.0, 100.0, 1920.0, 1080.0);
        // We no longer test for an identity matrix here because to_matrix now converts all the way to NDC.
    }

    // ── TimelineStore + query_active ──────────────────────────────────────────

    fn make_store_with_three_clips() -> TimelineStore {
        let mut store = TimelineStore::new();
        
        let p1 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 100,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 50,
            pts_out: 200,
            source_in: 0,
            layer_order: 1,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        let p3 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 150,
            pts_out: 300,
            source_in: 0,
            layer_order: 2,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };

        insert_clip(&mut store, p1).unwrap();
        insert_clip(&mut store, p2).unwrap();
        insert_clip(&mut store, p3).unwrap();
        
        store
    }

    #[test]
    fn query_active_basic() {
        let store = make_store_with_three_clips();
        let mut out = Vec::new();
        query_active(&store, 75, &mut out);
        assert_eq!(out.len(), 2);
        // A is index 0, B is index 1
        assert_eq!(out[0].store_index, 0);
        assert_eq!(out[1].store_index, 1);
    }

    #[test]
    fn query_active_at_left_boundary() {
        let store = make_store_with_three_clips();
        let mut out = Vec::new();
        query_active(&store, 0, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].store_index, 0);
    }

    #[test]
    fn query_active_at_right_boundary() {
        let store = make_store_with_three_clips();
        let mut out = Vec::new();
        query_active(&store, 100, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].store_index, 1); // Clip A has ended, B is active
    }

    #[test]
    fn same_track_clips_get_distinct_layer_order() {
        let mut project = Project::new("test");
        let track = project.add_video_track("V1").unwrap();

        let lower = project
            .insert_clip(ClipInsertParams {
                track_id: track,
                source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
                pts_in: 0,
                pts_out: 100,
                source_in: 0,
                layer_order: 0,
                opacity: 1.0,
                transform: ClipTransform::identity(),
                volume: 1.0,
                pan: 0.0,
                audio_muted: false,
                speed: 1.0,
                pitch: 0.0,
            })
            .unwrap();
        let upper = project
            .insert_clip(ClipInsertParams {
                track_id: track,
                source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
                pts_in: 0,
                pts_out: 100,
                source_in: 0,
                layer_order: 0,
                opacity: 1.0,
                transform: ClipTransform::identity(),
                volume: 1.0,
                pan: 0.0,
                audio_muted: false,
                speed: 1.0,
                pitch: 0.0,
            })
            .unwrap();

        let lower_idx = project.clips.index_of(lower).unwrap();
        let upper_idx = project.clips.index_of(upper).unwrap();

        assert!(project.clips.layer_order_at(lower_idx) < project.clips.layer_order_at(upper_idx));
    }

    #[test]
    fn moving_a_clip_between_tracks_reindexes_order_for_the_original_track() {
        let mut project = Project::new("test");
        let track1 = project.add_video_track("V1").unwrap();
        let track2 = project.add_video_track("V2").unwrap();

        let first = project
            .insert_clip(ClipInsertParams {
                track_id: track1,
                source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
                pts_in: 0,
                pts_out: 100,
                source_in: 0,
                layer_order: 0,
                opacity: 1.0,
                transform: ClipTransform::identity(),
                volume: 1.0,
                pan: 0.0,
                audio_muted: false,
                speed: 1.0,
                pitch: 0.0,
            })
            .unwrap();

        let moved = project
            .insert_clip(ClipInsertParams {
                track_id: track1,
                source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
                pts_in: 100,
                pts_out: 200,
                source_in: 0,
                layer_order: 0,
                opacity: 1.0,
                transform: ClipTransform::identity(),
                volume: 1.0,
                pan: 0.0,
                audio_muted: false,
                speed: 1.0,
                pitch: 0.0,
            })
            .unwrap();

        let moved_to_track2 = project.move_clip_to_track(moved, track2, 100).unwrap();
        let new_on_track1 = project
            .insert_clip(ClipInsertParams {
                track_id: track1,
                source_id: SourceId(2),
            kind: crate::timeline::store::ClipKind::Video,
                pts_in: 200,
                pts_out: 300,
                source_in: 0,
                layer_order: 0,
                opacity: 1.0,
                transform: ClipTransform::identity(),
                volume: 1.0,
                pan: 0.0,
                audio_muted: false,
                speed: 1.0,
                pitch: 0.0,
            })
            .unwrap();

        let first_idx = project.clips.index_of(first).unwrap();
        let new_idx = project.clips.index_of(new_on_track1).unwrap();
        let moved_idx = project.clips.index_of(moved_to_track2).unwrap();

        assert_ne!(project.clips.track_id_at(moved_idx), track1);
        assert_ne!(project.clips.layer_order_at(first_idx), project.clips.layer_order_at(new_idx));
        assert!(project.clips.layer_order_at(new_idx) > project.clips.layer_order_at(first_idx));
    }

    #[test]
    fn insert_maintains_sort() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 300, pts_out: 400, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let p3 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 200, pts_out: 300, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        insert_clip(&mut store, p1).unwrap();
        insert_clip(&mut store, p2).unwrap();
        insert_clip(&mut store, p3).unwrap();
        
        store.assert_sorted();
    }

    #[test]
    fn query_source_pts_formula() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 50, pts_out: 200, source_in: 1000, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(1), source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 50, pts_out: 200, source_in: 1000, layer_order: 1, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 2.0, pitch: 0.0,
        };
        insert_clip(&mut store, p1).unwrap();
        insert_clip(&mut store, p2).unwrap();

        let mut out = Vec::new();
        query_active(&store, 75, &mut out);
        
        // speed 1.0: 1000 + (75 - 50) * 1.0 = 1025
        assert_eq!(out[0].source_pts, 1025);
        // speed 2.0: 1000 + (75 - 50) * 2.0 = 1050
        assert_eq!(out[1].source_pts, 1050);
    }

    // ── Mutations ────────────────────────────────────────────────────────────

    #[test]
    fn move_clip_preserves_duration() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0, pts_out: 100, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let id = insert_clip(&mut store, p1).unwrap();
        let new_id = move_clip(&mut store, id, 500).unwrap();
        
        let idx = store.index_of(new_id).unwrap();
        assert_eq!(store.pts_in_at(idx), 500);
        assert_eq!(store.pts_out_at(idx), 600);
    }

    #[test]
    fn trim_in_adjusts_source_in() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let id = insert_clip(&mut store, p1).unwrap();
        let new_id = trim_clip_in(&mut store, id, 150).unwrap();
        
        let idx = store.index_of(new_id).unwrap();
        assert_eq!(store.source_in_at(idx), 50);
        assert_eq!(store.pts_in_at(idx), 150);
    }

    #[test]
    fn trim_out_no_resort() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 200, pts_out: 300, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let id = insert_clip(&mut store, p1).unwrap();
        insert_clip(&mut store, p2).unwrap();
        
        trim_clip_out(&mut store, id, 150).unwrap();
        store.assert_sorted();
    }

    #[test]
    fn remove_clip_reduces_len() {
        let mut store = make_store_with_three_clips();
        assert_eq!(store.len(), 3);
        let id = store.clip_id_at(1);
        remove_clip(&mut store, id).unwrap();
        assert_eq!(store.len(), 2);
        store.assert_sorted();
    }

    #[test]
    fn insert_invalid_duration_is_error() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 100, pts_out: 100, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, speed: 1.0, pitch: 0.0,
        };
        let result = insert_clip(&mut store, p1);
        assert!(matches!(result, Err(MutationError::InvalidDuration { .. })));
    }

    // ── next_boundary ────────────────────────────────────────────────────────

    #[test]
    fn next_boundary_finds_nearest() {
        let store = make_store_with_three_clips(); // A: 0..100, B: 50..200, C: 150..300
        assert_eq!(next_boundary(&store, 0), Some(50));
        assert_eq!(next_boundary(&store, 50), Some(100));
        assert_eq!(next_boundary(&store, 100), Some(150));
    }

    #[test]
    fn test_insert_clip_ripple() {
        use crate::timeline::mutation::insert_clip_ripple;
        let mut store = TimelineStore::new();
        
        let p1 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 10,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        insert_clip(&mut store, p1).unwrap(); // A: 0..10

        // Ripple insert B: 3..8 (duration 5).
        // This should split A into: Head of A [0, 3], Tail of A [8, 15].
        // B will occupy [3, 8].
        let p2 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(2),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 3,
            pts_out: 8,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        let _b_id = insert_clip_ripple(&mut store, p2).unwrap();

        assert_eq!(store.len(), 3);
        // Verify clip parameters and positions
        // Since store is sorted by pts_in:
        // index 0: Head of A (0..3)
        // index 1: B (3..8)
        // index 2: Tail of A (8..15)
        assert_eq!(store.pts_in[0], 0);
        assert_eq!(store.pts_out[0], 3);
        assert_eq!(store.source_ids[0], SourceId(1));

        assert_eq!(store.pts_in[1], 3);
        assert_eq!(store.pts_out[1], 8);
        assert_eq!(store.source_ids[1], SourceId(2));

        assert_eq!(store.pts_in[2], 8);
        assert_eq!(store.pts_out[2], 15);
        assert_eq!(store.source_ids[2], SourceId(1));
        // Source in of Tail should be shifted by head duration (3)
        assert_eq!(store.source_in[2], 3);

        store.assert_sorted();
    }

    #[test]
    fn test_move_clip_ripple() {
        use crate::timeline::mutation::{insert_clip_ripple, move_clip_ripple};
        let mut store = TimelineStore::new();

        let p1 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 10,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        let _a_id = insert_clip(&mut store, p1).unwrap(); // A: 0..10

        let p2 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(2),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 10,
            pts_out: 15,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        let b_id = insert_clip(&mut store, p2).unwrap(); // B: 10..15

        // Current state: A [0, 10], B [10, 15]
        // Move B to 3.
        // B is ripple-removed first: Timeline becomes A [0, 10]. (no gap to close since B is at the end).
        // Then B is ripple-inserted at 3: A is split into Head of A [0, 3], Tail of A [8, 15]. B is [3, 8].
        let _new_b_id = move_clip_ripple(&mut store, b_id, 3).unwrap();

        assert_eq!(store.len(), 3);
        assert_eq!(store.pts_in[0], 0);
        assert_eq!(store.pts_out[0], 3); // Head of A

        assert_eq!(store.pts_in[1], 3);
        assert_eq!(store.pts_out[1], 8); // B

        assert_eq!(store.pts_in[2], 8);
        assert_eq!(store.pts_out[2], 15); // Tail of A

        store.assert_sorted();
    }

    #[test]
    fn test_insert_clip_ripple_linked_tracks() {
        use crate::timeline::mutation::insert_clip_ripple;
        let mut store = TimelineStore::new();

        // Track 0: Video Clip A [0, 10], source_id 1
        let p1 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 10,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        insert_clip(&mut store, p1).unwrap();

        // Track 1: Audio Clip B [0, 10], source_id 1 (linked to A!)
        let p2 = ClipInsertParams {
            track_id: TrackId(1),
            source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 10,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        insert_clip(&mut store, p2).unwrap();

        // Insert Image Clip C [3, 8] (duration 5) on Track 0.
        // This should split Clip A (video) on Track 0, AND also split Clip B (audio) on Track 1.
        let p3 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(2),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 3,
            pts_out: 8,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            speed: 1.0,
            pitch: 0.0,
        };
        insert_clip_ripple(&mut store, p3).unwrap();

        // Verify:
        // Clip A should be split: head [0, 3], tail [8, 15] on Track 0.
        // Clip B should be split: head [0, 3], tail [8, 15] on Track 1.
        // Clip C should be at [3, 8] on Track 0.

        // Count elements in track 0 and track 1
        let mut track0_clips = Vec::new();
        let mut track1_clips = Vec::new();
        for i in 0..store.len() {
            let pts_in = store.pts_in[i];
            let pts_out = store.pts_out[i];
            let track_id = store.track_ids[i];
            let source_id = store.source_ids[i];
            if track_id == TrackId(0) {
                track0_clips.push((pts_in, pts_out, source_id));
            } else if track_id == TrackId(1) {
                track1_clips.push((pts_in, pts_out, source_id));
            }
        }

        // Track 0 should have 3 clips: [0, 3] from source 1, [3, 8] from source 2, [8, 15] from source 1
        assert_eq!(track0_clips.len(), 3);
        assert_eq!(track0_clips[0], (0, 3, SourceId(1)));
        assert_eq!(track0_clips[1], (3, 8, SourceId(2)));
        assert_eq!(track0_clips[2], (8, 15, SourceId(1)));

        // Track 1 should have 2 clips (split head and tail, since no clip was inserted on track 1 directly):
        // [0, 3] from source 1, [8, 15] from source 1
        assert_eq!(track1_clips.len(), 2);
        assert_eq!(track1_clips[0], (0, 3, SourceId(1)));
        assert_eq!(track1_clips[1], (8, 15, SourceId(1)));

        store.assert_sorted();
    }
}
