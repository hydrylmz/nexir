// src/timeline/tests.rs
#[cfg(test)]
mod tests {
    use crate::project::Project;
    use crate::timeline::rational::Rational;
    use crate::timeline::transform::ClipTransform;
    use crate::timeline::store::TimelineStore;
    use crate::timeline::ids::{TrackId, SourceId};
    use crate::timeline::query::{query_active, next_boundary};
    use crate::timeline::mutation::{insert_clip, remove_clip, move_clip, split_clip, trim_clip_in, trim_clip_out, ClipInsertParams, MutationError};

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
        let _m = ClipTransform::identity().to_matrix(100.0, 100.0, 1920.0, 1080.0);
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
                fade_in_pts: 0,
                fade_out_pts: 0,
                speed: 1.0,
                pitch: 0.0,
            ..Default::default()
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
                fade_in_pts: 0,
                fade_out_pts: 0,
                speed: 1.0,
                pitch: 0.0,
            ..Default::default()
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
                fade_in_pts: 0,
                fade_out_pts: 0,
                speed: 1.0,
                pitch: 0.0,
            ..Default::default()
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
                fade_in_pts: 0,
                fade_out_pts: 0,
                speed: 1.0,
                pitch: 0.0,
            ..Default::default()
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
                fade_in_pts: 0,
                fade_out_pts: 0,
                speed: 1.0,
                pitch: 0.0,
            ..Default::default()
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
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
        };
        let p3 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 200, pts_out: 300, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
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
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(1), source_id: SourceId(1),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 50, pts_out: 200, source_in: 1000, layer_order: 1, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 2.0, pitch: 0.0, ..Default::default()
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
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
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
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
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
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 200, pts_out: 300, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
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
            volume: 1.0, pan: 0.0, audio_muted: false, fade_in_pts: 0, fade_out_pts: 0, speed: 1.0, pitch: 0.0, ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
        use crate::timeline::mutation::move_clip_ripple;
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
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

    // ── ColorInfo ────────────────────────────────────────────────────────────

    #[test]
    fn color_info_presets() {
        use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients, TransferFunction, ColorPrimaries};

        let bt709 = ColorInfo::bt709();
        assert_eq!(bt709.matrix, MatrixCoefficients::Bt709);
        assert_eq!(bt709.range, ColorRange::Limited);
        assert_eq!(bt709.primaries, ColorPrimaries::Bt709);
        assert_eq!(bt709.transfer_fn, TransferFunction::Bt709);
        assert_eq!(bt709.bit_depth, 8);
        assert!(!bt709.is_hdr());

        let bt601 = ColorInfo::bt601();
        assert_eq!(bt601.matrix, MatrixCoefficients::Bt601);
        assert_eq!(bt601.range, ColorRange::Limited);

        let bt2020_hdr = ColorInfo::bt2020(true, 10);
        assert_eq!(bt2020_hdr.matrix, MatrixCoefficients::Bt2020);
        assert_eq!(bt2020_hdr.primaries, ColorPrimaries::Bt2020);
        assert_eq!(bt2020_hdr.transfer_fn, TransferFunction::Pq);
        assert_eq!(bt2020_hdr.bit_depth, 10);
        assert!(bt2020_hdr.is_hdr());

        let srgb = ColorInfo::srgb();
        assert_eq!(srgb.range, ColorRange::Full);
        assert_eq!(srgb.transfer_fn, TransferFunction::Srgb);
    }

    #[test]
    fn color_info_from_ffmpeg_parsing() {
        use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients, TransferFunction, ColorPrimaries};

        // BT.709 HD video
        let ci_hd = ColorInfo::from_ffmpeg(1, 1, 1, 1, 8, 1920, 1080);
        assert_eq!(ci_hd.matrix, MatrixCoefficients::Bt709);
        assert_eq!(ci_hd.range, ColorRange::Limited);
        assert_eq!(ci_hd.primaries, ColorPrimaries::Bt709);
        assert_eq!(ci_hd.transfer_fn, TransferFunction::Bt709);

        // BT.601 SD video
        let ci_sd = ColorInfo::from_ffmpeg(5, 1, 1, 1, 8, 720, 480);
        assert_eq!(ci_sd.matrix, MatrixCoefficients::Bt601);
        assert_eq!(ci_sd.range, ColorRange::Limited);

        // BT.2020 4K HDR (PQ)
        let ci_4k = ColorInfo::from_ffmpeg(9, 1, 16, 9, 10, 3840, 2160);
        assert_eq!(ci_4k.matrix, MatrixCoefficients::Bt2020);
        assert_eq!(ci_4k.primaries, ColorPrimaries::Bt2020);
        assert_eq!(ci_4k.transfer_fn, TransferFunction::Pq);
        assert_eq!(ci_4k.bit_depth, 10);
        assert!(ci_4k.is_hdr());
    }

    #[test]
    fn color_info_resolution_heuristics_for_unspecified() {
        use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients, TransferFunction, ColorPrimaries};

        // Unspecified SD (< 720p) -> heuristics pick BT.601
        let ci_unspecified_sd = ColorInfo::from_ffmpeg(2, 0, 2, 2, 8, 640, 480);
        assert_eq!(ci_unspecified_sd.matrix, MatrixCoefficients::Bt601);
        assert_eq!(ci_unspecified_sd.range, ColorRange::Limited);
        assert_eq!(ci_unspecified_sd.primaries, ColorPrimaries::Bt709);

        // Unspecified HD (1080p) -> heuristics pick BT.709
        let ci_unspecified_hd = ColorInfo::from_ffmpeg(2, 0, 2, 2, 8, 1920, 1080);
        assert_eq!(ci_unspecified_hd.matrix, MatrixCoefficients::Bt709);
        assert_eq!(ci_unspecified_hd.range, ColorRange::Limited);
        assert_eq!(ci_unspecified_hd.primaries, ColorPrimaries::Bt709);

        // Unspecified 4K -> heuristics pick BT.2020
        let ci_unspecified_4k = ColorInfo::from_ffmpeg(2, 0, 2, 2, 10, 3840, 2160);
        assert_eq!(ci_unspecified_4k.matrix, MatrixCoefficients::Bt2020);
        assert_eq!(ci_unspecified_4k.primaries, ColorPrimaries::Bt2020);
        assert_eq!(ci_unspecified_4k.transfer_fn, TransferFunction::Pq);
        assert!(ci_unspecified_4k.is_hdr());
    }

    // ── Phase 7: Timeline Correctness & Timestamp Suite ───────────────────────

    #[test]
    fn test_framerate_constants() {
        assert_eq!(Rational::FPS_23_976.num, 24_000);
        assert_eq!(Rational::FPS_23_976.den, 1_001);

        assert_eq!(Rational::FPS_24.num, 24);
        assert_eq!(Rational::FPS_24.den, 1);

        assert_eq!(Rational::FPS_25.num, 25);
        assert_eq!(Rational::FPS_25.den, 1);

        assert_eq!(Rational::FPS_29_97.num, 30_000);
        assert_eq!(Rational::FPS_29_97.den, 1_001);

        assert_eq!(Rational::FPS_30.num, 30);
        assert_eq!(Rational::FPS_30.den, 1);

        assert_eq!(Rational::FPS_50.num, 50);
        assert_eq!(Rational::FPS_50.den, 1);

        assert_eq!(Rational::FPS_59_94.num, 60_000);
        assert_eq!(Rational::FPS_59_94.den, 1_001);

        assert_eq!(Rational::FPS_60.num, 60);
        assert_eq!(Rational::FPS_60.den, 1);
    }

    #[test]
    fn test_frame_pts_roundtrip_all_standard_framerates() {
        use crate::timeline::rational::{frame_to_pts, pts_to_frame};
        let tb = Rational::TIMEBASE_90K;
        let framerates = [
            Rational::FPS_23_976,
            Rational::FPS_24,
            Rational::FPS_25,
            Rational::FPS_29_97,
            Rational::FPS_30,
            Rational::FPS_50,
            Rational::FPS_59_94,
            Rational::FPS_60,
        ];

        for fps in framerates {
            for frame in 0..10_000 {
                let pts = frame_to_pts(frame, fps, tb);
                let recovered_frame = pts_to_frame(pts, fps, tb);
                assert_eq!(
                    recovered_frame, frame,
                    "Failed round-trip at fps {}/{} for frame {}",
                    fps.num, fps.den, frame
                );
            }
        }
    }

    #[test]
    fn test_fractional_frame_rate_pts_precision() {
        use crate::timeline::rational::frame_to_pts;
        let tb = Rational::TIMEBASE_90K;

        // At 29.97 fps (30000/1001), 30000 frames is exactly 1001 seconds
        let fps_2997 = Rational::FPS_29_97;
        let pts_30000_frames = frame_to_pts(30_000, fps_2997, tb);
        let expected_pts = 1001 * 90_000;
        assert_eq!(pts_30000_frames, expected_pts);

        // At 23.976 fps (24000/1001), 24000 frames is exactly 1001 seconds
        let fps_23976 = Rational::FPS_23_976;
        let pts_24000_frames = frame_to_pts(24_000, fps_23976, tb);
        assert_eq!(pts_24000_frames, expected_pts);

        // At 59.94 fps (60000/1001), 60000 frames is exactly 1001 seconds
        let fps_5994 = Rational::FPS_59_94;
        let pts_60000_frames = frame_to_pts(60_000, fps_5994, tb);
        assert_eq!(pts_60000_frames, expected_pts);
    }

    #[test]
    fn test_vfr_timestamp_interop() {
        // Simulate a variable frame rate sequence with irregular presentation intervals
        let vfr_intervals_ms = [33, 34, 33, 40, 28, 33, 35, 33, 16, 50];
        let mut current_ms = 0;
        let mut pts_timestamps = Vec::new();

        for &dur in &vfr_intervals_ms {
            let pts = (current_ms as i64 * 90_000) / 1_000;
            pts_timestamps.push(pts);
            current_ms += dur;
        }

        // Verify strictly monotonic timestamps
        for i in 1..pts_timestamps.len() {
            assert!(
                pts_timestamps[i] > pts_timestamps[i - 1],
                "VFR PTS not strictly monotonic: {} vs {}",
                pts_timestamps[i],
                pts_timestamps[i - 1]
            );
        }
    }

    #[test]
    fn test_empty_timeline_queries() {
        let store = TimelineStore::new();
        let mut active = Vec::new();
        query_active(&store, 0, &mut active);
        assert!(active.is_empty());

        query_active(&store, -100, &mut active);
        assert!(active.is_empty());

        query_active(&store, 10_000_000, &mut active);
        assert!(active.is_empty());

        assert_eq!(next_boundary(&store, 0), None);
        assert_eq!(next_boundary(&store, 1000), None);

        let mut overlaps = Vec::new();
        crate::timeline::query::query_overlap(&store, 0, 1000, &mut overlaps);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn test_empty_track_query() {
        // Add only 1 clip on TrackId(0)
        let mut clean_store = TimelineStore::new();
        let p0 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 90_000,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        };
        insert_clip(&mut clean_store, p0).unwrap();

        let mut active = Vec::new();
        // Query on TrackId(1) (empty track)
        crate::timeline::query::query_active_on_track(&clean_store, 45_000, TrackId(1), &mut active);
        assert!(active.is_empty());

        // Query on TrackId(0)
        crate::timeline::query::query_active_on_track(&clean_store, 45_000, TrackId(0), &mut active);
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_zero_and_negative_duration_rejected() {
        let mut store = TimelineStore::new();
        let mut p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 100,
            pts_out: 100, // zero duration
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        };
        let res = insert_clip(&mut store, p.clone());
        assert!(matches!(res, Err(MutationError::InvalidDuration { .. })));

        // Negative duration (pts_out < pts_in)
        p.pts_out = 50;
        let res_neg = insert_clip(&mut store, p);
        assert!(matches!(res_neg, Err(MutationError::InvalidDuration { .. })));
    }

    #[test]
    fn test_extreme_and_very_long_clips() {
        let mut store = TimelineStore::new();
        // 24-hour clip at 90 kHz timebase = 24 * 3600 * 90,000 = 7,776,000,000 PTS
        let day_pts: i64 = 24 * 3600 * 90_000;
        let p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: day_pts,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        };
        let clip_id = insert_clip(&mut store, p).unwrap();

        let mut active = Vec::new();
        query_active(&store, day_pts / 2, &mut active);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].source_pts, day_pts / 2);

        // Splitting 12 hours in
        let _new_clip_id = split_clip(&mut store, clip_id, day_pts / 2).unwrap();
        assert_eq!(store.pts_in_at(0), 0);
        assert_eq!(store.pts_out_at(0), day_pts / 2);
        assert_eq!(store.pts_in_at(1), day_pts / 2);
        assert_eq!(store.pts_out_at(1), day_pts);
    }

    #[test]
    fn test_clip_splitting_edge_boundaries() {
        let mut store = TimelineStore::new();
        let p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 1000,
            pts_out: 2000,
            source_in: 500,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        };
        let clip_id = insert_clip(&mut store, p).unwrap();

        // Splitting before clip start -> Error
        assert!(matches!(
            split_clip(&mut store, clip_id, 999),
            Err(MutationError::TrimPastOppositeEnd)
        ));

        // Splitting exactly at clip start -> Error
        assert!(matches!(
            split_clip(&mut store, clip_id, 1000),
            Err(MutationError::TrimPastOppositeEnd)
        ));

        // Splitting exactly at clip end -> Error
        assert!(matches!(
            split_clip(&mut store, clip_id, 2000),
            Err(MutationError::TrimPastOppositeEnd)
        ));

        // Splitting after clip end -> Error
        assert!(matches!(
            split_clip(&mut store, clip_id, 2001),
            Err(MutationError::TrimPastOppositeEnd)
        ));

        // Valid split in middle
        let _new_clip_id = split_clip(&mut store, clip_id, 1500).unwrap();
        assert_eq!(store.pts_in_at(0), 1000);
        assert_eq!(store.pts_out_at(0), 1500);
        assert_eq!(store.source_in_at(0), 500);

        assert_eq!(store.pts_in_at(1), 1500);
        assert_eq!(store.pts_out_at(1), 2000);
        assert_eq!(store.source_in_at(1), 1000); // 500 + (1500 - 1000)
    }

    #[test]
    fn test_split_clip_with_speed() {
        let mut store = TimelineStore::new();
        let p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 2000,
            source_in: 100,
            speed: 2.0,
            ..Default::default()
        };
        let clip_id = insert_clip(&mut store, p).unwrap();

        let _new_clip_id = split_clip(&mut store, clip_id, 1000).unwrap();
        assert_eq!(store.pts_in_at(0), 0);
        assert_eq!(store.pts_out_at(0), 1000);
        assert_eq!(store.source_in_at(0), 100);

        assert_eq!(store.pts_in_at(1), 1000);
        assert_eq!(store.pts_out_at(1), 2000);
        // delta is 1000 timeline pts, at 2.0x speed -> 2000 source pts
        assert_eq!(store.source_in_at(1), 2100);
    }

    #[test]
    fn test_trimming_edge_cases() {
        let mut store = TimelineStore::new();
        let p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 1000,
            pts_out: 2000,
            source_in: 100,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        };
        let clip_id = insert_clip(&mut store, p).unwrap();

        // Trim in past or at pts_out -> Error
        assert!(matches!(
            trim_clip_in(&mut store, clip_id, 2000),
            Err(MutationError::TrimPastOppositeEnd)
        ));
        assert!(matches!(
            trim_clip_in(&mut store, clip_id, 2050),
            Err(MutationError::TrimPastOppositeEnd)
        ));

        // Trim in before source start (e.g. new_pts_in = 800 -> delta = -200, source_in = 100 - 200 = -100) -> Error
        assert!(matches!(
            trim_clip_in(&mut store, clip_id, 800),
            Err(MutationError::SourceInBeforeStart { .. })
        ));

        // Trim out before or at pts_in -> Error
        assert!(matches!(
            trim_clip_out(&mut store, clip_id, 1000),
            Err(MutationError::TrimPastOppositeEnd)
        ));
        assert!(matches!(
            trim_clip_out(&mut store, clip_id, 950),
            Err(MutationError::TrimPastOppositeEnd)
        ));

        // Valid trim in (returns new ClipId because insert/remove re-keys)
        let clip_id = trim_clip_in(&mut store, clip_id, 1300).unwrap();
        assert_eq!(store.pts_in_at(0), 1300);
        assert_eq!(store.pts_out_at(0), 2000);
        assert_eq!(store.source_in_at(0), 400); // 100 + (1300 - 1000)

        // Valid trim out
        trim_clip_out(&mut store, clip_id, 1800).unwrap();
        assert_eq!(store.pts_in_at(0), 1300);
        assert_eq!(store.pts_out_at(0), 1800);
        assert_eq!(store.source_in_at(0), 400);
    }

    #[test]
    fn test_variable_speed_source_pts_math() {
        let mut store = TimelineStore::new();
        let p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 1000,
            source_in: 500,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 2.0, // 2x speed
            pitch: 0.0,
            ..Default::default()
        };
        insert_clip(&mut store, p).unwrap();

        let mut active = Vec::new();
        query_active(&store, 200, &mut active);
        assert_eq!(active.len(), 1);
        // At 2x speed, after 200 timeline ticks, source advances 400 ticks -> 500 + 400 = 900
        assert_eq!(active[0].source_pts, 900);

        // Test 0.5x speed
        let mut store_half = TimelineStore::new();
        let p_half = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 0,
            pts_out: 1000,
            source_in: 500,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 0.5,
            pitch: 0.0,
            ..Default::default()
        };
        insert_clip(&mut store_half, p_half).unwrap();

        let mut active_half = Vec::new();
        query_active(&store_half, 200, &mut active_half);
        assert_eq!(active_half.len(), 1);
        // At 0.5x speed, after 200 timeline ticks, source advances 100 ticks -> 500 + 100 = 600
        assert_eq!(active_half[0].source_pts, 600);
    }

    #[test]
    fn test_query_active_boundary_precision() {
        let mut store = TimelineStore::new();
        // Clip [1000, 2000)
        let p = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in: 1000,
            pts_out: 2000,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        };
        insert_clip(&mut store, p).unwrap();

        let mut active = Vec::new();

        // 1 tick before start
        query_active(&store, 999, &mut active);
        assert!(active.is_empty());

        // Exactly at start
        query_active(&store, 1000, &mut active);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].source_pts, 0);

        // 1 tick before end
        active.clear();
        query_active(&store, 1999, &mut active);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].source_pts, 999);

        // Exactly at end (half-open interval [in, out) -> inactive)
        active.clear();
        query_active(&store, 2000, &mut active);
        assert!(active.is_empty());
    }

    #[test]
    fn test_missing_media_and_unknown_sources() {
        use crate::timeline::source::SourceRegistry;
        let sources = SourceRegistry::new();
        // SourceId(999) has not been registered
        let unknown_source = SourceId(999);
        assert!(sources.audio_info(unknown_source).is_err());
        assert!(sources.video_info(unknown_source).is_err());
        assert!(sources.path(unknown_source).is_none());
    }

    fn make_test_params(track: TrackId, pts_in: i64, pts_out: i64, source_in: i64) -> ClipInsertParams {
        ClipInsertParams {
            track_id: track,
            source_id: SourceId(0),
            kind: crate::timeline::store::ClipKind::Video,
            pts_in,
            pts_out,
            source_in,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            volume: 1.0,
            pan: 0.0,
            audio_muted: false,
            fade_in_pts: 0,
            fade_out_pts: 0,
            speed: 1.0,
            pitch: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_slip_clip() {
        use crate::timeline::mutation::slip_clip;
        let mut store = TimelineStore::new();
        let p = make_test_params(TrackId(0), 1000, 2000, 500);
        let id = insert_clip(&mut store, p).unwrap();

        // Slip +200 source PTS
        slip_clip(&mut store, id, 200).unwrap();
        let idx = store.index_of(id).unwrap();
        assert_eq!(store.pts_in[idx], 1000);
        assert_eq!(store.pts_out[idx], 2000);
        assert_eq!(store.source_in[idx], 700);

        // Slip negative past 0 fails
        assert!(slip_clip(&mut store, id, -800).is_err());
    }

    #[test]
    fn test_slide_clip() {
        use crate::timeline::mutation::slide_clip;
        let mut store = TimelineStore::new();
        let p1 = make_test_params(TrackId(0), 0, 1000, 0);
        let p2 = make_test_params(TrackId(0), 1000, 2000, 0);
        let p3 = make_test_params(TrackId(0), 2000, 3000, 0);
        let _id1 = insert_clip(&mut store, p1).unwrap();
        let id2 = insert_clip(&mut store, p2).unwrap();
        let _id3 = insert_clip(&mut store, p3).unwrap();

        // Slide clip 2 forward by 200 PTS (extends clip 1, shrinks clip 3)
        slide_clip(&mut store, id2, 200).unwrap();
        let idx1 = store.index_of(_id1).unwrap();
        let idx2 = store.index_of(id2).unwrap();
        let idx3 = store.index_of(_id3).unwrap();

        assert_eq!(store.pts_out[idx1], 1200);
        assert_eq!(store.pts_in[idx2], 1200);
        assert_eq!(store.pts_out[idx2], 2200);
        assert_eq!(store.pts_in[idx3], 2200);
        assert_eq!(store.source_in[idx3], 200);
    }

    #[test]
    fn test_roll_edit() {
        use crate::timeline::mutation::roll_edit;
        let mut store = TimelineStore::new();
        let p1 = make_test_params(TrackId(0), 0, 1000, 0);
        let p2 = make_test_params(TrackId(0), 1000, 2000, 100);
        let id1 = insert_clip(&mut store, p1).unwrap();
        let id2 = insert_clip(&mut store, p2).unwrap();

        // Move cut point from 1000 to 1200
        roll_edit(&mut store, id1, id2, 200).unwrap();
        let idx1 = store.index_of(id1).unwrap();
        let idx2 = store.index_of(id2).unwrap();

        assert_eq!(store.pts_out[idx1], 1200);
        assert_eq!(store.pts_in[idx2], 1200);
        assert_eq!(store.source_in[idx2], 300);
    }

    #[test]
    fn test_duplicate_and_batch_move() {
        use crate::timeline::mutation::{duplicate_clip_to, batch_move_clips};
        let mut store = TimelineStore::new();
        let p1 = make_test_params(TrackId(0), 0, 1000, 0);
        let id1 = insert_clip(&mut store, p1).unwrap();

        let id2 = duplicate_clip_to(&mut store, id1, TrackId(1), 500).unwrap();
        let idx2 = store.index_of(id2).unwrap();
        assert_eq!(store.track_ids[idx2], TrackId(1));
        assert_eq!(store.pts_in[idx2], 500);
        assert_eq!(store.pts_out[idx2], 1500);

        batch_move_clips(&mut store, &[id1, id2], 1000).unwrap();
        let idx1 = store.index_of(id1).unwrap();
        let idx2 = store.index_of(id2).unwrap();
        assert_eq!(store.pts_in[idx1], 1000);
        assert_eq!(store.pts_out[idx1], 2000);
        assert_eq!(store.pts_in[idx2], 1500);
        assert_eq!(store.pts_out[idx2], 2500);
    }

    // ── Phase 14: Transforms & Compositing ───────────────────────────────────

    #[test]
    fn test_blend_modes() {
        use crate::timeline::transform::{BlendMode, blend_channel, blend_pixel};

        // All 13 blend modes
        let modes = BlendMode::all();
        assert_eq!(modes.len(), 13);

        for &m in modes {
            let u = m.as_u32();
            assert_eq!(BlendMode::from_u32(u), m);
            assert!(!m.label().is_empty());
        }

        // Test math for key blend modes
        assert_eq!(blend_channel(BlendMode::Normal, 0.7, 0.3), 0.7);
        assert_eq!(blend_channel(BlendMode::Add, 0.4, 0.4), 0.8);
        assert_eq!(blend_channel(BlendMode::Add, 0.7, 0.8), 1.0);
        assert!((blend_channel(BlendMode::Multiply, 0.5, 0.5) - 0.25).abs() < 1e-5);
        assert!((blend_channel(BlendMode::Screen, 0.5, 0.5) - 0.75).abs() < 1e-5);
        assert_eq!(blend_channel(BlendMode::Darken, 0.3, 0.7), 0.3);
        assert_eq!(blend_channel(BlendMode::Lighten, 0.3, 0.7), 0.7);
        assert!((blend_channel(BlendMode::Difference, 0.8, 0.3) - 0.5).abs() < 1e-5);

        // Test full pixel blend
        let src = [1.0, 0.0, 0.0, 1.0];
        let dst = [0.0, 1.0, 0.0, 1.0];
        let out = blend_pixel(BlendMode::Add, src, dst);
        assert_eq!(out, [1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_crop_and_feather() {
        use crate::timeline::transform::CropRect;

        let mut crop = CropRect {
            left: 0.2,
            top: 0.1,
            right: 0.8,
            bottom: 0.9,
            feather: 0.0,
        };
        assert!(!crop.is_identity());
        assert_eq!(crop.alpha_at(0.5, 0.5), 1.0);
        assert_eq!(crop.alpha_at(0.1, 0.5), 0.0);
        assert_eq!(crop.alpha_at(0.5, 0.95), 0.0);

        // Feathered crop
        crop.feather = 0.2;
        let mid_alpha = crop.alpha_at(0.2, 0.5); // at the left edge
        assert!(mid_alpha >= 0.0 && mid_alpha <= 1.0);

        // Normalise swapped bounds
        let mut inverted = CropRect { left: 0.9, top: 0.8, right: 0.1, bottom: 0.2, feather: -0.1 };
        inverted.normalise();
        assert_eq!(inverted.left, 0.1);
        assert_eq!(inverted.right, 0.9);
        assert_eq!(inverted.top, 0.2);
        assert_eq!(inverted.bottom, 0.8);
        assert_eq!(inverted.feather, 0.0);
    }

    #[test]
    fn test_corner_pin() {
        use crate::timeline::transform::CornerPin;

        let pin = CornerPin::identity();
        assert!(pin.is_identity());
        let gpu = pin.to_gpu();
        assert_eq!(gpu.len(), 8);
        assert_eq!(gpu[0], 0.0);
        assert_eq!(gpu[2], 1.0);
    }

    #[test]
    fn test_matte_mode() {
        use crate::timeline::transform::MatteMode;

        let m_alpha = MatteMode::AlphaMatte;
        assert_eq!(m_alpha.evaluate_matte([1.0, 1.0, 1.0, 0.75]), 0.75);

        let m_alpha_inv = MatteMode::AlphaMatteInverted;
        assert!((m_alpha_inv.evaluate_matte([1.0, 1.0, 1.0, 0.75]) - 0.25).abs() < 1e-5);

        let m_luma = MatteMode::LumaMatte;
        let white_luma = m_luma.evaluate_matte([1.0, 1.0, 1.0, 1.0]);
        assert!((white_luma - 1.0).abs() < 1e-3);

        let black_luma = m_luma.evaluate_matte([0.0, 0.0, 0.0, 1.0]);
        assert!((black_luma - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_geometric_masks() {
        use crate::timeline::transform::{GeometricMask, CropRect};

        let rect_mask = GeometricMask::Rectangle(CropRect {
            left: 0.2, top: 0.2, right: 0.8, bottom: 0.8, feather: 0.0,
        });
        assert_eq!(rect_mask.alpha_at(0.5, 0.5), 1.0);
        assert_eq!(rect_mask.alpha_at(0.1, 0.5), 0.0);

        let ellipse_mask = GeometricMask::Ellipse {
            center: [0.5, 0.5],
            radius: [0.3, 0.3],
            feather: 0.0,
        };
        assert_eq!(ellipse_mask.alpha_at(0.5, 0.5), 1.0);
        assert_eq!(ellipse_mask.alpha_at(0.1, 0.1), 0.0);

        let poly_mask = GeometricMask::Polygon {
            points: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            feather: 0.0,
        };
        assert_eq!(poly_mask.alpha_at(0.5, 0.2), 1.0);
        assert_eq!(poly_mask.alpha_at(0.1, 0.9), 0.0);
    }

    #[test]
    fn test_transform_mutations() {
        use crate::timeline::mutation::{set_blend_mode, set_crop, set_corner_pin, set_matte_mode};
        use crate::timeline::transform::{BlendMode, CropRect, CornerPin, MatteMode};

        let mut store = TimelineStore::new();
        let p = make_test_params(TrackId(0), 0, 1000, 0);
        let id = insert_clip(&mut store, p).unwrap();

        set_blend_mode(&mut store, id, BlendMode::Overlay).unwrap();
        assert_eq!(store.blend_mode_at(0), BlendMode::Overlay);

        let crop = CropRect { left: 0.1, top: 0.1, right: 0.9, bottom: 0.9, feather: 0.05 };
        set_crop(&mut store, id, crop).unwrap();
        assert_eq!(*store.crop_at(0), crop);

        let pin = CornerPin {
            top_left: [0.1, 0.1],
            top_right: [0.9, 0.0],
            bottom_left: [0.0, 0.9],
            bottom_right: [0.9, 0.9],
        };
        set_corner_pin(&mut store, id, pin).unwrap();
        assert_eq!(*store.corner_pin_at(0), pin);

        set_matte_mode(&mut store, id, MatteMode::LumaMatte).unwrap();
        assert_eq!(store.matte_mode_at(0), MatteMode::LumaMatte);
    }
}


