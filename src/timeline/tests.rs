// src/timeline/tests.rs
#[cfg(test)]
mod tests {
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
            pts_in: 0,
            pts_out: 100,
            source_in: 0,
            layer_order: 0,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            speed: 1.0,
            pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            pts_in: 50,
            pts_out: 200,
            source_in: 0,
            layer_order: 1,
            opacity: 1.0,
            transform: ClipTransform::identity(),
            speed: 1.0,
            pitch: 0.0,
        };
        let p3 = ClipInsertParams {
            track_id: TrackId(0),
            source_id: SourceId(0),
            pts_in: 150,
            pts_out: 300,
            source_in: 0,
            layer_order: 2,
            opacity: 1.0,
            transform: ClipTransform::identity(),
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
    fn insert_maintains_sort() {
        let mut store = TimelineStore::new();
        let p1 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            pts_in: 300, pts_out: 400, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
        };
        let p3 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            pts_in: 200, pts_out: 300, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
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
            pts_in: 50, pts_out: 200, source_in: 1000, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(1), source_id: SourceId(1),
            pts_in: 50, pts_out: 200, source_in: 1000, layer_order: 1, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 2.0, pitch: 0.0,
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
            pts_in: 0, pts_out: 100, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
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
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
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
            pts_in: 100, pts_out: 200, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
        };
        let p2 = ClipInsertParams {
            track_id: TrackId(0), source_id: SourceId(0),
            pts_in: 200, pts_out: 300, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
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
            pts_in: 100, pts_out: 100, source_in: 0, layer_order: 0, opacity: 1.0, transform: ClipTransform::identity(),
            speed: 1.0, pitch: 0.0,
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
}
