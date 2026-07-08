use crate::timeline::store::TimelineStore;
use crate::timeline::ids::TrackId;

/// A resolved active-clip descriptor, ready for the scheduler to process.
#[derive(Debug, Clone)]
pub struct ActiveClip {
    /// Raw array index into TimelineStore (not a ClipId — avoids a second lookup).
    pub store_index: usize,
    /// PTS within the SOURCE file that should be decoded for this query PTS.
    /// source_pts = query_pts - pts_in + source_in   (all in project timebase)
    pub source_pts:  i64,
}

pub fn query_active(
    store:     &TimelineStore,
    query_pts: i64,
    out:       &mut Vec<ActiveClip>,
) {
    let boundary = store.pts_in_slice().partition_point(|&x| x <= query_pts);
    for i in 0..boundary {
        if store.pts_out_at(i) > query_pts {
            let offset = query_pts - store.pts_in_at(i);
            let source_pts = crate::timeline::rational::speed_scale_pts(offset, store.speed_at(i)) + store.source_in_at(i);
            out.push(ActiveClip { store_index: i, source_pts });
        }
    }
}

pub fn query_active_on_track(
    store:     &TimelineStore,
    query_pts: i64,
    track_id:  TrackId,
    out:       &mut Vec<ActiveClip>,
) {
    query_active(store, query_pts, out);
    out.retain(|entry| store.track_id_at(entry.store_index) == track_id);
}

pub fn next_boundary(store: &TimelineStore, query_pts: i64) -> Option<i64> {
    let mut min_bound: Option<i64> = None;
    for i in 0..store.len() {
        let pts_in = store.pts_in_at(i);
        let pts_out = store.pts_out_at(i);
        
        if pts_in > query_pts {
            min_bound = Some(min_bound.unwrap_or(i64::MAX).min(pts_in));
        }
        if pts_out > query_pts {
            min_bound = Some(min_bound.unwrap_or(i64::MAX).min(pts_out));
        }
    }
    min_bound
}

pub fn query_overlap(
    store: &TimelineStore,
    a:     i64,
    b:     i64,
    out:   &mut Vec<usize>,
) {
    for i in 0..store.len() {
        if store.pts_in_at(i) < b && store.pts_out_at(i) > a {
            out.push(i);
        }
    }
}
