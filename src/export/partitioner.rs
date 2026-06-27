use crate::export::job::ExportJob;

#[derive(Debug, Clone)]
pub struct ExportSegment {
    pub index:        usize,
    pub frame_start:  usize,
    pub frame_end:    usize,
    pub pts_start:    i64,
    pub pts_end:      i64,
}

impl ExportSegment {
    pub fn frame_count(&self) -> usize {
        self.frame_end - self.frame_start
    }
}

pub struct SegmentPartitioner;

impl SegmentPartitioner {
    pub fn partition(job: &ExportJob) -> Vec<ExportSegment> {
        let total = job.total_frames();
        if total == 0 {
            return vec![];
        }
        let n_segments = job.render_threads.min(total);
        let fps = (total + n_segments - 1) / n_segments;

        let mut segments = Vec::with_capacity(n_segments);
        for i in 0..n_segments {
            let frame_start = i * fps;
            let frame_end = (frame_start + fps).min(total);
            if frame_start >= total {
                break;
            }
            let pts_start = job.frame_pts(frame_start);
            let pts_end = if frame_end < total {
                job.frame_pts(frame_end)
            } else {
                job.pts_out
            };
            segments.push(ExportSegment {
                index: i,
                frame_start,
                frame_end,
                pts_start,
                pts_end,
            });
        }
        
        if !segments.is_empty() {
            debug_assert_eq!(segments.last().unwrap().frame_end, total);
            for i in 0..segments.len().saturating_sub(1) {
                debug_assert!(Self::are_contiguous(&segments[i], &segments[i + 1]));
            }
        }

        segments
    }

    pub fn are_contiguous(a: &ExportSegment, b: &ExportSegment) -> bool {
        a.frame_end == b.frame_start && a.pts_end == b.pts_start
    }
}
