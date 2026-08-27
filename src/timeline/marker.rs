// src/timeline/marker.rs
// Timeline and clip markers with color, duration, and range queries.

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimelineMarker {
    pub id: u32,
    pub pts: i64,
    pub name: String,
    pub color: [f32; 4],
    pub duration_pts: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClipMarker {
    pub id: u32,
    pub source_pts: i64,
    pub name: String,
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MarkerList {
    markers: Vec<TimelineMarker>,
    next_id: u32,
}

impl MarkerList {
    pub fn new() -> Self {
        Self {
            markers: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add(&mut self, pts: i64, name: String, color: [f32; 4], duration_pts: Option<i64>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.markers.push(TimelineMarker {
            id,
            pts,
            name,
            color,
            duration_pts,
        });
        self.markers.sort_by_key(|m| m.pts);
        id
    }

    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.markers.iter().position(|m| m.id == id) {
            self.markers.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn all(&self) -> &[TimelineMarker] {
        &self.markers
    }

    pub fn in_range(&self, start_pts: i64, end_pts: i64) -> Vec<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| {
                let m_end = m.pts + m.duration_pts.unwrap_or(0);
                m.pts < end_pts && m_end >= start_pts
            })
            .collect()
    }

    pub fn nearest(&self, pts: i64) -> Option<&TimelineMarker> {
        self.markers.iter().min_by_key(|m| (m.pts - pts).abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_add_remove_query() {
        let mut list = MarkerList::new();
        let m1 = list.add(10_000, "Intro".into(), [1.0, 0.0, 0.0, 1.0], None);
        let m2 = list.add(50_000, "Verse".into(), [0.0, 1.0, 0.0, 1.0], Some(10_000));
        let m3 = list.add(100_000, "Outro".into(), [0.0, 0.0, 1.0, 1.0], None);

        assert_eq!(list.all().len(), 3);
        assert_eq!(list.nearest(12_000).unwrap().id, m1);
        assert_eq!(list.nearest(48_000).unwrap().id, m2);

        let in_range = list.in_range(40_000, 70_000);
        assert_eq!(in_range.len(), 1);
        assert_eq!(in_range[0].id, m2);

        assert!(list.remove(m2));
        assert_eq!(list.all().len(), 2);
    }
}
