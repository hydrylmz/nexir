use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::project::{Project, ProjectSettings};
use crate::timeline::ids::SourceId;
use crate::timeline::source::{AudioStreamInfo, SourceRegistry, VideoStreamInfo};
use crate::timeline::store::TimelineStore;
use crate::timeline::track::TrackList;

/// The canonical .nexp file format version.
const FORMAT_VERSION: u32 = 1;

/// The on-disk representation of a Nexir project (.nexp).
///
/// # Format
/// JSON file with a `format_version` header for forward-compat.
/// All project data is serialized flat – the `SourceRegistry` uses plain
/// `PathBuf` instead of `Arc<PathBuf>` so serde can handle it natively.
#[derive(Serialize, Deserialize)]
pub struct ProjectFile {
    pub format_version: u32,
    pub name: String,
    pub settings: ProjectSettings,
    pub tracks: TrackList,
    pub clips: TimelineStore,
    pub sources: SourceRegistryFile,
}

/// Serializable mirror of `SourceRegistry` that uses `PathBuf` directly
/// (instead of `Arc<PathBuf>`) so serde derives work without extra feature flags.
#[derive(Serialize, Deserialize)]
pub struct SourceRegistryFile {
    pub ids: Vec<SourceId>,
    pub paths: Vec<PathBuf>,
    pub video_info: Vec<Option<VideoStreamInfo>>,
    pub audio_info: Vec<Option<AudioStreamInfo>>,
    pub proxy_ids: Vec<Option<SourceId>>,
    pub next_id: u32,
}

// ─────────────────────────────────────────────
// Project → ProjectFile conversion
// ─────────────────────────────────────────────

impl From<&Project> for ProjectFile {
    fn from(project: &Project) -> Self {
        let sources_lock = project.sources.read().unwrap();
        let source_file = SourceRegistryFile {
            ids: sources_lock.ids.clone(),
            paths: sources_lock.paths.iter().map(|p| (**p).clone()).collect(),
            video_info: sources_lock.video_info.clone(),
            audio_info: sources_lock.audio_info.clone(),
            proxy_ids: sources_lock.proxy_ids.clone(),
            next_id: sources_lock.next_id,
        };
        drop(sources_lock);

        ProjectFile {
            format_version: FORMAT_VERSION,
            name: project.name.clone(),
            settings: project.settings.clone(),
            tracks: project.tracks.clone(),
            clips: project.clips.clone(),
            sources: source_file,
        }
    }
}

// ─────────────────────────────────────────────
// ProjectFile → Project conversion
// ─────────────────────────────────────────────

impl From<ProjectFile> for Project {
    fn from(pf: ProjectFile) -> Self {
        let mut registry = SourceRegistry::empty();

        // Re-number IDs and rebuild arrays.
        // We do a simple push loop to keep `next_id` consistent.
        for idx in 0..pf.sources.ids.len() {
            let path = pf.sources.paths[idx].clone();
            let mut vi = pf.sources.video_info[idx].clone();
            let mut ai = pf.sources.audio_info[idx].clone();
            let is_still_image = crate::timeline::source::is_still_image_path(&path);

            // If the project file was generated externally without stream info, reprobe it now
            if vi.is_none() || ai.is_none() {
                if let Ok(demuxer) = crate::io::demuxer::Demuxer::open(&path) {
                    let project_tb = crate::timeline::rational::Rational {
                        num: 1,
                        den: 90_000,
                    };

                    if vi.is_none() {
                        vi = demuxer.video_stream.as_ref().map(|s| {
                            crate::timeline::source::VideoStreamInfo {
                                width: s.width.unwrap_or(1920),
                                height: s.height.unwrap_or(1080),
                                frame_rate: if is_still_image {
                                    crate::timeline::rational::Rational { num: 0, den: 1 }
                                } else {
                                    s.frame_rate.unwrap_or(crate::timeline::rational::Rational {
                                        num: 30,
                                        den: 1,
                                    })
                                },
                                pixel_fmt: crate::timeline::source::PixelFormat::Yuv420p,
                                color_info: s.color_info,
                                duration_pts: if is_still_image {
                                    0
                                } else {
                                    project_tb.from_pts(s.duration, s.time_base)
                                },
                                is_vfr: !is_still_image && s.is_vfr,
                                time_base: s.time_base,
                            }
                        });
                    }
                    if ai.is_none() {
                        ai = demuxer.audio_stream.as_ref().map(|s| {
                            crate::timeline::source::AudioStreamInfo {
                                sample_rate: 48000, // standard default
                                channels: 2,
                                sample_fmt: crate::timeline::source::SampleFormat::F32Interleaved,
                                duration_pts: project_tb.from_pts(s.duration, s.time_base),
                            }
                        });
                    }
                }
            }

            if is_still_image {
                if let Some(info) = vi.as_mut() {
                    info.frame_rate = crate::timeline::rational::Rational { num: 0, den: 1 };
                    info.duration_pts = 0;
                    info.is_vfr = false;
                }
            }

            let pid = pf.sources.proxy_ids[idx];
            let sid = registry.register(path, vi, ai);
            if let Some(proxy) = pid {
                let _ = registry.set_proxy(sid, proxy);
            }
        }

        // Verify data consistency.
        debug_assert_eq!(registry.len(), pf.sources.ids.len());

        let mut clips = pf.clips;
        if clips.kind.is_empty() && !clips.is_empty() {
            clips.kind.resize(clips.len(), crate::timeline::store::ClipKind::Video);
        }

        Project {
            name: pf.name,
            settings: pf.settings,
            tracks: pf.tracks,
            clips,
            sources: Arc::new(std::sync::RwLock::new(registry)),
        }
    }
}

// ─────────────────────────────────────────────
// High-level save / load helpers
// ─────────────────────────────────────────────

impl ProjectFile {
    /// Serialise a `Project` and write it to `path` as a `.nexp` JSON file.
    pub fn save(
        path: impl AsRef<Path>,
        project: &Project,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let pf = ProjectFile::from(project);
        let json = serde_json::to_string_pretty(&pf)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a `.nexp` file from `path` and reconstruct a `Project`.
    pub fn load(path: impl AsRef<Path>) -> Result<Project, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let pf: ProjectFile = serde_json::from_str(&json)?;

        // Basic forward-compat: reject files from newer format versions.
        if pf.format_version > FORMAT_VERSION {
            return Err(format!(
                "Project file uses format v{}, but this version of nexir only supports v{}",
                pf.format_version, FORMAT_VERSION,
            )
            .into());
        }

        Ok(Project::from(pf))
    }
}

// ═════════════════════════════════════════════
// Internal access helpers for serialization
// ═════════════════════════════════════════════

impl SourceRegistry {
    /// Create an empty registry (used during deserialization).
    pub(crate) fn empty() -> Self {
        SourceRegistry::new()
    }
}
