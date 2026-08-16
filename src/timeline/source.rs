use crate::timeline::ids::SourceId;
use crate::timeline::rational::Rational;
use std::path::PathBuf;
use std::sync::Arc;

pub fn is_still_image_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "tiff" | "tif" | "webp"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VideoStreamInfo {
    pub width: u32,
    pub height: u32,
    pub frame_rate: Rational,
    pub pixel_fmt: PixelFormat,
    pub color_info: ColorInfo,
    pub duration_pts: i64,
    pub is_vfr: bool,
    pub time_base: Rational,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AudioStreamInfo {
    pub sample_rate: u32,
    pub channels: u8,
    pub sample_fmt: SampleFormat,
    pub duration_pts: i64,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PixelFormat {
    Yuv420p,
    Yuv422p,
    Yuv444p,
    Nv12,
    Rgba8,
    Rgba16f,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TransferFunction {
    Bt709,
    Bt2020,
    Pq,
    Hlg,
    Srgb,
    Linear,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColorRange {
    Full,
    Limited,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MatrixCoefficients {
    Bt709,
    Bt601,
    Bt2020,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColorPrimaries {
    Bt709,
    Bt2020,
    Unknown,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ColorInfo {
    pub transfer_fn: TransferFunction,
    pub range: ColorRange,
    pub matrix: MatrixCoefficients,
    pub primaries: ColorPrimaries,
    pub bit_depth: u8,
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SampleFormat {
    F32Planar,
    F32Interleaved,
    I16Interleaved,
}

/// SoA source registry — all arrays parallel, indexed by SourceId.
#[derive(Debug, Clone)]
pub struct SourceRegistry {
    pub(crate) ids: Vec<SourceId>,
    pub(crate) paths: Vec<Arc<PathBuf>>,
    pub(crate) video_info: Vec<Option<VideoStreamInfo>>,
    pub(crate) audio_info: Vec<Option<AudioStreamInfo>>,
    pub(crate) proxy_ids: Vec<Option<SourceId>>,
    pub(crate) next_id: u32,
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceRegistry {
    pub fn new() -> Self {
        SourceRegistry {
            ids: Vec::new(),
            paths: Vec::new(),
            video_info: Vec::new(),
            audio_info: Vec::new(),
            proxy_ids: Vec::new(),
            next_id: 0,
        }
    }

    pub fn register(
        &mut self,
        path: PathBuf,
        video_info: Option<VideoStreamInfo>,
        audio_info: Option<AudioStreamInfo>,
    ) -> SourceId {
        let id = SourceId(self.next_id);
        self.next_id += 1;
        self.ids.push(id);
        self.paths.push(Arc::new(path));
        self.video_info.push(video_info);
        self.audio_info.push(audio_info);
        self.proxy_ids.push(None);
        // Log registration so UI import activity is visible in the main logs.
        if let Some(p) = self.paths.last() {
            log::debug!("SourceRegistry: registered {:?} -> {:?}", p, id);
        }
        id
    }

    pub fn set_proxy(
        &mut self,
        original_id: SourceId,
        proxy_id: SourceId,
    ) -> Result<(), SourceError> {
        let orig_idx = self
            .ids
            .iter()
            .position(|&id| id == original_id)
            .ok_or(SourceError::NotFound(original_id))?;
        let _proxy_idx = self
            .ids
            .iter()
            .position(|&id| id == proxy_id)
            .ok_or(SourceError::NotFound(proxy_id))?;
        self.proxy_ids[orig_idx] = Some(proxy_id);
        Ok(())
    }

    pub fn video_info(&self, id: SourceId) -> Result<&VideoStreamInfo, SourceError> {
        let idx = self
            .ids
            .iter()
            .position(|&x| x == id)
            .ok_or(SourceError::NotFound(id))?;
        self.video_info
            .get(idx)
            .and_then(|v| v.as_ref())
            .ok_or(SourceError::NoVideoStream(id))
    }

    pub fn audio_info(&self, id: SourceId) -> Result<&AudioStreamInfo, SourceError> {
        let idx = self
            .ids
            .iter()
            .position(|&x| x == id)
            .ok_or(SourceError::NotFound(id))?;
        self.audio_info
            .get(idx)
            .and_then(|v| v.as_ref())
            .ok_or(SourceError::NoAudioStream(id))
    }

    pub fn resolve_proxy(&self, id: SourceId) -> SourceId {
        let idx = self.ids.iter().position(|&x| x == id);
        if let Some(idx) = idx {
            self.proxy_ids.get(idx).and_then(|p| *p).unwrap_or(id)
        } else {
            id
        }
    }

    pub fn frame_size_bytes(&self, id: SourceId) -> Result<u64, SourceError> {
        let info = self.video_info(id)?;
        let pixels = (info.width as u64) * (info.height as u64);
        let chroma_w = (info.width as u64).div_ceil(2);
        let chroma_h = (info.height as u64).div_ceil(2);
        let bytes = match info.pixel_fmt {
            PixelFormat::Yuv420p => pixels + chroma_w * chroma_h * 2,
            PixelFormat::Yuv422p => pixels * 2,
            PixelFormat::Yuv444p => pixels * 3,
            PixelFormat::Nv12 => pixels + chroma_w * chroma_h * 2,
            PixelFormat::Rgba8 => pixels * 4,
            PixelFormat::Rgba16f => pixels * 8,
        };
        // Round up to nearest 256
        Ok((bytes + 255) & !255)
    }

    pub fn path(&self, id: SourceId) -> Option<Arc<PathBuf>> {
        let idx = self.ids.iter().position(|&x| x == id)?;
        self.paths.get(idx).cloned()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn path_registered(&self, path: &std::path::Path) -> bool {
        self.paths.iter().any(|p| p.as_ref() == path)
    }

    pub fn id_for_path(&self, path: &std::path::Path) -> Option<SourceId> {
        self.paths
            .iter()
            .position(|p| p.as_ref() == path)
            .map(|idx| self.ids[idx])
    }

    /// Return a clone of all registered source IDs. Used by the export
    /// pipeline to prime the prefetch queue before a render segment starts.
    pub fn all_source_ids(&self) -> Vec<SourceId> {
        self.ids.clone()
    }
}

#[derive(Debug)]
pub enum SourceError {
    NotFound(SourceId),
    NoVideoStream(SourceId),
    NoAudioStream(SourceId),
}
