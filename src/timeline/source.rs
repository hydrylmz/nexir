use std::path::PathBuf;
use std::sync::Arc;
use crate::timeline::ids::SourceId;
use crate::timeline::rational::Rational;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VideoStreamInfo {
    pub width:       u32,
    pub height:      u32,
    pub frame_rate:  Rational,   
    pub pixel_fmt:   PixelFormat,
    pub color_space: ColorSpace,
    pub duration_pts: i64,       
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AudioStreamInfo {
    pub sample_rate: u32,   
    pub channels:    u8,
    pub sample_fmt:  SampleFormat,
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
pub enum ColorSpace {
    Bt601,
    Bt709,
    Bt2020,
    Srgb,
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
    pub(crate) ids:        Vec<SourceId>,
    pub(crate) paths:      Vec<Arc<PathBuf>>,
    pub(crate) video_info: Vec<Option<VideoStreamInfo>>,
    pub(crate) audio_info: Vec<Option<AudioStreamInfo>>,
    pub(crate) proxy_ids:  Vec<Option<SourceId>>,
    pub(crate) next_id:    u32,
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
        path:       PathBuf,
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
        id
    }

    pub fn set_proxy(
        &mut self,
        original_id: SourceId,
        proxy_id:    SourceId,
    ) -> Result<(), SourceError> {
        let orig_idx = self.ids.iter().position(|&id| id == original_id).ok_or(SourceError::NotFound(original_id))?;
        let _proxy_idx = self.ids.iter().position(|&id| id == proxy_id).ok_or(SourceError::NotFound(proxy_id))?;
        self.proxy_ids[orig_idx] = Some(proxy_id);
        Ok(())
    }

    pub fn video_info(&self, id: SourceId) -> Result<&VideoStreamInfo, SourceError> {
        let idx = self.ids.iter().position(|&x| x == id).ok_or(SourceError::NotFound(id))?;
        self.video_info.get(idx).and_then(|v| v.as_ref()).ok_or(SourceError::NoVideoStream(id))
    }

    pub fn audio_info(&self, id: SourceId) -> Result<&AudioStreamInfo, SourceError> {
        let idx = self.ids.iter().position(|&x| x == id).ok_or(SourceError::NotFound(id))?;
        self.audio_info.get(idx).and_then(|v| v.as_ref()).ok_or(SourceError::NoAudioStream(id))
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
        let bytes = match info.pixel_fmt {
            PixelFormat::Yuv420p => pixels * 3 / 2,
            PixelFormat::Yuv422p => pixels * 2,
            PixelFormat::Yuv444p => pixels * 3,
            PixelFormat::Nv12    => pixels * 3 / 2,
            PixelFormat::Rgba8   => pixels * 4,
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
        self.paths.iter().position(|p| p.as_ref() == path).map(|idx| self.ids[idx])
    }
}

#[derive(Debug)]
pub enum SourceError {
    NotFound(SourceId),
    NoVideoStream(SourceId),
    NoAudioStream(SourceId),
}
