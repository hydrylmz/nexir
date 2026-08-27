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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum VideoRotation {
    #[default]
    None,
    Rotate90,
    Rotate180,
    Rotate270,
}

impl VideoRotation {
    pub fn from_degrees(deg: i32) -> Self {
        match ((deg % 360) + 360) % 360 {
            90 => VideoRotation::Rotate90,
            180 => VideoRotation::Rotate180,
            270 => VideoRotation::Rotate270,
            _ => VideoRotation::None,
        }
    }

    pub fn to_degrees(&self) -> i32 {
        match self {
            VideoRotation::None => 0,
            VideoRotation::Rotate90 => 90,
            VideoRotation::Rotate180 => 180,
            VideoRotation::Rotate270 => 270,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum MediaStatus {
    #[default]
    Available,
    Missing,
    Corrupted,
    UnsupportedFormat,
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
    #[serde(default)]
    pub rotation: VideoRotation,
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
    P010,
    Yuv420p10,
    Yuv422p10,
    Yuv444p10,
    Yuv420p12,
    Yuv444p12,
    Rgba8,
    Rgba16f,
}

impl PixelFormat {
    /// True if this pixel format uses 10-bit or greater depth per channel (e.g. P010, YUV420p10, 12-bit).
    pub fn is_10bit(&self) -> bool {
        matches!(
            self,
            PixelFormat::P010
                | PixelFormat::Yuv420p10
                | PixelFormat::Yuv422p10
                | PixelFormat::Yuv444p10
                | PixelFormat::Yuv420p12
                | PixelFormat::Yuv444p12
        )
    }

    /// Number of bytes used per sample channel in memory (1 for 8-bit, 2 for 10/12/16-bit).
    pub fn bytes_per_sample(&self) -> usize {
        if self.is_10bit() || *self == PixelFormat::Rgba16f {
            2
        } else {
            1
        }
    }

    /// Whether this pixel format is YUV (planar or semi-planar).
    pub fn is_yuv(&self) -> bool {
        !matches!(self, PixelFormat::Rgba8 | PixelFormat::Rgba16f)
    }

    /// Whether this is a semi-planar YUV format (NV12 or P010).
    pub fn is_semi_planar(&self) -> bool {
        matches!(self, PixelFormat::Nv12 | PixelFormat::P010)
    }
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

impl Default for ColorInfo {
    fn default() -> Self {
        Self::bt709()
    }
}

impl ColorInfo {
    pub fn bt709() -> Self {
        Self {
            transfer_fn: TransferFunction::Bt709,
            range: ColorRange::Limited,
            matrix: MatrixCoefficients::Bt709,
            primaries: ColorPrimaries::Bt709,
            bit_depth: 8,
        }
    }

    pub fn bt601() -> Self {
        Self {
            transfer_fn: TransferFunction::Bt709,
            range: ColorRange::Limited,
            matrix: MatrixCoefficients::Bt601,
            primaries: ColorPrimaries::Bt709,
            bit_depth: 8,
        }
    }

    pub fn bt2020(is_hdr: bool, bit_depth: u8) -> Self {
        Self {
            transfer_fn: if is_hdr { TransferFunction::Pq } else { TransferFunction::Bt2020 },
            range: ColorRange::Limited,
            matrix: MatrixCoefficients::Bt2020,
            primaries: ColorPrimaries::Bt2020,
            bit_depth,
        }
    }

    pub fn srgb() -> Self {
        Self {
            transfer_fn: TransferFunction::Srgb,
            range: ColorRange::Full,
            matrix: MatrixCoefficients::Bt709,
            primaries: ColorPrimaries::Bt709,
            bit_depth: 8,
        }
    }

    /// Construct `ColorInfo` from raw FFmpeg codecpar / frame values with intelligent fallbacks.
    pub fn from_ffmpeg(
        matrix_raw: i32,
        range_raw: i32,
        trc_raw: i32,
        pri_raw: i32,
        bit_depth: u8,
        width: u32,
        height: u32,
    ) -> Self {
        let range = match range_raw {
            1 => ColorRange::Limited,
            2 => ColorRange::Full,
            _ => ColorRange::Unknown,
        };

        let matrix = match matrix_raw {
            1 => MatrixCoefficients::Bt709,
            5 | 6 => MatrixCoefficients::Bt601,
            9 | 10 => MatrixCoefficients::Bt2020,
            _ => MatrixCoefficients::Unknown,
        };

        let primaries = match pri_raw {
            1 => ColorPrimaries::Bt709,
            9 => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unknown,
        };

        let transfer_fn = match trc_raw {
            1 => TransferFunction::Bt709,
            8 => TransferFunction::Linear,
            13 => TransferFunction::Srgb,
            14 | 15 => TransferFunction::Bt2020,
            16 => TransferFunction::Pq,
            18 => TransferFunction::Hlg,
            _ => TransferFunction::Unknown,
        };

        let mut ci = Self {
            transfer_fn,
            range,
            matrix,
            primaries,
            bit_depth: if bit_depth == 0 { 8 } else { bit_depth },
        };

        // If matrix is unknown, deduce it from resolution / primaries
        if ci.matrix == MatrixCoefficients::Unknown {
            ci.matrix = ci.effective_matrix(width, height);
        }
        if ci.range == ColorRange::Unknown {
            ci.range = ci.effective_range();
        }
        if ci.primaries == ColorPrimaries::Unknown {
            ci.primaries = ci.effective_primaries(width, height);
        }
        if ci.transfer_fn == TransferFunction::Unknown {
            ci.transfer_fn = if ci.matrix == MatrixCoefficients::Bt2020 && ci.bit_depth >= 10 {
                TransferFunction::Pq
            } else {
                TransferFunction::Bt709
            };
        }

        ci
    }

    /// Resolve effective matrix coefficients with resolution heuristics.
    pub fn effective_matrix(&self, width: u32, height: u32) -> MatrixCoefficients {
        match self.matrix {
            MatrixCoefficients::Unknown => {
                if self.primaries == ColorPrimaries::Bt2020
                    || matches!(self.transfer_fn, TransferFunction::Pq | TransferFunction::Hlg | TransferFunction::Bt2020)
                    || (width >= 3840 || height >= 2160)
                {
                    MatrixCoefficients::Bt2020
                } else if width > 0 && width < 1280 && height > 0 && height < 720 {
                    MatrixCoefficients::Bt601
                } else {
                    MatrixCoefficients::Bt709
                }
            }
            known => known,
        }
    }

    /// Resolve effective color range (defaults to Limited for standard broadcast video).
    pub fn effective_range(&self) -> ColorRange {
        match self.range {
            ColorRange::Unknown => ColorRange::Limited,
            known => known,
        }
    }

    /// Resolve effective color primaries.
    pub fn effective_primaries(&self, width: u32, height: u32) -> ColorPrimaries {
        match self.primaries {
            ColorPrimaries::Unknown => match self.effective_matrix(width, height) {
                MatrixCoefficients::Bt2020 => ColorPrimaries::Bt2020,
                _ => ColorPrimaries::Bt709,
            },
            known => known,
        }
    }

    /// Whether this color profile represents high dynamic range (HDR10, HLG, or 10-bit BT.2020).
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer_fn, TransferFunction::Pq | TransferFunction::Hlg)
            || (self.matrix == MatrixCoefficients::Bt2020 && self.bit_depth >= 10)
    }
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
    pub(crate) statuses: Vec<MediaStatus>,
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
            statuses: Vec::new(),
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
        self.statuses.push(MediaStatus::Available);
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

    pub fn media_status(&self, id: SourceId) -> MediaStatus {
        if let Some(idx) = self.ids.iter().position(|&x| x == id) {
            self.statuses.get(idx).copied().unwrap_or(MediaStatus::Missing)
        } else {
            MediaStatus::Missing
        }
    }

    pub fn set_media_status(&mut self, id: SourceId, status: MediaStatus) {
        if let Some(idx) = self.ids.iter().position(|&x| x == id) {
            if idx < self.statuses.len() {
                self.statuses[idx] = status;
            }
        }
    }

    pub fn is_available(&self, id: SourceId) -> bool {
        self.media_status(id) == MediaStatus::Available
    }

    /// Refresh file existence check for all registered media sources.
    pub fn check_media_availability(&mut self) {
        for i in 0..self.paths.len() {
            let exists = self.paths[i].exists();
            if !exists {
                self.statuses[i] = MediaStatus::Missing;
            } else if self.statuses[i] == MediaStatus::Missing {
                self.statuses[i] = MediaStatus::Available;
            }
        }
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
            PixelFormat::Yuv420p   => pixels + chroma_w * chroma_h * 2,
            PixelFormat::Yuv422p   => pixels * 2,
            PixelFormat::Yuv444p   => pixels * 3,
            PixelFormat::Nv12      => pixels + chroma_w * chroma_h * 2,
            // P010 and 10/12-bit formats are 2 bytes per sample
            PixelFormat::P010      => (pixels + chroma_w * chroma_h * 2) * 2,
            PixelFormat::Yuv420p10 => (pixels + chroma_w * chroma_h * 2) * 2,
            PixelFormat::Yuv422p10 => pixels * 4,
            PixelFormat::Yuv444p10 => pixels * 6,
            PixelFormat::Yuv420p12 => (pixels + chroma_w * chroma_h * 2) * 2,
            PixelFormat::Yuv444p12 => pixels * 6,
            PixelFormat::Rgba8     => pixels * 4,
            PixelFormat::Rgba16f   => pixels * 8,
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

/// Detects if a filename follows an image sequence pattern (e.g. `frame_0001.png`, `render.0042.exr`).
pub fn detect_image_sequence_pattern(filename: &str) -> Option<(String, usize, usize)> {
    let path = std::path::Path::new(filename);
    let stem = path.file_stem()?.to_str()?;
    let ext = path.extension()?.to_str()?;

    // Find trailing sequence digits in the file stem
    let digits_start = stem.rfind(|c: char| !c.is_ascii_digit()).map(|i| i + 1).unwrap_or(0);
    if digits_start >= stem.len() {
        return None;
    }
    let digits_str = &stem[digits_start..];
    let num_digits = digits_str.len();
    let current_index: usize = digits_str.parse().ok()?;
    let prefix = &stem[..digits_start];

    let pattern = format!("{}{{:0{}}}..{}", prefix, num_digits, ext);
    Some((pattern, current_index, num_digits))
}

#[derive(Debug)]
pub enum SourceError {
    NotFound(SourceId),
    NoVideoStream(SourceId),
    NoAudioStream(SourceId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_rotation() {
        assert_eq!(VideoRotation::from_degrees(0), VideoRotation::None);
        assert_eq!(VideoRotation::from_degrees(90), VideoRotation::Rotate90);
        assert_eq!(VideoRotation::from_degrees(180), VideoRotation::Rotate180);
        assert_eq!(VideoRotation::from_degrees(270), VideoRotation::Rotate270);
        assert_eq!(VideoRotation::from_degrees(450), VideoRotation::Rotate90); // 450 % 360 == 90
        assert_eq!(VideoRotation::from_degrees(-90), VideoRotation::Rotate270);

        assert_eq!(VideoRotation::Rotate90.to_degrees(), 90);
        assert_eq!(VideoRotation::Rotate180.to_degrees(), 180);
        assert_eq!(VideoRotation::Rotate270.to_degrees(), 270);
        assert_eq!(VideoRotation::None.to_degrees(), 0);
    }

    #[test]
    fn test_media_status_tracking() {
        let mut registry = SourceRegistry::new();
        let id = registry.register(PathBuf::from("test.mp4"), None, None);
        assert_eq!(registry.media_status(id), MediaStatus::Available);
        assert!(registry.is_available(id));

        registry.set_media_status(id, MediaStatus::Missing);
        assert_eq!(registry.media_status(id), MediaStatus::Missing);
        assert!(!registry.is_available(id));

        // Unknown source
        assert_eq!(registry.media_status(SourceId(999)), MediaStatus::Missing);
        assert!(!registry.is_available(SourceId(999)));
    }

    #[test]
    fn test_image_sequence_pattern_detection() {
        let res = detect_image_sequence_pattern("render_0042.png");
        assert!(res.is_some());
        let (pattern, idx, digits) = res.unwrap();
        assert_eq!(pattern, "render_{:04}..png");
        assert_eq!(idx, 42);
        assert_eq!(digits, 4);

        let non_seq = detect_image_sequence_pattern("photo.jpg");
        assert!(non_seq.is_none());
    }

    #[test]
    fn test_12bit_pixel_formats() {
        assert!(PixelFormat::Yuv420p12.is_10bit());
        assert!(PixelFormat::Yuv444p12.is_10bit());
        assert_eq!(PixelFormat::Yuv420p12.bytes_per_sample(), 2);
    }
}
