use std::path::PathBuf;
use crate::timeline::rational::Rational;
use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients};

/// Target output codec for video.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,    // AV_CODEC_ID_H264   = 27  — universal compatibility
    H265,    // AV_CODEC_ID_HEVC   = 173 — better compression, slower encode
    ProRes,  // AV_CODEC_ID_PRORES = 147 — lossless-ish, large files, fast encode
    Vp9,     // AV_CODEC_ID_VP9    = 167 — web delivery
}

impl VideoCodec {
    pub fn ffmpeg_id(self) -> u32 {
        match self {
            Self::H264 => 27,
            Self::H265 => 173,
            Self::ProRes => 147,
            Self::Vp9 => 167,
        }
    }

    pub fn supports_crf(self) -> bool {
        matches!(self, Self::H264 | Self::H265)
    }

    /// Whether this codec can carry 10-bit HDR in a way players actually honour.
    ///
    /// H.264 High10 is legal and libx264 will encode `yuv420p10le`, but no
    /// hardware H.264 encoder accepts 10-bit input and HDR10 in AVC is barely
    /// supported by players — so it is not offered as an HDR target.
    pub fn supports_hdr(self) -> bool {
        matches!(self, Self::H265 | Self::ProRes | Self::Vp9)
    }
}

/// Target output codec for audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,   // AV_CODEC_ID_AAC  = 86018 — universal
    Opus,  // AV_CODEC_ID_OPUS = 86076 — best quality/size ratio
    Pcm,   // AV_CODEC_ID_PCM_S16LE = 65536 — lossless, large
}

impl AudioCodec {
    pub fn ffmpeg_id(self) -> u32 {
        match self {
            Self::Aac => 86018,
            Self::Opus => 86076,
            Self::Pcm => 65536,
        }
    }
}

/// Container format for the output file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Mp4,  // "mp4"  — most compatible
    Mkv,  // "matroska" — flexible, good for H.265/VP9
    Mov,  // "mov"  — required for ProRes
}

impl Container {
    pub fn format_name(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "matroska",
            Self::Mov => "mov",
        }
    }
}

/// SMPTE ST 2086 mastering-display colour volume plus CTA-861.3 content light
/// level — the static metadata an HDR10 file must carry alongside its BT.2020/PQ
/// tags.
///
/// Values are held in the integer units the standards themselves use, so nothing
/// is rounded on the way to FFmpeg:
///
/// * chromaticities in increments of `0.00002` (denominator 50000)
/// * luminances in increments of `0.0001 cd/m²` (denominator 10000)
///
/// This is what `ffprobe -show_frames` reports as `mastering_display_metadata`
/// and `content_light_level`, and what a display reads to decide how to map the
/// signal it was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hdr10Metadata {
    /// Display primaries as numerators over 50000, in `R.x R.y G.x G.y B.x B.y`
    /// order.
    pub primaries: [i32; 6],
    /// White point as numerators over 50000, `x y`.
    pub white_point: [i32; 2],
    /// Minimum mastering-display luminance, numerator over 10000.
    pub min_luminance: i32,
    /// Maximum mastering-display luminance, numerator over 10000.
    pub max_luminance: i32,
    /// Maximum content light level, cd/m².
    pub max_cll: u32,
    /// Maximum frame-average light level, cd/m².
    pub max_fall: u32,
}

impl Hdr10Metadata {
    /// The conventional HDR10 grade: BT.2020 primaries, D65, 1000 nit peak.
    ///
    /// Identical to the `master-display=G(8500,39850)B(6550,2300)R(35400,14600)
    /// WP(15635,16450)L(10000000,1)` string used across the HDR tooling
    /// ecosystem, so a file exported with this matches what a 1000-nit grade is
    /// expected to declare.
    pub fn bt2020_1000_nits() -> Self {
        Self {
            // BT.2020: R(0.708,0.292) G(0.170,0.797) B(0.131,0.046)
            primaries:     [35400, 14600, 8500, 39850, 6550, 2300],
            // D65: (0.3127, 0.3290)
            white_point:   [15635, 16450],
            min_luminance: 1,          // 0.0001 cd/m²
            max_luminance: 10_000_000, // 1000 cd/m²
            max_cll:       1000,
            max_fall:      400,
        }
    }

    /// Peak mastering luminance in nits, for the tone-map / PQ encode stage.
    pub fn peak_nits(&self) -> f32 {
        self.max_luminance as f32 / 10_000.0
    }
}

/// Quality/bitrate mode for video encoding.
#[derive(Debug, Clone, Copy)]
pub enum VideoQuality {
    Crf(u32),
    TargetBitrate(u64),
}

/// CPU encoder preset (speed vs compression ratio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuPreset {
    Ultrafast,
    Superfast,
    Veryfast,
    Faster,
    Fast,
    Medium,
    Slow,
    Slower,
    Veryslow,
}

impl CpuPreset {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ultrafast => "ultrafast",
            Self::Superfast => "superfast",
            Self::Veryfast  => "veryfast",
            Self::Faster    => "faster",
            Self::Fast      => "fast",
            Self::Medium    => "medium",
            Self::Slow      => "slow",
            Self::Slower    => "slower",
            Self::Veryslow  => "veryslow",
        }
    }
}

/// Complete description of an export job.
#[derive(Debug, Clone)]
pub struct ExportJob {
    pub output_path:   PathBuf,
    pub container:     Container,
    pub video_codec:   VideoCodec,
    pub audio_codec:   AudioCodec,
    pub quality:       VideoQuality,
    pub audio_bitrate: u64,
    pub pts_in:        i64,
    pub pts_out:       i64,
    pub width:         u32,
    pub height:        u32,
    pub frame_rate:    Rational,
    pub project_tb:    Rational,
    pub render_threads: usize,
    pub cpu_preset:    CpuPreset,
    /// Colour description written onto the output stream (encoder VUI + container
    /// `colr` / Matroska colour element).
    ///
    /// This describes the pixels the encoder is *fed*, not the source clips. For
    /// an SDR export the renderer tone-maps HDR input down to Rec.709 first (see
    /// `ExportRenderer::compile_graph`), so the default is `bt709()`.
    ///
    /// Setting this to a PQ/HLG profile — most easily via [`Self::set_hdr10`] —
    /// switches the whole pipeline to HDR passthrough: the renderer stops
    /// tone-mapping, the encoder picks a 10-bit pixel format, and the mastering
    /// display / content light level side data below is attached. Setting it by
    /// hand and leaving `hdr10` at `None` is legal but produces a file with no
    /// static metadata, which many displays treat as an unmastered grade.
    pub output_color:  ColorInfo,
    /// HDR10 static metadata (SMPTE ST 2086 + CTA-861.3), attached to both the
    /// encoder and the container when present.
    ///
    /// `None` for every SDR export. [`Self::set_hdr10`] fills it in alongside
    /// `output_color`; nothing else in the pipeline writes it.
    pub hdr10:         Option<Hdr10Metadata>,
}

impl ExportJob {
    pub fn total_frames(&self) -> usize {
        let duration_pts = self.pts_out - self.pts_in;
        let numer = duration_pts as u64 * self.frame_rate.num as u64;
        let denom = self.project_tb.den as u64 * self.frame_rate.den as u64;
        numer.div_ceil(denom) as usize
    }

    pub fn frame_pts(&self, n: usize) -> i64 {
        let frame_dur = self.project_tb.den / self.frame_rate.num;
        self.pts_in + n as i64 * frame_dur
    }

    /// True when this job asks for an HDR (PQ or HLG) output.
    ///
    /// This is THE switch for the HDR path. Everything downstream keys off it:
    /// `ExportRenderer` skips its tone-map node, `VideoEncoder` selects a 10-bit
    /// pixel format and encodes the PQ/HLG curve, and the muxer attaches the
    /// static metadata. Reading `output_color.is_hdr()` directly elsewhere would
    /// work but would also fire for 10-bit SDR BT.2020, which is not a passthrough
    /// job — hence the narrower test here.
    pub fn is_hdr(&self) -> bool {
        use crate::timeline::source::TransferFunction;
        matches!(
            self.output_color.transfer_fn,
            TransferFunction::Pq | TransferFunction::Hlg
        )
    }

    /// Configure this job as a 10-bit HDR10 (BT.2020 + PQ) export.
    ///
    /// Sets `output_color` and `hdr10` together, which is the only combination
    /// that yields a correctly described file: the colour tags say how to
    /// interpret the samples and the static metadata says what they were graded
    /// against.
    ///
    /// Returns an error for a codec that cannot carry it, rather than silently
    /// producing an 8-bit file wearing HDR tags — the exact failure the SDR-only
    /// pipeline used to have.
    pub fn set_hdr10(&mut self, metadata: Hdr10Metadata) -> Result<(), JobError> {
        if !self.video_codec.supports_hdr() {
            return Err(JobError::HdrNotSupported(self.video_codec));
        }
        self.output_color = ColorInfo::bt2020(true, 10);
        self.hdr10 = Some(metadata);
        Ok(())
    }

    /// Bit depth the encoder must be configured for: 10 for an HDR job, 8 otherwise.
    ///
    /// Driven by `output_color.bit_depth` so a caller who sets the colour profile
    /// by hand gets a consistent encoder, but floored at 10 for any HDR job — PQ
    /// in 8 bits bands visibly and is out of spec.
    pub fn encode_bit_depth(&self) -> u8 {
        if self.is_hdr() {
            self.output_color.bit_depth.max(10)
        } else {
            8
        }
    }

    /// `AVPixelFormat` the video encoder should be opened with.
    ///
    /// `is_hw` selects the format a hardware encoder wants (NV12 / P010, which is
    /// what NVENC, AMF and QSV accept) rather than the planar format the software
    /// encoders take.
    ///
    /// ProRes is the one codec with no 8-bit option at all — it is 10-bit 4:2:2
    /// only, which is why it was already special-cased here.
    pub fn encoder_pix_fmt(&self, is_hw: bool) -> i32 {
        use crate::io::ffi::avutil::{
            AV_PIX_FMT_NV12, AV_PIX_FMT_P010LE, AV_PIX_FMT_YUV420P,
            AV_PIX_FMT_YUV420P10LE, AV_PIX_FMT_YUV422P10LE,
        };

        // ProRes: always 10-bit 4:2:2, HDR or not.
        if self.video_codec == VideoCodec::ProRes {
            return AV_PIX_FMT_YUV422P10LE;
        }

        match (is_hw, self.encode_bit_depth() >= 10) {
            (true,  true)  => AV_PIX_FMT_P010LE,
            (true,  false) => AV_PIX_FMT_NV12,
            (false, true)  => AV_PIX_FMT_YUV420P10LE,
            (false, false) => AV_PIX_FMT_YUV420P,
        }
    }

    /// swscale colour-space id (`SWS_CS_*`) matching `output_color`'s matrix.
    ///
    /// Without this swscale applies its default (BT.601) matrix regardless of what
    /// the stream is tagged as, so an HDR export would be tagged BT.2020 while its
    /// samples were converted with BT.601 coefficients — a real, visible hue
    /// error, not just wrong metadata.
    pub fn sws_colorspace(&self) -> i32 {
        // SWS_CS_ITU709 = 1, SWS_CS_ITU601 = 5, SWS_CS_BT2020 = 9.
        match self.output_color.matrix {
            MatrixCoefficients::Bt709   => 1,
            MatrixCoefficients::Bt601   => 5,
            MatrixCoefficients::Bt2020  => 9,
            MatrixCoefficients::Unknown => 1,
        }
    }

    /// swscale destination range flag: 1 = full/JPEG, 0 = limited/MPEG.
    pub fn sws_dst_range(&self) -> i32 {
        match self.output_color.range {
            ColorRange::Full => 1,
            _                => 0,
        }
    }

    pub fn validate(&self) -> Result<(), JobError> {
        if self.pts_in >= self.pts_out {
            return Err(JobError::NegativeDuration);
        }
        if self.width == 0 || self.height == 0 {
            return Err(JobError::ZeroDimension);
        }
        if self.frame_rate.num == 0 || self.frame_rate.den == 0 {
            return Err(JobError::ZeroFrameRate);
        }
        if self.render_threads == 0 {
            return Err(JobError::ZeroThreads);
        }
        if let VideoQuality::Crf(_) = self.quality {
            if !self.video_codec.supports_crf() {
                return Err(JobError::CrfNotSupported(self.video_codec));
            }
        }
        // An HDR job on a codec that cannot carry 10-bit would encode 8-bit
        // samples under HDR tags.  Refuse it here as well as in `set_hdr10`, so
        // a job assembled field-by-field cannot slip past.
        if self.is_hdr() && !self.video_codec.supports_hdr() {
            return Err(JobError::HdrNotSupported(self.video_codec));
        }
        if let Some(parent) = self.output_path.parent() {
            if !parent.exists() {
                return Err(JobError::BadOutputPath(self.output_path.clone()));
            }
        }
        Ok(())
    }
}

impl ExportJob {
    pub fn preset_web_h264(
        output_path: PathBuf,
        pts_in: i64, pts_out: i64,
        width: u32, height: u32,
        frame_rate: Rational,
        project_tb: Rational,
    ) -> Self {
        ExportJob {
            output_path,
            container: Container::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            quality: VideoQuality::Crf(23),
            audio_bitrate: 192_000,
            pts_in,
            pts_out,
            width,
            height,
            frame_rate,
            project_tb,
            render_threads: num_cpus::get().max(2) / 2,
            cpu_preset: CpuPreset::Ultrafast,
            output_color: ColorInfo::bt709(),
            hdr10: None,
        }
    }

    pub fn preset_prores_master(
        output_path: PathBuf,
        pts_in: i64, pts_out: i64,
        width: u32, height: u32,
        frame_rate: Rational,
        project_tb: Rational,
    ) -> Self {
        ExportJob {
            output_path,
            container: Container::Mov,
            video_codec: VideoCodec::ProRes,
            audio_codec: AudioCodec::Pcm,
            quality: VideoQuality::TargetBitrate(200_000_000), // Default high bitrate for ProRes
            audio_bitrate: 1_536_000, // uncompressed 48k stereo 16-bit
            pts_in,
            pts_out,
            width,
            height,
            frame_rate,
            project_tb,
            render_threads: num_cpus::get().max(2) / 2,
            cpu_preset: CpuPreset::Medium,
            output_color: ColorInfo::bt709(),
            hdr10: None,
        }
    }
}

#[derive(Debug)]
pub enum JobError {
    NegativeDuration,
    ZeroDimension,
    ZeroFrameRate,
    ZeroThreads,
    CrfNotSupported(VideoCodec),
    /// An HDR (PQ/HLG) output was requested for a codec that cannot carry 10-bit.
    HdrNotSupported(VideoCodec),
    BadOutputPath(PathBuf),
}
