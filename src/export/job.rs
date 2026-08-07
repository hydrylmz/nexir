use std::path::PathBuf;
use crate::timeline::rational::Rational;

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
}

impl ExportJob {
    pub fn total_frames(&self) -> usize {
        let duration_pts = self.pts_out - self.pts_in;
        let numer = duration_pts as u64 * self.frame_rate.num as u64;
        let denom = self.project_tb.den as u64 * self.frame_rate.den as u64;
        ((numer + denom - 1) / denom) as usize
    }

    pub fn frame_pts(&self, n: usize) -> i64 {
        let frame_dur = self.project_tb.den as i64 / self.frame_rate.num as i64;
        self.pts_in + n as i64 * frame_dur
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
    BadOutputPath(PathBuf),
}
