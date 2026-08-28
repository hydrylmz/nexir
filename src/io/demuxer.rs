// src/io/demuxer.rs
// Safe, owned wrapper around AVFormatContext.
// Demuxer is Send (moved to decode thread) but NOT Sync.

use std::ffi::CString;
use std::path::Path;
use std::ptr;
use crate::io::ffi::avformat::*;
use crate::io::ffi::avcodec::{
    avstream_get_codecpar, avcodecpar_get_codec_id,
    avcodecpar_get_width, avcodecpar_get_height,
    avcodecpar_get_color_space, avcodecpar_get_color_range,
    avcodecpar_get_color_trc, avcodecpar_get_color_primaries,
    avcodecpar_get_bit_depth
};
use crate::io::ffi::avutil::{
    AVPacket, av_packet_alloc, av_packet_free, av_packet_unref,
    AVERROR_EOF, av_err_to_string,
};
use crate::timeline::rational::Rational;

/// A demuxed packet: compressed data for one stream, with timing.
pub struct Packet {
    /// Owning pointer to the FFmpeg packet. Freed on drop.
    inner:            *mut AVPacket,
    pub stream_index: i32,
    /// PTS in stream timebase ticks. `AV_NOPTS_VALUE` if unknown.
    pub pts:          i64,
    /// Duration in stream timebase ticks.
    pub duration:     i64,
}

impl Packet {
    /// Raw const pointer for passing to the decoder.
    pub fn as_ptr(&self) -> *const AVPacket {
        self.inner as *const _
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn from_raw(src: *mut AVPacket) -> Self {
        // SAFETY: `src` is expected to come from FFmpeg. This constructor
        // immediately clones the packet into an owned AVPacket via av_packet_ref.
        unsafe {
            let inner = crate::io::ffi::avutil::av_packet_alloc();
            extern "C" { pub fn av_packet_ref(dst: *mut AVPacket, src: *const AVPacket) -> std::ffi::c_int; }
            av_packet_ref(inner, src);
            Self {
                inner,
                stream_index: (*inner).stream_index,
                pts: (*inner).pts,
                duration: (*inner).duration,
            }
        }
    }
}

impl Drop for Packet {
    fn drop(&mut self) {
        unsafe {
            av_packet_unref(self.inner);
            av_packet_free(&mut self.inner);
        }
    }
}

// SAFETY: Packet wraps a raw pointer that we treat as exclusively owned.
unsafe impl Send for Packet {}

/// Stream metadata extracted at open time.
#[derive(Clone, Debug)]
pub struct StreamInfo {
    pub index:      usize,
    pub time_base:  Rational,
    pub duration:   i64,
    pub codec_id:   u32,
    pub frame_rate: Option<Rational>,
    pub width:      Option<u32>,
    pub height:     Option<u32>,
    pub is_vfr:     bool,
    pub color_info: crate::timeline::source::ColorInfo,
    /// We need the raw codec parameters to initialize the decoder
    pub codecpar:   *mut crate::io::ffi::avcodec::AVCodecParameters,
}

// Ensure the pointer is thread-safe since we only use it safely.
unsafe impl Send for StreamInfo {}
unsafe impl Sync for StreamInfo {}

pub struct Demuxer {
    ctx:              *mut AVFormatContext,
    pub video_stream: Option<StreamInfo>,
    pub audio_stream: Option<StreamInfo>,
    /// Reusable packet buffer — avoids one allocation per packet.
    packet_buf:       *mut AVPacket,
}

// SAFETY: FFmpeg context is not thread-safe; Demuxer is Send (moved to decode thread) but not Sync.
unsafe impl Send for Demuxer {}

impl Demuxer {
    /// Open a media file and probe stream information.
    pub fn open(path: &Path) -> Result<Self, DemuxError> {
        // Step 1 — Null-terminate the path for FFmpeg
        let c_path = CString::new(
            path.to_str().ok_or(DemuxError::InvalidPath)?
        ).map_err(|_| DemuxError::InvalidPath)?;

        // Step 2 — Open the format context
        let mut ctx: *mut AVFormatContext = ptr::null_mut();
        let ret = unsafe {
            avformat_open_input(&mut ctx, c_path.as_ptr(), ptr::null(), ptr::null_mut())
        };
        if ret < 0 {
            return Err(DemuxError::Open(av_err_to_string(ret)));
        }

        // Step 3 — Find stream info (decodes a few frames to compute fps/duration)
        let ret = unsafe { avformat_find_stream_info(ctx, ptr::null_mut()) };
        if ret < 0 {
            unsafe { avformat_close_input(&mut ctx); }
            return Err(DemuxError::StreamInfo(av_err_to_string(ret)));
        }

        // Step 4 — Find best video stream
        let video_idx = unsafe {
            av_find_best_stream(
                ctx, AVMEDIA_TYPE_VIDEO, -1, -1, ptr::null_mut(), 0,
            )
        };
        let video_stream = if video_idx >= 0 {
            Some(Self::read_stream_info(ctx, video_idx as usize))
        } else {
            None
        };

        // Step 5 — Find best audio stream
        let audio_idx = unsafe {
            av_find_best_stream(
                ctx, AVMEDIA_TYPE_AUDIO, -1, -1, ptr::null_mut(), 0,
            )
        };
        let audio_stream = if audio_idx >= 0 {
            Some(Self::read_stream_info(ctx, audio_idx as usize))
        } else {
            None
        };

        // Step 6 — Allocate the reusable packet buffer
        let packet_buf = unsafe { av_packet_alloc() };
        if packet_buf.is_null() {
            unsafe { avformat_close_input(&mut ctx); }
            return Err(DemuxError::Alloc);
        }

        Ok(Demuxer { ctx, video_stream, audio_stream, packet_buf })
    }

    /// Video stream index (None if no video stream).
    fn video_stream_index(&self) -> Option<i32> {
        self.video_stream.as_ref().map(|s| s.index as i32)
    }

    /// Read the next packet from the video stream, skipping packets from other streams.
    pub fn next_video_packet(&mut self) -> Result<Option<Packet>, DemuxError> {
        let video_idx = match self.video_stream_index() {
            Some(i) => i,
            None    => return Ok(None),
        };

        loop {
            let ret = unsafe { av_read_frame(self.ctx, self.packet_buf) };
            if ret == AVERROR_EOF {
                return Ok(None);
            }
            if ret < 0 {
                return Err(DemuxError::Read(av_err_to_string(ret)));
            }

            // Read stream_index and pts from the opaque packet.
            // We expose these as accessible fields via a thin overlay struct.
            // Since AVPacket is opaque we call the accessor.
            let pkt_stream_idx = unsafe { av_packet_stream_index(self.packet_buf) };
            let pkt_pts         = unsafe { av_packet_pts(self.packet_buf) };
            let pkt_duration    = unsafe { av_packet_duration(self.packet_buf) };

            if pkt_stream_idx != video_idx {
                // Not our stream — unref and continue
                unsafe { av_packet_unref(self.packet_buf); }
                continue;
            }

            // Transfer ownership: allocate a fresh packet and move data via ref
            let owned_pkt = unsafe { av_packet_alloc() };
            if owned_pkt.is_null() {
                unsafe { av_packet_unref(self.packet_buf); }
                return Err(DemuxError::Alloc);
            }
            let ret = unsafe { av_packet_ref(owned_pkt, self.packet_buf) };
            unsafe { av_packet_unref(self.packet_buf); }
            if ret < 0 {
                unsafe { av_packet_free(&mut (owned_pkt as *mut _)); }
                return Err(DemuxError::Read(av_err_to_string(ret)));
            }

            return Ok(Some(Packet {
                inner:        owned_pkt,
                stream_index: pkt_stream_idx,
                pts:          pkt_pts,
                duration:     pkt_duration,
            }));
        }
    }

    pub fn audio_stream(&self) -> Option<&StreamInfo> {
        self.audio_stream.as_ref()
    }

    /// Read the next packet from the audio stream, ignoring others.
    pub fn next_audio_packet(&mut self) -> Result<Option<Packet>, DemuxError> {
        let stream_idx = match &self.audio_stream {
            Some(s) => s.index,
            None => return Ok(None),
        };

        loop {
            let ret = unsafe { av_read_frame(self.ctx, self.packet_buf) };
            if ret == AVERROR_EOF {
                return Ok(None);
            }
            if ret < 0 {
                return Err(DemuxError::Read(av_err_to_string(ret)));
            }

            let pkt_stream = unsafe { (*self.packet_buf).stream_index as usize };
            if pkt_stream == stream_idx {
                let owned_pkt = unsafe { av_packet_alloc() };
                if owned_pkt.is_null() {
                    unsafe { av_packet_unref(self.packet_buf); }
                    return Err(DemuxError::Alloc);
                }
                let ret = unsafe { av_packet_ref(owned_pkt, self.packet_buf) };
                unsafe { av_packet_unref(self.packet_buf); }
                if ret < 0 {
                    unsafe { av_packet_free(&mut (owned_pkt as *mut _)); }
                    return Err(DemuxError::Read(av_err_to_string(ret)));
                }

                return Ok(Some(Packet {
                    inner:        owned_pkt,
                    stream_index: unsafe { (*owned_pkt).stream_index },
                    pts:          unsafe { (*owned_pkt).pts },
                    duration:     unsafe { (*owned_pkt).duration },
                }));
            } else {
                unsafe { av_packet_unref(self.packet_buf); }
            }
        }
    }

    /// Seek the demuxer to just before the given PTS (in project timebase).
    /// Prefers the video stream but falls back to the audio stream for audio-only files.
    /// Returns the stream-timebase PTS that was actually seeked to.
    pub fn seek(&mut self, pts: i64, project_tb: Rational) -> Result<i64, DemuxError> {
        let stream_info = self.video_stream.as_ref()
            .or(self.audio_stream.as_ref())
            .ok_or(DemuxError::NoVideoStream)?;
        let stream_idx  = stream_info.index as i32;
        let stream_tb   = stream_info.time_base;

        // Convert project PTS to stream timebase ticks
        let stream_pts = project_tb.rescale_pts(pts, stream_tb);

        let ret = unsafe {
            av_seek_frame(self.ctx, stream_idx, stream_pts, AVSEEK_FLAG_BACKWARD)
        };
        if ret < 0 {
            return Err(DemuxError::Seek(av_err_to_string(ret)));
        }
        Ok(stream_pts)
    }

    /// Extract `StreamInfo` from a stream at the given index.
    fn read_stream_info(ctx: *mut AVFormatContext, index: usize) -> StreamInfo {
        unsafe {
            let stream   = avformat_get_stream(ctx, index as u32);
            let time_base = avstream_get_time_base(stream).to_rational();
            let avg_fr    = avstream_get_avg_frame_rate(stream);
            let r_fr      = crate::io::ffi::avformat::avstream_get_r_frame_rate(stream);
            
            let mut is_vfr = false;
            let frame_rate = if r_fr.is_valid() && avg_fr.is_valid() {
                if (r_fr.num as i64 * avg_fr.den as i64) != (avg_fr.num as i64 * r_fr.den as i64) {
                    is_vfr = true;
                }
                Some(r_fr.to_rational())
            } else if r_fr.is_valid() {
                Some(r_fr.to_rational())
            } else if avg_fr.is_valid() {
                Some(avg_fr.to_rational())
            } else {
                None
            };
            
            // Format duration is in AV_TIME_BASE (1,000,000) ticks. Convert to stream timebase.
            let duration_av = avformat_get_duration(ctx);
            let duration = if duration_av > 0 {
                let av_tb = Rational::new(1, 1_000_000);
                time_base.from_pts(duration_av, av_tb)
            } else {
                0
            };

            let codecpar  = avstream_get_codecpar(stream);
            let codec_id  = avcodecpar_get_codec_id(codecpar);
            let width     = avcodecpar_get_width(codecpar);
            let height    = avcodecpar_get_height(codecpar);
            
            let cs_raw = avcodecpar_get_color_space(codecpar);
            let cr_raw = avcodecpar_get_color_range(codecpar);
            let trc_raw = avcodecpar_get_color_trc(codecpar);
            let pri_raw = avcodecpar_get_color_primaries(codecpar);
            let bit_depth = avcodecpar_get_bit_depth(codecpar) as u8;

            use crate::timeline::source::ColorInfo;

            let w = if width > 0 { width as u32 } else { 0 };
            let h = if height > 0 { height as u32 } else { 0 };

            let color_info = ColorInfo::from_ffmpeg(
                cs_raw,
                cr_raw,
                trc_raw,
                pri_raw,
                bit_depth,
                w,
                h,
            );

            StreamInfo {
                index,
                time_base,
                duration,
                codec_id,
                frame_rate,
                width:  if width  > 0 { Some(width  as u32) } else { None },
                height: if height > 0 { Some(height as u32) } else { None },
                is_vfr,
                color_info,
                codecpar,
            }
        }
    }
}

impl Drop for Demuxer {
    fn drop(&mut self) {
        unsafe {
            // Free packet before context (packet data is owned by the context)
            if !self.packet_buf.is_null() {
                av_packet_free(&mut self.packet_buf);
            }
            if !self.ctx.is_null() {
                avformat_close_input(&mut self.ctx);
            }
        }
    }
}

#[derive(Debug)]
pub enum DemuxError {
    InvalidPath,
    Open(String),
    StreamInfo(String),
    NoVideoStream,
    Read(String),
    Seek(String),
    Alloc,
}

#[link(name = "avformat")]
unsafe extern "C" {
    fn av_packet_ref(dst: *mut AVPacket, src: *const AVPacket) -> i32;
}

extern "C" {
    fn av_packet_stream_index(pkt: *const AVPacket) -> i32;
    fn av_packet_pts(pkt: *const AVPacket) -> i64;
    fn av_packet_duration(pkt: *const AVPacket) -> i64;
}
