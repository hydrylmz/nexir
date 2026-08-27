// src/timeline/timecode.rs
// SMPTE timecode (HH:MM:SS:FF / HH:MM:SS;FF) parsing and formatting.

use crate::timeline::rational::Rational;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    pub hours: u32,
    pub minutes: u32,
    pub seconds: u32,
    pub frames: u32,
    pub is_drop_frame: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimecodeError {
    InvalidFormat,
    InvalidComponents,
    InvalidFramerate,
}

impl Timecode {
    pub fn new(hours: u32, minutes: u32, seconds: u32, frames: u32, is_drop_frame: bool) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            is_drop_frame,
        }
    }

    /// Converts a timeline PTS (at project timebase 1/90,000) and frame rate into SMPTE Timecode.
    pub fn from_pts(pts: i64, timebase: Rational, frame_rate: Rational) -> Self {
        let frame_number = frame_rate.pts_to_frame(pts, timebase).max(0) as u64;
        Self::from_frame_number(frame_number, frame_rate)
    }

    /// Converts total sequential frame number to Timecode.
    pub fn from_frame_number(mut frame_number: u64, frame_rate: Rational) -> Self {
        let fps_f64 = frame_rate.as_f64();
        let fps_nominal = fps_f64.round() as u64;

        let is_drop_frame = (frame_rate == Rational::FPS_29_97) || (frame_rate == Rational::FPS_59_94);

        if is_drop_frame && fps_nominal == 30 {
            // NTSC 29.97 Drop-Frame: Drop 2 frames every minute except every 10th minute
            let drop_frames = 2u64;
            let frames_per_min = 30 * 60 - drop_frames; // 1798
            let frames_per_10min = 30 * 60 * 10 - drop_frames * 9; // 17990

            let d = frame_number / frames_per_10min;
            let m = frame_number % frames_per_10min;

            if m > drop_frames {
                frame_number += drop_frames * 9 * d + drop_frames * ((m - drop_frames) / frames_per_min);
            } else {
                frame_number += drop_frames * 9 * d;
            }
        }

        let frames = (frame_number % fps_nominal) as u32;
        let total_seconds = frame_number / fps_nominal;
        let seconds = (total_seconds % 60) as u32;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u32;
        let hours = (total_minutes / 60) as u32;

        Self {
            hours,
            minutes,
            seconds,
            frames,
            is_drop_frame,
        }
    }

    /// Converts Timecode to timeline PTS (at project timebase 1/90,000).
    pub fn to_pts(&self, timebase: Rational, frame_rate: Rational) -> i64 {
        let frame_number = self.to_frame_number(frame_rate);
        frame_rate.frame_to_pts(frame_number as i64, timebase)
    }

    /// Converts Timecode to sequential frame number.
    pub fn to_frame_number(&self, frame_rate: Rational) -> u64 {
        let fps_f64 = frame_rate.as_f64();
        let fps_nominal = fps_f64.round() as u64;

        if self.is_drop_frame && fps_nominal == 30 {
            let total_minutes = self.hours as u64 * 60 + self.minutes as u64;
            let drop_frames = 2u64;
            let frame_number = (self.hours as u64 * 3600 + self.minutes as u64 * 60 + self.seconds as u64) * 30
                + self.frames as u64
                - drop_frames * (total_minutes - total_minutes / 10);
            return frame_number;
        }

        (self.hours as u64 * 3600 + self.minutes as u64 * 60 + self.seconds as u64) * fps_nominal
            + self.frames as u64
    }

    /// Formats Timecode as standard string (e.g. `01:00:00:00` or `01:00:00;00` for drop-frame).
    pub fn to_string_formatted(&self) -> String {
        let sep = if self.is_drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, sep, self.frames
        )
    }

    /// Parses standard SMPTE timecode string (`HH:MM:SS:FF` or `HH:MM:SS;FF`).
    pub fn parse(s: &str) -> Result<Self, TimecodeError> {
        let s = s.trim();
        if s.len() != 11 {
            return Err(TimecodeError::InvalidFormat);
        }
        let bytes = s.as_bytes();
        let is_drop_frame = bytes[8] == b';';
        if (bytes[2] != b':' || bytes[5] != b':') || (bytes[8] != b':' && bytes[8] != b';') {
            return Err(TimecodeError::InvalidFormat);
        }

        let hours: u32 = s[0..2].parse().map_err(|_| TimecodeError::InvalidComponents)?;
        let minutes: u32 = s[3..5].parse().map_err(|_| TimecodeError::InvalidComponents)?;
        let seconds: u32 = s[6..8].parse().map_err(|_| TimecodeError::InvalidComponents)?;
        let frames: u32 = s[9..11].parse().map_err(|_| TimecodeError::InvalidComponents)?;

        if minutes >= 60 || seconds >= 60 {
            return Err(TimecodeError::InvalidComponents);
        }

        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            is_drop_frame,
        })
    }
}

impl std::fmt::Display for Timecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_formatted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_formatting_and_parsing() {
        let tc = Timecode::new(1, 23, 45, 12, false);
        assert_eq!(tc.to_string_formatted(), "01:23:45:12");

        let parsed = Timecode::parse("01:23:45:12").unwrap();
        assert_eq!(parsed, tc);

        let df_tc = Timecode::new(0, 10, 0, 2, true);
        assert_eq!(df_tc.to_string_formatted(), "00:10:00;02");
        let parsed_df = Timecode::parse("00:10:00;02").unwrap();
        assert_eq!(parsed_df, df_tc);
    }

    #[test]
    fn test_timecode_pts_roundtrip() {
        let fps = Rational::FPS_30;
        let tb = Rational { num: 1, den: 90_000 };

        // 1 hour 0 seconds at 30 FPS = 108,000 frames = 324,000,000 PTS
        let pts = 324_000_000i64;
        let tc = Timecode::from_pts(pts, tb, fps);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);

        let roundtrip_pts = tc.to_pts(tb, fps);
        assert_eq!(roundtrip_pts, pts);
    }

    #[test]
    fn test_timecode_invalid_strings() {
        assert!(Timecode::parse("01:23:45").is_err());
        assert!(Timecode::parse("01:23:45:123").is_err());
        assert!(Timecode::parse("01:75:45:12").is_err()); // min >= 60
    }
}
