// src/audio/audio_mixer.rs

use std::f32::consts::{FRAC_1_SQRT_2, PI};

pub const MIXER_SAMPLE_RATE: u32 = 48_000;
pub const MIXER_CHANNELS: usize = 2;

/// Calculates stereo gain multiplier pair `(left, right)` using constant-power panning law.
/// `pan` ranges from -1.0 (full left) through 0.0 (center) to +1.0 (full right).
#[inline]
pub fn constant_power_pan(pan: f32) -> (f32, f32) {
    let clamped = pan.clamp(-1.0, 1.0);
    let angle = (clamped + 1.0) * PI / 4.0; // 0.0 (left) -> π/4 (center) -> π/2 (right)
    (angle.cos(), angle.sin())
}

/// Computes fade multiplier in range `[0.0, 1.0]` for a given timeline PTS timestamp.
#[inline]
pub fn compute_fade_multiplier(
    pos_pts: i64,
    clip_pts_in: i64,
    clip_pts_out: i64,
    fade_in_pts: i64,
    fade_out_pts: i64,
) -> f32 {
    if pos_pts < clip_pts_in || pos_pts >= clip_pts_out {
        return 0.0;
    }
    let mut factor = 1.0f32;
    if fade_in_pts > 0 {
        let delta = pos_pts - clip_pts_in;
        if delta < fade_in_pts {
            let t = (delta as f32 / fade_in_pts as f32).clamp(0.0, 1.0);
            factor = factor.min(t);
        }
    }
    if fade_out_pts > 0 {
        let delta = clip_pts_out - pos_pts;
        if delta < fade_out_pts {
            let t = (delta as f32 / fade_out_pts as f32).clamp(0.0, 1.0);
            factor = factor.min(t);
        }
    }
    factor
}

/// Master bus soft limiter / tanh saturator.
/// Prevents digital clipping over `[-1.0, 1.0]` while preserving dynamics below 0.8.
#[inline]
pub fn soft_limiter(sample: f32) -> f32 {
    if sample.abs() <= 0.8 {
        sample
    } else {
        sample.tanh()
    }
}

/// Applies soft limiting in-place to an audio slice.
pub fn soft_limit_buffer(buffer: &mut [f32]) {
    for s in buffer.iter_mut() {
        *s = soft_limiter(*s);
    }
}

/// Planar stereo 32-bit float audio buffer at 48 kHz.
#[derive(Debug, Clone, Default)]
pub struct PlanarAudioBuffer {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
}

impl PlanarAudioBuffer {
    pub fn new_zeros(sample_count: usize, sample_rate: u32) -> Self {
        Self {
            left: vec![0.0; sample_count],
            right: vec![0.0; sample_count],
            sample_rate,
        }
    }

    pub fn len(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    pub fn clear_to_zero(&mut self) {
        self.left.fill(0.0);
        self.right.fill(0.0);
    }

    /// Convert planar audio to interleaved `[L0, R0, L1, R1, ...]`.
    pub fn to_interleaved(&self) -> Vec<f32> {
        let n = self.len();
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            out.push(self.left[i]);
            out.push(self.right[i]);
        }
        out
    }
}

/// Multi-track audio mixer providing sample-accurate summing, clip/track DSP,
/// fade envelopes, and master bus soft limiting.
pub struct AudioMixer {
    pub master_volume: f32,
}

impl Default for AudioMixer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl AudioMixer {
    pub fn new(master_volume: f32) -> Self {
        Self { master_volume }
    }

    /// Mixes an already-decoded planar clip chunk into an accumulation buffer.
    pub fn mix_clip_chunk(
        accum_left: &mut [f32],
        accum_right: &mut [f32],
        chunk_left: &[f32],
        chunk_right: &[f32],
        chunk_offset_samples: usize,
        clip_pts_in: i64,
        clip_pts_out: i64,
        fade_in_pts: i64,
        fade_out_pts: i64,
        clip_volume: f32,
        clip_pan: f32,
        clip_muted: bool,
        track_gain: f32,
        track_pan: f32,
        track_active: bool,
        timeline_start_pts: i64,
    ) {
        if !track_active || clip_muted {
            return;
        }

        let (c_pan_l, c_pan_r) = constant_power_pan(clip_pan);
        let (t_pan_l, t_pan_r) = constant_power_pan(track_pan);

        let total_left_gain = clip_volume * c_pan_l * track_gain * t_pan_l;
        let total_right_gain = clip_volume * c_pan_r * track_gain * t_pan_r;

        let n = chunk_left.len().min(chunk_right.len());
        let available = accum_left.len().saturating_sub(chunk_offset_samples);
        let count = n.min(available);

        for i in 0..count {
            let out_idx = chunk_offset_samples + i;
            let sample_pts = timeline_start_pts + (out_idx as f64 * 90000.0 / MIXER_SAMPLE_RATE as f64) as i64;
            let fade = compute_fade_multiplier(sample_pts, clip_pts_in, clip_pts_out, fade_in_pts, fade_out_pts);

            accum_left[out_idx] += chunk_left[i] * total_left_gain * fade;
            accum_right[out_idx] += chunk_right[i] * total_right_gain * fade;
        }
    }

    /// Applies master volume and master soft limiting in-place.
    pub fn apply_master_bus(&self, buffer: &mut PlanarAudioBuffer) {
        let vol = self.master_volume;
        for s in buffer.left.iter_mut() {
            *s = soft_limiter(*s * vol);
        }
        for s in buffer.right.iter_mut() {
            *s = soft_limiter(*s * vol);
        }
    }
}

/// Audio channel layout for input audio sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Surround5_1,
    Surround7_1,
}

/// Downmixes arbitrary channel inputs (Mono, Stereo, 5.1, 7.1) into standard planar stereo (Left, Right).
pub fn downmix_channels_to_stereo(
    input_channels: &[&[f32]],
    layout: ChannelLayout,
    out_left: &mut [f32],
    out_right: &mut [f32],
) {
    let count = out_left.len().min(out_right.len());
    match layout {
        ChannelLayout::Mono => {
            if let Some(&mono) = input_channels.first() {
                let n = count.min(mono.len());
                out_left[..n].copy_from_slice(&mono[..n]);
                out_right[..n].copy_from_slice(&mono[..n]);
            }
        }
        ChannelLayout::Stereo => {
            if input_channels.len() >= 2 {
                let left_in = input_channels[0];
                let right_in = input_channels[1];
                let n = count.min(left_in.len()).min(right_in.len());
                out_left[..n].copy_from_slice(&left_in[..n]);
                out_right[..n].copy_from_slice(&right_in[..n]);
            } else if let Some(&mono) = input_channels.first() {
                let n = count.min(mono.len());
                out_left[..n].copy_from_slice(&mono[..n]);
                out_right[..n].copy_from_slice(&mono[..n]);
            }
        }
        ChannelLayout::Surround5_1 => {
            // 5.1 layout: 0=Left, 1=Right, 2=Center, 3=LFE, 4=Left Surround, 5=Right Surround
            let num_ch = input_channels.len();
            let c_gain = FRAC_1_SQRT_2;
            let s_gain = FRAC_1_SQRT_2;
            let lfe_gain = 0.5f32;
            for i in 0..count {
                let l = if num_ch > 0 && i < input_channels[0].len() { input_channels[0][i] } else { 0.0 };
                let r = if num_ch > 1 && i < input_channels[1].len() { input_channels[1][i] } else { 0.0 };
                let c = if num_ch > 2 && i < input_channels[2].len() { input_channels[2][i] } else { 0.0 };
                let lfe = if num_ch > 3 && i < input_channels[3].len() { input_channels[3][i] } else { 0.0 };
                let ls = if num_ch > 4 && i < input_channels[4].len() { input_channels[4][i] } else { 0.0 };
                let rs = if num_ch > 5 && i < input_channels[5].len() { input_channels[5][i] } else { 0.0 };

                out_left[i] = l + c * c_gain + ls * s_gain + lfe * lfe_gain;
                out_right[i] = r + c * c_gain + rs * s_gain + lfe * lfe_gain;
            }
        }
        ChannelLayout::Surround7_1 => {
            // 7.1 layout: 0=L, 1=R, 2=C, 3=LFE, 4=Ls, 5=Rs, 6=Rls, 7=Rrs
            let num_ch = input_channels.len();
            let c_gain = FRAC_1_SQRT_2;
            let s_gain = 0.5f32;
            let lfe_gain = 0.5f32;
            for i in 0..count {
                let l = if num_ch > 0 && i < input_channels[0].len() { input_channels[0][i] } else { 0.0 };
                let r = if num_ch > 1 && i < input_channels[1].len() { input_channels[1][i] } else { 0.0 };
                let c = if num_ch > 2 && i < input_channels[2].len() { input_channels[2][i] } else { 0.0 };
                let lfe = if num_ch > 3 && i < input_channels[3].len() { input_channels[3][i] } else { 0.0 };
                let ls = if num_ch > 4 && i < input_channels[4].len() { input_channels[4][i] } else { 0.0 };
                let rs = if num_ch > 5 && i < input_channels[5].len() { input_channels[5][i] } else { 0.0 };
                let rls = if num_ch > 6 && i < input_channels[6].len() { input_channels[6][i] } else { 0.0 };
                let rrs = if num_ch > 7 && i < input_channels[7].len() { input_channels[7][i] } else { 0.0 };

                out_left[i] = l + c * c_gain + (ls + rls) * s_gain + lfe * lfe_gain;
                out_right[i] = r + c * c_gain + (rs + rrs) * s_gain + lfe * lfe_gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_power_pan() {
        // Center pan (0.0): both channels should have equal power (cos(pi/4) = sin(pi/4) ≈ 0.7071)
        let (l_center, r_center) = constant_power_pan(0.0);
        assert!((l_center - r_center).abs() < 1e-4);
        assert!((l_center.powi(2) + r_center.powi(2) - 1.0).abs() < 1e-4);

        // Full left (-1.0): left = 1.0, right = 0.0
        let (l_left, r_left) = constant_power_pan(-1.0);
        assert!((l_left - 1.0).abs() < 1e-4);
        assert!(r_left.abs() < 1e-4);

        // Full right (1.0): left = 0.0, right = 1.0
        let (l_right, r_right) = constant_power_pan(1.0);
        assert!(l_right.abs() < 1e-4);
        assert!((r_right - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_fade_multiplier() {
        let pts_in = 0;
        let pts_out = 90_000; // 1 second
        let fade_in = 9_000;   // 0.1s
        let fade_out = 9_000;  // 0.1s

        // Before clip
        assert_eq!(compute_fade_multiplier(-10, pts_in, pts_out, fade_in, fade_out), 0.0);
        // After clip
        assert_eq!(compute_fade_multiplier(90_000, pts_in, pts_out, fade_in, fade_out), 0.0);

        // Fade in start
        assert_eq!(compute_fade_multiplier(0, pts_in, pts_out, fade_in, fade_out), 0.0);
        // Fade in half
        assert!((compute_fade_multiplier(4_500, pts_in, pts_out, fade_in, fade_out) - 0.5).abs() < 1e-4);
        // Fade in complete / sustain
        assert_eq!(compute_fade_multiplier(9_000, pts_in, pts_out, fade_in, fade_out), 1.0);
        assert_eq!(compute_fade_multiplier(45_000, pts_in, pts_out, fade_in, fade_out), 1.0);

        // Fade out half
        assert!((compute_fade_multiplier(85_500, pts_in, pts_out, fade_in, fade_out) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_soft_limiter() {
        // Linear region
        assert_eq!(soft_limiter(0.5), 0.5);
        assert_eq!(soft_limiter(-0.5), -0.5);

        // Saturation region (must never exceed 1.0)
        let saturated = soft_limiter(2.0);
        assert!(saturated > 0.8 && saturated < 1.0);

        let neg_saturated = soft_limiter(-5.0);
        assert!(neg_saturated < -0.8 && neg_saturated > -1.0);
    }

    #[test]
    fn test_mix_clip_chunk() {
        let mut accum = PlanarAudioBuffer::new_zeros(100, MIXER_SAMPLE_RATE);
        let chunk = vec![1.0f32; 50];

        AudioMixer::mix_clip_chunk(
            &mut accum.left,
            &mut accum.right,
            &chunk,
            &chunk,
            0,
            0,
            90_000,
            0,
            0,
            1.0,
            0.0,
            false,
            1.0,
            0.0,
            true,
            0,
        );

        let (c_l, c_r) = constant_power_pan(0.0);
        let expected_left_gain = c_l * c_l; // clip pan center * track pan center
        let expected_right_gain = c_r * c_r;

        assert!((accum.left[0] - expected_left_gain).abs() < 1e-4);
        assert!((accum.right[0] - expected_right_gain).abs() < 1e-4);
        assert_eq!(accum.left[50], 0.0); // beyond chunk length
    }

    #[test]
    fn test_downmix_channels() {
        let mut out_l = vec![0.0f32; 4];
        let mut out_r = vec![0.0f32; 4];

        // 1. Mono downmix
        let mono_in = vec![1.0f32; 4];
        downmix_channels_to_stereo(&[&mono_in], ChannelLayout::Mono, &mut out_l, &mut out_r);
        assert_eq!(out_l, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out_r, vec![1.0, 1.0, 1.0, 1.0]);

        // 2. 5.1 downmix (L, R, C, LFE, Ls, Rs)
        let ch_l = vec![1.0f32; 4];
        let ch_r = vec![0.5f32; 4];
        let ch_c = vec![1.0f32; 4];
        let ch_lfe = vec![0.0f32; 4];
        let ch_ls = vec![1.0f32; 4];
        let ch_rs = vec![0.0f32; 4];

        downmix_channels_to_stereo(
            &[&ch_l, &ch_r, &ch_c, &ch_lfe, &ch_ls, &ch_rs],
            ChannelLayout::Surround5_1,
            &mut out_l,
            &mut out_r,
        );

        // L = 1.0 + 1.0*0.7071 + 1.0*0.7071 = 2.4142
        // R = 0.5 + 1.0*0.7071 + 0.0*0.7071 = 1.2071
        assert!((out_l[0] - 2.4142).abs() < 1e-3);
        assert!((out_r[0] - 1.2071).abs() < 1e-3);
    }
}
