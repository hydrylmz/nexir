// src/colour/hdr.rs
// HDR10 / HLG mastering metadata, EOTF/OETF curves, and gamut conversions.

use serde::{Deserialize, Serialize};

/// SMPTE ST 2086 / CTA-861-G Mastering Display Color Volume & Light Level Metadata.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HdrMasteringMetadata {
    /// CIE 1931 xy coordinates of the Red primary (standard DCI-P3: 0.680, 0.320).
    pub red_primary: (f32, f32),
    /// CIE 1931 xy coordinates of the Green primary (standard DCI-P3: 0.265, 0.690).
    pub green_primary: (f32, f32),
    /// CIE 1931 xy coordinates of the Blue primary (standard DCI-P3: 0.150, 0.060).
    pub blue_primary: (f32, f32),
    /// CIE 1931 xy coordinates of the White Point (standard D65: 0.3127, 0.3290).
    pub white_point: (f32, f32),
    /// Peak display mastering luminance in cd/m² (nits), e.g. 1000.0 or 4000.0.
    pub max_luminance: f32,
    /// Minimum display mastering luminance in cd/m² (nits), e.g. 0.005 or 0.0001.
    pub min_luminance: f32,
    /// Maximum Content Light Level (MaxCLL) in nits.
    pub max_cll: u32,
    /// Maximum Frame-Average Light Level (MaxFALL) in nits.
    pub max_fall: u32,
}

impl Default for HdrMasteringMetadata {
    fn default() -> Self {
        Self::dci_p3_d65_1000nit()
    }
}

impl HdrMasteringMetadata {
    /// Standard mastering display profile: DCI-P3 color volume inside BT.2020 container, 1000 nits peak.
    pub fn dci_p3_d65_1000nit() -> Self {
        Self {
            red_primary: (0.680, 0.320),
            green_primary: (0.265, 0.690),
            blue_primary: (0.150, 0.060),
            white_point: (0.3127, 0.3290),
            max_luminance: 1000.0,
            min_luminance: 0.005,
            max_cll: 1000,
            max_fall: 400,
        }
    }

    /// Reference BT.2020 full color volume, 4000 nits peak.
    pub fn bt2020_d65_4000nit() -> Self {
        Self {
            red_primary: (0.708, 0.292),
            green_primary: (0.170, 0.797),
            blue_primary: (0.131, 0.046),
            white_point: (0.3127, 0.3290),
            max_luminance: 4000.0,
            min_luminance: 0.0001,
            max_cll: 4000,
            max_fall: 1000,
        }
    }
}

// ── SMPTE ST 2084 (PQ — Perceptual Quantizer) Constants ───────────────────────

const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = (2523.0 / 4096.0) * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = (2413.0 / 4096.0) * 32.0;
const PQ_C3: f32 = (2392.0 / 4096.0) * 32.0;

/// SMPTE ST 2084 (PQ) EOTF: Converts normalized non-linear signal N in [0, 1]
/// into absolute linear luminance in cd/m² (nits) in range [0, 10000].
pub fn pq_eotf(n: f32) -> f32 {
    if n <= 0.0 {
        return 0.0;
    }
    let n_m2 = n.powf(1.0 / PQ_M2);
    let num = (n_m2 - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * n_m2;
    if den <= 0.0 {
        return 10000.0;
    }
    (num / den).powf(1.0 / PQ_M1) * 10000.0
}

/// SMPTE ST 2084 (PQ) OETF: Converts absolute linear luminance in cd/m²
/// (nits) [0, 10000] into normalized non-linear signal N in [0, 1].
pub fn pq_oetf(l: f32) -> f32 {
    if l <= 0.0 {
        return 0.0;
    }
    let y = (l / 10000.0).clamp(0.0, 1.0);
    let y_m1 = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * y_m1;
    let den = 1.0 + PQ_C3 * y_m1;
    (num / den).powf(PQ_M2)
}

// ── ARIB STD-B67 (HLG — Hybrid Log-Gamma) Constants ──────────────────────────

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 1.0 - 4.0 * HLG_A;
const HLG_C: f32 = 0.559_910_7;

/// ARIB STD-B67 (HLG) EOTF: Converts normalized HLG signal E in [0, 1]
/// to linear relative luminance [0, 1] (scene-referred).
pub fn hlg_eotf(e: f32) -> f32 {
    let e = e.max(0.0);
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// ARIB STD-B67 (HLG) OETF: Converts linear relative luminance L in [0, 1]
/// to normalized HLG signal E in [0, 1].
pub fn hlg_oetf(l: f32) -> f32 {
    let l = l.max(0.0);
    if l <= 1.0 / 12.0 {
        (3.0 * l).sqrt()
    } else {
        HLG_A * (12.0 * l - HLG_B).ln() + HLG_C
    }
}

// ── Gamut Conversions (BT.2020 <-> BT.709) ───────────────────────────────────

/// Convert linear RGB in BT.2020 color primaries to linear RGB in BT.709 primaries.
pub fn bt2020_to_bt709(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    [
        1.660_491 * r - 0.5876411 * g - 0.0728499 * b,
       -0.1245505 * r + 1.1328999 * g - 0.0083494 * b,
       -0.0181508 * r - 0.1005789 * g + 1.1187297 * b,
    ]
}

/// Convert linear RGB in BT.709 color primaries to linear RGB in BT.2020 primaries.
pub fn bt709_to_bt2020(rgb: [f32; 3]) -> [f32; 3] {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    [
        0.627_404 * r + 0.329_282 * g + 0.0433136 * b,
        0.0690970 * r + 0.919_54 * g + 0.0113612 * b,
        0.0163916 * r + 0.0880132 * g + 0.895_595 * b,
    ]
}

// ── Tone Mapping Operators ───────────────────────────────────────────────────

/// ACES Filmic Tone Mapping Curve (Narkowicz / ACES fit).
/// Maps wide dynamic range linear scene luminance to [0.0, 1.0].
pub fn aces_filmic_tonemap(rgb: [f32; 3]) -> [f32; 3] {
    let tonemap_channel = |x: f32| -> f32 {
        let x = x.max(0.0);
        let a = 2.51;
        let b = 0.03;
        let c = 2.43;
        let d = 0.59;
        let e = 0.14;
        ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    };

    [
        tonemap_channel(rgb[0]),
        tonemap_channel(rgb[1]),
        tonemap_channel(rgb[2]),
    ]
}

// ── Unit Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── PQ EOTF / OETF round-trip ─────────────────────────────────────────

    #[test]
    fn pq_eotf_zero_is_zero() {
        assert!(approx_eq(pq_eotf(0.0), 0.0, EPSILON));
    }

    #[test]
    fn pq_eotf_one_is_10000_nits() {
        // PQ EOTF(1.0) should return 10000 cd/m²
        let nits = pq_eotf(1.0);
        assert!(approx_eq(nits, 10000.0, 5.0), "Expected ~10000, got {}", nits);
    }

    #[test]
    fn pq_eotf_oetf_round_trip() {
        for &l in &[0.0_f32, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 9999.0] {
            let n = pq_oetf(l);
            let l2 = pq_eotf(n);
            assert!(
                approx_eq(l, l2, l * 0.01 + 0.1),
                "PQ round-trip failed for {}: got {} -> {} -> {}",
                l, l, n, l2
            );
        }
    }

    // ── HLG EOTF / OETF ───────────────────────────────────────────────────

    #[test]
    fn hlg_eotf_zero_is_zero() {
        assert!(approx_eq(hlg_eotf(0.0), 0.0, EPSILON));
    }

    #[test]
    fn hlg_eotf_half_maps_to_expected_value() {
        // At e = 0.5 (boundary), HLG EOTF = (0.5 * 0.5) / 3 = 0.08333...
        let result = hlg_eotf(0.5);
        assert!(approx_eq(result, 1.0 / 12.0, EPSILON), "Expected 0.08333, got {}", result);
    }

    #[test]
    fn hlg_eotf_oetf_round_trip() {
        for &l in &[0.0_f32, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0] {
            let e = hlg_oetf(l);
            let l2 = hlg_eotf(e);
            assert!(
                approx_eq(l, l2, 0.002),
                "HLG round-trip failed for {}: got {} -> {} -> {}",
                l, l, e, l2
            );
        }
    }

    // ── Gamut conversion ──────────────────────────────────────────────────

    #[test]
    fn bt2020_to_bt709_white_point_invariant() {
        // D65 white [1.0, 1.0, 1.0] should stay [1.0, 1.0, 1.0] under any
        // proper chromatic adaptation matrix (equal energy white is invariant).
        let white = [1.0_f32, 1.0, 1.0];
        let result = bt2020_to_bt709(white);
        for (i, &v) in result.iter().enumerate() {
            assert!(approx_eq(v, 1.0, 0.01), "Channel {} = {}, expected ~1.0", i, v);
        }
    }

    #[test]
    fn bt709_to_bt2020_white_point_invariant() {
        let white = [1.0_f32, 1.0, 1.0];
        let result = bt709_to_bt2020(white);
        for (i, &v) in result.iter().enumerate() {
            assert!(approx_eq(v, 1.0, 0.01), "Channel {} = {}, expected ~1.0", i, v);
        }
    }

    #[test]
    fn gamut_round_trip() {
        let orig = [0.5_f32, 0.3, 0.7];
        let to_bt2020 = bt709_to_bt2020(orig);
        let back = bt2020_to_bt709(to_bt2020);
        for i in 0..3 {
            assert!(
                approx_eq(orig[i], back[i], 0.001),
                "Gamut round-trip failed channel {}: {} -> {} -> {}",
                i, orig[i], to_bt2020[i], back[i]
            );
        }
    }

    // ── ACES Filmic ───────────────────────────────────────────────────────

    #[test]
    fn aces_filmic_black_is_zero() {
        let result = aces_filmic_tonemap([0.0, 0.0, 0.0]);
        for v in result {
            assert!(approx_eq(v, 0.0, EPSILON));
        }
    }

    #[test]
    fn aces_filmic_very_bright_clamps_to_one() {
        let result = aces_filmic_tonemap([1000.0, 1000.0, 1000.0]);
        for v in result {
            assert!(approx_eq(v, 1.0, EPSILON), "Expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn aces_filmic_sdr_unity_is_reasonable() {
        // At 1.0, result should be between 0.8 and 1.0 (compressed but bright)
        let result = aces_filmic_tonemap([1.0, 1.0, 1.0]);
        for v in result {
            assert!(v > 0.7 && v <= 1.0, "Expected 0.7..1.0 at 1.0 input, got {}", v);
        }
    }

    // ── HdrMasteringMetadata ──────────────────────────────────────────────

    #[test]
    fn hdr_mastering_metadata_defaults_are_valid() {
        let meta = HdrMasteringMetadata::default();
        assert!(meta.max_luminance > meta.min_luminance);
        assert!(meta.max_cll >= meta.max_fall);
    }
}

