// src/colour/delta_e.rs
// ΔE2000 (CIEDE2000) colour difference formula. CPU-only, test use.

/// A colour in CIELAB space (D65 illuminant).
#[derive(Copy, Clone, Debug)]
pub struct Lab {
    pub l: f64, // Lightness: [0, 100]
    pub a: f64, // Green–red: [-128, 127]
    pub b: f64, // Blue–yellow: [-128, 127]
}

/// A colour in linear sRGB (D65).
#[derive(Copy, Clone, Debug)]
pub struct LinearRgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl LinearRgb {
    /// Convert linear sRGB to CIEXYZ (D65 illuminant).
    ///
    /// Matrix: sRGB to XYZ (IEC 61966-2-1)
    pub fn to_xyz(self) -> (f64, f64, f64) {
        let x = 0.4124564 * self.r + 0.3575761 * self.g + 0.1804375 * self.b;
        let y = 0.2126729 * self.r + 0.7151522 * self.g + 0.0721750 * self.b;
        let z = 0.0193339 * self.r + 0.1191920 * self.g + 0.9503041 * self.b;
        (x, y, z)
    }

    /// Convert linear sRGB to CIELAB (D65).
    pub fn to_lab(self) -> Lab {
        let (x, y, z) = self.to_xyz();

        // D65 white point
        let xn = 0.95047;
        let yn = 1.00000;
        let zn = 1.08883;

        fn f(t: f64) -> f64 {
            if t > 0.008856 {
                t.cbrt()
            } else {
                (29.0_f64 / 6.0_f64).powi(2) / 3.0 * t + 4.0 / 29.0
            }
        }

        let fx = f(x / xn);
        let fy = f(y / yn);
        let fz = f(z / zn);

        Lab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }
}

/// Compute the CIEDE2000 colour difference between two Lab colours.
///
/// Uses f64 throughout to avoid accumulation of floating-point error.
/// All angles in degrees; converted to radians only for trig functions.
pub fn delta_e_2000(c1: Lab, c2: Lab) -> f64 {
    use std::f64::consts::PI;

    // Step 1 — Adjust a* for chroma weighting
    let c1_ab = (c1.a.powi(2) + c1.b.powi(2)).sqrt();
    let c2_ab = (c2.a.powi(2) + c2.b.powi(2)).sqrt();
    let c_avg_ab = (c1_ab + c2_ab) / 2.0;
    let c_avg7 = c_avg_ab.powi(7);
    let g = 0.5 * (1.0 - (c_avg7 / (c_avg7 + 25.0_f64.powi(7))).sqrt());
    let a1p = c1.a * (1.0 + g);
    let a2p = c2.a * (1.0 + g);

    // Step 2 — Recompute C' and h' in adjusted space
    let c1p = (a1p.powi(2) + c1.b.powi(2)).sqrt();
    let c2p = (a2p.powi(2) + c2.b.powi(2)).sqrt();

    // h' in [0, 360°)
    let h1p = {
        let _h = a1p.atan2(c1.b).to_degrees(); // Note: atan2(y,x) — in CIE: atan2(b', a')
                                               // Wait, the standard says h' = atan2(b', a') but we flipped args above
                                               // atan2(b, a') gives angle in [-180, 180], then add 360 if < 0
        let h = c1.b.atan2(a1p).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    };
    let h2p = {
        let h = c2.b.atan2(a2p).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    };

    // Step 3 — Deltas
    let dl = c2.l - c1.l;
    let dcp = c2p - c1p;

    // Δh' (circular)
    let dhp = if c1p * c2p < 1e-12 {
        0.0
    } else {
        let diff = h2p - h1p;
        if diff.abs() <= 180.0 {
            diff
        } else if diff > 180.0 {
            diff - 360.0
        } else {
            diff + 360.0
        }
    };

    let dhp_half = dhp / 2.0;
    let big_dhp = 2.0 * (c1p * c2p).sqrt() * (dhp_half.to_radians()).sin();

    // Step 4 — Averages
    let l_avg = (c1.l + c2.l) / 2.0;
    let c_avgp = (c1p + c2p) / 2.0;

    let h_avgp = if c1p * c2p < 1e-12 {
        h1p + h2p
    } else {
        let diff = (h1p - h2p).abs();
        if diff <= 180.0 {
            (h1p + h2p) / 2.0
        } else if h1p + h2p < 360.0 {
            (h1p + h2p + 360.0) / 2.0
        } else {
            (h1p + h2p - 360.0) / 2.0
        }
    };

    // Step 5 — Weighting functions
    let l50 = l_avg - 50.0;
    let sl = 1.0 + 0.015 * l50.powi(2) / (20.0 + l50.powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_avgp;

    let t = 1.0 - 0.17 * ((h_avgp - 30.0).to_radians()).cos()
        + 0.24 * (2.0 * h_avgp.to_radians()).cos()
        + 0.32 * ((3.0 * h_avgp + 6.0).to_radians()).cos()
        - 0.20 * ((4.0 * h_avgp - 63.0).to_radians()).cos();

    let sh = 1.0 + 0.015 * c_avgp * t;

    let c_avg7p = c_avgp.powi(7);
    let rc = 2.0 * (c_avg7p / (c_avg7p + 25.0_f64.powi(7))).sqrt();
    let d_theta = 30.0 * (-(((h_avgp - 275.0) / 25.0).powi(2))).exp();
    let rt = -(2.0 * d_theta.to_radians()).sin() * rc;

    // Step 6 — Final ΔE2000
    let term_l = dl / sl;
    let term_c = dcp / sc;
    let term_h = big_dhp / sh;

    (term_l.powi(2) + term_c.powi(2) + term_h.powi(2) + rt * term_c * term_h).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sharma 2005 pair 1. Tolerance: < 0.0001.
    #[test]
    fn sharma_2005_pair_1() {
        let c1 = Lab {
            l: 50.0000,
            a: 2.6772,
            b: -79.7751,
        };
        let c2 = Lab {
            l: 50.0000,
            a: 0.0000,
            b: -82.7485,
        };
        let de = delta_e_2000(c1, c2);
        assert!((de - 2.0425).abs() < 0.0001, "got {de:.4}");
    }

    /// Sharma 2005 pair 17 (tests the RT rotation term).
    #[test]
    fn sharma_2005_pair_17() {
        let c1 = Lab {
            l: 50.0000,
            a: 3.1571,
            b: -77.2803,
        };
        let c2 = Lab {
            l: 50.0000,
            a: 2.8361,
            b: -74.0200,
        };
        let de = delta_e_2000(c1, c2);
        assert!((de - 0.6498).abs() < 0.0001, "got {de:.4}");
    }

    #[test]
    fn identical_colours_zero() {
        let c = Lab {
            l: 50.0,
            a: 20.0,
            b: -10.0,
        };
        let de = delta_e_2000(c, c);
        assert!(de < 1e-6, "identical colours should give 0, got {de}");
    }
}
