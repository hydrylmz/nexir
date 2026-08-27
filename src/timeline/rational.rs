// src/timeline/rational.rs

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    /// Standard 90 kHz broadcast timebase.
    pub const TIMEBASE_90K: Self = Self { num: 1, den: 90_000 };

    /// One microsecond timebase (useful for audio at 48 kHz: lcm(48000,1000000)=3000000).
    pub const TIMEBASE_1US: Self = Self { num: 1, den: 1_000_000 };

    // Standard video frame rates
    pub const FPS_23_976: Self = Self { num: 24_000, den: 1_001 };
    pub const FPS_24: Self = Self { num: 24, den: 1 };
    pub const FPS_25: Self = Self { num: 25, den: 1 };
    pub const FPS_29_97: Self = Self { num: 30_000, den: 1_001 };
    pub const FPS_30: Self = Self { num: 30, den: 1 };
    pub const FPS_50: Self = Self { num: 50, den: 1 };
    pub const FPS_59_94: Self = Self { num: 60_000, den: 1_001 };
    pub const FPS_60: Self = Self { num: 60, den: 1 };

    pub fn new(mut num: i64, mut den: i64) -> Self {
        let g = Self::gcd(num.abs(), den.abs());
        num /= g;
        den /= g;
        if den < 0 {
            num = -num;
            den = -den;
        }
        Self { num, den }    
    }

    pub fn from_pts(&self, source_pts: i64, source_tb: Rational) -> i64 {
        let a = source_pts as i128;
        let num = source_tb.num as i128 * self.den as i128;
        let den = source_tb.den as i128 * self.num as i128;
        (a * num / den) as i64
    }

    /// Euclidean GCD — building block for all rational arithmetic.
    pub fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a   
    }

    /// Add two rationals, reduce the result.
    pub fn add(self, rhs: Self) -> Self {
        let num = (self.num as i128 * rhs.den as i128) + (rhs.num as i128 * self.den as i128);
        let den = (self.den as i128) * (rhs.den as i128);
        let g = Self::gcd(num.abs() as i64, den.abs() as i64) as i128;
        let num = num / g;
        let den = den / g;
        Rational::new(num as i64, den as i64)
    }

    /// Multiply two rationals, reduce the result.
    pub fn mul(self, rhs: Self) -> Self {
        let num = (self.num as i128) * (rhs.num as i128);
        let den = (self.den as i128) * (rhs.den as i128);
        let g = Self::gcd(num.abs() as i64, den.abs() as i64) as i128;
        let num = num / g;
        let den = den / g;
        Rational::new(num as i64, den as i64)
    }

    /// Convert a PTS tick count to nanoseconds.
    pub fn pts_to_ns(&self, pts: i64) -> i64 {

        if self.den == 0 {
            panic!("Rational with zero denominator");
        }
        let pts_128 = pts as i128;
        let den_128 = self.den as i128;
        let ns_128 = (pts_128 * 1_000_000_000_i128) / den_128;
        ns_128 as i64
    }

    /// Convert nanoseconds to a PTS tick count (rounds toward zero).
    pub fn ns_to_pts(&self, ns: i64) -> i64 {
        let ns_128 = ns as i128;
        let den_128 = self.den as i128;
        let pts_128 = (ns_128 * den_128) / 1_000_000_000_i128;
        pts_128 as i64
    }

    /// Convert a PTS tick count in THIS timebase to a tick count in ANOTHER timebase.
    pub fn rescale_pts(&self, pts: i64, other: Rational) -> i64 {
        let pts_128 = pts as i128;
        let num_128 = other.den as i128;
        let den_128 = self.den as i128;
        let rescaled_128 = (pts_128 * num_128) / den_128;
        rescaled_128 as i64
    }

    /// Convert to f64 seconds. Used ONLY for display / debugging — never for timing math.
    pub fn to_f64_secs(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Converts a PTS timestamp in timebase `timebase` to frame index at this frame rate.
    pub fn pts_to_frame(&self, pts: i64, timebase: Rational) -> i64 {
        let pts_128 = pts as i128;
        let num_128 = self.num as i128 * timebase.num as i128;
        let den_128 = self.den as i128 * timebase.den as i128;
        (pts_128 * num_128 / den_128) as i64
    }

    /// Converts frame index at this frame rate to PTS timestamp in timebase `timebase`.
    pub fn frame_to_pts(&self, frame: i64, timebase: Rational) -> i64 {
        let f_128 = frame as i128;
        let num_128 = self.den as i128 * timebase.den as i128;
        let den_128 = self.num as i128 * timebase.num as i128;
        (f_128 * num_128 / den_128) as i64
    }
}

/// Scale a PTS offset by a speed factor using integer arithmetic.
/// `speed` is stored as f32 in the store; we convert to a rational approximation
/// with a denominator of 10_000 to avoid floating-point round-trip errors.
pub fn speed_scale_pts(offset: i64, speed: f32) -> i64 {
    let num = (speed * 10_000.0).round() as i64;
    let den = 10_000i64;
    // Round-half-away-from-zero
    let scaled = offset * num;
    if scaled >= 0 { (scaled + den / 2) / den }
    else           { (scaled - den / 2) / den }
}

/// Convert a frame index to a PTS timestamp in the given timebase.
///
/// Formula: `pts = round(frame * frame_rate.den * timebase.den / (frame_rate.num * timebase.num))`
pub fn frame_to_pts(frame: i64, frame_rate: Rational, timebase: Rational) -> i64 {
    let num = frame as i128 * frame_rate.den as i128 * timebase.den as i128;
    let den = frame_rate.num as i128 * timebase.num as i128;
    if den == 0 {
        return 0;
    }
    if num >= 0 {
        ((num + den / 2) / den) as i64
    } else {
        ((num - den / 2) / den) as i64
    }
}

/// Convert a PTS timestamp in the given timebase to the nearest frame index.
///
/// Formula: `frame = round(pts * timebase.num * frame_rate.num / (timebase.den * frame_rate.den))`
pub fn pts_to_frame(pts: i64, frame_rate: Rational, timebase: Rational) -> i64 {
    let num = pts as i128 * timebase.num as i128 * frame_rate.num as i128;
    let den = timebase.den as i128 * frame_rate.den as i128;
    if den == 0 {
        return 0;
    }
    if num >= 0 {
        ((num + den / 2) / den) as i64
    } else {
        ((num - den / 2) / den) as i64
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{}/{}", self.num, self.den))
    }
}