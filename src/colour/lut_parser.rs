// src/colour/lut_parser.rs

/// A parsed 3D LUT.
pub struct Lut3D {
    /// Cube size per axis (e.g. 33 for a 33³ LUT).
    pub size:       u32,
    /// Output RGB values, f32. Length = size³.
    /// Index: r + g * size + b * size²   (R varies fastest).
    pub data:       Vec<[f32; 3]>,
    /// Input domain min (usually [0,0,0]).
    pub domain_min: [f32; 3],
    /// Input domain max (usually [1,1,1]).
    pub domain_max: [f32; 3],
}

impl Lut3D {
    /// Parse a `.cube` file from a string.
    pub fn parse(input: &str) -> Result<Self, LutError> {
        let mut size: Option<u32> = None;
        let mut domain_min = [0.0f32; 3];
        let mut domain_max = [1.0f32; 3];
        let mut data: Vec<[f32; 3]> = Vec::new();

        // Step 1 — Scan lines: header keywords and data
        for (line_num, line) in input.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("LUT_3D_SIZE") {
                let n: u32 = rest.trim().parse().map_err(|_| {
                    LutError::InvalidHeader(format!("Invalid LUT_3D_SIZE on line {}", line_num + 1))
                })?;
                size = Some(n);
            } else if let Some(rest) = trimmed.strip_prefix("DOMAIN_MIN") {
                domain_min = parse_triple(rest, line_num)?;
            } else if let Some(rest) = trimmed.strip_prefix("DOMAIN_MAX") {
                domain_max = parse_triple(rest, line_num)?;
            } else if trimmed.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '-' || c == '+') {
                // Step 2 — Parse data line
                let vals: Vec<f32> = trimmed.split_whitespace()
                    .map(|s| s.parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| LutError::ParseError {
                        line: line_num + 1,
                        content: trimmed.to_string(),
                    })?;

                if vals.len() != 3 {
                    return Err(LutError::ParseError {
                        line: line_num + 1,
                        content: format!("Expected 3 values, got {}", vals.len()),
                    });
                }

                data.push([vals[0], vals[1], vals[2]]);
            } else {
                // Unknown header keyword — ignore gracefully (some exporters add extras)
            }
        }

        // Step 3 — Validate
        let n = size.ok_or(LutError::MissingSize)?;
        let expected = (n * n * n) as usize;
        if data.len() != expected {
            return Err(LutError::DataCountMismatch { expected, got: data.len() });
        }

        // Step 4 — Normalise domain if non-standard
        let needs_normalise = domain_min != [0.0, 0.0, 0.0] || domain_max != [1.0, 1.0, 1.0];
        if needs_normalise {
            for entry in &mut data {
                for ch in 0..3 {
                    let range = domain_max[ch] - domain_min[ch];
                    if range.abs() > 1e-6 {
                        entry[ch] = (entry[ch] - domain_min[ch]) / range;
                    }
                }
            }
        }

        Ok(Lut3D { size: n, data, domain_min, domain_max })
    }

    /// Look up the LUT output for a given (r, g, b) input, using CPU trilinear interpolation.
    /// Used in tests to verify the GPU sampler gives matching results.
    pub fn sample(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let n = (self.size - 1) as f32;

        let u = (r * n).clamp(0.0, n);
        let v = (g * n).clamp(0.0, n);
        let w = (b * n).clamp(0.0, n);

        let u0 = (u.floor() as usize).min(self.size as usize - 2);
        let v0 = (v.floor() as usize).min(self.size as usize - 2);
        let w0 = (w.floor() as usize).min(self.size as usize - 2);

        let du = u - u0 as f32;
        let dv = v - v0 as f32;
        let dw = w - w0 as f32;

        let s = self.size as usize;
        let idx = |ri: usize, gi: usize, bi: usize| ri + gi * s + bi * s * s;

        let c000 = self.data[idx(u0,   v0,   w0  )];
        let c001 = self.data[idx(u0,   v0,   w0+1)];
        let c010 = self.data[idx(u0,   v0+1, w0  )];
        let c011 = self.data[idx(u0,   v0+1, w0+1)];
        let c100 = self.data[idx(u0+1, v0,   w0  )];
        let c101 = self.data[idx(u0+1, v0,   w0+1)];
        let c110 = self.data[idx(u0+1, v0+1, w0  )];
        let c111 = self.data[idx(u0+1, v0+1, w0+1)];

        fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        }

        // Trilinear interpolation: dw first, then dv, then du
        let c00 = lerp3(c000, c001, dw);
        let c01 = lerp3(c010, c011, dw);
        let c10 = lerp3(c100, c101, dw);
        let c11 = lerp3(c110, c111, dw);
        let c0  = lerp3(c00, c01, dv);
        let c1  = lerp3(c10, c11, dv);
        lerp3(c0, c1, du)
    }
}

fn parse_triple(rest: &str, line_num: usize) -> Result<[f32; 3], LutError> {
    let parts: Vec<f32> = rest.split_whitespace()
        .map(|s| s.parse::<f32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| LutError::InvalidHeader(format!("Invalid triple on line {}", line_num + 1)))?;

    if parts.len() < 3 {
        return Err(LutError::InvalidHeader(format!(
            "Expected 3 values on line {}, got {}", line_num + 1, parts.len()
        )));
    }
    Ok([parts[0], parts[1], parts[2]])
}

#[derive(Debug)]
pub enum LutError {
    MissingSize,
    InvalidHeader(String),
    ParseError { line: usize, content: String },
    DataCountMismatch { expected: usize, got: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut_str(n: usize) -> String {
        let mut s = format!("LUT_3D_SIZE {}\n", n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let rf = r as f32 / (n - 1) as f32;
                    let gf = g as f32 / (n - 1) as f32;
                    let bf = b as f32 / (n - 1) as f32;
                    s.push_str(&format!("{:.6} {:.6} {:.6}\n", rf, gf, bf));
                }
            }
        }
        s
    }

    #[test]
    fn parse_identity_lut() {
        let src = identity_lut_str(4);
        let lut = Lut3D::parse(&src).expect("should parse");
        assert_eq!(lut.size, 4);
        assert_eq!(lut.data.len(), 64);
    }

    #[test]
    fn sample_identity_corners() {
        let src = identity_lut_str(17);
        let lut = Lut3D::parse(&src).expect("should parse");

        let out = lut.sample(0.0, 0.0, 0.0);
        assert!((out[0]).abs() < 0.01 && (out[1]).abs() < 0.01 && (out[2]).abs() < 0.01);

        let out = lut.sample(1.0, 1.0, 1.0);
        assert!((out[0] - 1.0).abs() < 0.01 && (out[1] - 1.0).abs() < 0.01 && (out[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn missing_size_returns_error() {
        let src = "# no size header\n0.0 0.0 0.0\n";
        assert!(matches!(Lut3D::parse(src), Err(LutError::MissingSize)));
    }

    #[test]
    fn data_count_mismatch_returns_error() {
        let src = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n";
        assert!(matches!(Lut3D::parse(src), Err(LutError::DataCountMismatch { .. })));
    }
}
