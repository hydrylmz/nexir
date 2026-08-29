// src/colour/yuv.rs
// YUV → RGB conversion parameters derived from a clip's actual colour metadata.
//
// P1.6 — this is the piece that turns "we have colour functions" into "the
// pipeline is colour managed".  The GPU shader used to hardcode three matrices
// and two range formulas with 8-BIT constants; every decision now happens here,
// in code a unit test can check without a GPU, and the shader just applies the
// numbers it is given.
//
// Two bugs this replaces, both invisible in an 8-bit BT.709 export and both
// wrong for anything else:
//
//  1. **10-bit planar YUV rendered near-black.**  `YuvUploadNode` puts 10-bit
//     samples in `R16Unorm`/`Rg16Unorm` textures, so `textureLoad` returns
//     `raw / 65535`.  For `yuv420p10le` the 10-bit code sits in the LOW bits, so
//     that is `code / 65535` — 1/64 of the intended `code / 1023`.  (P010 escapes
//     it because FFmpeg stores those codes shifted left by 6.)  Handled by
//     [`YuvConversion::sample_scale`].
//  2. **8-bit range constants applied to 10-bit data.**  Limited range is
//     `16 << (n-8) … 235 << (n-8)`, not `16 … 235`, at every depth.  Handled by
//     the offset/scale pair, which is computed from the real bit depth.
//
// The matrix is derived from the luma coefficients (Kr, Kb) of the signalled
// matrix coefficients rather than transcribed, so BT.601/709/2020 all come from
// one formula:
//
//     R = Y + 2(1-Kr)·Cr
//     B = Y + 2(1-Kb)·Cb
//     G = Y − (2·Kr(1-Kr)/Kg)·Cr − (2·Kb(1-Kb)/Kg)·Cb        where Kg = 1-Kr-Kb

use crate::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients};

/// Luma coefficients (Kr, Kb) for a set of matrix coefficients.
///
/// Values from ITU-R BT.601-7 §2.5.1, BT.709-6 §3.2 and BT.2020-2 Table 4.
pub fn luma_coefficients(matrix: MatrixCoefficients) -> (f32, f32) {
    match matrix {
        MatrixCoefficients::Bt601  => (0.299,  0.114),
        MatrixCoefficients::Bt2020 => (0.2627, 0.0593),
        // BT.709 is also the documented fallback for `Unknown`; callers should
        // resolve Unknown through `ColorInfo::effective_matrix` first so the
        // resolution heuristics apply, but landing on BT.709 here is safe.
        MatrixCoefficients::Bt709 | MatrixCoefficients::Unknown => (0.2126, 0.0722),
    }
}

/// Everything the YUV→RGB shader needs, fully resolved: no enums, no branches on
/// range, no assumption about bit depth.
///
/// Layout matches `struct YuvParams` in `yuv_to_rgb.wgsl` exactly — 80 bytes.
/// The matrix is stored as three padded `vec4`s because WGSL aligns `vec3<f32>`
/// to 16 bytes inside a struct; storing a bare `[f32; 9]` would make the shader
/// read the wrong words.
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct YuvConversion {
    /// Row 0 of the YCbCr→RGB matrix: `[1, c_cb, c_cr, pad]` for R.
    pub row_r: [f32; 4],
    /// Row 1: G.
    pub row_g: [f32; 4],
    /// Row 2: B.
    pub row_b: [f32; 4],
    /// Multiplier that turns a raw texture sample in `[0,1]` into a code value
    /// normalised against the format's own maximum.  1.0 for 8-bit, ~64 for
    /// low-aligned 10-bit in a 16-bit texture, ~1.0 for MSB-aligned P010.
    pub sample_scale: f32,
    /// Subtracted from the scaled luma before `luma_scale` is applied
    /// (`16 << (n-8) / max` for limited range, 0 for full).
    pub luma_offset: f32,
    /// Reciprocal of the luma code range (`max / (219 << (n-8))` limited,
    /// 1.0 full).
    pub luma_scale: f32,
    /// Subtracted from each scaled chroma sample to centre it on zero
    /// (`128 << (n-8) / max`, which is ~0.5 at every depth).
    pub chroma_offset: f32,
    /// Reciprocal of the chroma code range (`max / (224 << (n-8))` limited,
    /// 1.0 full).
    pub chroma_scale: f32,
    pub width:  u32,
    pub height: u32,
    pub _pad:   u32,
}

/// Byte size of the push-constant block.
///
/// 3 × vec4 (48) + 5 scalars (20) + width/height/pad (12) = 80, which is a
/// multiple of 16 as WGSL requires for a struct containing vec4s, and under the
/// 128-byte push-constant limit the device is created with.
const _: () = assert!(std::mem::size_of::<YuvConversion>() == 80);

impl YuvConversion {
    /// Build the conversion for a clip.
    ///
    /// `width`/`height` are the clip's pixel dimensions: they size the shader's
    /// bounds check and also feed `ColorInfo`'s resolution heuristics, which is
    /// how an unsignalled SD clip still gets BT.601 rather than BT.709.
    ///
    /// `is_semi_planar` distinguishes NV12/P010 (chroma interleaved in one plane,
    /// and for P010 the 10-bit codes are MSB-aligned) from planar YUV420P10.
    /// Getting it wrong is the 64× brightness error described at the top of this
    /// file, so it is a required argument rather than something inferred.
    pub fn new(color: ColorInfo, width: u32, height: u32, is_semi_planar: bool) -> Self {
        let (kr, kb) = luma_coefficients(color.effective_matrix(width, height));
        let kg = 1.0 - kr - kb;

        // Guard against a degenerate coefficient set (kg == 0) producing inf/NaN
        // that would paint the whole frame with garbage.  Not reachable from the
        // table above; cheap insurance if a future matrix is added wrongly.
        let kg = if kg.abs() < 1e-6 { 1.0 } else { kg };

        let cr_r = 2.0 * (1.0 - kr);
        let cb_b = 2.0 * (1.0 - kb);
        let cr_g = -2.0 * kr * (1.0 - kr) / kg;
        let cb_g = -2.0 * kb * (1.0 - kb) / kg;

        let depth = if color.bit_depth == 0 { 8 } else { color.bit_depth } as u32;
        let max_code = ((1u32 << depth) - 1) as f32;

        // How a raw texture sample maps back to `code / max_code`.
        //
        //  * 8-bit  → R8Unorm/Rg8Unorm: the sample already IS code/255.
        //  * 10/12-bit semi-planar (P010/P016-style): FFmpeg MSB-aligns the codes
        //    in 16-bit words, so the sample is `code << (16-depth) / 65535`;
        //    correcting it is a factor of `65535 / (max_code << (16-depth))`,
        //    which is within 0.1% of 1.0 — applied anyway so the white point is
        //    exact.
        //  * 10/12-bit planar: codes are LSB-aligned, so the sample is
        //    `code / 65535` and the factor is `65535 / max_code` (≈64 at 10-bit).
        let sample_scale = if depth <= 8 {
            1.0
        } else if is_semi_planar {
            let shift = 16 - depth;
            65535.0 / (max_code * (1u32 << shift) as f32)
        } else {
            65535.0 / max_code
        };

        // Limited ("MPEG"/broadcast) range: luma 16..235, chroma 128±112, both
        // scaled by 2^(depth-8).  Full ("JPEG") range uses the whole code space.
        let scale_to_depth = (1u32 << (depth - 8)) as f32;
        let (luma_offset, luma_scale, chroma_offset, chroma_scale) =
            match color.effective_range() {
                ColorRange::Full => (
                    0.0,
                    1.0,
                    // Full-range chroma is still centred on the midpoint, which is
                    // 128<<(n-8) — NOT 0.5 exactly, because the code space is odd.
                    (128.0 * scale_to_depth) / max_code,
                    1.0,
                ),
                // `effective_range` already resolves Unknown to Limited; matching
                // it explicitly documents that this is the deliberate default for
                // unsignalled video rather than an accident of ordering.
                ColorRange::Limited | ColorRange::Unknown => (
                    (16.0 * scale_to_depth) / max_code,
                    max_code / (219.0 * scale_to_depth),
                    (128.0 * scale_to_depth) / max_code,
                    max_code / (224.0 * scale_to_depth),
                ),
            };

        Self {
            row_r: [1.0, 0.0,  cr_r, 0.0],
            row_g: [1.0, cb_g, cr_g, 0.0],
            row_b: [1.0, cb_b, 0.0,  0.0],
            sample_scale,
            luma_offset,
            luma_scale,
            chroma_offset,
            chroma_scale,
            width,
            height,
            _pad: 0,
        }
    }

    /// Apply this conversion on the CPU, exactly as the shader does.
    ///
    /// Inputs are raw texture samples in `[0,1]` — i.e. what `textureLoad`
    /// returns — so a test can feed it the same numbers the GPU would see.
    /// Output is non-linear R'G'B' in the source's own transfer function, NOT
    /// clamped: HDR content legitimately exceeds 1.0 and clamping here is what
    /// would silently destroy it.
    pub fn apply(&self, y_sample: f32, cb_sample: f32, cr_sample: f32) -> [f32; 3] {
        let y  = (y_sample  * self.sample_scale - self.luma_offset) * self.luma_scale;
        let cb = (cb_sample * self.sample_scale - self.chroma_offset) * self.chroma_scale;
        let cr = (cr_sample * self.sample_scale - self.chroma_offset) * self.chroma_scale;

        [
            self.row_r[0] * y + self.row_r[1] * cb + self.row_r[2] * cr,
            self.row_g[0] * y + self.row_g[1] * cb + self.row_g[2] * cr,
            self.row_b[0] * y + self.row_b[1] * cb + self.row_b[2] * cr,
        ]
    }
}

/// The RGB→YUV direction, for the encode path.
///
/// P1.9 — this exists so the zero-copy NVENC export can hand the encoder NV12
/// that *we* converted, instead of packed RGB that the DRIVER converts. The
/// driver applies BT.601 with no way to tell it otherwise (writing
/// NV_ENC_CONFIG's per-codec VUI union means guessed offsets), so a BT.709 job
/// taking that path produced a file whose samples and tags disagreed — red
/// decoding as `[255, 25, 0]`. `ExportJob::nvenc_zero_copy_is_colour_safe`
/// currently routes every non-BT.601 job away from zero-copy for that reason.
/// Once the conversion happens here, our tags are authoritative by construction
/// and the gate can open for every matrix.
///
/// This is the exact inverse of [`YuvConversion`], and deliberately shares its
/// `(Kr, Kb)` derivation via [`luma_coefficients`] rather than transcribing a
/// second set of matrices: two independently-written matrices are two things to
/// keep in step, and the round-trip test below only means something if the pair
/// is a true inverse.
///
/// Layout matches `struct RgbToYuvParams` in the NV12 encode shader — 80 bytes,
/// same as `YuvConversion`, with the matrix stored as three padded `vec4`s
/// because WGSL aligns `vec3<f32>` to 16 bytes inside a struct.
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RgbToYuv {
    /// Row 0 of the RGB→YCbCr matrix: `[Kr, Kg, Kb, pad]`, producing Y.
    pub row_y:  [f32; 4],
    /// Row 1: produces Cb, centred on zero.
    pub row_cb: [f32; 4],
    /// Row 2: produces Cr, centred on zero.
    pub row_cr: [f32; 4],
    /// Multiplies Y before `luma_offset` is added
    /// (`219 << (n-8) / max` limited, 1.0 full).
    pub luma_scale:    f32,
    /// Added to the scaled luma (`16 << (n-8) / max` limited, 0 full).
    pub luma_offset:   f32,
    /// Multiplies each zero-centred chroma value
    /// (`224 << (n-8) / max` limited, 1.0 full).
    pub chroma_scale:  f32,
    /// Added to scaled chroma to move it off zero (`128 << (n-8) / max`).
    pub chroma_offset: f32,
    pub width:  u32,
    pub height: u32,
    pub _pad:   [u32; 2],
}

/// Same 80 bytes as [`YuvConversion`], and for the same reasons: a multiple of
/// 16 for WGSL's vec4 alignment, and inside the 128-byte push-constant limit the
/// device is created with.
const _: () = assert!(std::mem::size_of::<RgbToYuv>() == 80);

impl RgbToYuv {
    /// Build the forward conversion for an output colour description.
    ///
    /// `width`/`height` size the shader's bounds check and feed `ColorInfo`'s
    /// resolution heuristics, exactly as in [`YuvConversion::new`].
    ///
    /// Unlike the decode direction there is no `is_semi_planar` argument: this
    /// produces normalised samples in `[0, 1]`, and how they are packed into
    /// bytes (NV12) or MSB-aligned 16-bit words (P010) is the caller's business.
    pub fn new(color: ColorInfo, width: u32, height: u32) -> Self {
        let (kr, kb) = luma_coefficients(color.effective_matrix(width, height));
        let kg = 1.0 - kr - kb;
        // Same degenerate-coefficient guard as the inverse; unreachable from the
        // published table, cheap insurance against a future bad entry.
        let kg = if kg.abs() < 1e-6 { 1.0 } else { kg };

        // Cb = (B - Y) / (2(1-Kb)),  Cr = (R - Y) / (2(1-Kr)), expanded so each
        // is one dot product against RGB.
        let cb_den = 2.0 * (1.0 - kb);
        let cr_den = 2.0 * (1.0 - kr);

        let depth = if color.bit_depth == 0 { 8 } else { color.bit_depth } as u32;
        let max_code = ((1u32 << depth) - 1) as f32;
        let scale_to_depth = (1u32 << (depth - 8)) as f32;

        // Exact inverse of YuvConversion's (offset, scale) pair: it computes
        // `(sample - offset) * scale`, so this computes `value / scale + offset`.
        let (luma_scale, luma_offset, chroma_scale, chroma_offset) =
            match color.effective_range() {
                ColorRange::Full => (
                    1.0,
                    0.0,
                    1.0,
                    (128.0 * scale_to_depth) / max_code,
                ),
                ColorRange::Limited | ColorRange::Unknown => (
                    (219.0 * scale_to_depth) / max_code,
                    (16.0 * scale_to_depth) / max_code,
                    (224.0 * scale_to_depth) / max_code,
                    (128.0 * scale_to_depth) / max_code,
                ),
            };

        Self {
            row_y:  [kr, kg, kb, 0.0],
            row_cb: [-kr / cb_den, -kg / cb_den, (1.0 - kb) / cb_den, 0.0],
            row_cr: [(1.0 - kr) / cr_den, -kg / cr_den, -kb / cr_den, 0.0],
            luma_scale,
            luma_offset,
            chroma_scale,
            chroma_offset,
            width,
            height,
            _pad: [0; 2],
        }
    }

    /// Apply this conversion on the CPU, exactly as the shader does.
    ///
    /// Input is non-linear R'G'B' as the render graph produces it; output is
    /// `[y, cb, cr]` as normalised samples in `[0, 1]`, i.e. `code / max_code`.
    ///
    /// **RGB is clamped to `[0, 1]` before the matrix, not after.**  That ordering
    /// is deliberate and matters twice over:
    ///
    ///  * Clamping Y/Cb/Cr independently afterwards would move the result off the
    ///    gamut boundary and **shift the hue** — the three components are not
    ///    independent, so saturating one without the others is a colour error, not
    ///    a clip.  Clamping the input keeps an out-of-range colour on the boundary
    ///    it was heading for.
    ///  * With the input in `[0, 1]`, limited-range output lands inside
    ///    16…235 `<< (n-8)` **by construction**, so no legal-range enforcement is
    ///    needed on the way out and super-white cannot occupy the reserved codes.
    ///
    /// It also matches what `ABGR10_REPACK_WGSL` already does with its input, so
    /// both encode shaders treat out-of-range values the same way.
    ///
    /// Tone mapping runs earlier in the graph, so in practice the clamp is a
    /// no-op; it is here so an HDR value that slipped through degrades to a
    /// saturated colour rather than a wrapped byte.
    pub fn apply(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        let y  = self.row_y[0]  * r + self.row_y[1]  * g + self.row_y[2]  * b;
        let cb = self.row_cb[0] * r + self.row_cb[1] * g + self.row_cb[2] * b;
        let cr = self.row_cr[0] * r + self.row_cr[1] * g + self.row_cr[2] * b;

        [
            y  * self.luma_scale   + self.luma_offset,
            cb * self.chroma_scale + self.chroma_offset,
            cr * self.chroma_scale + self.chroma_offset,
        ]
    }

    /// [`Self::apply`] quantised to 8-bit codes, which is what an NV12 plane
    /// holds.  Rounds half-up, matching the shader's `+ 0.5` truncation.
    pub fn apply_u8(&self, r: f32, g: f32, b: f32) -> [u8; 3] {
        let s = self.apply(r, g, b);
        [
            (s[0] * 255.0 + 0.5) as u8,
            (s[1] * 255.0 + 0.5) as u8,
            (s[2] * 255.0 + 0.5) as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::source::{ColorPrimaries, TransferFunction};

    const HD_W: u32 = 1920;
    const HD_H: u32 = 1080;

    fn info(matrix: MatrixCoefficients, range: ColorRange, bit_depth: u8) -> ColorInfo {
        ColorInfo {
            transfer_fn: TransferFunction::Bt709,
            range,
            matrix,
            primaries: ColorPrimaries::Bt709,
            bit_depth,
        }
    }

    /// An 8-bit code as the shader would see it out of an R8Unorm texture.
    fn s8(code: u8) -> f32 {
        code as f32 / 255.0
    }

    /// A 10-bit code as the shader sees it out of an R16Unorm texture holding
    /// LSB-aligned planar data (yuv420p10le).
    fn s10_planar(code: u16) -> f32 {
        code as f32 / 65535.0
    }

    /// A 10-bit code as the shader sees it out of an R16Unorm texture holding
    /// MSB-aligned semi-planar data (P010).
    fn s10_p010(code: u16) -> f32 {
        ((code as u32) << 6) as f32 / 65535.0
    }

    fn assert_rgb(got: [f32; 3], want: [f32; 3], tol: f32, label: &str) {
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() <= tol,
                "{label}: channel {i} = {}, expected {} (tolerance {tol}); full {got:?} vs {want:?}",
                got[i], want[i]
            );
        }
    }

    // ── Matrix coefficients ───────────────────────────────────────────────────

    #[test]
    fn derived_matrices_match_the_published_coefficients() {
        // The numbers the WGSL shader used to hardcode, which are the ones the
        // standards publish.  Derived here from (Kr, Kb) instead.
        let bt709 = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H, false,
        );
        assert!((bt709.row_r[2] - 1.5748).abs() < 1e-4, "709 Cr→R: {}", bt709.row_r[2]);
        assert!((bt709.row_b[1] - 1.8556).abs() < 1e-4, "709 Cb→B: {}", bt709.row_b[1]);
        assert!((bt709.row_g[1] + 0.187324).abs() < 1e-4, "709 Cb→G: {}", bt709.row_g[1]);
        assert!((bt709.row_g[2] + 0.468124).abs() < 1e-4, "709 Cr→G: {}", bt709.row_g[2]);

        let bt601 = YuvConversion::new(
            info(MatrixCoefficients::Bt601, ColorRange::Limited, 8), 720, 480, false,
        );
        assert!((bt601.row_r[2] - 1.402).abs() < 1e-4, "601 Cr→R: {}", bt601.row_r[2]);
        assert!((bt601.row_b[1] - 1.772).abs() < 1e-4, "601 Cb→B: {}", bt601.row_b[1]);
        assert!((bt601.row_g[1] + 0.344136).abs() < 1e-4, "601 Cb→G: {}", bt601.row_g[1]);
        assert!((bt601.row_g[2] + 0.714136).abs() < 1e-4, "601 Cr→G: {}", bt601.row_g[2]);

        let bt2020 = YuvConversion::new(
            info(MatrixCoefficients::Bt2020, ColorRange::Limited, 10), 3840, 2160, false,
        );
        assert!((bt2020.row_r[2] - 1.4746).abs() < 1e-4, "2020 Cr→R: {}", bt2020.row_r[2]);
        assert!((bt2020.row_b[1] - 1.8814).abs() < 1e-4, "2020 Cb→B: {}", bt2020.row_b[1]);
        assert!((bt2020.row_g[1] + 0.164553).abs() < 1e-4, "2020 Cb→G: {}", bt2020.row_g[1]);
        assert!((bt2020.row_g[2] + 0.571353).abs() < 1e-4, "2020 Cr→G: {}", bt2020.row_g[2]);
    }

    #[test]
    fn every_matrix_maps_neutral_grey_to_neutral_rgb() {
        // Chroma at its midpoint must produce R == G == B for ANY matrix: the
        // Cb/Cr columns only ever move colour, never luma.
        for matrix in [
            MatrixCoefficients::Bt601,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Bt2020,
            MatrixCoefficients::Unknown,
        ] {
            for range in [ColorRange::Limited, ColorRange::Full] {
                let c = YuvConversion::new(info(matrix, range, 8), HD_W, HD_H, false);
                let rgb = c.apply(s8(128), s8(128), s8(128));
                assert!(
                    (rgb[0] - rgb[1]).abs() < 1e-4 && (rgb[1] - rgb[2]).abs() < 1e-4,
                    "{matrix:?}/{range:?}: mid-grey is not neutral: {rgb:?}"
                );
            }
        }
    }

    // ── Limited vs full range ─────────────────────────────────────────────────

    #[test]
    fn limited_range_8bit_maps_16_and_235_to_black_and_white() {
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H, false,
        );
        assert_rgb(c.apply(s8(16),  s8(128), s8(128)), [0.0; 3], 1e-4, "limited black");
        assert_rgb(c.apply(s8(235), s8(128), s8(128)), [1.0; 3], 1e-4, "limited white");
    }

    #[test]
    fn full_range_8bit_maps_0_and_255_to_black_and_white() {
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Full, 8), HD_W, HD_H, false,
        );
        assert_rgb(c.apply(s8(0),   s8(128), s8(128)), [0.0; 3], 1e-4, "full black");
        assert_rgb(c.apply(s8(255), s8(128), s8(128)), [1.0; 3], 1e-3, "full white");
    }

    #[test]
    fn range_handling_is_not_interchangeable() {
        // Reading limited-range video as full range is the classic washed-out /
        // crushed-black bug.  This pins the difference so a regression in the
        // range plumbing cannot pass unnoticed: code 16 is black in one and a
        // visible dark grey in the other.
        let limited = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H, false,
        );
        let full = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Full, 8), HD_W, HD_H, false,
        );
        let as_limited = limited.apply(s8(16), s8(128), s8(128))[0];
        let as_full    = full.apply(s8(16), s8(128), s8(128))[0];
        assert!(as_limited.abs() < 1e-4, "limited: {as_limited}");
        assert!(
            (as_full - 16.0 / 255.0).abs() < 1e-3,
            "full range must not remap black: {as_full}"
        );
    }

    #[test]
    fn unknown_range_is_treated_as_limited() {
        // Explicit behaviour for unknown metadata (P1.6): broadcast video is the
        // overwhelmingly common case, so Unknown means Limited.
        let unknown = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Unknown, 8), HD_W, HD_H, false,
        );
        let limited = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H, false,
        );
        assert_eq!(unknown.luma_offset, limited.luma_offset);
        assert_eq!(unknown.luma_scale,  limited.luma_scale);
    }

    // ── Bit depth ─────────────────────────────────────────────────────────────

    #[test]
    fn planar_10bit_is_not_64x_too_dark() {
        // The regression test for the bug this module fixes.  Peak white in
        // yuv420p10le limited range is code 940; before `sample_scale` the shader
        // computed ≈0.0156 for it, i.e. near-black.
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt2020, ColorRange::Limited, 10), 3840, 2160, false,
        );
        assert!(
            (c.sample_scale - 64.06).abs() < 0.1,
            "planar 10-bit sample_scale should be ~64, got {}",
            c.sample_scale
        );
        assert_rgb(
            c.apply(s10_planar(940), s10_planar(512), s10_planar(512)),
            [1.0; 3], 2e-3, "10-bit planar white",
        );
        assert_rgb(
            c.apply(s10_planar(64), s10_planar(512), s10_planar(512)),
            [0.0; 3], 2e-3, "10-bit planar black",
        );
    }

    #[test]
    fn p010_semi_planar_10bit_is_msb_aligned() {
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt2020, ColorRange::Limited, 10), 3840, 2160, true,
        );
        assert!(
            (c.sample_scale - 1.0).abs() < 0.01,
            "P010 sample_scale should be ~1.0, got {}",
            c.sample_scale
        );
        assert_rgb(
            c.apply(s10_p010(940), s10_p010(512), s10_p010(512)),
            [1.0; 3], 2e-3, "P010 white",
        );
        assert_rgb(
            c.apply(s10_p010(64), s10_p010(512), s10_p010(512)),
            [0.0; 3], 2e-3, "P010 black",
        );
    }

    #[test]
    fn twelve_bit_planar_round_trips_black_and_white() {
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt2020, ColorRange::Limited, 12), 3840, 2160, false,
        );
        let s = |code: u16| code as f32 / 65535.0;
        // Limited range at 12-bit: 16<<4 = 256 black, 235<<4 = 3760 white.
        assert_rgb(c.apply(s(256),  s(2048), s(2048)), [0.0; 3], 2e-3, "12-bit black");
        assert_rgb(c.apply(s(3760), s(2048), s(2048)), [1.0; 3], 2e-3, "12-bit white");
    }

    // ── Primary colours, per matrix ───────────────────────────────────────────

    #[test]
    fn primary_colours_decode_to_the_right_dominant_channel() {
        // Full-range 8-bit YCbCr for pure red/green/blue, per matrix.  A swapped
        // Cb/Cr column passes every grey test above but fails here.
        for matrix in [
            MatrixCoefficients::Bt601,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Bt2020,
        ] {
            let (kr, kb) = luma_coefficients(matrix);
            let kg = 1.0 - kr - kb;
            let c = YuvConversion::new(info(matrix, ColorRange::Full, 8), HD_W, HD_H, false);

            // Forward transform (full range): Y = Kr·R + Kg·G + Kb·B,
            // Cb = (B-Y)/(2(1-Kb)), Cr = (R-Y)/(2(1-Kr)), chroma offset +0.5.
            let encode = |r: f32, g: f32, b: f32| {
                let y = kr * r + kg * g + kb * b;
                let cb = (b - y) / (2.0 * (1.0 - kb)) + 128.0 / 255.0;
                let cr = (r - y) / (2.0 * (1.0 - kr)) + 128.0 / 255.0;
                (y, cb, cr)
            };

            for (label, rgb) in [
                ("red",   [1.0, 0.0, 0.0]),
                ("green", [0.0, 1.0, 0.0]),
                ("blue",  [0.0, 0.0, 1.0]),
                ("grey",  [0.5, 0.5, 0.5]),
            ] {
                let (y, cb, cr) = encode(rgb[0], rgb[1], rgb[2]);
                let out = c.apply(y, cb, cr);
                assert_rgb(out, rgb, 4e-3, &format!("{matrix:?} {label}"));
            }
        }
    }

    #[test]
    fn matrices_are_measurably_different() {
        // If ColorInfo were ignored and everything fell through to BT.709, every
        // test above would still pass.  This one would not.
        let red_ish = (s8(120), s8(90), s8(200));
        let bt601 = YuvConversion::new(
            info(MatrixCoefficients::Bt601, ColorRange::Limited, 8), 720, 480, false,
        )
        .apply(red_ish.0, red_ish.1, red_ish.2);
        let bt709 = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H, false,
        )
        .apply(red_ish.0, red_ish.1, red_ish.2);
        let bt2020 = YuvConversion::new(
            info(MatrixCoefficients::Bt2020, ColorRange::Limited, 8), 3840, 2160, false,
        )
        .apply(red_ish.0, red_ish.1, red_ish.2);

        assert!(
            (bt601[1] - bt709[1]).abs() > 0.01,
            "BT.601 and BT.709 green must differ: {bt601:?} vs {bt709:?}"
        );
        assert!(
            (bt709[1] - bt2020[1]).abs() > 0.01,
            "BT.709 and BT.2020 green must differ: {bt709:?} vs {bt2020:?}"
        );
    }

    #[test]
    fn hdr_values_above_one_are_not_clamped() {
        // Tone mapping is a later node's job; clamping here would throw away
        // highlight detail before it ever gets there.
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt2020, ColorRange::Limited, 10), 3840, 2160, false,
        );
        // Code 1023 is above the limited-range white point (940).
        let out = c.apply(s10_planar(1023), s10_planar(512), s10_planar(512));
        assert!(out[0] > 1.0, "super-white must stay >1.0, got {out:?}");
    }

    #[test]
    fn zero_bit_depth_is_treated_as_eight() {
        // `bit_depth: 0` reaches here from sources whose codecpar had no pixel
        // format; it must not produce a division by zero or an all-black frame.
        let c = YuvConversion::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 0), HD_W, HD_H, false,
        );
        assert_eq!(c.sample_scale, 1.0);
        assert_rgb(c.apply(s8(235), s8(128), s8(128)), [1.0; 3], 1e-3, "depth 0 white");
    }

    // ── RGB → YUV (the encode direction) ─────────────────────────────────────

    /// The published BT.709 forward coefficients, from ITU-R BT.709-6 §3.2:
    ///
    ///     Y  = 0.2126 R + 0.7152 G + 0.0722 B
    ///     Cb = (B - Y) / 1.8556
    ///     Cr = (R - Y) / 1.5748
    ///
    /// Written out as literals rather than derived, so this disagrees with
    /// `RgbToYuv` if the derivation is wrong.
    #[test]
    fn forward_matrix_matches_the_published_bt709_coefficients() {
        let c = RgbToYuv::new(
            info(MatrixCoefficients::Bt709, ColorRange::Full, 8), HD_W, HD_H,
        );

        assert!((c.row_y[0] - 0.2126).abs() < 1e-4, "Kr: {}", c.row_y[0]);
        assert!((c.row_y[1] - 0.7152).abs() < 1e-4, "Kg: {}", c.row_y[1]);
        assert!((c.row_y[2] - 0.0722).abs() < 1e-4, "Kb: {}", c.row_y[2]);

        // Cb row: [-Kr/1.8556, -Kg/1.8556, (1-Kb)/1.8556]
        assert!((c.row_cb[0] + 0.114572).abs() < 1e-4, "R→Cb: {}", c.row_cb[0]);
        assert!((c.row_cb[1] + 0.385428).abs() < 1e-4, "G→Cb: {}", c.row_cb[1]);
        assert!((c.row_cb[2] - 0.5).abs()      < 1e-4, "B→Cb: {}", c.row_cb[2]);

        // Cr row: [(1-Kr)/1.5748, -Kg/1.5748, -Kb/1.5748]
        assert!((c.row_cr[0] - 0.5).abs()      < 1e-4, "R→Cr: {}", c.row_cr[0]);
        assert!((c.row_cr[1] + 0.454153).abs() < 1e-4, "G→Cr: {}", c.row_cr[1]);
        assert!((c.row_cr[2] + 0.045847).abs() < 1e-4, "B→Cr: {}", c.row_cr[2]);
    }

    /// Hardcoded 8-bit BT.709 limited-range codes for the primaries.
    ///
    /// These are NOT derived from anything in this module: they are the values a
    /// standalone C probe computed and fed to NVENC, whose output decoded back to
    /// exactly them (`nvchk/nv12_probe.c`, verified against the decoded NV12).
    /// So they cross-check this code against a separate implementation *and*
    /// against what the hardware agreed the pattern was.
    #[test]
    fn primaries_quantise_to_the_known_8bit_bt709_limited_codes() {
        let c = RgbToYuv::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H,
        );

        // (label, rgb, expected Y/U/V)
        let cases: [(&str, [f32; 3], [u8; 3]); 4] = [
            ("red",   [1.0, 0.0, 0.0], [63,  102, 240]),
            ("green", [0.0, 1.0, 0.0], [173, 42,  26]),
            ("blue",  [0.0, 0.0, 1.0], [32,  240, 118]),
            ("white", [1.0, 1.0, 1.0], [235, 128, 128]),
        ];

        for (label, rgb, want) in cases {
            let got = c.apply_u8(rgb[0], rgb[1], rgb[2]);
            assert_eq!(
                got, want,
                "{label}: RGB{rgb:?} encoded to Y/U/V {got:?}, expected {want:?}"
            );
        }
    }

    #[test]
    fn limited_range_maps_black_and_white_to_16_and_235() {
        let c = RgbToYuv::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H,
        );
        assert_eq!(c.apply_u8(0.0, 0.0, 0.0)[0], 16,  "limited black luma");
        assert_eq!(c.apply_u8(1.0, 1.0, 1.0)[0], 235, "limited white luma");
        // Neutral input must land exactly on the chroma midpoint.
        assert_eq!(c.apply_u8(0.5, 0.5, 0.5)[1], 128, "grey Cb");
        assert_eq!(c.apply_u8(0.5, 0.5, 0.5)[2], 128, "grey Cr");
    }

    #[test]
    fn full_range_maps_black_and_white_to_0_and_255() {
        let c = RgbToYuv::new(
            info(MatrixCoefficients::Bt709, ColorRange::Full, 8), HD_W, HD_H,
        );
        assert_eq!(c.apply_u8(0.0, 0.0, 0.0)[0], 0,   "full black luma");
        assert_eq!(c.apply_u8(1.0, 1.0, 1.0)[0], 255, "full white luma");
    }

    /// The assertion that makes the pair trustworthy: encoding then decoding must
    /// return the original RGB, for every matrix, both ranges, and depths 8/10/12.
    ///
    /// A transposed row, a swapped Cb/Cr column, or a range formula that is not a
    /// true inverse all survive the individual checks above for at least one
    /// case; none of them survive this.
    #[test]
    fn rgb_to_yuv_round_trips_through_yuv_to_rgb() {
        let colours: [[f32; 3]; 9] = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.75, 0.5],
            [0.9, 0.1, 0.35],
            [0.05, 0.6, 0.95],
        ];

        for matrix in [
            MatrixCoefficients::Bt601,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Bt2020,
        ] {
            for range in [ColorRange::Limited, ColorRange::Full] {
                for depth in [8u8, 10, 12] {
                    let ci = info(matrix, range, depth);
                    let fwd = RgbToYuv::new(ci, HD_W, HD_H);
                    // The inverse reads raw texture samples, so tell it the data
                    // is planar (LSB-aligned) and hand it `code / 65535` for
                    // depths above 8 — which is what `sample_scale` undoes.
                    let inv = YuvConversion::new(ci, HD_W, HD_H, false);
                    let max_code = ((1u32 << depth) - 1) as f32;

                    for rgb in colours {
                        let yuv = fwd.apply(rgb[0], rgb[1], rgb[2]);
                        // Normalised sample -> code -> the raw sample the shader
                        // would read out of a 16-bit planar texture.
                        let to_raw = |s: f32| {
                            let code = (s * max_code).round();
                            if depth <= 8 { code / 255.0 } else { code / 65535.0 }
                        };
                        let back = inv.apply(to_raw(yuv[0]), to_raw(yuv[1]), to_raw(yuv[2]));

                        // Tolerance DERIVED from the quantisation, not guessed.
                        //
                        // The only lossy step is rounding each of Y/Cb/Cr to an
                        // integer code, so the error budget is at most half a code
                        // step on each, propagated through the inverse matrix:
                        //
                        //   one luma   code  -> luma_scale/max_code   in RGB
                        //   one chroma code  -> chroma_scale/max_code, times that
                        //                       channel's Cb/Cr coefficient
                        //
                        // 2.0 bounds the largest coefficient in any of the three
                        // matrices (BT.2020's Cb→B is 1.8814), and both chroma
                        // components can err in the same direction, hence 2 * 0.5.
                        let luma_step   = inv.luma_scale   / max_code;
                        let chroma_step = inv.chroma_scale / max_code;
                        let tol = 0.5 * luma_step + chroma_step * 2.0 + 1e-5;
                        for i in 0..3 {
                            assert!(
                                (back[i] - rgb[i]).abs() <= tol,
                                "{matrix:?}/{range:?}/{depth}-bit: channel {i} \
                                 round-tripped {rgb:?} -> {yuv:?} -> {back:?} \
                                 (tolerance {tol:.2e}, derived from one code step \
                                 at this depth and range)"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn forward_matrices_are_measurably_different() {
        // If `RgbToYuv` ignored ColorInfo and always used BT.709, every test
        // above except this one would still pass.
        let rgb = [0.8f32, 0.3, 0.15];
        let enc = |m| {
            RgbToYuv::new(info(m, ColorRange::Limited, 8), HD_W, HD_H)
                .apply_u8(rgb[0], rgb[1], rgb[2])
        };
        let bt601  = enc(MatrixCoefficients::Bt601);
        let bt709  = enc(MatrixCoefficients::Bt709);
        let bt2020 = enc(MatrixCoefficients::Bt2020);

        assert!(
            bt601[0].abs_diff(bt709[0]) > 2,
            "BT.601 and BT.709 luma must differ: {bt601:?} vs {bt709:?}"
        );
        assert!(
            bt709[0].abs_diff(bt2020[0]) > 2,
            "BT.709 and BT.2020 luma must differ: {bt709:?} vs {bt2020:?}"
        );
    }

    /// The bug this whole change exists to fix, pinned as an assertion.
    ///
    /// Encoding BT.709 red with the BT.601 matrix — which is what the NVENC
    /// driver does to RGB input — and then decoding it as BT.709 (what the file's
    /// tags say) must visibly corrupt the colour. The reported symptom was red
    /// arriving as `[255, 25, 0]`: green leaking in.
    #[test]
    fn mismatched_encode_and_decode_matrices_corrupt_the_colour() {
        let ci_709 = info(MatrixCoefficients::Bt709, ColorRange::Limited, 8);
        let ci_601 = info(MatrixCoefficients::Bt601, ColorRange::Limited, 8);

        // Encoded with 601 (the driver), decoded as 709 (the tags).
        let encoded = RgbToYuv::new(ci_601, HD_W, HD_H).apply(1.0, 0.0, 0.0);
        let decoded = YuvConversion::new(ci_709, HD_W, HD_H, false)
            .apply(encoded[0], encoded[1], encoded[2]);

        let green_leak = decoded[1];
        assert!(
            green_leak > 0.02,
            "a 601-encode/709-decode mismatch must leak visible green into pure \
             red, got {decoded:?} — if this is ~0 the two matrices are no longer \
             distinguishable and this test has stopped testing anything"
        );

        // And the matching pair must NOT corrupt it, which is the whole point of
        // doing the conversion ourselves.
        let matched_enc = RgbToYuv::new(ci_709, HD_W, HD_H).apply(1.0, 0.0, 0.0);
        let matched_dec = YuvConversion::new(ci_709, HD_W, HD_H, false)
            .apply(matched_enc[0], matched_enc[1], matched_enc[2]);
        assert_rgb(matched_dec, [1.0, 0.0, 0.0], 6e-3, "709 encode + 709 decode");
    }

    #[test]
    fn out_of_range_input_is_clamped_not_wrapped() {
        // Tone mapping runs earlier in the graph, so anything reaching the encode
        // shader should already be in [0,1] — but an HDR value that slipped
        // through must saturate, not wrap around to black.
        //
        // The clamp is applied to RGB *before* the matrix (see `apply`), so
        // super-white becomes white: luma 235, chroma neutral.  Clamping Y/Cb/Cr
        // afterwards instead would let super-white reach 255, occupying codes
        // limited range reserves, and would shift hue for a colour that was out of
        // range in only one channel.
        let c = RgbToYuv::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 8), HD_W, HD_H,
        );
        assert_eq!(
            c.apply_u8(4.0, 4.0, 4.0), [235, 128, 128],
            "super-white must clamp to legal white, not exceed 235"
        );
        assert_eq!(
            c.apply_u8(-2.0, -2.0, -2.0), [16, 128, 128],
            "sub-black must clamp to legal black, not fall below 16"
        );

        // A colour out of range in ONE channel keeps its hue direction: red
        // saturates to exactly the same codes as legal pure red rather than
        // drifting, which is what a post-matrix clamp would do.
        assert_eq!(
            c.apply_u8(3.0, 0.0, 0.0), c.apply_u8(1.0, 0.0, 0.0),
            "clamping must happen in RGB, so over-bright red matches pure red"
        );

        // And every output stays inside the limited-range legal box.
        for rgb in [[4.0f32, -1.0, 0.5], [-3.0, 9.0, 2.0], [1.5, 1.5, -0.2]] {
            let [y, u, v] = c.apply_u8(rgb[0], rgb[1], rgb[2]);
            assert!((16..=235).contains(&y), "luma {y} out of 16..=235 for {rgb:?}");
            assert!((16..=240).contains(&u), "Cb {u} out of 16..=240 for {rgb:?}");
            assert!((16..=240).contains(&v), "Cr {v} out of 16..=240 for {rgb:?}");
        }
    }

    #[test]
    fn forward_conversion_handles_zero_bit_depth() {
        // Same defensive case as the inverse: no divide-by-zero, no black frame.
        let c = RgbToYuv::new(
            info(MatrixCoefficients::Bt709, ColorRange::Limited, 0), HD_W, HD_H,
        );
        assert_eq!(c.apply_u8(1.0, 1.0, 1.0)[0], 235, "depth 0 white luma");
    }
}
