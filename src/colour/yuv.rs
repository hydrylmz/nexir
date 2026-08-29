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
}
