// fused_grade.wgsl
//
// Colour correction + 3D LUT + chroma key in ONE compute pass — P2.3.
//
// WHAT THIS SAVES, AND WHAT IT DOES NOT. The maths here is character-for-character
// the maths in `color_correction.wgsl`, `lut.wgsl` and `chroma_key.wgsl`, in that
// order. The saving is BANDWIDTH: the three-pass chain reads and writes a full
// canvas-sized `rgba16float` texture three times over (6 × 66.4 MB at 4K), and this
// reads once and writes once. Nothing about the arithmetic changes, so a workload
// that is not bandwidth-bound will not move — which is why the three passes were
// measured for bandwidth-boundedness first (AGENTS.md gotcha 27: the same shaders
// cost 5.881 ms on compressible content and 11.366 on noisy, and their implied
// per-pass GB/s crosses this card's 224 GB/s bus rate).
//
// THE ONE BEHAVIOURAL DIFFERENCE, stated because it is real. The chain stores each
// intermediate as `rgba16float` and therefore ROUNDS TO f16 twice on the way through;
// this keeps f32 in registers. So the fused result is not bit-identical to the
// chain's — it is the same computation carried at higher precision, and
// `tests::fused_grade` asserts agreement within f16 quantisation rather than
// equality. A tolerance tighter than f16 would fail for the right reason and the
// wrong cause.
//
// PUSH CONSTANTS. The three nodes carry 80 + 16 + 32 = 128 bytes between them, which
// is exactly the device limit this crate requests, and three copies of `width`/
// `height` among them. Packing to 112 bytes here is not an economy — it is what
// leaves headroom for the limit to stay 128, and it means the pair of dimensions is
// declared ONCE so the three stages cannot disagree about the frame they are
// processing. Keep `struct FusedGradeParams` byte-identical to
// `nodes::fused_grade::FusedGradeParams`.

struct FusedGradeParams {
    // ── colour correction ─────────────────────────────────────────────────────
    lift:   vec4<f32>,   // [R, G, B, unused]
    gamma:  vec4<f32>,   // [R, G, B, unused]   1.0 = identity
    gain:   vec4<f32>,   // [R, G, B, unused]   1.0 = identity
    // [saturation, brightness, contrast, hue_shift]; 1, 0, 1, 0 = identity
    grade:  vec4<f32>,
    // ── chroma key ────────────────────────────────────────────────────────────
    // [key_hue (deg), tolerance (deg), softness (deg), min_saturation]
    key_a:  vec4<f32>,
    // [min_value, spill_suppress, lut_strength, unused]
    key_b:  vec4<f32>,
    // ── geometry, once for all three stages ───────────────────────────────────
    width:  u32,
    height: u32,
    _pad0:  u32,
    _pad1:  u32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var lut_tex: texture_3d<f32>;
@group(0) @binding(3) var lut_smp: sampler;

var<push_constant> params: FusedGradeParams;

// Verbatim from `color_correction.wgsl`.
fn rotate_hue(rgb: vec3<f32>, theta: f32) -> vec3<f32> {
    if abs(theta) < 0.001 { return rgb; }
    let cos_a = cos(theta);
    let sin_a = sin(theta);
    let k0 = 0.2126;
    let k1 = 0.7152;
    let k2 = 0.0722;

    let r = rgb.r * (cos_a + (1.0 - cos_a) * k0) +
            rgb.g * (k0 * (1.0 - cos_a) - k2 * sin_a) +
            rgb.b * (k0 * (1.0 - cos_a) + k1 * sin_a);

    let g = rgb.r * (k1 * (1.0 - cos_a) + k2 * sin_a) +
            rgb.g * (cos_a + (1.0 - cos_a) * k1) +
            rgb.b * (k1 * (1.0 - cos_a) - k0 * sin_a);

    let b = rgb.r * (k2 * (1.0 - cos_a) - k1 * sin_a) +
            rgb.g * (k2 * (1.0 - cos_a) + k0 * sin_a) +
            rgb.b * (cos_a + (1.0 - cos_a) * k2);

    return max(vec3<f32>(0.0), vec3<f32>(r, g, b));
}

// Verbatim from `chroma_key.wgsl`.
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let cmax = max(rgb.r, max(rgb.g, rgb.b));
    let cmin = min(rgb.r, min(rgb.g, rgb.b));
    let delta = cmax - cmin;

    let v = cmax;
    let s = select(0.0, delta / cmax, cmax > 0.0001);

    var h: f32 = 0.0;
    if delta >= 0.0001 {
        if cmax == rgb.r {
            let raw = (rgb.g - rgb.b) / delta;
            h = 60.0 * ((raw % 6.0 + 6.0) % 6.0);
        } else if cmax == rgb.g {
            h = 60.0 * ((rgb.b - rgb.r) / delta + 2.0);
        } else {
            h = 60.0 * ((rgb.r - rgb.g) / delta + 4.0);
        }
        if h < 0.0 { h += 360.0; }
    }

    return vec3(h, s, v);
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);

    // ── ONE read. The whole point of this shader. ──────────────────────────────
    var c = textureLoad(in_tex, coord);

    // ══ STAGE 1 — colour correction (color_correction.wgsl, steps 3-8) ════════

    let saturation = params.grade.x;
    let brightness = params.grade.y;
    let contrast   = params.grade.z;
    let hue_shift  = params.grade.w;

    // Lift
    c = vec4(
        c.r + params.lift.r * (1.0 - c.r),
        c.g + params.lift.g * (1.0 - c.g),
        c.b + params.lift.b * (1.0 - c.b),
        c.a
    );

    // Gamma
    var gr = c.r;
    var gg = c.g;
    var gb = c.b;
    if params.gamma.r > 0.0 {
        gr = pow(max(c.r, 0.0), 1.0 / params.gamma.r);
    }
    if params.gamma.g > 0.0 {
        gg = pow(max(c.g, 0.0), 1.0 / params.gamma.g);
    }
    if params.gamma.b > 0.0 {
        gb = pow(max(c.b, 0.0), 1.0 / params.gamma.b);
    }
    c = vec4(gr, gg, gb, c.a);

    // Gain
    c = vec4(c.r * params.gain.r, c.g * params.gain.g, c.b * params.gain.b, c.a);

    // Brightness & contrast
    if abs(contrast - 1.0) > 0.001 || abs(brightness) > 0.001 {
        let adj_rgb = (c.rgb - vec3<f32>(0.5)) * max(contrast, 0.0)
                    + vec3<f32>(0.5) + vec3<f32>(brightness);
        c = vec4(max(vec3<f32>(0.0), adj_rgb), c.a);
    }

    // Saturation (BT.709 luma weights)
    if abs(saturation - 1.0) > 0.001 {
        let L = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
        let sat_rgb = mix(vec3(L), c.rgb, max(saturation, 0.0));
        c = vec4(max(vec3<f32>(0.0), sat_rgb), c.a);
    }

    // Hue shift
    if abs(hue_shift) > 0.001 {
        c = vec4(rotate_hue(c.rgb, hue_shift), c.a);
    }

    // ══ STAGE 2 — 3D LUT (lut.wgsl) ═══════════════════════════════════════════
    //
    // The sample is unconditional and sits ahead of the chroma key's early exit,
    // so it is never in non-uniform control flow.

    let lut_strength = params.key_b.z;
    let lut_uvw = clamp(c.rgb, vec3(0.0), vec3(1.0));
    let lut_out = textureSampleLevel(lut_tex, lut_smp, lut_uvw, 0.0);
    c = vec4(mix(c.rgb, lut_out.rgb, lut_strength), c.a);

    // ══ STAGE 3 — chroma key (chroma_key.wgsl) ════════════════════════════════

    let key_hue        = params.key_a.x;
    let tolerance      = params.key_a.y;
    let softness       = params.key_a.z;
    let min_saturation = params.key_a.w;
    let min_value      = params.key_b.x;
    let spill_suppress = params.key_b.y;

    let hsv = rgb_to_hsv(c.rgb);

    let raw_dist = abs(hsv.x - key_hue);
    let hue_dist = min(raw_dist, 360.0 - raw_dist);

    let in_range = hsv.y > min_saturation && hsv.z > min_value;
    if !in_range {
        // The chain's early return, kept as a store-and-exit for the same reason:
        // a pixel outside the key's saturation/value gate passes through untouched.
        textureStore(out_tex, coord, c);
        return;
    }

    let t = clamp((tolerance - hue_dist) / max(softness, 0.0001), 0.0, 1.0);
    let mask = t * t * (3.0 - 2.0 * t);

    c.a = c.a * (1.0 - mask);

    if spill_suppress > 0.0 && mask > 0.0 {
        if key_hue > 60.0 && key_hue < 180.0 {
            // Green key — suppress green
            let avg = (c.r + c.b) / 2.0;
            c.g = mix(c.g, min(c.g, avg), spill_suppress * mask);
        } else {
            // Blue key — suppress blue
            let avg = (c.r + c.g) / 2.0;
            c.b = mix(c.b, min(c.b, avg), spill_suppress * mask);
        }
    }

    // ── ONE write. ────────────────────────────────────────────────────────────
    textureStore(out_tex, coord, c);
}
