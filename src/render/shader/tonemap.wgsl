// tonemap.wgsl
// Colour-space transform node: transfer functions (PQ, HLG, sRGB/BT.709), gamut
// conversion (BT.2020 <-> BT.709), and tone mapping.
//
// P1.7 — this shader used to be HDR->SDR only: it decoded an HDR transfer
// function to linear light, tone-mapped, and wrote LINEAR values out.  Two things
// were wrong with that once a real HDR export existed:
//
//   1. It could not encode TO an HDR curve, so there was no way to put PQ pixels
//      in front of a 10-bit encoder.
//   2. Writing linear light and then handing it to a chain that treats the
//      texture as display-encoded (the RGBA16F -> RGBA8 step in the CPU encoder,
//      and the preview blit) made every tone-mapped HDR->SDR result too dark.
//      Nothing else in the pipeline hit this, because a clip that skips this node
//      keeps the R'G'B' that `yuv_to_rgb.wgsl` produced, which IS display-encoded.
//
// So the pass is now symmetric: decode `transfer_fn` to linear, convert gamut,
// tone-map, then ENCODE with `output_transfer_fn`.  Linear light is normalised so
// 1.0 = `sdr_reference_nits` (100 nits by convention), which is what makes the PQ
// and HLG scalings below commute.

struct ToneMapParams {
    transfer_fn:        u32,   // input EOTF:  0=Linear, 1=PQ, 2=HLG, 3=sRGB/BT.709
    gamut_conv:         u32,   // 0=None, 1=BT.2020->BT.709, 2=BT.709->BT.2020
    tonemap_mode:       u32,   // 0=Clamp, 1=ACES Filmic, 2=Reinhard, 3=Passthrough
    peak_nits:          f32,   // reference peak luminance in cd/m² (e.g. 1000.0)
    width:              u32,
    height:             u32,
    exposure:           f32,   // exposure multiplier (default 1.0)
    output_transfer_fn: u32,   // output OETF: 0=Linear, 1=PQ, 2=HLG, 3=sRGB/BT.709
    sdr_reference_nits: f32,   // cd/m² that linear 1.0 represents (default 100.0)
    _pad0:              f32,
    _pad1:              f32,
    _pad2:              f32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: ToneMapParams;

// PQ (SMPTE ST 2084) constants
const PQ_M1: f32 = 0.1593017578125;       // 2610 / 16384
const PQ_M2: f32 = 78.84375;             // (2523 / 4096) * 128
const PQ_C1: f32 = 0.8359375;             // 3424 / 4096
const PQ_C2: f32 = 18.8515625;           // (2413 / 4096) * 32
const PQ_C3: f32 = 18.6875;              // (2392 / 4096) * 32

/// PQ code value [0,1] -> linear, where 1.0 out = 10000 nits.
fn pq_to_linear_scalar(n: f32) -> f32 {
    if (n <= 0.0) { return 0.0; }
    let n_m2 = pow(n, 1.0 / PQ_M2);
    let num = max(n_m2 - PQ_C1, 0.0);
    let den = max(PQ_C2 - PQ_C3 * n_m2, 1e-6);
    return pow(num / den, 1.0 / PQ_M1); // Linear in [0, 1] representing 0 to 10000 nits
}

fn pq_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        pq_to_linear_scalar(c.r),
        pq_to_linear_scalar(c.g),
        pq_to_linear_scalar(c.b)
    );
}

/// Inverse of `pq_to_linear_scalar` — the ST 2084 OETF.  Input is linear where
/// 1.0 = 10000 nits; output is a PQ code value in [0,1].
fn linear_to_pq_scalar(l: f32) -> f32 {
    let v = clamp(l, 0.0, 1.0);
    if (v <= 0.0) { return 0.0; }
    let v_m1 = pow(v, PQ_M1);
    let num = PQ_C1 + PQ_C2 * v_m1;
    let den = 1.0 + PQ_C3 * v_m1;
    return pow(num / den, PQ_M2);
}

fn linear_to_pq(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_pq_scalar(c.r),
        linear_to_pq_scalar(c.g),
        linear_to_pq_scalar(c.b)
    );
}

// HLG (ARIB STD-B67) constants
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;

fn hlg_to_linear_scalar(e: f32) -> f32 {
    let val = max(e, 0.0);
    if (val <= 0.5) {
        return (val * val) / 3.0;
    } else {
        return (exp((val - HLG_C) / HLG_A) + HLG_B) / 12.0;
    }
}

fn hlg_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        hlg_to_linear_scalar(c.r),
        hlg_to_linear_scalar(c.g),
        hlg_to_linear_scalar(c.b)
    );
}

/// Inverse of `hlg_to_linear_scalar` — the ARIB STD-B67 OETF.  Input is scene
/// linear normalised to [0,1]; output is an HLG code value in [0,1].
fn linear_to_hlg_scalar(l: f32) -> f32 {
    let v = clamp(l, 0.0, 1.0);
    if (v <= 1.0 / 12.0) {
        return sqrt(3.0 * v);
    } else {
        return HLG_A * log(12.0 * v - HLG_B) + HLG_C;
    }
}

fn linear_to_hlg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_hlg_scalar(c.r),
        linear_to_hlg_scalar(c.g),
        linear_to_hlg_scalar(c.b)
    );
}

// sRGB / BT.709-ish display transfer function.  The sRGB piecewise curve is used
// (rather than BT.709's own OETF) because every other 8-bit path in the engine —
// the still-image loader, the preview blit, the RGBA16F -> RGBA8 step in the CPU
// encoder — already treats display-encoded values as sRGB.
fn srgb_to_linear_scalar(e: f32) -> f32 {
    let v = clamp(e, 0.0, 1.0);
    if (v <= 0.04045) {
        return v / 12.92;
    } else {
        return pow((v + 0.055) / 1.055, 2.4);
    }
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        srgb_to_linear_scalar(c.r),
        srgb_to_linear_scalar(c.g),
        srgb_to_linear_scalar(c.b)
    );
}

fn linear_to_srgb_scalar(l: f32) -> f32 {
    let v = clamp(l, 0.0, 1.0);
    if (v <= 0.0031308) {
        return v * 12.92;
    } else {
        return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
    }
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_srgb_scalar(c.r),
        linear_to_srgb_scalar(c.g),
        linear_to_srgb_scalar(c.b)
    );
}

// BT.2020 -> BT.709 Chromatic Adaptation & Primaries Matrix
fn bt2020_to_bt709(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r;
    let g = rgb.g;
    let b = rgb.b;

    let out_r =  1.6604910 * r - 0.5876411 * g - 0.0728499 * b;
    let out_g = -0.1245505 * r + 1.1328999 * g - 0.0083494 * b;
    let out_b = -0.0181508 * r - 0.1005789 * g + 1.1187297 * b;

    return vec3<f32>(out_r, out_g, out_b);
}

// BT.709 -> BT.2020, the exact inverse of the matrix above.  Needed to place an
// SDR Rec.709 clip inside an HDR BT.2020 timeline: without it the clip's samples
// would be reinterpreted against wider primaries and every saturated colour would
// be pulled outward.
fn bt709_to_bt2020(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r;
    let g = rgb.g;
    let b = rgb.b;

    let out_r = 0.6274039 * r + 0.3292830 * g + 0.0433131 * b;
    let out_g = 0.0690973 * r + 0.9195404 * g + 0.0113623 * b;
    let out_b = 0.0163914 * r + 0.0880133 * g + 0.8955953 * b;

    return vec3<f32>(out_r, out_g, out_b);
}

// ACES Filmic Tone Mapping Curve (Narkowicz / ACES fit)
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let v = max(x, vec3<f32>(0.0));
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Reinhard Tone Mapping (Extended with peak luminance)
fn reinhard(c: vec3<f32>) -> vec3<f32> {
    let v = max(c, vec3<f32>(0.0));
    return v / (vec3<f32>(1.0) + v);
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Step 1 — Bounds check
    if (id.x >= params.width || id.y >= params.height) { return; }

    let coord = vec2<i32>(id.xy);

    // Step 2 — Load the pixel as the upstream node wrote it.
    let pixel = textureLoad(in_tex, coord);
    var color = pixel.rgb * params.exposure;

    // Step 3 — Decode the input transfer function to linear light, normalised so
    // that 1.0 = `sdr_reference_nits`.
    let ref_nits = max(params.sdr_reference_nits, 1.0);
    if (params.transfer_fn == 1u) {
        // PQ ST 2084: the curve's 1.0 is 10000 nits.
        color = pq_to_linear(color) * (10000.0 / ref_nits);
    } else if (params.transfer_fn == 2u) {
        // HLG: normalised scene linear, scaled by the signal's own peak.
        color = hlg_to_linear(color) * (params.peak_nits / ref_nits);
    } else if (params.transfer_fn == 3u) {
        // SDR display-encoded: diffuse white is the reference, so 1.0 stays 1.0.
        color = srgb_to_linear(color);
    }

    // Step 4 — Gamut conversion.
    if (params.gamut_conv == 1u) {
        color = bt2020_to_bt709(color);
    } else if (params.gamut_conv == 2u) {
        color = bt709_to_bt2020(color);
    }

    // Step 5 — Tone mapping.  Modes 0..2 compress into [0,1] for an SDR target;
    // mode 3 leaves the linear values untouched, which is what an HDR target
    // needs (clamping here would flatten every highlight above diffuse white).
    if (params.tonemap_mode == 1u) {
        color = aces_filmic(color);
    } else if (params.tonemap_mode == 2u) {
        color = reinhard(color);
    } else if (params.tonemap_mode == 0u) {
        color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    } else {
        color = max(color, vec3<f32>(0.0));
    }

    // Step 6 — Encode the output transfer function.
    if (params.output_transfer_fn == 1u) {
        // PQ expects linear where 1.0 = 10000 nits.
        color = linear_to_pq(color * (ref_nits / 10000.0));
    } else if (params.output_transfer_fn == 2u) {
        color = linear_to_hlg(color * (ref_nits / max(params.peak_nits, 1.0)));
    } else if (params.output_transfer_fn == 3u) {
        color = linear_to_srgb(color);
    }

    // Step 7 — Write output (preserving alpha)
    textureStore(out_tex, coord, vec4<f32>(color, pixel.a));
}
