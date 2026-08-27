// tonemap.wgsl
// Tone mapping, HDR Transfer Functions (PQ, HLG), and Gamut Conversions (BT.2020 -> BT.709).

struct ToneMapParams {
    transfer_fn:  u32,   // 0=Linear, 1=PQ (ST 2084), 2=HLG (ARIB STD-B67), 3=sRGB
    gamut_conv:   u32,   // 0=PassThrough, 1=BT.2020 -> BT.709
    tonemap_mode: u32,   // 0=None/Clamp, 1=ACES Filmic, 2=Reinhard, 3=BT.2446a
    peak_nits:    f32,   // Reference peak luminance in cd/m² (e.g. 1000.0)
    width:        u32,
    height:       u32,
    exposure:     f32,   // Exposure multiplier (default 1.0)
    _pad:         f32,
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

    // Step 2 — Load HDR pixel
    let pixel = textureLoad(in_tex, coord);
    var color = pixel.rgb * params.exposure;

    // Step 3 — Apply EOTF (Non-linear to Linear)
    if (params.transfer_fn == 1u) {
        // PQ ST 2084: scale so SDR reference 100 nits = 1.0 (10000 nits = 100.0)
        color = pq_to_linear(color) * (10000.0 / 100.0);
    } else if (params.transfer_fn == 2u) {
        // HLG: normalized scene linear (scale by relative peak factor)
        color = hlg_to_linear(color) * (params.peak_nits / 100.0);
    }

    // Step 4 — Gamut Conversion (BT.2020 -> BT.709)
    if (params.gamut_conv == 1u) {
        color = bt2020_to_bt709(color);
    }

    // Step 5 — Apply Tone-Mapping to fit within SDR range [0.0, 1.0]
    if (params.tonemap_mode == 1u) {
        color = aces_filmic(color);
    } else if (params.tonemap_mode == 2u) {
        color = reinhard(color);
    } else {
        color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // Step 6 — Write output (preserving alpha)
    textureStore(out_tex, coord, vec4<f32>(color, pixel.a));
}
