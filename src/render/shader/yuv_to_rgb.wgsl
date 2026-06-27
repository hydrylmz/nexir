// yuv_to_rgb.wgsl
// Converts YUV420p/NV12 GPU textures to RGBA16Float.
// Handles BT.601, BT.709, BT.2020; full range and limited range.

struct YuvParams {
    color_space:   u32,   // 0=BT.601, 1=BT.709, 2=BT.2020
    limited_range: u32,   // 0=full, 1=limited
    width:         u32,
    height:        u32,
}

// Ordinary textures: read-only access, exact integer-coordinate access via textureLoad(tex, coord, mip_level).
@group(0) @binding(0) var y_plane:  texture_2d<f32>;
@group(0) @binding(1) var uv_plane: texture_2d<f32>;
@group(0) @binding(2) var out_tex:  texture_storage_2d<rgba16float, write>;

var<push_constant> params: YuvParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Step 1 — Bounds check
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);

    // Step 2 — Load Y sample (r8unorm auto-normalises [0,255] -> [0.0, 1.0])
    let y_raw = textureLoad(y_plane, coord, 0).r;

    // Step 3 — Load UV sample (UV plane is half resolution for YUV420)
    let uv_coord = coord / 2;
    let uv_raw = textureLoad(uv_plane, uv_coord, 0).rg;

    // Step 4 — Denormalise Y, Cb, Cr based on range
    var Y:  f32;
    var Cb: f32;
    var Cr: f32;

    if params.limited_range == 1u {
        // Limited range (broadcast, hardware decoders)
        Y  = (y_raw  - 16.0/255.0) / (219.0/255.0);
        Cb = (uv_raw.r - 128.0/255.0) / (224.0/255.0);
        Cr = (uv_raw.g - 128.0/255.0) / (224.0/255.0);
        Y  = clamp(Y, 0.0, 1.0);
    } else {
        // Full range (JPEG, most software decoders)
        Y  = y_raw;
        Cb = uv_raw.r - 0.5;
        Cr = uv_raw.g - 0.5;
    }

    // Step 5 — Apply colour matrix
    var R: f32;
    var G: f32;
    var B: f32;

    if params.color_space == 0u {
        // BT.601 (SD, legacy)
        R = Y + 1.402000 * Cr;
        G = Y - 0.344136 * Cb - 0.714136 * Cr;
        B = Y + 1.772000 * Cb;
    } else if params.color_space == 2u {
        // BT.2020 (4K HDR)
        R = Y + 1.474600 * Cr;
        G = Y - 0.164553 * Cb - 0.571353 * Cr;
        B = Y + 1.881400 * Cb;
    } else {
        // BT.709 (default, HD video)
        R = Y + 1.574800 * Cr;
        G = Y - 0.187324 * Cb - 0.468124 * Cr;
        B = Y + 1.855600 * Cb;
    }

    // Step 6 — Store RGBA16Float output (no clamping — HDR may exceed 1.0)
    textureStore(out_tex, coord, vec4(R, G, B, 1.0));
}
