// yuv_to_rgb.wgsl
// Converts YUV420p / NV12 / P010 / 10-12-bit planar GPU textures to RGBA16Float.
//
// P1.6 — this shader no longer decides anything about colour.  It used to carry
// three hardcoded matrices plus 8-BIT range constants, which silently mis-decoded
// 10-bit sources (both the code alignment and the range endpoints scale with bit
// depth).  Everything is now resolved on the CPU in `colour::yuv::YuvConversion`,
// unit-tested without a GPU, and handed over as push constants; the shader just
// applies them.
//
// Keep `struct YuvParams` byte-identical to `colour::yuv::YuvConversion`.  The
// three matrix rows are vec4, not vec3, because WGSL aligns vec3<f32> to 16
// bytes inside a struct — a bare 9-float array would be read at the wrong
// offsets.

struct YuvParams {
    // YCbCr -> R'G'B' matrix, one padded row each: [1, c_cb, c_cr, pad].
    row_r:         vec4<f32>,
    row_g:         vec4<f32>,
    row_b:         vec4<f32>,
    // Raw sample -> code/max_code.  1.0 for 8-bit and for MSB-aligned P010,
    // ~64 for LSB-aligned 10-bit planar data sitting in a 16-bit texture.
    sample_scale:  f32,
    // Limited-range luma: subtract 16<<(n-8), then scale by max/(219<<(n-8)).
    // Full range: 0.0 and 1.0.
    luma_offset:   f32,
    luma_scale:    f32,
    // Chroma: subtract the 128<<(n-8) midpoint, then scale by max/(224<<(n-8))
    // for limited range (1.0 for full).
    chroma_offset: f32,
    chroma_scale:  f32,
    width:         u32,
    height:        u32,
    _pad:          u32,
}

// Ordinary textures: read-only, exact integer-coordinate access via textureLoad.
// Y is R8Unorm or R16Unorm; UV is Rg8Unorm or Rg16Unorm — the normalisation
// difference between those is what `sample_scale` corrects.
@group(0) @binding(0) var y_plane:  texture_2d<f32>;
@group(0) @binding(1) var uv_plane: texture_2d<f32>;
@group(0) @binding(2) var out_tex:  texture_storage_2d<rgba16float, write>;

var<push_constant> params: YuvParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Step 1 — Bounds check
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);

    // Step 2 — Load the luma sample.
    let y_raw = textureLoad(y_plane, coord, 0).r;

    // Step 3 — Load the chroma pair.  The UV plane is half resolution in both
    // axes for 4:2:0, and clamping keeps the last odd row/column in range.
    let uv_dims  = textureDimensions(uv_plane);
    let uv_coord = min(coord / 2, vec2<i32>(uv_dims) - vec2<i32>(1, 1));
    let uv_raw   = textureLoad(uv_plane, uv_coord, 0).rg;

    // Step 4 — Normalise to code space, then to Y in [0,1] and Cb/Cr in [-0.5,0.5].
    // This is the step that used to hardcode 8-bit constants.
    let y  = (y_raw     * params.sample_scale - params.luma_offset)   * params.luma_scale;
    let cb = (uv_raw.r  * params.sample_scale - params.chroma_offset) * params.chroma_scale;
    let cr = (uv_raw.g  * params.sample_scale - params.chroma_offset) * params.chroma_scale;

    let ycc = vec3<f32>(y, cb, cr);

    // Step 5 — Apply the CPU-derived colour matrix.
    let r = dot(params.row_r.xyz, ycc);
    let g = dot(params.row_g.xyz, ycc);
    let b = dot(params.row_b.xyz, ycc);

    // Step 6 — Store R'G'B' (still in the source's transfer function).  NOT
    // clamped: HDR content legitimately exceeds 1.0 and the tone-map node
    // downstream needs those values intact.
    textureStore(out_tex, coord, vec4(r, g, b, 1.0));
}
