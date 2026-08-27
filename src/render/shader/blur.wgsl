// src/render/shader/blur.wgsl
// Separable 1D Gaussian blur compute kernel.

struct BlurParams {
    direction: vec2<f32>, // [1.0, 0.0] for horizontal, [0.0, 1.0] for vertical
    radius:    f32,       // e.g. 1.0 .. 64.0
    sigma:     f32,       // e.g. radius / 2.0
    width:     u32,
    height:    u32,
    _pad0:     f32,
    _pad1:     f32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: BlurParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);
    let r = i32(clamp(params.radius, 1.0, 64.0));
    let sigma = max(params.sigma, 0.5);
    let two_sigma_sq = 2.0 * sigma * sigma;

    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    let dir = vec2<i32>(params.direction);

    for (var i = -r; i <= r; i = i + 1) {
        let sample_coord = clamp(
            coord + dir * i,
            vec2<i32>(0, 0),
            vec2<i32>(i32(params.width) - 1, i32(params.height) - 1)
        );
        let dist = f32(i);
        let w = exp(-(dist * dist) / two_sigma_sq);
        let c = textureLoad(in_tex, sample_coord);
        color_sum = color_sum + c * w;
        weight_sum = weight_sum + w;
    }

    let final_color = color_sum / max(weight_sum, 1e-5);
    textureStore(out_tex, coord, final_color);
}
