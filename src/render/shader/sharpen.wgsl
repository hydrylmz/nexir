// src/render/shader/sharpen.wgsl
// High-pass 3x3 unsharp convolution compute filter.

struct SharpenParams {
    amount: f32, // Sharpen strength: 0.0 (identity) to 2.0+
    width:  u32,
    height: u32,
    _pad:   f32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: SharpenParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);
    let center = textureLoad(in_tex, coord);

    if params.amount <= 0.001 {
        textureStore(out_tex, coord, center);
        return;
    }

    let max_x = i32(params.width) - 1;
    let max_y = i32(params.height) - 1;

    let left  = textureLoad(in_tex, vec2<i32>(max(coord.x - 1, 0), coord.y));
    let right = textureLoad(in_tex, vec2<i32>(min(coord.x + 1, max_x), coord.y));
    let up    = textureLoad(in_tex, vec2<i32>(coord.x, max(coord.y - 1, 0)));
    let down  = textureLoad(in_tex, vec2<i32>(coord.x, min(coord.y + 1, max_y)));

    let neighbors_avg = (left.rgb + right.rgb + up.rgb + down.rgb) * 0.25;
    let high_pass = center.rgb - neighbors_avg;
    let sharpened = max(vec3<f32>(0.0), center.rgb + high_pass * params.amount);

    textureStore(out_tex, coord, vec4<f32>(sharpened, center.a));
}
