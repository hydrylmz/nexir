// color_correction.wgsl
// Lift / gamma / gain + saturation in linear light.

struct ColorParams {
    lift:       vec4<f32>,    // [R, G, B, unused]
    gamma:      vec4<f32>,    // [R, G, B, unused]  value 1.0 = identity
    gain:       vec4<f32>,    // [R, G, B, unused]  value 1.0 = identity
    saturation: f32,
    width:      u32,
    height:     u32,
    _pad:       f32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: ColorParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Step 1 — Bounds check
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);

    // Step 2 — Load pixel
    var c = textureLoad(in_tex, coord);

    // Step 3 — Apply lift: c.rgb = c.rgb + lift * (1.0 - c.rgb)
    c = vec4(
        c.r + params.lift.r * (1.0 - c.r),
        c.g + params.lift.g * (1.0 - c.g),
        c.b + params.lift.b * (1.0 - c.b),
        c.a
    );

    // Step 4 — Apply gamma: pow(max(x, 0), 1/gamma) per channel
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

    // Step 5 — Apply gain
    c = vec4(c.r * params.gain.r, c.g * params.gain.g, c.b * params.gain.b, c.a);

    // Step 6 — Apply saturation (BT.709 luma weights)
    let L = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    let sat_rgb = mix(vec3(L), c.rgb, params.saturation);
    c = vec4(sat_rgb, c.a);

    // Step 7 — Write output (preserve alpha)
    textureStore(out_tex, coord, vec4(c.rgb, c.a));
}
