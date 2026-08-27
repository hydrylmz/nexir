// color_correction.wgsl
// Lift / gamma / gain + brightness / contrast / saturation / hue in linear light.

struct ColorParams {
    lift:       vec4<f32>,    // [R, G, B, unused]
    gamma:      vec4<f32>,    // [R, G, B, unused]  value 1.0 = identity
    gain:       vec4<f32>,    // [R, G, B, unused]  value 1.0 = identity
    saturation: f32,          // value 1.0 = identity
    brightness: f32,          // value 0.0 = identity
    contrast:   f32,          // value 1.0 = identity
    hue_shift:  f32,          // value 0.0 = identity (radians)
    width:      u32,
    height:     u32,
    _pad0:      f32,
    _pad1:      f32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: ColorParams;

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

    // Step 6 — Apply brightness & contrast: (rgb - 0.5) * contrast + 0.5 + brightness
    if abs(params.contrast - 1.0) > 0.001 || abs(params.brightness) > 0.001 {
        let adj_rgb = (c.rgb - vec3<f32>(0.5)) * max(params.contrast, 0.0) + vec3<f32>(0.5) + vec3<f32>(params.brightness);
        c = vec4(max(vec3<f32>(0.0), adj_rgb), c.a);
    }

    // Step 7 — Apply saturation (BT.709 luma weights)
    if abs(params.saturation - 1.0) > 0.001 {
        let L = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
        let sat_rgb = mix(vec3(L), c.rgb, max(params.saturation, 0.0));
        c = vec4(max(vec3<f32>(0.0), sat_rgb), c.a);
    }

    // Step 8 — Apply hue shift
    if abs(params.hue_shift) > 0.001 {
        c = vec4(rotate_hue(c.rgb, params.hue_shift), c.a);
    }

    // Step 9 — Write output (preserve alpha)
    textureStore(out_tex, coord, vec4(c.rgb, c.a));
}
