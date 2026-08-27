// src/render/shader/vignette.wgsl
// Radial smoothstep falloff vignette compute filter.

struct VignetteParams {
    intensity: f32, // Darkening factor: 0.0 (none) to 1.0 (full black at edges)
    radius:    f32, // Normalized inner radius (default 0.75)
    softness:  f32, // Feather width (default 0.45)
    roundness: f32, // 1.0 = circular (aspect corrected), 0.0 = oval (fits frame)
    center_x:  f32, // default 0.5
    center_y:  f32, // default 0.5
    width:     u32,
    height:    u32,
}

@group(0) @binding(0) var in_tex:  texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: VignetteParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);
    let c = textureLoad(in_tex, coord);

    if params.intensity <= 0.001 {
        textureStore(out_tex, coord, c);
        return;
    }

    let uv = vec2<f32>(f32(id.x) / f32(params.width), f32(id.y) / f32(params.height));
    let center = vec2<f32>(params.center_x, params.center_y);
    let d = uv - center;

    // Aspect ratio adjustment for circular vignette
    let aspect = f32(params.width) / max(f32(params.height), 1.0);
    let circular_d = vec2<f32>(d.x * aspect, d.y);
    let dist = mix(length(d) * 2.0, length(circular_d) * 2.0 / aspect, params.roundness);

    let inner = max(params.radius - params.softness, 0.0);
    let outer = params.radius + params.softness;
    let factor = smoothstep(inner, outer, dist);
    let vignette_mult = 1.0 - factor * params.intensity;

    let out_rgb = c.rgb * clamp(vignette_mult, 0.0, 1.0);
    textureStore(out_tex, coord, vec4<f32>(out_rgb, c.a));
}
