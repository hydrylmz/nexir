// lut.wgsl — 3D LUT application with GPU trilinear interpolation

struct LutParams {
    strength: f32,
    width:    u32,
    height:   u32,
    _pad:     f32,
}

@group(0) @binding(0) var in_tex:   texture_storage_2d<rgba16float, read>;
@group(0) @binding(1) var out_tex:  texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var lut_tex:  texture_3d<f32>;
@group(0) @binding(3) var lut_smp:  sampler;

var<push_constant> params: LutParams;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);
    let c = textureLoad(in_tex, coord);

    // Sample 3D LUT using hardware trilinear interpolation
    // Input RGB is the UVW coordinate into the LUT
    let lut_uvw = clamp(c.rgb, vec3(0.0), vec3(1.0));
    let lut_out = textureSampleLevel(lut_tex, lut_smp, lut_uvw, 0.0);

    // Blend: lerp between original and LUT output by strength
    let blended = mix(c.rgb, lut_out.rgb, params.strength);

    textureStore(out_tex, coord, vec4(blended, c.a));
}
