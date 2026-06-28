struct ClipInstance {
    transform: mat3x4<f32>,
    opacity:   f32,
    tex_index: u32,
    _pad:      vec2<f32>,
}

@group(0) @binding(0) var<storage, read> instances: array<ClipInstance>;
@group(0) @binding(1) var clip_tex: binding_array<texture_2d<f32>>;
@group(0) @binding(2) var clip_smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
    @location(1)       opacity: f32,
    @location(2)       tex_index: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vid:  u32,
    @builtin(instance_index) iid:  u32,
) -> VOut {
    var out: VOut;
    // Generate quad using 6 vertices
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    let uv = uvs[vid];

    // transform is a 3x3 matrix in the first 3 columns/rows of mat3x4.
    // We treat local_pos as vec3(u, v, 1.0).
    // result is vec4. x,y are ndc, z is 0.0, w is 1.0 (from our construction).
    let local_pos = vec3<f32>(uv.x, uv.y, 1.0);
    let ndc_pos = instances[iid].transform * local_pos;

    out.pos = vec4(ndc_pos.x, ndc_pos.y, 0.0, 1.0);
    out.uv = uv;
    out.opacity = instances[iid].opacity;
    out.tex_index = instances[iid].tex_index;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    var color = textureSample(clip_tex[in.tex_index], clip_smp, in.uv);
    color *= in.opacity;
    return color;
}
