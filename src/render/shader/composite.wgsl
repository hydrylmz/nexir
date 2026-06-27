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
    // Generate full-screen triangle
    var x = -1.0;
    var y = -1.0;
    var u = 0.0;
    var v = 1.0;

    if (vid == 1u) {
        x = 3.0;
        u = 2.0;
    } else if (vid == 2u) {
        y = 3.0;
        v = -1.0;
    }

    out.pos = vec4(x, y, 0.0, 1.0);
    out.uv = vec2(u, v);
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
