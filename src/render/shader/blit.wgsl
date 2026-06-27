// blit.wgsl — RTT → swapchain with ACES tone-map + sRGB gamma

@group(0) @binding(0) var rtt:     texture_2d<f32>;
@group(0) @binding(1) var rtt_smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
    var out: VOut;
    var x = -1.0;
    var y = -1.0;
    var u = 0.0;
    var v = 1.0;

    if vid == 1u {
        x = 3.0;
        u = 2.0;
    } else if vid == 2u {
        y = 3.0;
        v = -1.0;
    }

    out.pos = vec4(x, y, 0.0, 1.0);
    out.uv = vec2(u, v);
    return out;
}

// Convert sRGB encoded value to linear light
fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        return s / 12.92;
    } else {
        return pow((s + 0.055) / 1.055, 2.4);
    }
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Step 1 — Sample RTT (YUV conversion already outputs sRGB/Rec709 encoded values)
    var c = textureSample(rtt, rtt_smp, in.uv);

    // Step 2 — Convert sRGB to Linear because our render target is Rgba8UnormSrgb
    // (wgpu will automatically apply the sRGB gamma curve on write)
    c.r = srgb_to_linear(c.r);
    c.g = srgb_to_linear(c.g);
    c.b = srgb_to_linear(c.b);

    return vec4(c.rgb, 1.0);
}
