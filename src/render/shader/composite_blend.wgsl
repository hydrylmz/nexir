/// composite_blend.wgsl
/// Single-clip compositing shader that reads both the clip texture and
/// the accumulated background, applies the correct blend-mode formula,
/// and outputs the composited result.  No hardware blending is used —
/// the Porter-Duff "over" plus the blend formula is done entirely in WGSL.

struct ClipInstance {
    transform:    mat3x4<f32>,
    crop:         vec4<f32>,
    opacity:      f32,
    tex_index:    u32,
    blend_mode:   u32,
    crop_feather: f32,
}

// Blend-mode constants (must match BlendMode::as_u32())
const BLEND_NORMAL:      u32 = 0u;
const BLEND_ADD:         u32 = 1u;
const BLEND_MULTIPLY:    u32 = 2u;
const BLEND_SCREEN:      u32 = 3u;
const BLEND_OVERLAY:     u32 = 4u;
const BLEND_DARKEN:      u32 = 5u;
const BLEND_LIGHTEN:     u32 = 6u;
const BLEND_COLOR_DODGE: u32 = 7u;
const BLEND_COLOR_BURN:  u32 = 8u;
const BLEND_HARD_LIGHT:  u32 = 9u;
const BLEND_SOFT_LIGHT:  u32 = 10u;
const BLEND_DIFFERENCE:  u32 = 11u;
const BLEND_EXCLUSION:   u32 = 12u;

// Bindings: instance data and the two textures
@group(0) @binding(0) var<storage, read> instances: array<ClipInstance>;
@group(0) @binding(1) var clip_tex: texture_2d<f32>;
@group(0) @binding(2) var bg_tex:   texture_2d<f32>;
@group(0) @binding(3) var smp:      sampler;

struct VOut {
    @builtin(position) pos:         vec4<f32>,
    @location(0)       uv:          vec2<f32>,
    @location(1)       screen_uv:   vec2<f32>,
    @location(2)       opacity:     f32,
    @location(3)       blend_mode:  u32,
    @location(4)       crop:        vec4<f32>,
    @location(5)       crop_feather: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32,
) -> VOut {
    var out: VOut;
    var uvs = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    let uv = uvs[vid];
    let local_pos = vec3<f32>(uv.x, uv.y, 1.0);
    let ndc_pos   = instances[iid].transform * local_pos;

    out.pos          = vec4(ndc_pos.x, ndc_pos.y, 0.0, 1.0);
    out.uv           = uv;
    // screen_uv goes 0..1 in framebuffer space (y flipped from NDC)
    out.screen_uv    = vec2((ndc_pos.x + 1.0) * 0.5, (1.0 - ndc_pos.y) * 0.5);
    out.opacity      = instances[iid].opacity;
    out.blend_mode   = instances[iid].blend_mode;
    out.crop         = instances[iid].crop;
    out.crop_feather = instances[iid].crop_feather;
    return out;
}

// ── blend helpers (component-wise, linear light) ─────────────────────────────

fn blend_add(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return min(s + d, vec3(1.0));
}
fn blend_multiply(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return s * d;
}
fn blend_screen(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return s + d - s * d;
}
fn blend_overlay_chan(s: f32, d: f32) -> f32 {
    if d < 0.5 { return 2.0 * s * d; }
    return 1.0 - 2.0 * (1.0 - s) * (1.0 - d);
}
fn blend_overlay(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return vec3(blend_overlay_chan(s.x, d.x),
                blend_overlay_chan(s.y, d.y),
                blend_overlay_chan(s.z, d.z));
}
fn blend_darken(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return min(s, d);
}
fn blend_lighten(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return max(s, d);
}
fn blend_color_dodge_chan(s: f32, d: f32) -> f32 {
    if s >= 1.0 { return 1.0; }
    return min(1.0, d / (1.0 - s));
}
fn blend_color_dodge(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return vec3(blend_color_dodge_chan(s.x, d.x),
                blend_color_dodge_chan(s.y, d.y),
                blend_color_dodge_chan(s.z, d.z));
}
fn blend_color_burn_chan(s: f32, d: f32) -> f32 {
    if s <= 0.0 { return 0.0; }
    return 1.0 - min(1.0, (1.0 - d) / s);
}
fn blend_color_burn(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return vec3(blend_color_burn_chan(s.x, d.x),
                blend_color_burn_chan(s.y, d.y),
                blend_color_burn_chan(s.z, d.z));
}
fn blend_hard_light(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    // Same formula as Overlay but src/dst swapped
    return blend_overlay(d, s);
}
fn blend_soft_light_chan(s: f32, d: f32) -> f32 {
    if s <= 0.5 {
        return d - (1.0 - 2.0 * s) * d * (1.0 - d);
    }
    var g: f32;
    if d <= 0.25 {
        g = ((16.0 * d - 12.0) * d + 4.0) * d;
    } else {
        g = sqrt(d);
    }
    return d + (2.0 * s - 1.0) * (g - d);
}
fn blend_soft_light(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return vec3(blend_soft_light_chan(s.x, d.x),
                blend_soft_light_chan(s.y, d.y),
                blend_soft_light_chan(s.z, d.z));
}
fn blend_difference(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return abs(s - d);
}
fn blend_exclusion(s: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return s + d - 2.0 * s * d;
}

// ── Porter-Duff "over" with blend-mode applied to RGB ───────────────────────
//
//   src  = clip sample (straight alpha)
//   dst  = accumulated background (straight alpha)
//   mode = blend-mode id
//
//   1. Compute the "mixed" RGB using the blend formula (applied to pre-demultiplied colors)
//   2. Porter-Duff Over:  out.rgb = mixed * sa + dst.rgb * da * (1 - sa)  (norm by out.a)
//                          out.a  = sa + da * (1 - sa)
//
fn composite(src: vec4<f32>, dst: vec4<f32>, mode: u32) -> vec4<f32> {
    let sa = src.a;
    let da = dst.a;
    let out_a = sa + da * (1.0 - sa);
    if out_a < 0.0001 { return vec4(0.0); }

    let sc = src.rgb;          // straight-alpha clip color
    let dc = dst.rgb;          // straight-alpha bg color

    var mixed: vec3<f32>;
    switch mode {
        case BLEND_ADD:         { mixed = blend_add(sc, dc); }
        case BLEND_MULTIPLY:    { mixed = blend_multiply(sc, dc); }
        case BLEND_SCREEN:      { mixed = blend_screen(sc, dc); }
        case BLEND_OVERLAY:     { mixed = blend_overlay(sc, dc); }
        case BLEND_DARKEN:      { mixed = blend_darken(sc, dc); }
        case BLEND_LIGHTEN:     { mixed = blend_lighten(sc, dc); }
        case BLEND_COLOR_DODGE: { mixed = blend_color_dodge(sc, dc); }
        case BLEND_COLOR_BURN:  { mixed = blend_color_burn(sc, dc); }
        case BLEND_HARD_LIGHT:  { mixed = blend_hard_light(sc, dc); }
        case BLEND_SOFT_LIGHT:  { mixed = blend_soft_light(sc, dc); }
        case BLEND_DIFFERENCE:  { mixed = blend_difference(sc, dc); }
        case BLEND_EXCLUSION:   { mixed = blend_exclusion(sc, dc); }
        default:                { mixed = sc; }   // NORMAL — Porter-Duff handles it
    }

    // Porter-Duff over with blended RGB
    let out_rgb = (mixed * sa + dc * da * (1.0 - sa)) / out_a;
    return vec4(out_rgb, out_a);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // ── Crop / Feather ────────────────────────────────────────────────────────
    var crop_alpha = 1.0;
    let crop = in.crop;
    if in.crop_feather <= 0.0001 {
        if in.uv.x < crop.x || in.uv.x > crop.z || in.uv.y < crop.y || in.uv.y > crop.w {
            // Outside crop: just pass through the background unchanged
            let bg = textureSample(bg_tex, smp, in.screen_uv);
            return bg;
        }
    } else {
        let half_f = in.crop_feather * 0.5;
        let al = clamp((in.uv.x - crop.x) / half_f, 0.0, 1.0);
        let ar = clamp((crop.z   - in.uv.x) / half_f, 0.0, 1.0);
        let at = clamp((in.uv.y - crop.y) / half_f, 0.0, 1.0);
        let ab = clamp((crop.w   - in.uv.y) / half_f, 0.0, 1.0);
        crop_alpha = min(min(al, ar), min(at, ab));
    }

    // Sample clip and background
    var src = textureSample(clip_tex, smp, in.uv);
    src.a   *= in.opacity * crop_alpha;
    let dst  = textureSample(bg_tex,  smp, in.screen_uv);

    return composite(src, dst, in.blend_mode);
}
