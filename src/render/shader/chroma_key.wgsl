// chroma_key.wgsl
// HSV chroma key with smoothstep feathering and spill suppression.

struct KeyParams {
    key_hue:        f32,
    tolerance:      f32,
    softness:       f32,
    min_saturation: f32,
    min_value:      f32,
    spill_suppress: f32,
    width:          u32,
    height:         u32,
}

@group(0) @binding(0) var in_tex:  texture_2d<f32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba16float, write>;

var<push_constant> params: KeyParams;

// Helper: convert linear RGB to HSV.
// Returns vec3(H in [0,360), S in [0,1], V in [0,1]).
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let cmax = max(rgb.r, max(rgb.g, rgb.b));
    let cmin = min(rgb.r, min(rgb.g, rgb.b));
    let delta = cmax - cmin;

    let v = cmax;
    let s = select(0.0, delta / cmax, cmax > 0.0001);

    var h: f32 = 0.0;
    if delta >= 0.0001 {
        if cmax == rgb.r {
            // Use fmod-style for robustness: force non-negative
            let raw = (rgb.g - rgb.b) / delta;
            h = 60.0 * ((raw % 6.0 + 6.0) % 6.0);
        } else if cmax == rgb.g {
            h = 60.0 * ((rgb.b - rgb.r) / delta + 2.0);
        } else {
            h = 60.0 * ((rgb.r - rgb.g) / delta + 4.0);
        }
        if h < 0.0 { h += 360.0; }
    }

    return vec3(h, s, v);
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Step 1 — Bounds check
    if id.x >= params.width || id.y >= params.height { return; }

    let coord = vec2<i32>(id.xy);

    // Step 2 — Load pixel
    var c = textureLoad(in_tex, coord, 0);

    // Step 3 — Convert to HSV
    let hsv = rgb_to_hsv(c.rgb);

    // Step 4 — Compute hue distance (circular)
    let raw_dist = abs(hsv.x - params.key_hue);
    let hue_dist = min(raw_dist, 360.0 - raw_dist);

    // Step 5 — Check saturation and value thresholds
    let in_range = hsv.y > params.min_saturation && hsv.z > params.min_value;
    if !in_range {
        textureStore(out_tex, coord, c);
        return;
    }

    // Step 6 — Compute smoothstep mask (Hermite cubic)
    let t = clamp((params.tolerance - hue_dist) / max(params.softness, 0.0001), 0.0, 1.0);
    let mask = t * t * (3.0 - 2.0 * t);

    // Step 7 — Apply mask to alpha
    c.a = c.a * (1.0 - mask);

    // Step 8 — Spill suppression
    if params.spill_suppress > 0.0 && mask > 0.0 {
        if params.key_hue > 60.0 && params.key_hue < 180.0 {
            // Green key — suppress green
            let avg = (c.r + c.b) / 2.0;
            c.g = mix(c.g, min(c.g, avg), params.spill_suppress * mask);
        } else {
            // Blue key — suppress blue
            let avg = (c.r + c.g) / 2.0;
            c.b = mix(c.b, min(c.b, avg), params.spill_suppress * mask);
        }
    }

    // Step 9 — Write output
    textureStore(out_tex, coord, c);
}
