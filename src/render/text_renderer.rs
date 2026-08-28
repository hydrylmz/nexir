use rusttype::{Font, Scale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

const DEFAULT_FONT: &[u8] = include_bytes!("../../assets/Roboto-Regular.ttf");

/// Measure the pixel dimensions of `text` at `font_size` **including** padding
/// that will be added when stroke or background is present.
pub fn measure_text(
    text: &str,
    font_size: f32,
    stroke_width: f32,
    has_stroke: bool,
    background_active: bool,
    bg_padding: f32,
) -> (u32, u32) {
    if text.is_empty() {
        return (1, 1);
    }
    let font = Font::try_from_bytes(DEFAULT_FONT).unwrap();
    let scale = Scale::uniform(font_size);
    let (text_w, text_h) = imageproc::drawing::text_size(scale, &font, text);
    if text_w <= 0 || text_h <= 0 {
        return (1, 1);
    }

    let stroke_pad = if has_stroke { stroke_width.ceil() as i32 } else { 0 };
    let bg_pad = if background_active {
        let min_bg_pad = (font_size * 0.10).ceil();
        bg_padding.max(min_bg_pad).ceil() as i32
    } else {
        0
    };
    let pad = stroke_pad + bg_pad;

    let width = (text_w + pad * 2).max(1) as u32;
    let height = (text_h + pad * 2).max(1) as u32;
    (width, height)
}

/// Full-featured rasterisation using morphological dilation for stroke.
/// Returns (width, height, rgba8_bytes).
pub fn rasterize_text(
    text: &str,
    font_size: f32,
    color: [f32; 4],
    stroke_color: Option<[f32; 4]>,
    stroke_width: f32,
    background_color: Option<[f32; 4]>,
    bg_padding: f32,
) -> (u32, u32, Vec<u8>) {
    if text.is_empty() {
        return (1, 1, vec![0; 4]);
    }

    let font = Font::try_from_bytes(DEFAULT_FONT).unwrap();
    let scale = Scale::uniform(font_size);

    let (text_w, text_h) = imageproc::drawing::text_size(scale, &font, text);
    if text_w <= 0 || text_h <= 0 {
        return (1, 1, vec![0; 4]);
    }

    let stroke_pad = if stroke_color.is_some() { stroke_width.ceil() as i32 } else { 0 };
    let bg_pad = if background_color.is_some() {
        let min_bg_pad = (font_size * 0.10).ceil();
        bg_padding.max(min_bg_pad).ceil() as i32
    } else {
        0
    };
    let pad = stroke_pad + bg_pad;

    let width = (text_w + pad * 2).max(1) as u32;
    let height = (text_h + pad * 2).max(1) as u32;
    let w_i32 = width as i32;
    let h_i32 = height as i32;

    // 1. Draw glyph alpha mask
    let mut glyph_img = RgbaImage::new(width, height);
    let y_offset = (font_size * 0.10).ceil() as i32;
    draw_text_mut(&mut glyph_img, Rgba([255, 255, 255, 255]), pad, pad - y_offset, scale, &font, text);

    let mut alpha_in = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            alpha_in[(y * width + x) as usize] = glyph_img.get_pixel(x, y)[3];
        }
    }

    // 2. Compute stroke mask via circular dilation
    let alpha_stroke = if stroke_color.is_some() && stroke_width > 0.0 {
        let mut stroke_mask = vec![0u8; (width * height) as usize];
        let r = stroke_width;
        let r_int = r.ceil() as i32;
        let r_sq = r * r;

        let mut offsets = Vec::new();
        for dy in -r_int..=r_int {
            for dx in -r_int..=r_int {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= r_sq {
                    offsets.push((dx, dy));
                }
            }
        }

        for y in 0..h_i32 {
            for x in 0..w_i32 {
                let mut max_a = 0u8;
                for &(dx, dy) in &offsets {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0 && nx < w_i32 && ny >= 0 && ny < h_i32 {
                        let a = alpha_in[(ny as u32 * width + nx as u32) as usize];
                        if a > max_a {
                            max_a = a;
                            if max_a == 255 {
                                break;
                            }
                        }
                    }
                }
                stroke_mask[(y as u32 * width + x as u32) as usize] = max_a;
            }
        }
        stroke_mask
    } else {
        Vec::new()
    };

    // 3. Composite output image (Background -> Stroke -> Main Text)
    let mut out_img = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let text_a = alpha_in[idx] as f32 / 255.0 * color[3];

            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;
            let mut a = 0.0f32;

            if let Some(bg) = background_color {
                r = bg[0];
                g = bg[1];
                b = bg[2];
                a = bg[3];
            }

            if let Some(sc) = stroke_color {
                let stroke_a = alpha_stroke[idx] as f32 / 255.0 * sc[3];
                if stroke_a > 0.0 {
                    let inv_s = 1.0 - stroke_a;
                    r = sc[0] * stroke_a + r * inv_s;
                    g = sc[1] * stroke_a + g * inv_s;
                    b = sc[2] * stroke_a + b * inv_s;
                    a = stroke_a + a * inv_s;
                }
            }

            if text_a > 0.0 {
                let inv_t = 1.0 - text_a;
                r = color[0] * text_a + r * inv_t;
                g = color[1] * text_a + g * inv_t;
                b = color[2] * text_a + b * inv_t;
                a = text_a + a * inv_t;
            }

            let px = Rgba([
                (r * 255.0).clamp(0.0, 255.0) as u8,
                (g * 255.0).clamp(0.0, 255.0) as u8,
                (b * 255.0).clamp(0.0, 255.0) as u8,
                (a * 255.0).clamp(0.0, 255.0) as u8,
            ]);
            out_img.put_pixel(x, y, px);
        }
    }

    (width, height, out_img.into_raw())
}
