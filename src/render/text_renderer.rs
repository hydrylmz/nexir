use rusttype::{Font, Scale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

const DEFAULT_FONT: &[u8] = include_bytes!("../../assets/Roboto-Regular.ttf");

/// Background padding relative to font_size (adds breathing room around text).
const BG_PADDING_RATIO: f32 = 0.15;

/// Number of directions used for stroke simulation.
const STROKE_ANGLES: usize = 16;

/// Measure the pixel dimensions of `text` at `font_size` **including** padding
/// that will be added when stroke or background is present. This ensures the
/// scheduler gives us the correct clip_width/clip_height.
pub fn measure_text(text: &str, font_size: f32) -> (u32, u32) {
    if text.is_empty() {
        return (1, 1);
    }
    let font = Font::try_from_bytes(DEFAULT_FONT).unwrap();
    let scale = Scale::uniform(font_size);
    let (w, h) = imageproc::drawing::text_size(scale, &font, text);
    (w.max(1) as u32, h.max(1) as u32)
}

/// Full-featured rasterisation: optional background fill, optional stroke outline, text.
/// Returns (width, height, rgba8_bytes).
pub fn rasterize_text(
    text: &str,
    font_size: f32,
    color: [f32; 4],
    stroke_color: Option<[f32; 4]>,
    stroke_width: f32,
    background_color: Option<[f32; 4]>,
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

    // Compute total padding: stroke bleed + optional background breathing room.
    let stroke_pad = if stroke_color.is_some() { stroke_width.ceil() as i32 } else { 0 };
    let bg_pad = if background_color.is_some() {
        (font_size * BG_PADDING_RATIO).ceil() as i32
    } else {
        0
    };
    let pad = (stroke_pad + bg_pad).max(0);

    let width  = (text_w + pad * 2).max(1) as u32;
    let height = (text_h + pad * 2).max(1) as u32;

    // Start fully transparent.
    let mut image = RgbaImage::new(width, height);

    // 1. Background fill.
    if let Some(bg) = background_color {
        let bg_pixel = to_rgba8(bg);
        for pixel in image.pixels_mut() {
            *pixel = bg_pixel;
        }
    }

    // 2. Stroke — draw text offset in STROKE_ANGLES directions.
    if let Some(sc) = stroke_color {
        let stroke_pixel = to_rgba8(sc);
        let sw = stroke_width.max(1.0);
        for i in 0..STROKE_ANGLES {
            let angle = (i as f32) * std::f32::consts::TAU / STROKE_ANGLES as f32;
            let ox = (angle.cos() * sw).round() as i32;
            let oy = (angle.sin() * sw).round() as i32;
            draw_text_mut(
                &mut image,
                stroke_pixel,
                pad + ox,
                pad + oy,
                scale,
                &font,
                text,
            );
        }
    }

    // 3. Main text on top.
    let text_pixel = Rgba([
        (color[0] * 255.0).clamp(0.0, 255.0) as u8,
        (color[1] * 255.0).clamp(0.0, 255.0) as u8,
        (color[2] * 255.0).clamp(0.0, 255.0) as u8,
        255u8,
    ]);
    draw_text_mut(&mut image, text_pixel, pad, pad, scale, &font, text);

    (width, height, image.into_raw())
}

fn to_rgba8(c: [f32; 4]) -> Rgba<u8> {
    Rgba([
        (c[0] * 255.0).clamp(0.0, 255.0) as u8,
        (c[1] * 255.0).clamp(0.0, 255.0) as u8,
        (c[2] * 255.0).clamp(0.0, 255.0) as u8,
        (c[3] * 255.0).clamp(0.0, 255.0) as u8,
    ])
}
