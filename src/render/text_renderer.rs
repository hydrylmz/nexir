use rusttype::{Font, Scale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

const DEFAULT_FONT: &[u8] = include_bytes!("../../assets/Roboto-Regular.ttf");

/// Measure the pixel dimensions of `text` rendered at `font_size` without
/// allocating a pixel buffer. Used by the scheduler hot path.
pub fn measure_text(text: &str, font_size: f32) -> (u32, u32) {
    if text.is_empty() {
        return (1, 1);
    }
    let font = Font::try_from_bytes(DEFAULT_FONT).unwrap();
    let scale = Scale::uniform(font_size);
    let (w, h) = imageproc::drawing::text_size(scale, &font, text);
    (w.max(1) as u32, h.max(1) as u32)
}

/// Rasterize `text` into a tight-cropped RGBA8 buffer.
/// Returns (width, height, rgba_bytes).
/// Background pixels are fully transparent (alpha = 0).
pub fn rasterize_text(text: &str, font_size: f32, color: [f32; 4]) -> (u32, u32, Vec<u8>) {
    if text.is_empty() {
        return (1, 1, vec![0; 4]);
    }

    let font = Font::try_from_bytes(DEFAULT_FONT).unwrap();
    let scale = Scale::uniform(font_size);

    let (width, height) = imageproc::drawing::text_size(scale, &font, text);
    if width <= 0 || height <= 0 {
        return (1, 1, vec![0; 4]);
    }

    let width_u32 = width as u32;
    let height_u32 = height as u32;

    // RgbaImage is zero-initialised → background pixels are (0,0,0,0) transparent.
    let mut image = RgbaImage::new(width_u32, height_u32);

    let rgba_color = Rgba([
        (color[0] * 255.0).clamp(0.0, 255.0) as u8,
        (color[1] * 255.0).clamp(0.0, 255.0) as u8,
        (color[2] * 255.0).clamp(0.0, 255.0) as u8,
        // Use full opacity for the text pixels themselves; rusttype's
        // sub-pixel rendering writes partial-alpha values for antialiasing.
        255u8,
    ]);

    draw_text_mut(&mut image, rgba_color, 0, 0, scale, &font, text);

    (width_u32, height_u32, image.into_raw())
}
