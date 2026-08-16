use rusttype::{Font, Scale};
use image::{Rgba, RgbaImage};
use imageproc::drawing::draw_text_mut;

const DEFAULT_FONT: &[u8] = include_bytes!("../../assets/Roboto-Regular.ttf");

/// Rasterize text into a 2D RGBA buffer, tightly cropped to the text size.
pub fn rasterize_text(text: &str, font_size: f32, color: [f32; 4]) -> (u32, u32, Vec<u8>) {
    let font = Font::try_from_bytes(DEFAULT_FONT).unwrap();
    let scale = Scale::uniform(font_size);

    // Measure text size
    let (width, height) = imageproc::drawing::text_size(scale, &font, text);
    
    // If text is empty or measurement fails (returns 0), return a 1x1 transparent image
    if width == 0 || height == 0 {
        return (1, 1, vec![0; 4]);
    }

    let width_u32 = width as u32;
    let height_u32 = height as u32;

    let mut image = RgbaImage::new(width_u32, height_u32);

    let rgba_color = Rgba([
        (color[0] * 255.0).clamp(0.0, 255.0) as u8,
        (color[1] * 255.0).clamp(0.0, 255.0) as u8,
        (color[2] * 255.0).clamp(0.0, 255.0) as u8,
        (color[3] * 255.0).clamp(0.0, 255.0) as u8,
    ]);

    draw_text_mut(&mut image, rgba_color, 0, 0, scale, &font, text);

    (width_u32, height_u32, image.into_raw())
}
