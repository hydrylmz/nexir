use std::collections::HashMap;

use half::f16;

use crate::render::context::RenderContext;
use crate::render::device::GpuDevice;
use crate::render::frame_state::FrameState;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceDescriptor, ResourceId, ResolutionSource};
use crate::render::text_renderer::rasterize_text;

fn round_up_256(n: u32) -> u32 {
    (n + 255) & !255
}

#[derive(Clone)]
pub struct CachedText {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub staging: std::sync::Arc<wgpu::Buffer>,
}

/// Cache key covers every parameter that affects the rasterized output.
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct TextKey {
    pub text: String,
    pub font_size_bits: u32,
    pub color_bits: [u32; 4],
    // Stroke
    pub has_stroke: bool,
    pub stroke_color_bits: [u32; 4],
    pub stroke_width_bits: u32,
    // Background
    pub has_bg: bool,
    pub bg_color_bits: [u32; 4],
}

impl TextKey {
    pub fn new(
        text: &str,
        font_size: f32,
        color: [f32; 4],
        stroke_color: Option<[f32; 4]>,
        stroke_width: f32,
        background_color: Option<[f32; 4]>,
    ) -> Self {
        Self {
            text: text.to_string(),
            font_size_bits: font_size.to_bits(),
            color_bits: [
                color[0].to_bits(), color[1].to_bits(),
                color[2].to_bits(), color[3].to_bits(),
            ],
            has_stroke: stroke_color.is_some(),
            stroke_color_bits: stroke_color.unwrap_or([0.0; 4]).map(f32::to_bits),
            stroke_width_bits: stroke_width.to_bits(),
            has_bg: background_color.is_some(),
            bg_color_bits: background_color.unwrap_or([0.0; 4]).map(f32::to_bits),
        }
    }
}

/// Loads and caches rasterized text on the GPU.
#[derive(Default)]
pub struct TextCache {
    cache: HashMap<TextKey, CachedText>,
}

impl TextCache {
    pub fn get_or_create(
        &mut self,
        device: &GpuDevice,
        text: &str,
        font_size: f32,
        color: [f32; 4],
        stroke_color: Option<[f32; 4]>,
        stroke_width: f32,
        background_color: Option<[f32; 4]>,
    ) -> CachedText {
        let key = TextKey::new(text, font_size, color, stroke_color, stroke_width, background_color);
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let (width, height, rgba_data) =
            rasterize_text(text, font_size, color, stroke_color, stroke_width, background_color);

        // Pack pixels as Rgba16Float with 256-byte row alignment.
        let bytes_per_pixel = 8u32; // 4 channels × 2 bytes (f16)
        let row_bytes = width * bytes_per_pixel;
        let bytes_per_row = round_up_256(row_bytes);
        let mut upload = vec![0u8; (bytes_per_row * height) as usize];

        for y in 0..height {
            let src_row = (y * width * 4) as usize;
            let dst_row = (y * bytes_per_row) as usize;
            for x in 0..width {
                let src = src_row + (x * 4) as usize;
                let dst = dst_row + (x * bytes_per_pixel) as usize;

                let sr = rgba_data[src]     as f32 / 255.0;
                let sg = rgba_data[src + 1] as f32 / 255.0;
                let sb = rgba_data[src + 2] as f32 / 255.0;
                let sa = rgba_data[src + 3] as f32 / 255.0;

                let comps = [sr, sg, sb, sa];
                for c in 0..4 {
                    let value = f16::from_f32(comps[c]).to_le_bytes();
                    upload[dst + c * 2]     = value[0];
                    upload[dst + c * 2 + 1] = value[1];
                }
            }
        }

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text_rgba16_staging"),
            size: upload.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&staging, 0, &upload);

        let cached = CachedText {
            width,
            height,
            bytes_per_row,
            staging: std::sync::Arc::new(staging),
        };
        self.cache.insert(key, cached.clone());
        cached
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

pub struct TextUploadNode {
    text: CachedText,
    out_rgba: ResourceId,
}

impl TextUploadNode {
    pub fn new(text: CachedText, out_rgba: ResourceId) -> Self {
        Self { text, out_rgba }
    }
}

impl RenderNode for TextUploadNode {
    fn name(&self) -> &str {
        "TextUpload"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.creates.push((
            self.out_rgba,
            ResourceDescriptor {
                label: Some(format!("Text_{}", self.out_rgba.0)),
                size: ResolutionSource::Fixed(self.text.width, self.text.height),
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING,
            },
        ));
        builder.write(self.out_rgba);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext,
        _frame: &FrameState,
    ) {
        let rgba = ctx.get(self.out_rgba);
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.text.staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.text.bytes_per_row),
                    rows_per_image: Some(self.text.height),
                },
            },
            wgpu::ImageCopyTexture {
                texture: rgba.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.text.width,
                height: self.text.height,
                depth_or_array_layers: 1,
            },
        );
    }
}
