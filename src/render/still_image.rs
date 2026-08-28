// src/render/still_image.rs
//
// Still-image loading, caching, and GPU upload for use in both the preview
// renderer (ui crate) and the export pipeline (nexir library crate).
//
// Images are decoded once via the `image` crate into an Rgba8 buffer, then
// converted to Rgba16Float (half-precision) and uploaded into a wgpu staging
// buffer. The `StillImageUploadNode` copies that staging buffer into the
// `Rgba16Float` texture resource that the composite node reads from.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::f16;

use crate::render::context::RenderContext;
use crate::render::device::GpuDevice;
use crate::render::frame_state::FrameState;
use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};

fn round_up_256(n: u32) -> u32 {
    (n + 255) & !255
}

/// A decoded still image uploaded into a wgpu staging buffer, ready to be
/// blitted into a render texture each frame.
#[derive(Clone)]
pub struct CachedStillImage {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub staging: std::sync::Arc<wgpu::Buffer>,
}

/// Loads and caches still images (PNG, JPEG, etc.) on the GPU.
///
/// Each unique file path is decoded and uploaded exactly once; subsequent
/// requests return a cheap `Arc` clone of the already-uploaded staging buffer.
#[derive(Default)]
pub struct StillImageCache {
    images: HashMap<PathBuf, CachedStillImage>,
}

impl StillImageCache {
    /// Return a cached entry for `path`, loading and uploading it if this is
    /// the first request for that file.
    pub fn get_or_load(&mut self, device: &GpuDevice, path: &Path) -> Option<CachedStillImage> {
        if let Some(cached) = self.images.get(path) {
            return Some(cached.clone());
        }

        let reader = match image::io::Reader::open(path) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("[still_image] failed to open {:?}: {:?}", path, e);
                return None;
            }
        };
        let reader: image::io::Reader<std::io::BufReader<std::fs::File>> = match reader.with_guessed_format() {
            Ok(r) => r,
            Err(e) => {
                log::warn!("[still_image] failed to guess format {:?}: {:?}", path, e);
                return None;
            }
        };
        let decoded: image::DynamicImage = match reader.decode() {
            Ok(d) => d,
            Err(e) => {
                log::warn!("[still_image] failed to decode {:?}: {:?}", path, e);
                return None;
            }
        };
        let rgba = decoded.to_rgba8();

        let (width, height) = rgba.dimensions();
        if width == 0 || height == 0 {
            return None;
        }

        // Diagnostics: detect fully-transparent images and force-opaque them.
        let mut min_a = 255u8;
        let mut max_a = 0u8;
        for p in rgba.pixels() {
            let a = p.0[3];
            if a < min_a { min_a = a; }
            if a > max_a { max_a = a; }
        }
        log::debug!(
            "[still_image] load {:?} {}x{} alpha min={} max={}",
            path, width, height, min_a, max_a
        );
        let force_opaque = max_a == 0;

        // Pack pixels as Rgba16Float with 256-byte row alignment.
        let bytes_per_pixel = 8u32; // 4 channels × 2 bytes (f16)
        let row_bytes = width * bytes_per_pixel;
        let bytes_per_row = round_up_256(row_bytes);
        let mut upload = vec![0u8; (bytes_per_row * height) as usize];

        for y in 0..height {
            let dst_row = (y * bytes_per_row) as usize;
            for x in 0..width {
                let pixel = rgba.get_pixel(x, y).0;
                let dst = dst_row + (x * bytes_per_pixel) as usize;

                // Store sRGB-encoded values normalised to [0,1] as f16.
                // The composite/blit shader applies the sRGB decode; we must
                // NOT pre-convert here to avoid double-applying gamma.
                let sr = pixel[0] as f32 / 255.0;
                let sg = pixel[1] as f32 / 255.0;
                let sb = pixel[2] as f32 / 255.0;
                let sa = if force_opaque { 1.0 } else { pixel[3] as f32 / 255.0 };

                let comps = [sr, sg, sb, sa];
                for c in 0..4 {
                    let value = f16::from_f32(comps[c]).to_le_bytes();
                    upload[dst + c * 2]     = value[0];
                    upload[dst + c * 2 + 1] = value[1];
                }
            }
        }

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("still_image_rgba16_staging"),
            size: upload.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&staging, 0, &upload);

        let cached = CachedStillImage {
            width,
            height,
            bytes_per_row,
            staging: std::sync::Arc::new(staging),
        };
        self.images.insert(path.to_path_buf(), cached.clone());
        Some(cached)
    }

    /// Remove all cached images (e.g. on project close).
    pub fn clear(&mut self) {
        self.images.clear();
    }

    /// Evict a single cached entry by path.
    pub fn evict(&mut self, path: &Path) {
        self.images.remove(path);
    }
}

/// Render node that blits a pre-loaded still image into an `Rgba16Float`
/// texture resource so the composite node can sample it like any other clip.
pub struct StillImageUploadNode {
    image: CachedStillImage,
    out_rgba: ResourceId,
}

impl StillImageUploadNode {
    pub fn new(image: CachedStillImage, out_rgba: ResourceId) -> Self {
        Self { image, out_rgba }
    }
}

impl RenderNode for StillImageUploadNode {
    fn name(&self) -> &str {
        "StillImageUpload"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((
            self.out_rgba,
            ResourceDescriptor {
                label: Some(format!("StillImage_{}", self.out_rgba.0)),
                size: ResolutionSource::Fixed(self.image.width, self.image.height),
                format: wgpu::TextureFormat::Rgba16Float,
            },
        ));
        builder.write(self.out_rgba, TextureAccess::CopyDst);
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
                buffer: &self.image.staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.image.bytes_per_row),
                    rows_per_image: Some(self.image.height),
                },
            },
            wgpu::ImageCopyTexture {
                texture: rgba.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.image.width,
                height: self.image.height,
                depth_or_array_layers: 1,
            },
        );
    }
}
