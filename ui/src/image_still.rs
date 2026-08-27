use std::collections::HashMap;
use std::path::{Path, PathBuf};

use half::f16;
use nexir::render::context::RenderContext;
use nexir::render::device::GpuDevice;
use nexir::render::graph::RenderNode;
use nexir::render::resource::{ResourceBuilder, ResourceDescriptor, ResourceId, ResolutionSource};

fn round_up_256(n: u32) -> u32 {
    (n + 255) & !255
}

#[derive(Clone)]
pub struct CachedStillImage {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub staging: std::sync::Arc<wgpu::Buffer>,
}

#[derive(Default)]
pub struct StillImageCache {
    images: HashMap<PathBuf, CachedStillImage>,
}

impl StillImageCache {
    pub fn get_or_load(&mut self, device: &GpuDevice, path: &Path) -> Option<CachedStillImage> {
        if path.to_string_lossy().starts_with("nexir://") {
            return None;
        }
        if let Some(cached) = self.images.get(path) {
            return Some(cached.clone());
        }

        let reader = match image::ImageReader::open(path) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("StillImage: failed to open {:?}: {:?}", path, e);
                return None;
            }
        };
        let reader = match reader.with_guessed_format() {
            Ok(r) => r,
            Err(e) => {
                log::warn!("StillImage: failed to guess format {:?}: {:?}", path, e);
                return None;
            }
        };
        let decoded = match reader.decode() {
            Ok(d) => d,
            Err(e) => {
                log::warn!("StillImage: failed to decode {:?}: {:?}", path, e);
                return None;
            }
        };
        let rgba = decoded.to_rgba8();

        let (width, height) = rgba.dimensions();
        if width == 0 || height == 0 {
            return None;
        }

        // Quick diagnostics: compute alpha min/max to detect fully-transparent images.
        let mut min_a = 255u8;
        let mut max_a = 0u8;
        for p in rgba.pixels() {
            let a = p.0[3];
            if a < min_a { min_a = a; }
            if a > max_a { max_a = a; }
        }
        log::debug!("StillImage load: {:?} {}x{} alpha min={} max={}", path, width, height, min_a, max_a);
        let force_opaque = max_a == 0;

        let bytes_per_pixel = 8u32;
        let row_bytes = width * bytes_per_pixel;
        let bytes_per_row = round_up_256(row_bytes);
        let mut upload = vec![0u8; (bytes_per_row * height) as usize];

        for y in 0..height {
            let dst_row = (y * bytes_per_row) as usize;
            for x in 0..width {
                let pixel = rgba.get_pixel(x, y).0;
                let dst = dst_row + (x * bytes_per_pixel) as usize;
                    // Store sRGB-encoded values normalised to [0,1] as f16.
                    // The blit shader will apply srgb_to_linear once, and wgpu
                    // applies the sRGB gamma curve on write to Rgba8UnormSrgb.
                    // Do NOT pre-convert here — doing so would double-apply the
                    // gamma decode and darken the image noticeably.
                    let sr = pixel[0] as f32 / 255.0;
                    let sg = pixel[1] as f32 / 255.0;
                    let sb = pixel[2] as f32 / 255.0;
                    let sa = if force_opaque { 1.0 } else { pixel[3] as f32 / 255.0 };

                    let comps = [sr, sg, sb, sa];
                    for c in 0..4 {
                        let value = f16::from_f32(comps[c]).to_le_bytes();
                        upload[dst + c * 2] = value[0];
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

    pub fn clear(&mut self) {
        self.images.clear();
    }

    /// Evict a single cached still image by path.
    pub fn evict(&mut self, path: &Path) {
        self.images.remove(path);
    }

    /// Lightweight probe that attempts to open and decode the image using the
    /// CPU-only `image` crate and logs diagnostics. This does not allocate GPU
    /// resources and is safe to call from import-time code where no `device` is
    /// available.
    pub fn probe_decode(&mut self, path: &Path) {
        if path.to_string_lossy().starts_with("nexir://") {
            return;
        }
        match image::ImageReader::open(path) {
            Ok(r) => match r.with_guessed_format() {
                Ok(r2) => match r2.decode() {
                    Ok(img) => {
                        let rgba = img.to_rgba8();
                        let (w, h) = rgba.dimensions();
                        let mut min_a = 255u8;
                        let mut max_a = 0u8;
                        for p in rgba.pixels() {
                            let a = p.0[3];
                            if a < min_a { min_a = a; }
                            if a > max_a { max_a = a; }
                        }
                        log::debug!("StillImage probe: {:?} {}x{} alpha min={} max={}", path, w, h, min_a, max_a);
                    }
                    Err(e) => {
                        log::warn!("StillImage probe: failed to decode {:?}: {:?}", path, e);
                    }
                },
                Err(e) => {
                    log::warn!("StillImage probe: failed to guess format {:?}: {:?}", path, e);
                }
            },
            Err(e) => {
                log::warn!("StillImage probe: failed to open {:?}: {:?}", path, e);
            }
        }
    }
}

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
        builder.creates.push((
            self.out_rgba,
            ResourceDescriptor {
                label: Some(format!("StillImage_{}", self.out_rgba.0)),
                size: ResolutionSource::Fixed(self.image.width, self.image.height),
                format: wgpu::TextureFormat::Rgba16Float,
            },
        ));
        builder.write(self.out_rgba, nexir::render::resource::TextureAccess::CopyDst);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &RenderContext,
        _frame: &nexir::render::frame_state::FrameState,
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
