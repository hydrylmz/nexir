// src/render/nodes/yuv_upload.rs

use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use std::sync::atomic::{AtomicU32, Ordering};

fn round_up_256(n: u32) -> u32 {
    (n + 255) & !255
}

/// Uploads CPU YUV data into a pair of GPU textures (Y plane + UV plane).
///
/// Staging buffers are written via `queue.write_buffer` (non-blocking, no
/// map_async / poll(Wait) calls) so that this node never holds a lock while
/// waiting for the GPU.  The `upload_frame` method may be called from any
/// thread without causing device-wide GPU stalls.
pub struct YuvUploadNode {
    pub clip_slot:    u32,
    /// Maximum dimensions the staging buffers were allocated for.
    pub width:        u32,
    pub height:       u32,
    /// Actual dimensions of the most-recently-uploaded frame (may be ≤ max).
    current_width:    AtomicU32,
    current_height:   AtomicU32,
    pub out_y:        ResourceId,
    pub out_uv:       ResourceId,
    y_staging:        wgpu::Buffer,
    uv_staging:       wgpu::Buffer,
    queue:            std::sync::Arc<wgpu::Queue>,
}

impl YuvUploadNode {
    pub fn new(
        device:    &GpuDevice,
        clip_slot: u32,
        width:     u32,
        height:    u32,
        out_y:     ResourceId,
        out_uv:    ResourceId,
    ) -> Self {
        let y_bytes_per_row  = round_up_256(width);
        let uv_bytes_per_row = round_up_256(width);

        let y_size  = y_bytes_per_row as u64 * height as u64;
        let uv_size = uv_bytes_per_row as u64 * (height / 2) as u64;

        // COPY_DST | COPY_SRC: written by queue.write_buffer, read by copy_buffer_to_texture.
        // No MAP_WRITE needed — write_buffer uses wgpu's internal upload ring.
        let y_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y_staging"),
            size:  y_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let uv_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uv_staging"),
            size:  uv_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            clip_slot,
            width,
            height,
            current_width:  AtomicU32::new(width),
            current_height: AtomicU32::new(height),
            out_y,
            out_uv,
            y_staging,
            uv_staging,
            queue: std::sync::Arc::clone(&device.queue),
        }
    }

    /// Upload a YUV frame into the staging buffers using the wgpu queue's
    /// internal upload ring (**non-blocking, no poll(Wait)**).
    ///
    /// `frame_width` and `frame_height` are the actual source frame dimensions,
    /// which may differ from the node's maximum allocation dimensions.
    /// Frames larger than the pre-allocated buffer are cropped to fit.
    pub fn upload_frame(
        &self,
        yuv_data:     &[u8],
        nv12:         bool,
        frame_width:  u32,
        frame_height: u32,
    ) {
        // Clamp to what we actually allocated
        let w = frame_width.min(self.width);
        let h = frame_height.min(self.height);


        let y_size        = (frame_width * frame_height) as usize;
        let uv_plane_size = ((frame_width / 2) * (frame_height / 2)) as usize;

        let y_bytes_per_row  = round_up_256(self.width) as usize;
        let uv_bytes_per_row = round_up_256(self.width) as usize;

        // ── Y plane ──────────────────────────────────────────────────────────
        if y_size > yuv_data.len() {
            log::warn!("[upload] slot={} Y data too small ({} < {}), skipping",
                self.clip_slot, yuv_data.len(), y_size);
            return;
        }
        let src_y = &yuv_data[..y_size];

        // Build a row-padded CPU buffer matching the staging buffer layout.
        let mut y_buf = vec![0u8; y_bytes_per_row * self.height as usize];
        for row in 0..h as usize {
            let src_start = row * frame_width as usize;
            let dst_start = row * y_bytes_per_row;
            y_buf[dst_start..dst_start + w as usize]
                .copy_from_slice(&src_y[src_start..src_start + w as usize]);
        }
        self.queue.write_buffer(&self.y_staging, 0, &y_buf);

        // ── UV plane ─────────────────────────────────────────────────────────
        let mut uv_buf = vec![0u8; uv_bytes_per_row * (self.height / 2) as usize];
        let uv_rows    = (h / 2) as usize;

        if nv12 {
            let src_uv_start = y_size;
            let src_uv_len   = (frame_width * (frame_height / 2)) as usize;
            if yuv_data.len() < src_uv_start + src_uv_len {
                log::warn!("[upload] slot={} NV12 UV data too small, skipping UV", self.clip_slot);
            } else {
                let src_uv = &yuv_data[src_uv_start..src_uv_start + src_uv_len];
                for row in 0..uv_rows {
                    let src_start = row * frame_width as usize;
                    let dst_start = row * uv_bytes_per_row;
                    uv_buf[dst_start..dst_start + w as usize]
                        .copy_from_slice(&src_uv[src_start..src_start + w as usize]);
                }
            }
        } else {
            // I420: U and V planes separate
            if yuv_data.len() < y_size + 2 * uv_plane_size {
                log::warn!("[upload] slot={} I420 UV data too small, skipping UV", self.clip_slot);
            } else {
                let u_plane = &yuv_data[y_size..y_size + uv_plane_size];
                let v_plane = &yuv_data[y_size + uv_plane_size..y_size + 2 * uv_plane_size];
                let uv_w    = (frame_width / 2) as usize;
                for row in 0..uv_rows {
                    let dst_start = row * uv_bytes_per_row;
                    for col in 0..(w / 2) as usize {
                        let src_idx = row * uv_w + col;
                        uv_buf[dst_start + col * 2]     = u_plane[src_idx];
                        uv_buf[dst_start + col * 2 + 1] = v_plane[src_idx];
                    }
                }
            }
        }
        self.queue.write_buffer(&self.uv_staging, 0, &uv_buf);

        // Record the actual dimensions so record() uses the right copy extent.
        self.current_width.store(w, Ordering::Relaxed);
        self.current_height.store(h, Ordering::Relaxed);
    }
}

impl RenderNode for YuvUploadNode {
    fn name(&self) -> &str {
        "YuvUpload"
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource, TextureAccess};
        builder.creates.push((self.out_y, ResourceDescriptor {
            label: Some(format!("Yuv_Y_{}", self.out_y.0)),
            size: ResolutionSource::Fixed(self.width, self.height),
            format: wgpu::TextureFormat::R8Unorm,
        }));
        builder.creates.push((self.out_uv, ResourceDescriptor {
            label: Some(format!("Yuv_UV_{}", self.out_uv.0)),
            size: ResolutionSource::Fixed(self.width / 2, self.height / 2),
            format: wgpu::TextureFormat::Rg8Unorm,
        }));
        builder.write(self.out_y, TextureAccess::CopyDst);
        builder.write(self.out_uv, TextureAccess::CopyDst);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        // Use the dimensions from the most recent upload_frame call.
        let cw = self.current_width.load(Ordering::Relaxed);
        let ch = self.current_height.load(Ordering::Relaxed);

        let y_res            = ctx.get(self.out_y);
        let y_bytes_per_row  = round_up_256(self.width);

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.y_staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(y_bytes_per_row),
                    rows_per_image: Some(ch),
                },
            },
            wgpu::ImageCopyTexture {
                texture:   y_res.texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
        );

        let uv_res            = ctx.get(self.out_uv);
        let uv_bytes_per_row  = round_up_256(self.width);

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.uv_staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(uv_bytes_per_row),
                    rows_per_image: Some(ch / 2),
                },
            },
            wgpu::ImageCopyTexture {
                texture:   uv_res.texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: cw / 2, height: ch / 2, depth_or_array_layers: 1 },
        );
    }
}
