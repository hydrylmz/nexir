// src/render/nodes/yuv_upload.rs

use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;

fn round_up_256(n: u32) -> u32 {
    (n + 255) & !255
}

/// Uploads CPU YUV data into a pair of GPU textures (Y plane + UV plane).
pub struct YuvUploadNode {
    pub clip_slot:     u32,
    pub width:         u32,
    pub height:        u32,
    pub out_y:         ResourceId,
    pub out_uv:        ResourceId,
    y_staging:         wgpu::Buffer,
    uv_staging:        wgpu::Buffer,
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
        let y_bytes_per_row = round_up_256(width);
        let uv_bytes_per_row = round_up_256(width); // Since UV is Rg8Unorm, width/2 * 2 = width

        let y_size = (y_bytes_per_row * height) as u64;
        let uv_size = (uv_bytes_per_row * (height / 2)) as u64;

        let y_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y_staging"),
            size: y_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let uv_staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uv_staging"),
            size: uv_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        Self {
            clip_slot,
            width,
            height,
            out_y,
            out_uv,
            y_staging,
            uv_staging,
        }
    }

    pub fn upload_frame(
        &self,
        device:   &GpuDevice,
        yuv_data: &[u8],
        nv12:     bool,
    ) {
        let y_size = (self.width * self.height) as usize;
        let uv_plane_size = ((self.width / 2) * (self.height / 2)) as usize;

        let y_bytes_per_row = round_up_256(self.width) as usize;
        let uv_bytes_per_row = round_up_256(self.width) as usize; // width/2 * 2 = width

        // 1. Upload Y plane
        let y_slice = self.y_staging.slice(..);
        y_slice.map_async(wgpu::MapMode::Write, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        
        let mut mapped = y_slice.get_mapped_range_mut();
        // Handle padding row by row
        let src_y = &yuv_data[0..y_size];
        for row in 0..self.height as usize {
            let src_start = row * self.width as usize;
            let src_end = src_start + self.width as usize;
            let dst_start = row * y_bytes_per_row;
            let dst_end = dst_start + self.width as usize;
            mapped[dst_start..dst_end].copy_from_slice(&src_y[src_start..src_end]);
        }
        drop(mapped);
        self.y_staging.unmap();

        // 2. Upload UV plane
        let uv_slice = self.uv_staging.slice(..);
        uv_slice.map_async(wgpu::MapMode::Write, |_| {});
        device.device.poll(wgpu::Maintain::Wait);
        
        let mut mapped_uv = uv_slice.get_mapped_range_mut();
        if nv12 {
            let src_uv = &yuv_data[y_size..];
            for row in 0..(self.height / 2) as usize {
                let src_start = row * self.width as usize;
                let src_end = src_start + self.width as usize;
                let dst_start = row * uv_bytes_per_row;
                let dst_end = dst_start + self.width as usize;
                mapped_uv[dst_start..dst_end].copy_from_slice(&src_uv[src_start..src_end]);
            }
        } else {
            let u_plane = &yuv_data[y_size .. y_size + uv_plane_size];
            let v_plane = &yuv_data[y_size + uv_plane_size ..];
            
            for row in 0..(self.height / 2) as usize {
                let dst_start = row * uv_bytes_per_row;
                for col in 0..(self.width / 2) as usize {
                    let src_idx = row * (self.width / 2) as usize + col;
                    mapped_uv[dst_start + col * 2] = u_plane[src_idx];
                    mapped_uv[dst_start + col * 2 + 1] = v_plane[src_idx];
                }
            }
        }
        drop(mapped_uv);
        self.uv_staging.unmap();
    }
}

impl RenderNode for YuvUploadNode {
    fn name(&self) -> &str {
        "YuvUpload"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        use crate::render::resource::{ResourceDescriptor, ResolutionSource};
        builder.creates.push((self.out_y, ResourceDescriptor {
            label: Some(format!("Yuv_Y_{}", self.out_y.0)),
            size: ResolutionSource::Fixed(self.width, self.height),
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        }));
        builder.creates.push((self.out_uv, ResourceDescriptor {
            label: Some(format!("Yuv_UV_{}", self.out_uv.0)),
            size: ResolutionSource::Fixed(self.width / 2, self.height / 2),
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        }));
        builder.write(self.out_y);
        builder.write(self.out_uv);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx:     &RenderContext,
        _frame:  &FrameState,
    ) {
        let y_res = ctx.get(self.out_y);
        let y_bytes_per_row = round_up_256(self.width);

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.y_staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(y_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::ImageCopyTexture {
                texture: y_res.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );

        let uv_res = ctx.get(self.out_uv);
        let uv_bytes_per_row = round_up_256(self.width);

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &self.uv_staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(uv_bytes_per_row),
                    rows_per_image: Some(self.height / 2),
                },
            },
            wgpu::ImageCopyTexture {
                texture: uv_res.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: self.width / 2, height: self.height / 2, depth_or_array_layers: 1 },
        );
    }
}
