use crate::render::device::GpuDevice;

/// Two ping-pong staging buffers for GPU→CPU frame readback.
pub struct FrameReadback {
    buffers:      [wgpu::Buffer; 2],
    width:        u32,
    height:       u32,
    bytes_per_row: u32,
    #[allow(dead_code)]
    buffer_size:  u64,
}

impl FrameReadback {
    pub fn new(device: &GpuDevice, width: u32, height: u32) -> Result<Self, String> {
        let raw_bpr = width * 8u32;
        let bytes_per_row = (raw_bpr + 255) & !255;
        let buffer_size = bytes_per_row as u64 * height as u64;

        let buf0 = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_0"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let buf1 = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_1"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffers: [buf0, buf1],
            width,
            height,
            bytes_per_row,
            buffer_size,
        })
    }

    pub fn record_copy(
        &self,
        encoder:     &mut wgpu::CommandEncoder,
        rtt_texture: &wgpu::Texture,
        slot:        usize,
    ) {
        let src = wgpu::ImageCopyTexture {
            texture: rtt_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };

        let dst = wgpu::ImageCopyBuffer {
            buffer: &self.buffers[slot],
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.bytes_per_row),
                rows_per_image: Some(self.height),
            },
        };

        encoder.copy_texture_to_buffer(src, dst, wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        });
    }

    pub fn map_read<'a>(
        &'a self,
        slot:   usize,
        device: &GpuDevice,
    ) -> Result<wgpu::BufferView<'a>, String> {
        let slice = self.buffers[slot].slice(..);
        
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| { tx.send(v).unwrap(); });

        device.device.poll(wgpu::Maintain::Wait);
        
        if let Err(_) = rx.recv().map_err(|e| e.to_string())? {
            return Err("Failed to map buffer".to_string());
        }

        Ok(slice.get_mapped_range())
    }

    pub fn unmap(&self, slot: usize) {
        self.buffers[slot].unmap();
    }

    pub fn strip_padding(&self, padded_data: &[u8]) -> Vec<u8> {
        let raw_bpr = (self.width * 8) as usize;
        let pad_bpr = self.bytes_per_row as usize;
        let mut out = Vec::with_capacity(raw_bpr * self.height as usize);
        
        for r in 0..self.height as usize {
            let src_start = r * pad_bpr;
            out.extend_from_slice(&padded_data[src_start .. src_start + raw_bpr]);
        }
        out
    }
}
