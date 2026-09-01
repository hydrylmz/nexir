use crate::render::device::GpuDevice;

/// Two ping-pong staging buffers for GPU→CPU frame readback.
///
/// **This is the FFmpeg-encoder path only, and it must stay that way.** P2.6 —
/// the audit asked whether the export pipeline should adopt this readback; it must
/// not. `ExportBackend::GpuNvenc` converts to NV12 straight into the buffer NVENC
/// reads (`src/interop/nv12_encode.rs`), so a frame never touches the CPU; routing
/// it through here would add a full-frame `copy_texture_to_buffer`, a
/// `map_async` + `poll(Wait)` stall and two host copies per frame — measured at 4K
/// as `strip_padding` alone moving 66 MB. The separate `ExecutionPath` is
/// deliberate: this file exists for the encoders that can only accept host memory.
pub struct FrameReadback {
    buffers:      [wgpu::Buffer; 2],
    width:        u32,
    height:       u32,
    bytes_per_row: u32,
    #[allow(dead_code)]
    buffer_size:  u64,
    /// Pre-allocated output buffer for strip_padding — avoids per-frame 16 MB allocation.
    strip_buf:    Vec<u8>,
}

impl FrameReadback {
    pub fn new(device: &GpuDevice, width: u32, height: u32) -> Result<Self, String> {
        let raw_bpr = width * 8u32; // 8 bytes/pixel for Rgba16Float
        let bytes_per_row = (raw_bpr + 255) & !255;
        let buffer_size = bytes_per_row as u64 * height as u64;
        let strip_buf_size = (raw_bpr * height) as usize;

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
            strip_buf: vec![0u8; strip_buf_size],
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

    /// Map the readback buffer for slot `slot` and block until the GPU copy
    /// identified by `sid` completes.  Using `WaitForSubmissionIndex` rather
    /// than the blanket `Maintain::Wait` means we only stall on the specific
    /// command buffer that wrote into this slot, not on any later renders.
    pub fn map_read<'a>(
        &'a self,
        slot:   usize,
        device: &GpuDevice,
        sid:    wgpu::SubmissionIndex,
    ) -> Result<wgpu::BufferView<'a>, String> {
        let slice = self.buffers[slot].slice(..);
        
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            if tx.send(v).is_err() {
                log::warn!("[readback] map_async channel closed before result was delivered");
            }
        });

        // Only wait for the specific submission that wrote this buffer, not all
        // pending work (which would include the *next* frame's render commands).
        device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        
        if rx.recv().map_err(|e| e.to_string())?.is_err() {
            return Err("Failed to map buffer".to_string());
        }

        Ok(slice.get_mapped_range())
    }

    pub fn unmap(&self, slot: usize) {
        self.buffers[slot].unmap();
    }

    /// Strip GPU row padding and write into the pre-allocated `strip_buf`.
    /// Returns a slice into that buffer — zero allocation.
    pub fn strip_padding<'a>(&'a mut self, padded_data: &[u8]) -> &'a [u8] {
        let raw_bpr = (self.width * 8) as usize;
        let pad_bpr = self.bytes_per_row as usize;
        for r in 0..self.height as usize {
            let src = r * pad_bpr;
            let dst = r * raw_bpr;
            self.strip_buf[dst..dst + raw_bpr]
                .copy_from_slice(&padded_data[src..src + raw_bpr]);
        }
        &self.strip_buf
    }

    /// Zero-copy readback: map → strip padding directly from mapped view → unmap.
    ///
    /// Combines `map_read`, `strip_padding`, and `unmap` into a single call that
    /// avoids the intermediate `Vec<u8>` allocation. The mapped GPU buffer is read
    /// directly into the pre-allocated `strip_buf`, then unmapped immediately.
    /// Returns owned bytes suitable for sending to the encoder queue.
    ///
    /// This eliminates 2 of the 3 full-frame copies that the old path performed:
    ///   OLD: map → to_vec() [COPY 1] → strip_padding [COPY 2] → to_owned() [COPY 3]
    ///   NEW: map → strip directly from view [COPY 1] → to_vec() [COPY 2 — unavoidable for queue ownership]
    pub fn map_strip_unmap(
        &mut self,
        slot:   usize,
        device: &GpuDevice,
        sid:    wgpu::SubmissionIndex,
    ) -> Result<Vec<u8>, String> {
        let raw_bpr = (self.width * 8) as usize;
        let pad_bpr = self.bytes_per_row as usize;

        // Map the readback buffer.
        let slice = self.buffers[slot].slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            if tx.send(v).is_err() {
                log::warn!("[readback] map_strip_unmap channel closed before result was delivered");
            }
        });
        device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
        if rx.recv().map_err(|e| e.to_string())?.is_err() {
            return Err("Failed to map buffer".to_string());
        }
        let view = slice.get_mapped_range();

        // Strip padding directly from the mapped GPU buffer into strip_buf.
        // No intermediate Vec<u8> allocation — we read from the mapped view.
        let padded_data: &[u8] = &view;
        if raw_bpr == pad_bpr {
            // No padding — bulk copy in one shot.
            self.strip_buf.copy_from_slice(&padded_data[..raw_bpr * self.height as usize]);
        } else {
            for r in 0..self.height as usize {
                let src = r * pad_bpr;
                let dst = r * raw_bpr;
                self.strip_buf[dst..dst + raw_bpr]
                    .copy_from_slice(&padded_data[src..src + raw_bpr]);
            }
        }

        // Release the mapped view and unmap before returning.
        drop(view);
        self.buffers[slot].unmap();

        // Return owned bytes for the encoder queue.
        Ok(self.strip_buf.to_vec())
    }
}
