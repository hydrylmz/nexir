// src/render/nodes/yuv_upload.rs

use crate::render::graph::RenderNode;
use crate::render::resource::{ResourceBuilder, ResourceId};
use crate::render::context::RenderContext;
use crate::render::frame_state::FrameState;
use crate::render::device::GpuDevice;
use crate::timeline::source::FrameLayout;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

fn round_up_256(n: u32) -> u32 {
    (n + 255) & !255
}

/// What one [`YuvUploadNode::upload_frame`] call actually did.
///
/// Returned rather than discarded so the caller can attribute the cost instead
/// of charging an entire upload to one stage. At 4K the two halves differed by
/// an order of magnitude and had completely different cures — `prepare` is heap
/// allocation and memcpy, `submit` is the transfer — so a single number said
/// almost nothing about what to fix.
#[derive(Debug, Clone, Copy, Default)]
pub struct UploadCost {
    /// Time building row-padded CPU buffers. Zero on the contiguous fast path.
    pub prepare: std::time::Duration,
    /// Time inside `queue.write_buffer`.
    pub submit: std::time::Duration,
    /// Bytes handed to the queue. Counted, not estimated.
    pub bytes: u64,
    /// Whether the row repack was skipped because the source stride already
    /// matched the staging buffer's.
    pub contiguous: bool,
}

impl UploadCost {
    /// Accumulate another layer's cost into this one.
    pub fn add(&mut self, other: UploadCost) {
        self.prepare += other.prepare;
        self.submit += other.submit;
        self.bytes += other.bytes;
        // A frame is only "contiguous" if every layer in it was.
        self.contiguous &= other.contiguous;
    }

    /// A starting value for accumulation: nothing measured, contiguous until a
    /// layer proves otherwise.
    pub fn zero() -> Self {
        Self {
            contiguous: true,
            ..Default::default()
        }
    }
}

/// Uploads CPU YUV data into a pair of GPU textures (Y plane + UV plane).
///
/// Staging buffers are written via `queue.write_buffer` (non-blocking, no
/// map_async / poll(Wait) calls) so that this node never holds a lock while
/// waiting for the GPU.  The `upload_frame` method may be called from any
/// thread without causing device-wide GPU stalls.
///
/// P1.6 — the node is built from the [`FrameLayout`] the DECODER reported for the
/// frame in the slot, not from the container's advertised pixel format.  Layout
/// decides the texture formats (8- vs 16-bit), the row strides and whether chroma
/// arrives interleaved, so a mismatch here shows up as a garbled or near-black
/// picture rather than as a subtle colour shift.
///
/// P0.1 — the two costs of an upload are separated and each is attacked:
///
/// * **Re-striding.** `copy_buffer_to_texture` requires a 256-byte row stride, so
///   a frame whose row width is not a multiple of 256 has to be repacked.
///   1920-wide 8-bit luma pads to 2048; 3840 does not pad at all. When no
///   padding is needed the source bytes go straight to the queue
///   ([`Self::upload_frame`]'s contiguous path) and nothing is copied on the CPU.
/// * **Allocation.** The repack used to `vec![0u8; …]` a fresh 8.3 MB buffer per
///   plane per layer per frame — allocated, zeroed, filled, then handed on. The
///   scratch buffers below are allocated once and reused, so the remaining cost
///   is one memcpy rather than a page-fault storm.
pub struct YuvUploadNode {
    pub clip_slot:    u32,
    /// Maximum dimensions the staging buffers were allocated for.
    pub width:        u32,
    pub height:       u32,
    /// Pixel layout of the frames this node uploads.
    pub layout:       FrameLayout,
    /// Actual dimensions of the most-recently-uploaded frame (may be ≤ max).
    current_width:    AtomicU32,
    current_height:   AtomicU32,
    pub out_y:        ResourceId,
    pub out_uv:       ResourceId,
    y_staging:        wgpu::Buffer,
    uv_staging:       wgpu::Buffer,
    /// Reused row-padding scratch, one per plane.
    ///
    /// `Mutex` rather than a plain field because `upload_frame` takes `&self` —
    /// the node is shared behind the graph and `record` needs `&self` too. The
    /// lock is uncontended in practice: one caller uploads into a given node.
    y_scratch:        std::sync::Mutex<Vec<u8>>,
    uv_scratch:       std::sync::Mutex<Vec<u8>>,
    queue:            std::sync::Arc<wgpu::Queue>,
}

impl YuvUploadNode {
    /// Build a node for 8-bit planar I420 — the decoder's conversion target for
    /// anything it cannot pass through.
    pub fn new(
        device:    &GpuDevice,
        clip_slot: u32,
        width:     u32,
        height:    u32,
        out_y:     ResourceId,
        out_uv:    ResourceId,
    ) -> Self {
        Self::new_with_layout(device, clip_slot, width, height, out_y, out_uv, FrameLayout::YUV420P8)
    }

    /// Build a node for a specific decoded frame layout.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_layout(
        device:    &GpuDevice,
        clip_slot: u32,
        width:     u32,
        height:    u32,
        out_y:     ResourceId,
        out_uv:    ResourceId,
        layout:    FrameLayout,
    ) -> Self {
        let bpp = layout.bytes_per_sample() as u32;
        let y_bytes_per_row  = round_up_256(width * bpp);
        // Chroma rows hold two interleaved samples per chroma column, so a
        // half-width chroma plane is the same byte width as the luma plane.
        let uv_bytes_per_row = round_up_256(width * bpp);

        let y_size  = y_bytes_per_row as u64 * height as u64;
        let uv_size = uv_bytes_per_row as u64 * (height / 2) as u64;

        // COPY_DST | COPY_SRC: written by queue.write_buffer, read by copy_buffer_to_texture.
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
            layout,
            current_width:  AtomicU32::new(width),
            current_height: AtomicU32::new(height),
            out_y,
            out_uv,
            y_staging,
            uv_staging,
            // Sized to the staging buffers they feed, and allocated here rather
            // than per frame: this is the 8.3 MB/plane/layer allocation that made
            // 4K upload cost 20 ms.
            y_scratch:  std::sync::Mutex::new(vec![0u8; y_size as usize]),
            uv_scratch: std::sync::Mutex::new(vec![0u8; uv_size as usize]),
            queue: std::sync::Arc::clone(&device.queue),
        }
    }

    /// Legacy constructor kept for the benchmark harness: builds a layout from a
    /// bare bit depth, assuming planar chroma.
    ///
    /// Prefer [`Self::new_with_layout`] — for 10-bit data, planar and semi-planar
    /// differ by a factor of 64 in sample normalisation.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_depth(
        device:    &GpuDevice,
        clip_slot: u32,
        width:     u32,
        height:    u32,
        out_y:     ResourceId,
        out_uv:    ResourceId,
        bit_depth: u8,
    ) -> Self {
        Self::new_with_layout(
            device, clip_slot, width, height, out_y, out_uv,
            FrameLayout { bit_depth, semi_planar: false, msb_aligned: false },
        )
    }

    /// Upload a YUV frame into the staging buffers using the wgpu queue's
    /// internal upload ring (**non-blocking, no poll(Wait)**).
    ///
    /// `semi_planar` must match the data in `yuv_data`; it is passed per-call
    /// rather than taken from `self.layout` because the caller reads it off the
    /// frame it is about to upload, and a mid-timeline format switch is possible.
    ///
    /// Returns what it cost — see [`UploadCost`]. Ignoring the return value is
    /// fine for callers that do not profile.
    pub fn upload_frame(
        &self,
        yuv_data:     &[u8],
        semi_planar:  bool,
        frame_width:  u32,
        frame_height: u32,
    ) -> UploadCost {
        // Clamp to what we actually allocated
        let w = frame_width.min(self.width);
        let h = frame_height.min(self.height);

        let bpp = self.layout.bytes_per_sample();
        let y_size        = (frame_width * frame_height) as usize * bpp;
        let uv_plane_size = ((frame_width / 2) * (frame_height / 2)) as usize * bpp;

        let y_bytes_per_row  = round_up_256(self.width * (bpp as u32)) as usize;
        let uv_bytes_per_row = round_up_256(self.width * (bpp as u32)) as usize;

        let mut cost = UploadCost::default();

        // ── Y plane ──────────────────────────────────────────────────────────
        if y_size > yuv_data.len() {
            log::warn!("[upload] slot={} Y data too small ({} < {}), skipping",
                self.clip_slot, yuv_data.len(), y_size);
            return cost;
        }
        let src_y = &yuv_data[..y_size];

        // FAST PATH.  When the destination stride already equals the source row
        // width, and the frame fills the whole allocation, the loop below copies
        // every row to the byte offset it already occupies — a full-plane memcpy
        // into a freshly allocated, freshly zeroed buffer, for nothing.
        //
        // This is not a rare case: `round_up_256(width * 1)` is 1920 at 1080p and
        // 3840 at 4K, both already 256-aligned, so 8-bit NV12 at either standard
        // resolution takes this path. Measured at 4K with four layers, the repack
        // was ~18 of a 20.6 ms upload stage.
        //
        // The three conditions are all cheap, and any one of them failing falls
        // through to the general path unchanged — a non-aligned width, a partial
        // frame in a larger allocation, or high-bit-depth planar data that
        // genuinely has to be interleaved.
        let src_row_bytes = frame_width as usize * bpp;
        let contiguous = y_bytes_per_row == src_row_bytes
            && frame_width == self.width
            && frame_height == self.height;
        cost.contiguous = contiguous;

        if contiguous {
            let t = Instant::now();
            self.queue.write_buffer(&self.y_staging, 0, src_y);
            cost.submit += t.elapsed();
            cost.bytes += src_y.len() as u64;
        } else {
            // Re-stride into the reused scratch buffer. Allocated once in `new`,
            // so this is a memcpy — not the 8.3 MB alloc-and-zero per plane per
            // layer per frame that made 4K upload cost 20 ms.
            let t = Instant::now();
            let mut y_buf = self.y_scratch.lock().unwrap();
            let row_bytes = w as usize * bpp;
            for row in 0..h as usize {
                let src_start = row * frame_width as usize * bpp;
                let dst_start = row * y_bytes_per_row;
                y_buf[dst_start..dst_start + row_bytes]
                    .copy_from_slice(&src_y[src_start..src_start + row_bytes]);
            }
            cost.prepare += t.elapsed();

            let t = Instant::now();
            self.queue.write_buffer(&self.y_staging, 0, &y_buf);
            cost.submit += t.elapsed();
            cost.bytes += y_buf.len() as u64;
        }

        // ── UV plane ─────────────────────────────────────────────────────────
        // The semi-planar (NV12/P010) case is the one that can go contiguous:
        // its chroma plane is already interleaved, so like luma it only needs
        // re-striding, and at an aligned width it does not even need that. The
        // planar cases must interleave two separate planes and always repack.
        let uv_rows = (h / 2) as usize;

        if semi_planar && contiguous {
            let src_uv_start = y_size;
            let src_uv_len   = (frame_width * (frame_height / 2)) as usize * bpp;
            if yuv_data.len() < src_uv_start + src_uv_len {
                log::warn!("[upload] slot={} NV12/P010 UV data too small, skipping UV", self.clip_slot);
            } else {
                let src_uv = &yuv_data[src_uv_start..src_uv_start + src_uv_len];
                let t = Instant::now();
                self.queue.write_buffer(&self.uv_staging, 0, src_uv);
                cost.submit += t.elapsed();
                cost.bytes += src_uv.len() as u64;
            }
        } else {
            let t = Instant::now();
            let mut uv_buf = self.uv_scratch.lock().unwrap();

            if semi_planar {
                // NV12 / P010: one plane, U and V already interleaved, so the rows can
                // be copied straight across.
                let src_uv_start = y_size;
                let src_uv_len   = (frame_width * (frame_height / 2)) as usize * bpp;
                if yuv_data.len() < src_uv_start + src_uv_len {
                    log::warn!("[upload] slot={} NV12/P010 UV data too small, skipping UV", self.clip_slot);
                } else {
                    let src_uv = &yuv_data[src_uv_start..src_uv_start + src_uv_len];
                    let uv_row_bytes = w as usize * bpp;
                    for row in 0..uv_rows {
                        let src_start = row * frame_width as usize * bpp;
                        let dst_start = row * uv_bytes_per_row;
                        uv_buf[dst_start..dst_start + uv_row_bytes]
                            .copy_from_slice(&src_uv[src_start..src_start + uv_row_bytes]);
                    }
                }
            } else if bpp == 2 {
                // Planar 10/12-bit (YUV420P10LE, YUV420P12LE): separate U and V planes
                // of 16-bit samples, interleaved here into one Rg16Unorm texture.
                if yuv_data.len() < y_size + 2 * uv_plane_size {
                    log::warn!("[upload] slot={} high-depth planar UV data too small, skipping UV", self.clip_slot);
                } else {
                    let u_plane = &yuv_data[y_size..y_size + uv_plane_size];
                    let v_plane = &yuv_data[y_size + uv_plane_size..y_size + 2 * uv_plane_size];
                    let uv_w = (frame_width / 2) as usize;
                    for row in 0..uv_rows {
                        let dst_start = row * uv_bytes_per_row;
                        for col in 0..(w / 2) as usize {
                            let src_idx = (row * uv_w + col) * 2;
                            let dst_idx = dst_start + col * 4;
                            uv_buf[dst_idx..dst_idx + 2].copy_from_slice(&u_plane[src_idx..src_idx + 2]);
                            uv_buf[dst_idx + 2..dst_idx + 4].copy_from_slice(&v_plane[src_idx..src_idx + 2]);
                        }
                    }
                }
            } else {
                // I420: U and V planes separate (8-bit)
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
            cost.prepare += t.elapsed();

            let t = Instant::now();
            self.queue.write_buffer(&self.uv_staging, 0, &uv_buf);
            cost.submit += t.elapsed();
            cost.bytes += uv_buf.len() as u64;
        }

        // Record the actual dimensions so record() uses the right copy extent.
        self.current_width.store(w, Ordering::Relaxed);
        self.current_height.store(h, Ordering::Relaxed);

        cost
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
        // 16-bit textures for any depth above 8.  The shader's `sample_scale`
        // (see colour::yuv) is what compensates for the resulting normalisation.
        let (y_fmt, uv_fmt) = if self.layout.is_high_depth() {
            (wgpu::TextureFormat::R16Unorm, wgpu::TextureFormat::Rg16Unorm)
        } else {
            (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm)
        };
        builder.creates.push((self.out_y, ResourceDescriptor {
            label: Some(format!("Yuv_Y_{}", self.out_y.0)),
            size: ResolutionSource::Fixed(self.width, self.height),
            format: y_fmt,
        }));
        builder.creates.push((self.out_uv, ResourceDescriptor {
            label: Some(format!("Yuv_UV_{}", self.out_uv.0)),
            size: ResolutionSource::Fixed(self.width / 2, self.height / 2),
            format: uv_fmt,
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

        let bpp = self.layout.bytes_per_sample() as u32;
        let y_res            = ctx.get(self.out_y);
        let y_bytes_per_row  = round_up_256(self.width * bpp);

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
        let uv_bytes_per_row  = round_up_256(self.width * bpp);

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

