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

/// Which mechanism an upload uses to get pixels into the GPU texture.
///
/// Two are viable, they win on different inputs, and the measurement below is why
/// neither was deleted:
///
/// * [`UploadPath::StagingBuffer`] - `queue.write_buffer` into a persistent buffer,
///   then `copy_buffer_to_texture` inside the graph. wgpu allocates its own
///   internal staging buffer per `write_buffer` call
///   (wgpu-core-0.19.4 `device/queue.rs:405-431`), so the bytes are copied on the
///   CPU once here and once by wgpu, then moved on the GPU twice (into our buffer,
///   then into the texture). **`copy_buffer_to_texture` demands a 256-aligned row
///   stride, so an unaligned width forces a CPU repack first.**
/// * [`UploadPath::WriteTexture`] - `queue.write_texture` straight to the texture.
///   One CPU copy into wgpu's staging buffer and one GPU copy, and **wgpu does the
///   re-striding itself, row by row, inside its own staging allocation**
///   (`queue.rs:815-830`) — so an unaligned width costs no repack of ours.
///
/// MEASURED, 3 repeats each, RTX 3050 / DX12 (`target/bench_I3_*.txt`):
///
/// | Benchmark | staging | write_texture |
/// |---|---:|---:|
/// | 1080p 1 layer, render only | 577 FPS | **920** |
/// | 1080p 1 layer, NVENC | 325 | **402** |
/// | 1080p 1 layer, readback | 318 | **404** |
/// | 1080p 3 layers + CC | 196 | **231** |
/// | 4K 4 distinct sources | **49.7** | 46.4 |
/// | 4K 4 layers sharing 1 source | 75.7 | 76.4 (tie) |
///
/// The pattern is exactly the alignment: 1080p 8-bit luma pads 1920 → 2048, so the
/// staging path repacks every row on the CPU and `write_texture` does not. At 4K,
/// 3840 is already aligned, the staging path is a straight memcpy, and the extra
/// GPU hop costs less than `write_texture`'s per-row handling — 4K CPU-per-frame
/// went 6.7 → 8.7 ms on the `write_texture` run.
///
/// Hence [`UploadPath::Auto`], which is the default: pick per node from the
/// arithmetic that decides it. Deleting the "loser" would have made every 1080p row
/// 25-40% slower or every 4K row ~7% slower, depending on which was kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadPath {
    /// Choose per node: `WriteTexture` when the staging path would have to repack
    /// rows, `StagingBuffer` when it would not. Resolved once in the constructor.
    Auto,
    /// `write_buffer` + `copy_buffer_to_texture`. Works for every caller.
    StagingBuffer,
    /// `write_texture` at record time. Needs [`YuvUploadNode::upload_frame_shared`]
    /// and semi-planar chroma; falls back to staging otherwise.
    WriteTexture,
}

impl UploadPath {
    /// Read the path from `NEXIR_UPLOAD_PATH`, defaulting to [`Self::Auto`].
    ///
    /// An environment override rather than a constant so both paths can be
    /// measured from one binary - the point of having two is to compare them, and
    /// a rebuild between the two runs would change more than the path.
    pub fn from_env() -> Self {
        match std::env::var("NEXIR_UPLOAD_PATH").as_deref() {
            Ok("write_texture") => Self::WriteTexture,
            Ok("staging") => Self::StagingBuffer,
            _ => Self::Auto,
        }
    }

    /// Resolve [`Self::Auto`] for a node whose staging stride is `dst_row_bytes`
    /// and whose source rows are `src_row_bytes` wide.
    ///
    /// `Auto` picks `WriteTexture` exactly when the staging path would repack, i.e.
    /// when the two disagree. That is the condition the measurement in this type's
    /// docs splits on, so the rule is the measurement rather than an intuition
    /// about which API is cheaper.
    pub fn resolve(self, src_row_bytes: u32, dst_row_bytes: u32) -> Self {
        match self {
            Self::Auto => {
                if src_row_bytes == dst_row_bytes {
                    Self::StagingBuffer
                } else {
                    Self::WriteTexture
                }
            }
            other => other,
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
    /// Which upload mechanism this node uses. See [`UploadPath`].
    upload_path:      UploadPath,
    /// On the `WriteTexture` path: the frame bytes handed to the most recent
    /// `upload_frame_shared`, held until `record` can write them to the texture.
    ///
    /// An `Arc` rather than a copy, because copying here would reintroduce exactly
    /// the per-frame 12 MB memcpy the contiguous path exists to avoid — and the
    /// bench's pattern frames are immutable and shared anyway. `None` on the
    /// staging path, and on the first frame before anything has been uploaded.
    pending_frame:    std::sync::Mutex<Option<PendingFrame>>,
}

/// One frame's bytes, waiting for `record` to write them into the texture.
///
/// Always semi-planar: [`YuvUploadNode::upload_frame_shared`] routes planar input
/// to the staging path, so nothing here has to carry a layout flag or decide
/// between two chroma arrangements at record time.
struct PendingFrame {
    data: std::sync::Arc<Vec<u8>>,
    width: u32,
    height: u32,
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
            // Resolved HERE, not per frame: the decision is pure arithmetic on the
            // node's own dimensions, so re-deciding it per upload would give the
            // same answer at a per-frame cost — and a node that changed mechanism
            // mid-run would make its own `UploadCost` figures incomparable.
            upload_path: UploadPath::from_env().resolve(width * bpp, y_bytes_per_row),
            pending_frame: std::sync::Mutex::new(None),
        }
    }

    /// Which upload mechanism this node will use.
    pub fn upload_path(&self) -> UploadPath {
        self.upload_path
    }

    /// Force a specific upload path, overriding the environment default.
    ///
    /// For tests that must exercise both paths in one process, and for callers
    /// that cannot satisfy `WriteTexture`'s requirement of shared frame bytes.
    pub fn set_upload_path(&mut self, path: UploadPath) {
        self.upload_path = path;
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

    /// Upload a YUV frame whose bytes the caller can SHARE, allowing the
    /// `write_texture` path.
    ///
    /// Identical to [`Self::upload_frame`] on [`UploadPath::StagingBuffer`]. On
    /// [`UploadPath::WriteTexture`] it stores the `Arc` and defers the actual
    /// transfer to [`RenderNode::record`], because `write_texture` needs the
    /// destination texture and the graph only produces that during recording.
    ///
    /// The deferral is why this takes an `Arc` rather than a slice: the bytes have
    /// to outlive this call, and copying them here would put back the per-frame
    /// 12 MB memcpy the contiguous path exists to remove - the copy would then be
    /// charged to `prepare` and the comparison between the two paths would be
    /// measuring the copy rather than the transfer.
    ///
    /// The reported `submit` time on the deferred path is the time spent storing a
    /// handle, i.e. ~0. The transfer lands in the `GpuTransfer` span, exactly as
    /// `write_buffer`'s does: both are prepended to the next submission by wgpu's
    /// `pending_writes`.
    ///
    /// **Planar chroma always falls back to staging.** `write_texture_planes` can
    /// hand a semi-planar UV plane straight to wgpu, but planar U and V have to be
    /// woven together first — that is CPU work `upload_frame` already does
    /// correctly, and duplicating it here would be a second interleaver to keep in
    /// step with the first.
    pub fn upload_frame_shared(
        &self,
        yuv_data: std::sync::Arc<Vec<u8>>,
        semi_planar: bool,
        frame_width: u32,
        frame_height: u32,
    ) -> UploadCost {
        if self.upload_path != UploadPath::WriteTexture || !semi_planar {
            return self.upload_frame(&yuv_data, semi_planar, frame_width, frame_height);
        }

        let t = Instant::now();
        let bpp = self.layout.bytes_per_sample();
        // Counted the same way as the staging path: luma plus the interleaved
        // chroma plane, i.e. what actually crosses the bus. NOT the padded stride -
        // `write_texture` re-strides internally only when it must, and at 4K the
        // source rows already match.
        let y_bytes = (frame_width as u64) * (frame_height as u64) * bpp as u64;
        let uv_bytes = (frame_width as u64) * ((frame_height / 2) as u64) * bpp as u64;

        *self.pending_frame.lock().unwrap() = Some(PendingFrame {
            data: yuv_data,
            width: frame_width,
            height: frame_height,
        });
        self.current_width
            .store(frame_width.min(self.width), Ordering::Relaxed);
        self.current_height
            .store(frame_height.min(self.height), Ordering::Relaxed);

        UploadCost {
            prepare: std::time::Duration::ZERO,
            submit: t.elapsed(),
            bytes: y_bytes + uv_bytes,
            // No repack happens on this path at all - wgpu decides internally
            // whether the rows need re-striding.
            contiguous: true,
        }
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
        // Clear any deferred `write_texture` frame: this upload supersedes it, and
        // `record` branches on whether one is pending. Leaving a stale one behind
        // would have `record` re-upload the previous frame's pixels and silently
        // drop these — see the comment in `record`.
        *self.pending_frame.lock().unwrap() = None;

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
        // This is not a rare case, but it is narrower than it looks:
        // `round_up_256(width * bpp)` is 3840 at 4K (aligned, contiguous) and
        // 2048 at 1080p — 8-bit luma at 1920 pads, so 1080p still repacks and only
        // gains from the scratch reuse. `which_widths_can_skip_the_repack` pins the
        // real arithmetic, because loosening this guard to cover 1920 would ship a
        // sheared picture with no error anywhere. Measured at 4K with four layers,
        // the repack was ~18 of a 20.6 ms upload stage.
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

    /// Write both planes straight to their textures with `queue.write_texture`.
    ///
    /// Split out of `record` so the `pending_frame` borrow stays short and the
    /// slicing arithmetic lives in one place.
    ///
    /// `bytes_per_row` here is the SOURCE stride (unpadded), not the 256-aligned
    /// one: wgpu re-strides internally when the destination needs it
    /// (wgpu-core-0.19.4 `device/queue.rs:815-830`), and misreporting the source
    /// stride shears the picture exactly the way a wrong staging pitch does.
    fn write_texture_planes(&self, ctx: &RenderContext, frame: &PendingFrame) {
        let bpp = self.layout.bytes_per_sample();
        let w = frame.width.min(self.width);
        let h = frame.height.min(self.height);
        let y_size = (frame.width as usize) * (frame.height as usize) * bpp;

        if y_size > frame.data.len() {
            log::warn!(
                "[upload] slot={} write_texture: Y data too small ({} < {}), skipping",
                self.clip_slot,
                frame.data.len(),
                y_size
            );
            return;
        }

        let y_res = ctx.get(self.out_y);
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: y_res.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.data[..y_size],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(frame.width * bpp as u32),
                rows_per_image: Some(frame.height),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        // Chroma. Semi-planar by construction — `upload_frame_shared` routes planar
        // input to the staging path rather than reimplementing its U/V interleave
        // here, so there is no layout branch at this point.
        let uv_len = (frame.width as usize) * ((frame.height / 2) as usize) * bpp;
        if frame.data.len() < y_size + uv_len {
            log::warn!(
                "[upload] slot={} write_texture: UV data too small, skipping UV",
                self.clip_slot
            );
            return;
        }

        let uv_res = ctx.get(self.out_uv);
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: uv_res.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame.data[y_size..y_size + uv_len],
            wgpu::ImageDataLayout {
                offset: 0,
                // One row of interleaved UV spans the full luma width in bytes.
                bytes_per_row: Some(frame.width * bpp as u32),
                rows_per_image: Some(frame.height / 2),
            },
            wgpu::Extent3d {
                width: w / 2,
                height: h / 2,
                depth_or_array_layers: 1,
            },
        );
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
        // WRITE_TEXTURE PATH.  The transfer happens here rather than at upload
        // time because `queue.write_texture` needs the destination texture, and the
        // graph only hands that over during recording — it comes from the transient
        // pool and may be a different texture each frame.
        //
        // Note what gets recorded into `encoder`: nothing. `write_texture` is a
        // queue operation, so wgpu appends it to its own `pending_writes` encoder
        // and prepends that to the next `queue.submit` — the same mechanism
        // `write_buffer` uses. That is why both paths' transfer cost lands in the
        // `GpuTransfer` span between two frames' graphs, and why neither can be
        // bracketed directly.
        //
        // THE BRANCH IS ON `pending_frame`, NOT ON `self.upload_path`, and that
        // distinction is load-bearing. A node configured for `WriteTexture` still
        // serves callers that hand over a borrowed slice (`ui/src/app.rs`,
        // `export/renderer.rs`) or planar chroma — both of which
        // `upload_frame_shared` routes to the staging buffers. Branching on the
        // configured path would then skip the `copy_buffer_to_texture` for data that
        // is sitting in staging waiting for it, and the frame would render BLACK
        // with no error anywhere. `upload_frame` clears `pending_frame` for exactly
        // this reason, so whichever upload ran most recently is the one `record`
        // completes.
        //
        // Peeked rather than taken: a `record` with no new upload behind it (a graph
        // recompile, a paused playhead) then re-writes the last frame it was given,
        // which is what the staging path does too — its buffer still holds the last
        // upload. Taking it would make that case a black frame.
        let pending = self.pending_frame.lock().unwrap();
        if let Some(frame) = pending.as_ref() {
            self.write_texture_planes(ctx, frame);
            return;
        }
        drop(pending);

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

