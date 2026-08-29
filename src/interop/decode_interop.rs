// src/interop/decode_interop.rs
// Replaces Phase 4's av_hwframe_transfer_data CPU round-trip for CUDA/NVDEC frames.
// Copies NVDEC CUdeviceptr output directly into wgpu textures, staying on-GPU the whole way.

use crate::interop::cuda_context::{CudaContext, CudaError};
use crate::interop::external_texture::SharedTexture;
use crate::interop::ffi::cuda_gl_vk_interop::{CudaMemcpy2D, cuMemcpy2DAsync_v2};
use crate::interop::ffi::cuda_driver::{CUdeviceptr, CUDA_SUCCESS, cuStreamSynchronize};
use crate::render::device::GpuDevice;
use crate::io::ffi::avutil::{AVFrame, av_frame_get_data, av_frame_get_linesize};

/// A wgpu texture pair (Y plane, UV plane) whose memory is also CUDA-accessible,
/// ready to receive an NVDEC frame directly without any CPU involvement.
pub struct DecodeInteropTarget {
    y_slot:  SharedTexture,
    uv_slot: SharedTexture,
}

impl DecodeInteropTarget {
    /// Allocate an export-compatible Y/UV texture pair and import both into CUDA.
    ///
    /// The allocations are created as shared D3D12 resources by
    /// [`SharedTexture::new`] — an ordinary `device.create_texture` cannot be
    /// imported by CUDA at all, since the export flag has to be set at allocation
    /// time.
    ///
    /// Takes the context by `Arc` because each `SharedTexture` keeps a share of
    /// it: a CUDA import must not outlive the context it was made against.
    pub fn new(
        cuda_ctx:  std::sync::Arc<CudaContext>,
        device:    &GpuDevice,
        transport: crate::interop::capability::InteropTransport,
        width:     u32,
        height:    u32,
    ) -> Result<Self, CudaError> {
        // Y plane: R8Unorm, full resolution.  UV plane: Rg8Unorm, half resolution
        // (NV12 layout).  These match Phase 2/3's YuvUploadNode formats exactly,
        // so no shader changes are needed.
        const PLANE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
            .union(wgpu::TextureUsages::COPY_DST)
            .union(wgpu::TextureUsages::STORAGE_BINDING);

        let y_slot = SharedTexture::new(
            std::sync::Arc::clone(&cuda_ctx), device, transport,
            "DecodeInterop Y Plane",
            width, height,
            wgpu::TextureFormat::R8Unorm,
            PLANE_USAGE,
        )?;
        let uv_slot = SharedTexture::new(
            std::sync::Arc::clone(&cuda_ctx), device, transport,
            "DecodeInterop UV Plane",
            width / 2, height / 2,
            wgpu::TextureFormat::Rg8Unorm,
            PLANE_USAGE,
        )?;

        Ok(Self { y_slot, uv_slot })
    }

    /// The wgpu view of the luma plane, for binding in the render graph.
    pub fn y_texture(&self) -> &wgpu::Texture {
        &self.y_slot.texture
    }

    /// The wgpu view of the interleaved chroma plane.
    pub fn uv_texture(&self) -> &wgpu::Texture {
        &self.uv_slot.texture
    }

    /// Copy an NVDEC-decoded frame directly into this target's Y/UV textures.
    /// Called instead of Decoder::decode_into's CPU-copy path whenever
    /// InteropCapability::is_available() is true and the source hw_type is Cuda.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn copy_from_nvdec_frame(
        &self,
        cuda_ctx: &CudaContext,
        frame:    *const AVFrame,
        width:    u32,
        height:   u32,
    ) -> Result<(), CudaError> {
        // SAFETY: `frame` is supplied by the decoder while the decoded AVFrame
        // is still alive; we only read FFmpeg's data and linesize arrays.
        // Step 1 — Extract NVDEC's output device pointers and pitch from the AVFrame.
        let (y_devptr, uv_devptr, pitch) = unsafe {
            let data = av_frame_get_data(frame);
            let linesize = av_frame_get_linesize(frame);
            let y_ptr  = *data.add(0) as u64;
            let uv_ptr = *data.add(1) as u64;
            let pitch  = *linesize.add(0) as usize;
            (y_ptr as CUdeviceptr, uv_ptr as CUdeviceptr, pitch)
        };

        let y_array  = self.y_slot.external.cuda_array();
        let uv_array = self.uv_slot.external.cuda_array();

        // Step 2 — Issue device-to-array copies for each plane via cuMemcpy2DAsync.
        // This is a GPU-side copy (device memory → CUDA array), with zero CPU/PCIe involvement.
        let ret = cuda_ctx.with_context(|stream| unsafe {
            // Y plane — full resolution
            let y_copy = CudaMemcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_member_type: 2, // CU_MEMORYTYPE_DEVICE
                src_host: std::ptr::null(),
                src_device: y_devptr,
                src_array: std::ptr::null_mut(),
                src_pitch: pitch,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_member_type: 3, // CU_MEMORYTYPE_ARRAY
                dst_host: std::ptr::null_mut(),
                dst_device: 0,
                dst_array: y_array,
                dst_pitch: 0,
                width_in_bytes: width as usize,
                height: height as usize,
            };
            let r = cuMemcpy2DAsync_v2(&y_copy, stream);
            if r != CUDA_SUCCESS { return r; }

            // UV plane — half resolution (NV12: interleaved U and V)
            let uv_copy = CudaMemcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_member_type: 2,
                src_host: std::ptr::null(),
                src_device: uv_devptr,
                src_array: std::ptr::null_mut(),
                src_pitch: pitch,  // NV12 UV pitch is same as Y pitch
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_member_type: 3,
                dst_host: std::ptr::null_mut(),
                dst_device: 0,
                dst_array: uv_array,
                dst_pitch: 0,
                width_in_bytes: (width as usize), // UV interleaved: same row width in bytes
                height: (height / 2) as usize,
            };
            cuMemcpy2DAsync_v2(&uv_copy, stream)
        });

        if ret != CUDA_SUCCESS {
            return Err(CudaError::Import(
                crate::interop::ffi::cuda_driver::cu_err_to_string(ret),
            ));
        }

        // Step 3 — Synchronize before the texture is used by the render graph.
        // wgpu has no visibility into CUDA stream completion, so we must sync explicitly.
        // A future optimization would use exported CUDA semaphores for timeline sync.
        cuda_ctx.with_context(|stream| unsafe {
            cuStreamSynchronize(stream);
        });

        Ok(())
    }
}
