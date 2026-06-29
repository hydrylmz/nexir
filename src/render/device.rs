// src/render/device.rs

use std::sync::{Arc, Mutex};

/// Owns the wgpu instance, adapter, device, and queue.
/// Clone-safe via Arc internals. Pass `&GpuDevice` everywhere — never clone the device.
pub struct GpuDevice {
    pub instance: wgpu::Instance,
    pub adapter:  wgpu::Adapter,
    pub device:   Arc<wgpu::Device>,
    pub queue:    Arc<wgpu::Queue>,
    /// Best texture format supported by the adapter for RGBA16Float rendering.
    pub hdr_format: wgpu::TextureFormat,
    /// Surface format (the swapchain's pixel format, always SDR).
    /// Uses Mutex for interior mutability so GpuDevice can be shared behind Arc across threads.
    pub surface_format: Mutex<wgpu::TextureFormat>,
}

impl GpuDevice {
    /// Create a GpuDevice without any window (for headless/test use).
    pub async fn new_headless() -> Result<Self, DeviceError> {
        // Step 1: Create wgpu instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Step 2: Request adapter
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.ok_or(DeviceError::NoAdapter)?;

        // Step 3: Request device and queue
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("video_engine_device"),
                required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                                 | wgpu::Features::PUSH_CONSTANTS
                                 | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..wgpu::Limits::default()
                },
            },
            None,
        ).await.map_err(DeviceError::NoDevice)?;

        // Step 4: Detect hdr_format
        let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        let hdr_format = if format_features.allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING) {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        // Step 5: surface_format placeholder
        let surface_format = wgpu::TextureFormat::Bgra8UnormSrgb;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            hdr_format,
            surface_format: Mutex::new(surface_format),
        })
    }

    /// Create a GpuDevice with an OS window surface.
    pub async fn new_with_surface<'window>(
        target: wgpu::SurfaceTarget<'window>,
        window_width: u32,
        window_height: u32,
    ) -> Result<(Self, wgpu::Surface<'window>), DeviceError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // The surface needs to be created before we request the adapter,
        // so we can request an adapter compatible with this surface.
        let surface = instance.create_surface(target).map_err(|_| DeviceError::SurfaceIncompatible)?;

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or(DeviceError::NoAdapter)?;

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("video_engine_device"),
                required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                                 | wgpu::Features::PUSH_CONSTANTS
                                 | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..wgpu::Limits::default()
                },
            },
            None,
        ).await.map_err(DeviceError::NoDevice)?;

        let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        let hdr_format = if format_features.allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING) {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        // Initialize our struct with a dummy surface format, then call configure_surface to set it properly
        let gpu_device = Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            hdr_format,
            surface_format: Mutex::new(wgpu::TextureFormat::Bgra8UnormSrgb),
        };

        gpu_device.configure_surface(&surface, window_width, window_height);

        Ok((gpu_device, surface))
    }

    /// Configure (or reconfigure) the swapchain for a given surface and size.
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface,
        width:   u32,
        height:  u32,
    ) {
        // Step 1: Get surface formats
        let capabilities = surface.get_capabilities(&self.adapter);
        let chosen_format = capabilities.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(capabilities.formats[0]);

        // Step 2: Configure surface
        surface.configure(&self.device, &wgpu::SurfaceConfiguration {
            usage:        wgpu::TextureUsages::RENDER_ATTACHMENT,
            format:       chosen_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode:   wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        });

        // Step 3: Update format
        *self.surface_format.lock().unwrap() = chosen_format;
    }

    /// Begin a frame: create a CommandEncoder.
    pub fn begin_frame(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame_encoder"),
        })
    }

    /// Submit a completed CommandEncoder to the GPU queue.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Allocate a GPU texture with standard parameters for this engine.
    pub fn create_texture(
        &self,
        label:  Option<&str>,
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage:  wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }
}

#[derive(Debug)]
pub enum DeviceError {
    NoAdapter,
    NoDevice(wgpu::RequestDeviceError),
    SurfaceIncompatible,
}
