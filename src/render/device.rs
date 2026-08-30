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
    /// Whether the device supports TEXTURE_BINDING_ARRAY and SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
    pub has_binding_arrays: bool,
    /// Whether the device was created with `TIMESTAMP_QUERY`, i.e. whether
    /// `CommandEncoder::write_timestamp` may be called at all.
    ///
    /// Requested opportunistically: it is a profiling feature, so a device that
    /// lacks it must still work — every GPU-time column then prints `n/a`
    /// rather than a CPU-side number wearing a GPU label.
    pub has_timestamp_queries: bool,
}

impl GpuDevice {
    fn select_best_adapter(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Option<wgpu::Adapter> {
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        for adapter in &adapters {
            let info = adapter.get_info();
            log::info!("Found GPU Adapter: {} (Vendor: 0x{:X}, Backend: {:?})", info.name, info.vendor, info.backend);
        }

        let mut chosen_idx = None;

        // Pass 1: NVIDIA + Preferred backend (Dx12 on Windows, Vulkan on Linux)
        for (i, adapter) in adapters.iter().enumerate() {
            let info = adapter.get_info();
            if info.vendor == 0x10DE {
                let is_preferred = if cfg!(target_os = "windows") {
                    info.backend == wgpu::Backend::Dx12
                } else if cfg!(target_os = "linux") {
                    info.backend == wgpu::Backend::Vulkan
                } else {
                    true
                };
                if is_preferred {
                    if let Some(surf) = compatible_surface {
                        if adapter.is_surface_supported(surf) {
                            chosen_idx = Some(i);
                            break;
                        }
                    } else {
                        chosen_idx = Some(i);
                        break;
                    }
                }
            }
        }

        // Pass 2: NVIDIA + Any backend
        if chosen_idx.is_none() {
            for (i, adapter) in adapters.iter().enumerate() {
                let info = adapter.get_info();
                if info.vendor == 0x10DE {
                    if let Some(surf) = compatible_surface {
                        if adapter.is_surface_supported(surf) {
                            chosen_idx = Some(i);
                            break;
                        }
                    } else {
                        chosen_idx = Some(i);
                        break;
                    }
                }
            }
        }

        if let Some(idx) = chosen_idx {
            let mut adapters_mut = adapters;
            return Some(adapters_mut.remove(idx));
        }

        None
    }

    /// The preferred wgpu backend for this OS.
    /// On Windows we force DX12 because CUDA/NVENC interop uses D3D12Win32Handle —
    /// a Vulkan surface cannot be shared with a DX12 CUDA interop context.
    /// On Linux Vulkan is preferred for the same reason (VkExternalMemoryFd).
    fn preferred_backends() -> wgpu::Backends {
        if cfg!(target_os = "windows") {
            wgpu::Backends::DX12
        } else if cfg!(target_os = "linux") {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        }
    }

    /// Create a GpuDevice without any window (for headless/test use).
    pub async fn new_headless() -> Result<Self, DeviceError> {
        // Step 1: Create wgpu instance — use the OS-preferred backend so that
        // CUDA/NVENC interop works (DX12 on Windows, Vulkan on Linux).
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Self::preferred_backends(),
            ..Default::default()
        });

        // Step 2: Request adapter
        let chosen_adapter = Self::select_best_adapter(&instance, None);

        let adapter = match chosen_adapter {
            Some(a) => a,
            None => instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await.ok_or(DeviceError::NoAdapter)?
        };

        log::info!("Selected GPU Adapter: {} (Vendor: 0x{:X}, Backend: {:?})", adapter.get_info().name, adapter.get_info().vendor, adapter.get_info().backend);

        let adapter_features = adapter.features();
        let has_binding_arrays = adapter_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
            && adapter_features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

        let mut required_features = wgpu::Features::PUSH_CONSTANTS;
        if has_binding_arrays {
            required_features |= wgpu::Features::TEXTURE_BINDING_ARRAY
                               | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        }
        if adapter_features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }

        // Profiling only, so it is requested when offered and never required:
        // `GpuTimer` degrades to reporting nothing when this is false.
        let has_timestamp_queries = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if has_timestamp_queries {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        // Step 3: Request device and queue
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("video_engine_device"),
                required_features,
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
            has_binding_arrays,
            has_timestamp_queries,
        })
    }

    /// Create a GpuDevice with an OS window surface.
    pub async fn new_with_surface<'window>(
        target: wgpu::SurfaceTarget<'window>,
        window_width: u32,
        window_height: u32,
    ) -> Result<(Self, wgpu::Surface<'window>), DeviceError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Use OS-preferred backend so the surface and adapter share the same
            // backend, enabling CUDA/NVENC interop (DX12 on Windows).
            backends: Self::preferred_backends(),
            ..Default::default()
        });

        // The surface needs to be created before we request the adapter,
        // so we can request an adapter compatible with this surface.
        let surface = instance.create_surface(target).map_err(|_| DeviceError::SurfaceIncompatible)?;

        let chosen_adapter = Self::select_best_adapter(&instance, Some(&surface));

        let adapter = match chosen_adapter {
            Some(a) => a,
            None => instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            }).await.ok_or(DeviceError::NoAdapter)?
        };

        log::info!("Selected GPU Adapter: {} (Vendor: 0x{:X}, Backend: {:?})", adapter.get_info().name, adapter.get_info().vendor, adapter.get_info().backend);

        let adapter_features = adapter.features();
        let has_binding_arrays = adapter_features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
            && adapter_features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

        let mut required_features = wgpu::Features::PUSH_CONSTANTS;
        if has_binding_arrays {
            required_features |= wgpu::Features::TEXTURE_BINDING_ARRAY
                               | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
        }
        if adapter_features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }

        // Same opportunistic request as `new_headless` — the two constructors
        // build `required_features` independently, so a feature added to one and
        // not the other is a bug that only shows up in the UI or only in the
        // bench.
        let has_timestamp_queries = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if has_timestamp_queries {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("video_engine_device"),
                required_features,
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
            has_binding_arrays,
            has_timestamp_queries,
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

    /// Submit a completed CommandEncoder to the GPU queue. Returns the
    /// submission index so callers can use `poll(WaitForSubmissionIndex(idx))`
    /// to wait only for this specific batch of work rather than all pending work.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.queue.submit(std::iter::once(encoder.finish()))
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
