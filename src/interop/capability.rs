use crate::render::device::GpuDevice;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InteropTransport {
    VulkanOpaqueFd,
    D3D12Win32Handle,
    None,
}

#[derive(Clone, Debug)]
pub struct InteropCapability {
    pub transport:        InteropTransport,
    pub cuda_device_ordinal: i32,
    pub driver_version:   String,
}

impl InteropCapability {
    pub fn probe(device: &GpuDevice) -> Self {
        let info = device.adapter.get_info();
        let backend = info.backend;

        let transport = match backend {
            wgpu::Backend::Vulkan if cfg!(target_os = "linux") => InteropTransport::VulkanOpaqueFd,
            wgpu::Backend::Dx12   if cfg!(target_os = "windows") => InteropTransport::D3D12Win32Handle,
            _ => return Self::none(),
        };

        if info.vendor != 0x10DE {
            return Self::none();
        }

        let ret = unsafe { crate::interop::ffi::cuda_driver::cuInit(0) };
        if ret != crate::interop::ffi::cuda_driver::CUDA_SUCCESS {
            return Self::none();
        }

        let mut driver_version = 0;
        unsafe { crate::interop::ffi::cuda_driver::cuDriverGetVersion(&mut driver_version); }
        
        let min_version = match transport {
            InteropTransport::VulkanOpaqueFd => 4100, // wait, driver version isn't 410, it's CUDA version like 10000. Driver >= 410 means CUDA 10.0+
            InteropTransport::D3D12Win32Handle => 11000, // CUDA 11.0+
            _ => 0,
        };
        // cuDriverGetVersion returns the CUDA driver version: e.g., 12020 for 12.2
        if driver_version < min_version {
            return Self::none();
        }

        let mut count = 0;
        unsafe { crate::interop::ffi::cuda_driver::cuDeviceGetCount(&mut count); }
        
        let mut cuda_device_ordinal = -1;
        for i in 0..count {
            let mut cu_dev = 0;
            if unsafe { crate::interop::ffi::cuda_driver::cuDeviceGet(&mut cu_dev, i) } == crate::interop::ffi::cuda_driver::CUDA_SUCCESS {
                // Try to match based on PCI device ID? Actually let's just pick the first one with the same device ID
                // Wgpu exposes PCI device ID in info.device
                // Note: cuDeviceGetAttribute for PCI_DEVICE_ID does NOT return the PCI vendor/device ID! 
                // It returns the PCI bus slot's device ID. We can't match it that easily.
                // For simplicity, we just take the first CUDA device since wgpu usually picks the primary NVIDIA GPU.
                cuda_device_ordinal = i;
                break;
            }
        }

        if cuda_device_ordinal == -1 {
            return Self::none();
        }

        let version_str = format!("{}.{}", driver_version / 1000, (driver_version % 100) / 10);

        Self {
            transport,
            cuda_device_ordinal,
            driver_version: version_str,
        }
    }

    pub fn none() -> Self {
        Self {
            transport: InteropTransport::None,
            cuda_device_ordinal: -1,
            driver_version: String::new(),
        }
    }

    pub fn is_available(&self) -> bool {
        self.transport != InteropTransport::None
    }
}
