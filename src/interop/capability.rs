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
            log::warn!("InteropCapability::probe rejected: vendor is 0x{:X}, not NVIDIA (0x10DE)", info.vendor);
            return Self::none();
        }

        // Before calling any CUDA driver function we must verify that cuda.dll
        // is loadable. Our cuda.lib (from the FFmpeg dev package) was built against
        // cuda.dll, so the /DELAYLOAD resolver will look for *exactly* "cuda.dll".
        // Most consumer NVIDIA driver installs ship the CUDA runtime as "nvcuda.dll"
        // in System32, NOT as "cuda.dll". If "cuda.dll" is absent the first call to
        // cuInit() raises SEH 0xc06d007e (delay-load failure) and kills the process.
        // We probe here with LoadLibraryA so we can bail out gracefully instead.
        #[cfg(target_os = "windows")]
        {
            #[link(name = "kernel32")]
            extern "system" {
                fn LoadLibraryA(lp_lib_file_name: *const u8) -> *mut std::ffi::c_void;
                fn FreeLibrary(h_lib_module: *mut std::ffi::c_void) -> i32;
            }

            let h = unsafe { LoadLibraryA(b"cuda.dll\0".as_ptr()) };
            if h.is_null() {
                log::warn!("InteropCapability::probe rejected: cuda.dll not found (CUDA Toolkit not installed)");
                return Self::none();
            }
            unsafe { FreeLibrary(h) };
        }

        let ret = unsafe { crate::interop::ffi::cuda_driver::cuInit(0) };
        if ret != crate::interop::ffi::cuda_driver::CUDA_SUCCESS {
            log::warn!("InteropCapability::probe rejected: cuInit(0) failed with error code {:?}", ret);
            return Self::none();
        }

        let mut driver_version = 0;
        unsafe { crate::interop::ffi::cuda_driver::cuDriverGetVersion(&mut driver_version); }
        
        let min_version = match transport {
            InteropTransport::VulkanOpaqueFd => 4100,
            InteropTransport::D3D12Win32Handle => 11000,
            _ => 0,
        };
        if driver_version < min_version {
            log::warn!("InteropCapability::probe rejected: driver version {} < min version {}", driver_version, min_version);
            return Self::none();
        }

        let mut count = 0;
        unsafe { crate::interop::ffi::cuda_driver::cuDeviceGetCount(&mut count); }
        
        let mut cuda_device_ordinal = -1;
        for i in 0..count {
            let mut cu_dev = 0;
            if unsafe { crate::interop::ffi::cuda_driver::cuDeviceGet(&mut cu_dev, i) } == crate::interop::ffi::cuda_driver::CUDA_SUCCESS {
                cuda_device_ordinal = i;
                break;
            }
        }

        if cuda_device_ordinal == -1 {
            log::warn!("InteropCapability::probe rejected: could not get any CUDA devices");
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
