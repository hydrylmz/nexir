use nexir::interop::capability::InteropCapability;
use nexir::interop::ffi::cuda_driver::*;
use nexir::render::device::GpuDevice;

fn main() {
    let device = pollster::block_on(GpuDevice::new_headless()).unwrap();

    println!("Probing InteropCapability...");
    let cap = InteropCapability::probe(&device);
    println!("InteropCapability available: {}", cap.is_available());
    println!("InteropCapability transport: {:?}", cap.transport);

    println!("\nDetailed CUDA Probe:");
    unsafe {
        let res = cuInit(0);
        println!("cuInit(0) = {:?}", res);

        let mut count = 0;
        let res2 = cuDeviceGetCount(&mut count);
        println!("cuDeviceGetCount() = {:?}, count = {}", res2, count);

        if count > 0 {
            let mut dev = 0;
            let res3 = cuDeviceGet(&mut dev, 0);
            println!("cuDeviceGet(0) = {:?}, dev = {}", res3, dev);

            // Query compute capability as a proxy for "which GPU"
            let mut major = 0i32;
            let mut minor = 0i32;
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
            cuDeviceGetAttribute(&mut major, 75, dev);
            cuDeviceGetAttribute(&mut minor, 76, dev);
            println!("Compute capability: {}.{}", major, minor);
        }
    }
}
