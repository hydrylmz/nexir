use nexir::interop::capability::InteropCapability;
use nexir::interop::ffi::cuda::*;
use std::ffi::CStr;

fn main() {
    println!("Probing InteropCapability...");
    let cap = InteropCapability::probe();
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

            let mut name = [0i8; 128];
            let res4 = cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, dev);
            if res4 == CUresult::CUDA_SUCCESS {
                let c_str = CStr::from_ptr(name.as_ptr());
                println!("GPU Name: {:?}", c_str.to_string_lossy());
            }
        }
    }
}
