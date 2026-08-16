use nexir::interop::capability::{InteropCapability, InteropTransport};
use nexir::interop::cuda_context::CudaContext;
use nexir::interop::ffi::nvenc::*;

type OpenEncodeSessionEx =
    unsafe extern "C" fn(*const NvEncOpenEncodeSessionExParams, *mut NvEncodeSession) -> i32;

fn main() {
    println!("=== Testing NVENC API Initialization ===");

    // Initialize CUDA driver
    unsafe {
        let ret = nexir::interop::ffi::cuda_driver::cuInit(0);
        if ret != nexir::interop::ffi::cuda_driver::CUDA_SUCCESS {
            println!("Failed to initialize CUDA driver: {:?}", ret);
            return;
        }
    }

    // Mock InteropCapability for CUDA initialization
    let cap = InteropCapability {
        transport: InteropTransport::D3D12Win32Handle,
        cuda_device_ordinal: 0,
        driver_version: "530".to_string(),
    };

    // Initialize CUDA
    let cuda_ctx = match CudaContext::new(&cap) {
        Ok(ctx) => {
            println!("CUDA Context created successfully on device 0.");
            ctx
        }
        Err(e) => {
            println!("Failed to create CUDA Context: {:?}", e);
            return;
        }
    };

    // Load the function table
    let fn_table_size = std::mem::size_of::<usize>() * 64;
    let mut fn_table_raw = vec![0u8; fn_table_size];
    let mut success = false;
    let mut discovered_major = 0;

    for major_ver in (8..=15).rev() {
        let version = (major_ver as u32) | (2 << 16) | (0x7 << 28);
        unsafe {
            let ptr = fn_table_raw.as_mut_ptr() as *mut u32;
            *ptr = version;
        }
        let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;
        let ret = unsafe { NvEncodeAPICreateInstance(function_list) };
        if ret == NV_ENC_SUCCESS {
            success = true;
            discovered_major = major_ver;
            println!(
                "NvEncodeAPICreateInstance SUCCEEDED with major version: {}",
                major_ver
            );
            break;
        }
    }

    if !success {
        println!("NvEncodeAPICreateInstance failed for all major versions 8..=15.");
        return;
    }

    // Cast function list
    let function_list = fn_table_raw.as_ptr() as NV_ENCODE_API_FUNCTION_LIST;
    let base = function_list as *const usize;

    // We only need open_session for this test
    let open_session_ptr = unsafe { *base.add(1) };
    if open_session_ptr == 0 {
        println!("Error: open_session function pointer is NULL!");
        return;
    }

    let open_session: OpenEncodeSessionEx = unsafe { std::mem::transmute(open_session_ptr) };

    println!("Attempting to call OpenEncodeSessionEx...");

    // Let's try different combinations of version and api_version
    // 1. Original hardcoded values
    let orig_ver = (2 << 16) | 0x6001001;
    let orig_api = 14;

    let mut session: NvEncodeSession = std::ptr::null_mut();
    let params = NvEncOpenEncodeSessionExParams {
        version: orig_ver,
        device_type: NV_ENC_DEVICE_TYPE_CUDA,
        device: cuda_ctx.raw_context() as *mut _,
        reserved: std::ptr::null_mut(),
        api_version: orig_api,
        reserved1: [0u32; 253],
        reserved2: [std::ptr::null_mut(); 64],
    };

    let ret = unsafe { open_session(&params, &mut session) };
    println!(
        "Test 1 (Original: struct_ver=0x{:X}, api_ver={}): ret = {}",
        orig_ver, orig_api, ret
    );

    // 2. Discover version-based values
    // NVENCAPI_STRUCT_VERSION(ver) = (NVENCAPI_VERSION | (ver << 16) | (0x7 << 28))
    for test_major in (8..=15).rev() {
        let test_api = test_major;
        // Try struct version 1 and 2
        for struct_ver in 1..=2 {
            let test_struct_ver = test_api | (struct_ver << 16) | (0x7 << 28);

            let mut test_session: NvEncodeSession = std::ptr::null_mut();
            let test_params = NvEncOpenEncodeSessionExParams {
                version: test_struct_ver,
                device_type: NV_ENC_DEVICE_TYPE_CUDA,
                device: cuda_ctx.raw_context() as *mut _,
                reserved: std::ptr::null_mut(),
                api_version: test_api,
                reserved1: [0u32; 253],
                reserved2: [std::ptr::null_mut(); 64],
            };

            let ret = unsafe { open_session(&test_params, &mut test_session) };
            if ret == NV_ENC_SUCCESS {
                println!("--> Test SUCCESS: test_major={}, struct_ver={} (struct_ver_val=0x{:X}): ret = {}", test_major, struct_ver, test_struct_ver, ret);
            } else {
                // Filter out unsupported device/no device if possible
                if ret != 4 && ret != 1 {
                    println!(
                        "Test (major={}, struct_ver={}): ret = {}",
                        test_major, struct_ver, ret
                    );
                }
            }
        }
    }
}
