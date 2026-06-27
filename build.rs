// build.rs
// Instructs cargo to link against the system FFmpeg libraries.
fn main() {
    println!("cargo:rustc-link-lib=avformat");
    println!("cargo:rustc-link-lib=avcodec");
    println!("cargo:rustc-link-lib=avutil");
    println!("cargo:rustc-link-lib=swscale");
    println!("cargo:rustc-link-lib=swresample");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shim.c");

    cc::Build::new()
        .file("src/shim.c")
        .include("C:/ffmpeg/include")
        .compile("shim");

    // On Windows: set the search path to where FFmpeg libs are installed.
    // Supports the VE_FFMPEG_LIB_DIR env var for CI overriding.
    #[cfg(target_os = "windows")]
    {
        if let Ok(dir) = std::env::var("VE_FFMPEG_LIB_DIR") {
            println!("cargo:rustc-link-search={dir}");
        } else {
            // Common Chocolatey / MSYS2 / vcpkg locations
            for dir in &[
                "C:/ffmpeg/lib",
                "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin",
                "C:/msys64/mingw64/lib",
            ] {
                if std::path::Path::new(dir).exists() {
                    println!("cargo:rustc-link-search={dir}");
                    break;
                }
            }
        }
    }

    // On macOS (Homebrew):
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search=/opt/homebrew/lib");
        println!("cargo:rustc-link-search=/usr/local/lib");
    }

    // On Linux: prefer pkg-config for portability.
    // The FFI layer compiles regardless; linking happens at final link time.

    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=nvidia-encode");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    #[cfg(target_os = "windows")]
    println!("cargo:rustc-link-search=C:/Program Files/NVIDIA Corporation/CUDA/v12.0/lib/x64");
}
