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

    #[cfg(not(target_os = "windows"))]
    println!("cargo:rustc-link-lib=nvidia-encode");

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    println!("cargo:rustc-link-lib=cuda");

    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-search=C:/Program Files/NVIDIA Corporation/CUDA/v12.0/lib/x64");
        link_cuda_windows();
        link_nvenc_windows();
    }
}

/// Link the CUDA driver on Windows, building the import library from
/// `build/cuda.def` when possible.
///
/// There is no `cuda.lib` on a machine with only the consumer NVIDIA driver: the
/// runtime ships as `nvcuda.dll`, and `cuda.lib` comes with the CUDA Toolkit.
/// What this project linked against instead was a stub import library sitting in
/// `C:/ffmpeg/lib`, generated once from a `.def` that existed on one developer's
/// disk and listed exactly the twenty symbols the crate used that day.  Declaring
/// a twenty-first CUDA function in Rust then failed at LINK with `unresolved
/// external symbol __imp_<name>`, and nothing in the repository explained why.
///
/// Generating it here from a checked-in `.def` makes the symbol list part of the
/// source tree.  The generated library is deliberately NOT called `cuda.lib`:
/// `C:/ffmpeg/lib` is already on the link-search path and appears there first, so
/// a same-named file would be shadowed by the stale one.  It is `cudaimp.lib`,
/// linked by that name, and its `LIBRARY cuda.dll` line means the import records
/// still resolve against `cuda.dll` exactly as before.
///
/// Falls back to `-l cuda` with a warning when `lib.exe` is unavailable, so a
/// machine that has a real Toolkit `cuda.lib` keeps building.
#[cfg(target_os = "windows")]
fn link_cuda_windows() {
    println!("cargo:rerun-if-changed=build/cuda.def");

    let fall_back = |reason: String| {
        println!("cargo:warning={reason}");
        println!("cargo:rustc-link-lib=cuda");
    };

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set for a build script");
    let def_path = std::path::Path::new("build/cuda.def");
    if !def_path.exists() {
        fall_back("build/cuda.def is missing; linking any prebuilt cuda.lib instead".into());
        return;
    }

    let target = std::env::var("TARGET").unwrap_or_default();
    // `find` hands back a ready std::process::Command with the MSVC environment
    // already applied (`find_tool` would return a Tool needing `.to_command()`).
    let Some(mut cmd) = cc::windows_registry::find(&target, "lib.exe") else {
        fall_back(format!(
            "lib.exe not found for target {target}; cannot build the CUDA import library \
             from build/cuda.def, linking any prebuilt cuda.lib instead"
        ));
        return;
    };

    let machine = if target.starts_with("i686") {
        "X86"
    } else {
        "X64"
    };
    let out_lib = std::path::Path::new(&out_dir).join("cudaimp.lib");

    cmd.arg("/NOLOGO")
        .arg(format!("/DEF:{}", def_path.display()))
        .arg(format!("/MACHINE:{machine}"))
        .arg(format!("/OUT:{}", out_lib.display()));

    match cmd.output() {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=cudaimp");
        }
        Ok(output) => fall_back(format!(
            "lib.exe failed to build the CUDA import library from build/cuda.def ({}): {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
        )),
        Err(e) => fall_back(format!("could not run lib.exe: {e}")),
    }
}

/// Generate the one-symbol NVENC import library used by the FFI declarations.
#[cfg(target_os = "windows")]
fn link_nvenc_windows() {
    println!("cargo:rerun-if-changed=build/nvidia-encode.def");
    let target = std::env::var("TARGET").unwrap_or_default();
    let Some(mut cmd) = cc::windows_registry::find(&target, "lib.exe") else {
        println!("cargo:warning=lib.exe not found; linking any prebuilt nvidia-encode.lib instead");
        println!("cargo:rustc-link-lib=nvidia-encode");
        return;
    };
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is always set for a build script");
    let def_path = std::path::Path::new("build/nvidia-encode.def");
    let out_lib = std::path::Path::new(&out_dir).join("nvidia-encode.lib");
    let machine = if target.starts_with("i686") {
        "X86"
    } else {
        "X64"
    };
    cmd.arg("/NOLOGO")
        .arg(format!("/DEF:{}", def_path.display()))
        .arg(format!("/MACHINE:{machine}"))
        .arg(format!("/OUT:{}", out_lib.display()));
    match cmd.output() {
        Ok(output) if output.status.success() => {
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=nvidia-encode");
        }
        Ok(output) => panic!(
            "lib.exe failed to build nvidia-encode.lib ({}): {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
        ),
        Err(error) => panic!("could not run lib.exe for nvidia-encode.lib: {error}"),
    }
}
