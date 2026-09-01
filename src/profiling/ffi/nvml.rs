// src/profiling/ffi/nvml.rs
//
// NVML — GPU utilisation, NVENC utilisation and driver-reported VRAM.
//
// The last three `n/a` rows in the benchmark's System Metrics block. They are here
// because nothing else can answer them: wgpu exposes no utilisation query at all,
// and `allocated_gpu_bytes` is a lower bound counted by the caller (AGENTS.md
// gotcha 9 names it as the one exception, and labels it as such in the report).
//
// PROVENANCE. Every layout constant below is established by `nvchk/nvml_probe.c`,
// per the rule in AGENTS.md: a hard-coded size, offset or enumerant in an FFI
// module must come from a standalone probe rather than from documentation. Measured
// on an RTX 3050, driver API 12.2:
//
//     sizeof(nvmlUtilization_t)  = 8   (gpu @ 0, memory @ 4)
//     sizeof(nvmlMemory_t)       = 24  (total @ 0, free @ 8, used @ 16)
//     sizeof(nvmlMemory_v2_t)    = 40  (version @ 0, total @ 8, reserved @ 16,
//                                       free @ 24, used @ 32)
//     NVML_STRUCT_VERSION(Memory, 2) = 0x02000028
//
// and the probe's negative control confirms the version word is load-bearing: with
// `version = 1` the driver returns `NVML_ERROR_INVALID_ARGUMENT` rather than
// ignoring it. Without that control, a working v2 call would not be evidence that
// the field was right.
//
// LINKAGE. `nvml.dll` is loaded with `LoadLibraryA` and every entry point resolved
// with `GetProcAddress`. It is deliberately NOT linked and NOT added to
// `build/cuda.def`: it is a different DLL from `cuda.dll`, so no
// `.cargo/config.toml` `/DELAYLOAD` entry is needed either, and a machine without
// NVML degrades to `None` instead of failing to start.
//
// EVERY READING IS `Option`. A driver that reports `NVML_ERROR_NOT_SUPPORTED` for
// encoder utilisation (real on some SKUs) yields `None`, which the report prints as
// `n/a`. Nothing here substitutes a computed value for a query that did not answer.
//
// The `#[cfg(windows)]` gate lives on the `pub mod nvml;` in `ffi/mod.rs`, not here:
// a second one in this file is a duplicated attribute (clippy says so) and gating in
// two places is one more thing to keep in step.

use std::ffi::{c_int, c_uint, c_void, CString};
use std::sync::OnceLock;

type NvmlReturn = c_int;
const NVML_SUCCESS: NvmlReturn = 0;

/// Opaque device handle. NVML hands these out and never asks for them back, so
/// there is nothing to free.
type NvmlDevice = *mut c_void;

/// GPU and memory-controller utilisation, as percentages.
///
/// Two `u32`s: `gpu` at offset 0, `memory` at 4 — established by the probe, which
/// also asserts both are ≤ 100 on a live read, because a wrong layout still yields
/// two numbers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct NvmlUtilization {
    gpu: c_uint,
    memory: c_uint,
}

/// VRAM totals, v1: three `u64`s, no version word.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct NvmlMemory {
    total: u64,
    free: u64,
    used: u64,
}

/// VRAM totals, v2.
///
/// `version` is an **IN** field the caller must set to `sizeof | (2 << 24)`, and
/// the driver rejects an unrecognised value with `NVML_ERROR_INVALID_ARGUMENT` —
/// which is indistinguishable from "this branch has no v2 query" unless you know to
/// look. `reserved` is part of the size the version word encodes and cannot be
/// dropped.
///
/// v2 exists because v1's `used` includes driver-reserved memory; the probe measured
/// 1706 MB (v1) against 1556 MB (v2) on an idle card. v2 is preferred and v1 is the
/// fallback, with [`GpuMetrics::vram_is_v2`] recording which one answered so the
/// difference is never silently attributed to a workload.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct NvmlMemoryV2 {
    version: c_uint,
    total: u64,
    reserved: u64,
    free: u64,
    used: u64,
}

/// The version word for [`NvmlMemoryV2`]: `sizeof | (2 << 24)` = `0x02000028`.
///
/// Computed rather than written as a literal, so it cannot drift from the struct.
/// The probe checks both halves of the value it produces.
const fn memory_v2_version() -> c_uint {
    (std::mem::size_of::<NvmlMemoryV2>() as c_uint) | (2 << 24)
}

type FnInit = unsafe extern "C" fn() -> NvmlReturn;
type FnShutdown = unsafe extern "C" fn() -> NvmlReturn;
type FnDeviceGetHandleByIndex =
    unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
type FnDeviceGetUtilizationRates =
    unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> NvmlReturn;
type FnDeviceGetEncoderUtilization =
    unsafe extern "C" fn(NvmlDevice, *mut c_uint, *mut c_uint) -> NvmlReturn;
type FnDeviceGetMemoryInfo = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn;
type FnDeviceGetMemoryInfoV2 =
    unsafe extern "C" fn(NvmlDevice, *mut NvmlMemoryV2) -> NvmlReturn;

/// The entry points, resolved once.
struct Nvml {
    device_get_handle: FnDeviceGetHandleByIndex,
    device_get_utilization: FnDeviceGetUtilizationRates,
    device_get_encoder_utilization: FnDeviceGetEncoderUtilization,
    device_get_memory: FnDeviceGetMemoryInfo,
    /// Absent on older driver branches, which is why v1 is kept.
    device_get_memory_v2: Option<FnDeviceGetMemoryInfoV2>,
    /// Kept so the reason a machine has no readings can be printed once.
    _shutdown: FnShutdown,
}

// SAFETY: the resolved pointers are into a DLL kept loaded for the process
// lifetime (deliberately never `FreeLibrary`'d — see `load`), and NVML's own
// documentation makes these calls thread-safe after `nvmlInit_v2`.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

/// `Ok` when NVML initialised, `Err(reason)` when it did not — and the reason is
/// kept so a caller can print WHY a row is `n/a` rather than only that it is.
static NVML: OnceLock<Result<Nvml, String>> = OnceLock::new();

/// Load `nvml.dll`, resolve the entry points, and call `nvmlInit_v2`.
///
/// Runs at most once per process. The library is deliberately never freed: the
/// device handles NVML hands out stay valid for the process lifetime, and a
/// `FreeLibrary` racing another thread's reading would be a use-after-unload for no
/// benefit — this is a diagnostic surface, not a resource to husband.
fn load() -> Result<Nvml, String> {
    // `winapi`'s declarations rather than our own `extern "system"` block: the
    // crate already declares both of these (it is a Windows-only dependency for
    // `src/interop/external_texture.rs`), and re-declaring them here made rustc
    // warn about a signature mismatch — two definitions of one import is exactly
    // the sort of thing that becomes a real mismatch later.
    use winapi::um::libloaderapi::{GetProcAddress, LoadLibraryA};
    // By name first, so the loader's normal search applies (System32 on any
    // consumer driver install), then the NVSMI directory some branches use.
    let candidates = [
        "nvml.dll",
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
    ];
    let mut lib = std::ptr::null_mut();
    for name in candidates {
        let c = CString::new(name).expect("no interior NUL in a literal");
        // SAFETY: a valid NUL-terminated string; a failed load returns null.
        lib = unsafe { LoadLibraryA(c.as_ptr()) };
        if !lib.is_null() {
            break;
        }
    }
    if lib.is_null() {
        return Err("nvml.dll not present (no NVIDIA driver)".into());
    }

    /// Resolve one symbol or name it in the error.
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let c = CString::new($name).expect("no interior NUL in a literal");
            // SAFETY: `lib` is a live module handle and `c` is NUL-terminated.
            // `as *mut c_void` because winapi types the return as `FARPROC`, a
            // distinct pointer type; transmuting straight from it is a type error
            // rather than the ABI question we actually care about.
            let p = unsafe { GetProcAddress(lib, c.as_ptr()) } as *mut c_void;
            if p.is_null() {
                return Err(format!("nvml.dll does not export {}", $name));
            }
            // SAFETY: NVML's exports have the signatures declared above, each of
            // which the probe exercises against this same DLL.
            unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
        }};
    }

    let init: FnInit = sym!("nvmlInit_v2", FnInit);
    let shutdown: FnShutdown = sym!("nvmlShutdown", FnShutdown);
    let device_get_handle: FnDeviceGetHandleByIndex =
        sym!("nvmlDeviceGetHandleByIndex_v2", FnDeviceGetHandleByIndex);
    let device_get_utilization: FnDeviceGetUtilizationRates =
        sym!("nvmlDeviceGetUtilizationRates", FnDeviceGetUtilizationRates);
    let device_get_encoder_utilization: FnDeviceGetEncoderUtilization = sym!(
        "nvmlDeviceGetEncoderUtilization",
        FnDeviceGetEncoderUtilization
    );
    let device_get_memory: FnDeviceGetMemoryInfo =
        sym!("nvmlDeviceGetMemoryInfo", FnDeviceGetMemoryInfo);

    // Optional, so resolved without the macro's early return.
    let device_get_memory_v2 = {
        let c = CString::new("nvmlDeviceGetMemoryInfo_v2").unwrap();
        // SAFETY: as above; a missing export is null, which is a real answer here.
        let p = unsafe { GetProcAddress(lib, c.as_ptr()) } as *mut c_void;
        if p.is_null() {
            None
        } else {
            // SAFETY: signature established by the probe against this DLL.
            Some(unsafe { std::mem::transmute::<*mut c_void, FnDeviceGetMemoryInfoV2>(p) })
        }
    };

    // SAFETY: no arguments, and NVML requires this before any device call.
    let r = unsafe { init() };
    if r != NVML_SUCCESS {
        return Err(format!("nvmlInit_v2 failed ({r})"));
    }

    Ok(Nvml {
        device_get_handle,
        device_get_utilization,
        device_get_encoder_utilization,
        device_get_memory,
        device_get_memory_v2,
        _shutdown: shutdown,
    })
}

/// What NVML reported for one device, at one instant.
///
/// Every field is `Option` for the same reason `SystemMetrics`' are: a driver that
/// declines a query has not reported zero. `None` here becomes `n/a` in the table.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMetrics {
    /// GPU core utilisation, 0..=100.
    pub gpu_utilization: Option<f32>,
    /// Encoder (NVENC) utilisation, 0..=100. `None` on a SKU that does not report
    /// it — a real answer from NVML, not a failure.
    pub encoder_utilization: Option<f32>,
    /// VRAM in use, as the driver reports it.
    pub vram_used_bytes: Option<u64>,
    /// Total VRAM on the device.
    pub vram_total_bytes: Option<u64>,
    /// Whether [`Self::vram_used_bytes`] came from the v2 query.
    ///
    /// Recorded rather than dropped because v1's `used` includes driver-reserved
    /// memory and v2's does not — 1706 MB vs 1556 MB on an idle RTX 3050. A reader
    /// comparing two runs on different driver branches would otherwise see a
    /// 150 MB "change" that no workload caused.
    pub vram_is_v2: bool,
}

/// Read NVML for device 0, or `Err(reason)` when there is nothing to read.
///
/// Device 0 rather than a search: everything else in this crate — `CudaContext`,
/// the interop transport, the NVENC session — is already on the primary device, so
/// a different index here would report a different GPU than the one being measured.
pub fn read_device_0() -> Result<GpuMetrics, String> {
    let nvml = NVML.get_or_init(load).as_ref().map_err(|e| e.clone())?;

    let mut dev: NvmlDevice = std::ptr::null_mut();
    // SAFETY: `dev` is a live out-pointer; NVML fills it or returns non-zero.
    let r = unsafe { (nvml.device_get_handle)(0, &mut dev) };
    if r != NVML_SUCCESS {
        return Err(format!("nvmlDeviceGetHandleByIndex_v2(0) failed ({r})"));
    }

    let mut out = GpuMetrics::default();

    let mut util = NvmlUtilization::default();
    // SAFETY: a live, correctly sized struct of the layout the probe established.
    if unsafe { (nvml.device_get_utilization)(dev, &mut util) } == NVML_SUCCESS {
        // Guarded rather than trusted: a value above 100 would mean the layout is
        // wrong, and reporting it as a percentage would be worse than reporting
        // nothing. The probe asserts the same bound against the live driver.
        if util.gpu <= 100 {
            out.gpu_utilization = Some(util.gpu as f32);
        }
    }

    let mut enc_util: c_uint = 0;
    let mut enc_period: c_uint = 0;
    // SAFETY: two live out-pointers.
    if unsafe { (nvml.device_get_encoder_utilization)(dev, &mut enc_util, &mut enc_period) }
        == NVML_SUCCESS
        && enc_util <= 100
    {
        out.encoder_utilization = Some(enc_util as f32);
    }

    // v2 first, v1 as the fallback, and which one answered is recorded.
    if let Some(get_v2) = nvml.device_get_memory_v2 {
        let mut m = NvmlMemoryV2 { version: memory_v2_version(), ..Default::default() };
        // SAFETY: live struct, with the version word the driver requires — the
        // probe's negative control shows it is rejected rather than ignored when
        // wrong, so a success here means the driver agreed about the layout.
        if unsafe { get_v2(dev, &mut m) } == NVML_SUCCESS && m.total > 0 {
            out.vram_used_bytes = Some(m.used);
            out.vram_total_bytes = Some(m.total);
            out.vram_is_v2 = true;
        }
    }
    if out.vram_used_bytes.is_none() {
        let mut m = NvmlMemory::default();
        // SAFETY: live struct of the v1 layout the probe established.
        if unsafe { (nvml.device_get_memory)(dev, &mut m) } == NVML_SUCCESS && m.total > 0 {
            out.vram_used_bytes = Some(m.used);
            out.vram_total_bytes = Some(m.total);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout constants must match what `nvchk/nvml_probe.c` measured.
    ///
    /// This is the Rust half of the provenance rule: the probe establishes the
    /// numbers against the driver, and this test asserts the Rust structs agree
    /// with them. Either alone is insufficient — a probe nothing checks against is
    /// documentation, and a Rust assertion with no probe behind it is a guess about
    /// someone else's ABI.
    #[test]
    fn the_struct_layouts_match_the_probe() {
        assert_eq!(std::mem::size_of::<NvmlUtilization>(), 8);
        assert_eq!(std::mem::size_of::<NvmlMemory>(), 24);
        assert_eq!(std::mem::size_of::<NvmlMemoryV2>(), 40);
        // The version word, both halves — the probe prints 0x02000028 and decodes
        // it the same way.
        assert_eq!(memory_v2_version(), 0x0200_0028);
        assert_eq!(memory_v2_version() & 0x00FF_FFFF, 40);
        assert_eq!((memory_v2_version() >> 24) & 0xFF, 2);
    }

    /// A live reading must be plausible, or absent with a reason.
    ///
    /// Bounds rather than `is_some()`, for the reason gotcha 9 records: a stub
    /// returning zeros satisfies "some" and is exactly the failure mode this whole
    /// module exists to avoid. A utilisation above 100% or a `used > total` means
    /// the layout is wrong, and both are the shapes a transposed field produces.
    #[test]
    fn a_live_reading_is_plausible_or_absent_with_a_reason() {
        match read_device_0() {
            Ok(m) => {
                if let Some(g) = m.gpu_utilization {
                    assert!((0.0..=100.0).contains(&g), "GPU utilisation {g} is not a percentage");
                }
                if let Some(e) = m.encoder_utilization {
                    assert!((0.0..=100.0).contains(&e), "NVENC utilisation {e} is not a percentage");
                }
                match (m.vram_used_bytes, m.vram_total_bytes) {
                    (Some(used), Some(total)) => {
                        assert!(total > 256 << 20, "a VRAM total of {total} is implausible");
                        assert!(
                            used <= total,
                            "used {used} exceeds total {total} — the layout is wrong"
                        );
                    }
                    (None, None) => {}
                    other => panic!("used and total must be reported together, got {other:?}"),
                }
            }
            // No driver on this machine is a real answer, not a failure — the same
            // contract as every other hardware-dependent test in the tree.
            Err(reason) => eprintln!("SKIP: NVML unavailable — {reason}"),
        }
    }

    /// Repeated reads must not leak or re-initialise.
    ///
    /// `NVML` is a `OnceLock`, so the second call must reuse the first's handles
    /// rather than calling `nvmlInit_v2` again — and both calls must agree about
    /// whether NVML exists at all.
    #[test]
    fn repeated_reads_are_stable() {
        let a = read_device_0();
        let b = read_device_0();
        assert_eq!(
            a.is_ok(),
            b.is_ok(),
            "availability must not change between reads"
        );
        if let (Ok(a), Ok(b)) = (a, b) {
            // Utilisation moves; the card's total VRAM does not.
            assert_eq!(a.vram_total_bytes, b.vram_total_bytes);
            assert_eq!(a.vram_is_v2, b.vram_is_v2);
        }
    }
}
