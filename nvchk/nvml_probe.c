// nvchk/nvml_probe.c — what NVML's structs actually look like, and what it reports.
//
// WHY THIS EXISTS. `src/profiling/ffi/nvml.rs` hard-codes struct sizes and the
// layout of `nvmlUtilization_t` / `nvmlMemory_v2_t`. AGENTS.md's rule for the tree
// is that a hard-coded offset, size, `_VER` word or enumerant in an FFI module must
// be established by a standalone probe rather than by a comment citing
// documentation — an unfalsifiable layout claim is what turned an out-of-scope
// pitch measurement into a process kill (see README.md and nv12_pitch_probe.c).
//
// NVML is a softer target than NVENC: there is no versioned struct with a `_VER`
// word to get wrong, and the two structs this probe cares about are small. But
// `nvmlMemory_v2_t` carries a `version` field that the caller MUST set before the
// call, and `nvmlDeviceGetMemoryInfo_v2` returns
// `NVML_ERROR_INVALID_ARGUMENT` — not a layout crash — when it is wrong, so getting
// it wrong looks exactly like "this driver has no v2 memory query". That is the
// specific confusion this probe removes.
//
// DELIBERATELY INDEPENDENT OF THE CARGO BUILD, like every probe here: `nvml.dll` is
// opened with LoadLibraryA and every entry point resolved with GetProcAddress, so
// nothing about build.rs, build/cuda.def or the /DELAYLOAD flags can influence the
// result. Note that unlike cuda.dll, NVML needs NO import library and NO delay-load
// entry anywhere — it is a different DLL from cuda.dll and is not in build/cuda.def.
//
// Build:  ./build_probes.sh      (or: gcc -O1 -o nvml_probe.exe nvml_probe.c)
// Run:    ./nvml_probe.exe
//
// Exit 0 when every layout claim held and a live reading came back; 1 otherwise.
// No NVIDIA driver is not a failure of the claims — it prints a skip and exits 0,
// mirroring "missing hardware is a printed skip, never a zero row".

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <windows.h>

// ---------------------------------------------------------------------------
// The NVML types, declared here rather than included.
//
// nvml.h ships with the CUDA Toolkit, which this repo deliberately does not
// require (build.rs generates its import library from a .def file for exactly that
// reason). So the probe declares what it believes the ABI to be and then CHECKS
// those beliefs against the driver — which is the stronger test anyway: including
// the vendor header would make the sizes agree by construction and prove nothing
// about the Rust side, which also declares them by hand.
// ---------------------------------------------------------------------------

typedef int nvmlReturn_t;
#define NVML_SUCCESS                    0
#define NVML_ERROR_UNINITIALIZED        1
#define NVML_ERROR_INVALID_ARGUMENT     2
#define NVML_ERROR_NOT_SUPPORTED        3
#define NVML_ERROR_NOT_FOUND            6
#define NVML_ERROR_FUNCTION_NOT_FOUND   13

typedef void *nvmlDevice_t;

// Two unsigned ints, in this order: GPU core, then memory-controller. The order
// matters and is not self-evident — swapping them yields a plausible-looking
// number, which is why the live reading below prints both.
typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

// The v1 memory struct: three unsigned long longs, no version word.
typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

// The v2 memory struct. `version` is an IN field the caller sets to
// `sizeof(nvmlMemory_v2_t) | (2 << 24)`; the driver rejects the call outright when
// it does not recognise the value. `reserved` exists in the vendor header and is
// part of the size the version word encodes, so it cannot be dropped.
typedef struct {
    unsigned int version;
    unsigned long long total;
    unsigned long long reserved;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_v2_t;

// The version word, spelled the way the vendor header's macro spells it.
#define NVML_STRUCT_VERSION(data, ver) \
    (unsigned int)(sizeof(nvml##data##_v##ver##_t) | (ver << 24u))

typedef nvmlReturn_t (*PFN_nvmlInit_v2)(void);
typedef nvmlReturn_t (*PFN_nvmlShutdown)(void);
typedef const char * (*PFN_nvmlErrorString)(nvmlReturn_t);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetCount_v2)(unsigned int *);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetEncoderUtilization)(nvmlDevice_t, unsigned int *, unsigned int *);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*PFN_nvmlDeviceGetMemoryInfo_v2)(nvmlDevice_t, nvmlMemory_v2_t *);

static int failures = 0;

static void check_eq(const char *what, size_t got, size_t want)
{
    if (got == want) {
        printf("  ok    %-46s %zu\n", what, got);
    } else {
        printf("  FAIL  %-46s %zu (expected %zu)\n", what, got, want);
        failures++;
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("nvml_probe — NVML struct layout and a live reading\n\n");

    // ── Layout, before touching the driver ────────────────────────────────
    //
    // These are the numbers `src/profiling/ffi/nvml.rs` hard-codes. Printed as
    // sizes AND offsets: a struct can have the right total size with two fields
    // transposed, and a transposed `free`/`used` reads as a plausible VRAM figure.
    printf("Layout (x86-64, MSVC/MinGW ABI):\n");
    check_eq("sizeof(nvmlUtilization_t)", sizeof(nvmlUtilization_t), 8);
    check_eq("offsetof(nvmlUtilization_t, gpu)", offsetof(nvmlUtilization_t, gpu), 0);
    check_eq("offsetof(nvmlUtilization_t, memory)", offsetof(nvmlUtilization_t, memory), 4);

    check_eq("sizeof(nvmlMemory_t)", sizeof(nvmlMemory_t), 24);
    check_eq("offsetof(nvmlMemory_t, total)", offsetof(nvmlMemory_t, total), 0);
    check_eq("offsetof(nvmlMemory_t, free)", offsetof(nvmlMemory_t, free), 8);
    check_eq("offsetof(nvmlMemory_t, used)", offsetof(nvmlMemory_t, used), 16);

    // 4-byte version + 4 bytes of padding, then four 8-byte fields = 40.
    check_eq("sizeof(nvmlMemory_v2_t)", sizeof(nvmlMemory_v2_t), 40);
    check_eq("offsetof(nvmlMemory_v2_t, version)", offsetof(nvmlMemory_v2_t, version), 0);
    check_eq("offsetof(nvmlMemory_v2_t, total)", offsetof(nvmlMemory_v2_t, total), 8);
    check_eq("offsetof(nvmlMemory_v2_t, reserved)", offsetof(nvmlMemory_v2_t, reserved), 16);
    check_eq("offsetof(nvmlMemory_v2_t, free)", offsetof(nvmlMemory_v2_t, free), 24);
    check_eq("offsetof(nvmlMemory_v2_t, used)", offsetof(nvmlMemory_v2_t, used), 32);

    unsigned int mem_v2_version = NVML_STRUCT_VERSION(Memory, 2);
    printf("  info  %-46s 0x%08X (= sizeof|2<<24 = %u|0x2000000)\n",
           "NVML_STRUCT_VERSION(Memory, 2)", mem_v2_version,
           (unsigned)sizeof(nvmlMemory_v2_t));
    check_eq("  ...decoded size half", mem_v2_version & 0x00FFFFFFu, 40);
    check_eq("  ...decoded version half", (mem_v2_version >> 24) & 0xFFu, 2);

    // ── The driver ────────────────────────────────────────────────────────
    printf("\nDriver:\n");
    // System32 by name only — no path, so the loader's normal search applies, the
    // same as the Rust side does.
    HMODULE lib = LoadLibraryA("nvml.dll");
    if (!lib) {
        // The vendor also installs it here on some driver branches.
        lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    }
    if (!lib) {
        printf("  SKIP: nvml.dll not present — no NVIDIA driver on this machine.\n");
        printf("\n%s\n", failures ? "LAYOUT FAILURES ABOVE" : "layout claims held; no live reading taken");
        return failures ? 1 : 0;
    }
    printf("  ok    nvml.dll loaded\n");

#define RESOLVE(var, type, name)                                            \
    type var = (type)(void *)GetProcAddress(lib, name);                     \
    if (!var) {                                                             \
        printf("  FAIL  %s not exported\n", name);                          \
        failures++;                                                         \
    } else {                                                                \
        printf("  ok    %s resolved\n", name);                              \
    }

    RESOLVE(pInit, PFN_nvmlInit_v2, "nvmlInit_v2")
    RESOLVE(pShutdown, PFN_nvmlShutdown, "nvmlShutdown")
    RESOLVE(pErrStr, PFN_nvmlErrorString, "nvmlErrorString")
    RESOLVE(pCount, PFN_nvmlDeviceGetCount_v2, "nvmlDeviceGetCount_v2")
    RESOLVE(pHandle, PFN_nvmlDeviceGetHandleByIndex_v2, "nvmlDeviceGetHandleByIndex_v2")
    RESOLVE(pName, PFN_nvmlDeviceGetName, "nvmlDeviceGetName")
    RESOLVE(pUtil, PFN_nvmlDeviceGetUtilizationRates, "nvmlDeviceGetUtilizationRates")
    RESOLVE(pEncUtil, PFN_nvmlDeviceGetEncoderUtilization, "nvmlDeviceGetEncoderUtilization")
    RESOLVE(pMem, PFN_nvmlDeviceGetMemoryInfo, "nvmlDeviceGetMemoryInfo")
    // v2 is genuinely optional: older driver branches export only v1, and the
    // Rust side falls back. So a missing v2 is INFO, not a failure.
    PFN_nvmlDeviceGetMemoryInfo_v2 pMem2 =
        (PFN_nvmlDeviceGetMemoryInfo_v2)(void *)GetProcAddress(lib, "nvmlDeviceGetMemoryInfo_v2");
    printf("  %s nvmlDeviceGetMemoryInfo_v2 %s\n",
           pMem2 ? "ok   " : "info ",
           pMem2 ? "resolved" : "NOT exported (older branch; v1 fallback applies)");

    if (!pInit || !pCount || !pHandle || !pUtil || !pEncUtil || !pMem) {
        printf("\nFAILED: a required entry point is missing.\n");
        return 1;
    }

    const char *(*errstr)(nvmlReturn_t) = pErrStr;
#define ESTR(r) (errstr ? errstr(r) : "?")

    nvmlReturn_t r = pInit();
    if (r != NVML_SUCCESS) {
        printf("  SKIP: nvmlInit_v2 failed (%d: %s) — driver present but not usable.\n",
               r, ESTR(r));
        FreeLibrary(lib);
        return failures ? 1 : 0;
    }
    printf("  ok    nvmlInit_v2\n");

    unsigned int count = 0;
    r = pCount(&count);
    printf("  %s nvmlDeviceGetCount_v2 -> %u (%d: %s)\n",
           r == NVML_SUCCESS ? "ok   " : "FAIL ", count, r, ESTR(r));
    if (r != NVML_SUCCESS || count == 0) {
        printf("  SKIP: no NVML devices.\n");
        pShutdown();
        FreeLibrary(lib);
        return failures ? 1 : 0;
    }

    nvmlDevice_t dev = NULL;
    r = pHandle(0, &dev);
    if (r != NVML_SUCCESS) {
        printf("  FAIL  nvmlDeviceGetHandleByIndex_v2(0) (%d: %s)\n", r, ESTR(r));
        pShutdown();
        FreeLibrary(lib);
        return 1;
    }
    char name[96] = {0};
    if (pName && pName(dev, name, sizeof(name)) == NVML_SUCCESS) {
        printf("  ok    device 0: %s\n", name);
    }

    // ── Live readings ─────────────────────────────────────────────────────
    //
    // The point of printing them: a utilisation struct read through a wrong layout
    // still yields two numbers. These have to be sanity-checkable — percentages in
    // 0..=100, and a VRAM total that matches the card.
    printf("\nLive readings:\n");

    nvmlUtilization_t u;
    memset(&u, 0xAB, sizeof(u)); // poison, so an untouched field is obvious
    r = pUtil(dev, &u);
    if (r == NVML_SUCCESS) {
        printf("  ok    utilisation: gpu=%u%%  memory-controller=%u%%\n", u.gpu, u.memory);
        if (u.gpu > 100 || u.memory > 100) {
            printf("  FAIL  a utilisation above 100%% means the struct layout is wrong\n");
            failures++;
        }
    } else {
        printf("  info  nvmlDeviceGetUtilizationRates (%d: %s)\n", r, ESTR(r));
    }

    unsigned int enc_util = 0xABABABAB, enc_period = 0xABABABAB;
    r = pEncUtil(dev, &enc_util, &enc_period);
    if (r == NVML_SUCCESS) {
        printf("  ok    encoder: util=%u%%  sampling period=%u us\n", enc_util, enc_period);
        if (enc_util > 100) {
            printf("  FAIL  encoder utilisation above 100%%\n");
            failures++;
        }
    } else {
        // NOT_SUPPORTED is a real answer on some SKUs and is not a failure.
        printf("  info  nvmlDeviceGetEncoderUtilization (%d: %s)\n", r, ESTR(r));
    }

    nvmlMemory_t m1;
    memset(&m1, 0xAB, sizeof(m1));
    r = pMem(dev, &m1);
    if (r == NVML_SUCCESS) {
        printf("  ok    memory v1: total=%llu MB  used=%llu MB  free=%llu MB\n",
               m1.total >> 20, m1.used >> 20, m1.free >> 20);
        // total = used + free is the invariant that catches a transposition: with
        // `free` and `used` swapped the sum still holds, but with a shifted layout
        // it does not.
        if (m1.used + m1.free != m1.total) {
            printf("  FAIL  used+free != total — the layout is wrong\n");
            failures++;
        }
    } else {
        printf("  FAIL  nvmlDeviceGetMemoryInfo (%d: %s)\n", r, ESTR(r));
        failures++;
    }

    if (pMem2) {
        nvmlMemory_v2_t m2;
        memset(&m2, 0xAB, sizeof(m2));
        m2.version = mem_v2_version;
        r = pMem2(dev, &m2);
        if (r == NVML_SUCCESS) {
            printf("  ok    memory v2: total=%llu MB  used=%llu MB  free=%llu MB\n",
                   m2.total >> 20, m2.used >> 20, m2.free >> 20);
            if (m2.used + m2.free > m2.total) {
                printf("  FAIL  v2 used+free exceeds total — the layout is wrong\n");
                failures++;
            }
        } else {
            printf("  FAIL  nvmlDeviceGetMemoryInfo_v2 with version=0x%08X (%d: %s)\n",
                   mem_v2_version, r, ESTR(r));
            failures++;
        }

        // NEGATIVE CONTROL. Without this the success above is not evidence that the
        // version word matters: a driver ignoring it would look identical. A wrong
        // version must be REJECTED, and rejected with an error rather than by
        // writing through a struct it has mis-sized.
        nvmlMemory_v2_t bad;
        memset(&bad, 0xAB, sizeof(bad));
        bad.version = 1; // not sizeof|2<<24
        r = pMem2(dev, &bad);
        if (r == NVML_SUCCESS) {
            printf("  FAIL  v2 accepted version=1 — the version word is NOT load-bearing,\n"
                   "        so this probe cannot establish that it is\n");
            failures++;
        } else {
            printf("  ok    negative control: version=1 rejected (%d: %s)\n", r, ESTR(r));
        }
    }

    pShutdown();
    FreeLibrary(lib);

    printf("\n%s\n", failures ? "FAILURES ABOVE" : "every claim held.");
    return failures ? 1 : 0;
}
