// d3d12_buf_probe.c — the allocation shape the NV12 zero-copy path needs, end to end.
//
// WHAT THIS DECIDES.  NV12 is two planes in one allocation, and the current
// interop path (SharedTexture) allocates a single-format D3D12 *texture* and maps
// it as a CUarray.  nv12_probe.c showed an over-tall CUarray does work, but it
// also showed pitch=0 kills the process, and an over-tall texture leaves the
// stride entirely to whatever CUDA infers from the D3D12 footprint.
//
// The alternative is a D3D12 committed BUFFER, imported with
// cuExternalMemoryGetMappedBuffer to get a plain CUdeviceptr, and registered as
// NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR.  That is what libavcodec's
// h264_nvenc does (nvenc.c:2269-2288), the pitch semantics are documented
// unambiguously, and the chroma plane offset is ours to choose rather than
// inferred.  But three things about it are unmeasured:
//
//   1. Does CUDA import a D3D12 *buffer* resource at all (as opposed to a
//      texture), and does cuExternalMemoryGetMappedBuffer accept it?
//   2. Does NVENC read the chroma plane at `pitch * height`, or at
//      `width * height`?  Every test so far used pitch == width, where the two
//      are identical and indistinguishable.
//   3. Is a pitch larger than width accepted, and does it need alignment?
//
// Rung C answers 1.  Rungs C-pitch answer 2 and 3 by using pitch = width + 64
// and placing chroma at pitch*height: if NVENC looked at width*height instead,
// the chroma plane it reads is 64*height bytes into the luma plane and the
// decoded colours are visibly wrong rather than subtly off.
//
// Build: gcc -O1 -o d3d12_buf_probe.exe d3d12_buf_probe.c -ld3d12 -ldxgi
// Run:   ./d3d12_buf_probe.exe [pitch]

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include "ffnvcodec/nvEncodeAPI_n12.2.72.0.h"

#define PROBE_W 256
#define PROBE_H 128

// ---------------------------------------------------------------------------
// CUDA driver API, loaded dynamically.
// ---------------------------------------------------------------------------
typedef int   CUresult_t;
typedef void *CUcontext_t;
typedef void *CUextMem_t;
typedef int   CUdevice_t;
typedef unsigned long long CUdeviceptr_t;

typedef struct { void *handle; const void *name; } CUext_win32_t;

typedef struct {
    unsigned type;
    union {
        int fd;
        CUext_win32_t win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned flags;
    unsigned reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC_t;

typedef struct {
    unsigned long long offset;
    unsigned long long size;
    unsigned flags;
    unsigned reserved[16];
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC_t;

#define CU_EXT_MEM_HANDLE_TYPE_D3D12_RESOURCE_T 4u
#define CUDA_EXTERNAL_MEMORY_DEDICATED_T        1u

typedef CUresult_t(__stdcall *pfn_cuInit)(unsigned);
typedef CUresult_t(__stdcall *pfn_cuDeviceGet)(CUdevice_t *, int);
typedef CUresult_t(__stdcall *pfn_cuCtxCreate)(CUcontext_t *, unsigned, CUdevice_t);
typedef CUresult_t(__stdcall *pfn_cuCtxDestroy)(CUcontext_t);
typedef CUresult_t(__stdcall *pfn_cuImportExternalMemory)(CUextMem_t *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_t *);
typedef CUresult_t(__stdcall *pfn_cuExternalMemoryGetMappedBuffer)(CUdeviceptr_t *, CUextMem_t, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC_t *);
typedef CUresult_t(__stdcall *pfn_cuDestroyExternalMemory)(CUextMem_t);
typedef CUresult_t(__stdcall *pfn_cuMemcpyHtoD)(CUdeviceptr_t, const void *, size_t);
typedef CUresult_t(__stdcall *pfn_cuCtxSynchronize)(void);
typedef CUresult_t(__stdcall *pfn_cuGetErrorName)(CUresult_t, const char **);

static pfn_cuGetErrorName cuGetErrorName_;
static const char *cu_name(CUresult_t r) {
    const char *n = NULL;
    if (cuGetErrorName_ && cuGetErrorName_(r, &n) == 0 && n) return n;
    return "(unknown CUresult)";
}

static const char *nv_name(NVENCSTATUS s) {
    switch (s) {
    case NV_ENC_SUCCESS:                      return "NV_ENC_SUCCESS";
    case NV_ENC_ERR_INVALID_PARAM:            return "NV_ENC_ERR_INVALID_PARAM";
    case NV_ENC_ERR_INVALID_CALL:             return "NV_ENC_ERR_INVALID_CALL";
    case NV_ENC_ERR_INVALID_PTR:              return "NV_ENC_ERR_INVALID_PTR";
    case NV_ENC_ERR_OUT_OF_MEMORY:            return "NV_ENC_ERR_OUT_OF_MEMORY";
    case NV_ENC_ERR_UNSUPPORTED_PARAM:        return "NV_ENC_ERR_UNSUPPORTED_PARAM";
    case NV_ENC_ERR_NEED_MORE_INPUT:          return "NV_ENC_ERR_NEED_MORE_INPUT";
    case NV_ENC_ERR_MAP_FAILED:               return "NV_ENC_ERR_MAP_FAILED";
    case NV_ENC_ERR_RESOURCE_REGISTER_FAILED: return "NV_ENC_ERR_RESOURCE_REGISTER_FAILED";
    case NV_ENC_ERR_GENERIC:                  return "NV_ENC_ERR_GENERIC";
    default:                                  return "(unnamed)";
    }
}

static const GUID CODEC_H264 = { 0x6bc82762, 0x4e63, 0x4ca4,
    { 0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf } };
static const GUID PRESET_P4  = { 0x90a7b826, 0xdf06, 0x4862,
    { 0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81 } };

static void bt709_limited(double r, double g, double b, int *Y, int *U, int *V) {
    const double kr = 0.2126, kb = 0.0722, kg = 1.0 - kr - kb;
    double y  = kr * r + kg * g + kb * b;
    double cb = (b - y) / (2.0 * (1.0 - kb));
    double cr = (r - y) / (2.0 * (1.0 - kr));
    *Y = (int)(16.0 + 219.0 * y + 0.5);
    *U = (int)(128.0 + 224.0 * cb + 0.5);
    *V = (int)(128.0 + 224.0 * cr + 0.5);
}

static const double BARS[4][3] = {
    { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 1 },
};

/// NV12 at an arbitrary pitch, chroma plane starting at `pitch * height`.
/// Bytes between `width` and `pitch` on each row are filled with 0xAA — a value
/// no bar produces, so if NVENC ever reads them as pixels the decode shows it.
static void build_nv12_pitched(uint8_t *out, int w, int h, int pitch) {
    memset(out, 0xAA, (size_t)pitch * h * 3 / 2);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int Y, U, V, bar = (x * 4) / w;
            bt709_limited(BARS[bar][0], BARS[bar][1], BARS[bar][2], &Y, &U, &V);
            out[(size_t)y * pitch + x] = (uint8_t)Y;
        }
    uint8_t *chroma = out + (size_t)pitch * h;
    for (int y = 0; y < h / 2; y++)
        for (int x = 0; x < w / 2; x++) {
            int Y, U, V, bar = (x * 2 * 4) / w;
            bt709_limited(BARS[bar][0], BARS[bar][1], BARS[bar][2], &Y, &U, &V);
            chroma[(size_t)y * pitch + x * 2 + 0] = (uint8_t)U;
            chroma[(size_t)y * pitch + x * 2 + 1] = (uint8_t)V;
        }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    uint32_t pitch = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : PROBE_W;
    if (pitch < PROBE_W) pitch = PROBE_W;
    printf("d3d12_buf_probe: %dx%d NV12 via D3D12 shared BUFFER -> CUdeviceptr, pitch=%u\n",
           PROBE_W, PROBE_H, pitch);
    printf("  (chroma plane placed at pitch*height = %u)\n\n", pitch * PROBE_H);

    size_t total = (size_t)pitch * PROBE_H * 3 / 2;

    // ---------------- 1. D3D12 device ----------------
    HMODULE d3d = LoadLibraryA("d3d12.dll");
    if (!d3d) { printf("FAIL LoadLibrary(d3d12.dll) err=%lu\n", GetLastError()); return 1; }
    typedef HRESULT (WINAPI *pfn_create_dev)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
    pfn_create_dev D3D12CreateDevice_ = (pfn_create_dev)GetProcAddress(d3d, "D3D12CreateDevice");
    if (!D3D12CreateDevice_) { printf("FAIL GetProcAddress(D3D12CreateDevice)\n"); return 1; }

    ID3D12Device *dev12 = NULL;
    HRESULT hr = D3D12CreateDevice_(NULL, D3D_FEATURE_LEVEL_11_0,
                                    &IID_ID3D12Device, (void **)&dev12);
    printf("D3D12CreateDevice -> 0x%08lX %s\n", hr, hr >= 0 ? "OK" : "FAILED");
    if (hr < 0) return 1;

    // ---------------- 2. Shared committed BUFFER ----------------
    D3D12_HEAP_PROPERTIES hp;
    memset(&hp, 0, sizeof(hp));
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = 0;
    rd.Width  = total;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // ALLOW_UNORDERED_ACCESS is what a compute shader writing this buffer needs;
    // included here so the probe measures the same resource production will use.
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // MinGW's C vtable for a struct-returning method takes a hidden out-pointer
    // first; the C++ signature does not match.
    D3D12_RESOURCE_ALLOCATION_INFO ai;
    dev12->lpVtbl->GetResourceAllocationInfo(dev12, &ai, 0, 1, &rd);
    printf("GetResourceAllocationInfo -> SizeInBytes=%llu Alignment=%llu\n",
           (unsigned long long)ai.SizeInBytes, (unsigned long long)ai.Alignment);
    if (ai.SizeInBytes == 0 || ai.SizeInBytes == (UINT64)-1) {
        printf("FAIL: D3D12 rejected the buffer descriptor\n"); return 1;
    }

    ID3D12Resource *res = NULL;
    hr = dev12->lpVtbl->CreateCommittedResource(
        dev12, &hp, D3D12_HEAP_FLAG_SHARED, &rd,
        D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&res);
    printf("CreateCommittedResource(HEAP_FLAG_SHARED, BUFFER %zu bytes) -> 0x%08lX %s\n",
           total, hr, hr >= 0 ? "OK" : "FAILED");
    if (hr < 0) return 1;

    HANDLE nt = NULL;
    hr = dev12->lpVtbl->CreateSharedHandle(dev12, (ID3D12DeviceChild *)res,
                                           NULL, GENERIC_ALL, NULL, &nt);
    printf("CreateSharedHandle -> 0x%08lX handle=%p\n", hr, nt);
    if (hr < 0 || !nt) { printf("FAIL: no shared handle\n"); return 1; }

    // ---------------- 3. CUDA import ----------------
    HMODULE cu = LoadLibraryA("nvcuda.dll");
    pfn_cuInit        cuInit_     = (pfn_cuInit)GetProcAddress(cu, "cuInit");
    pfn_cuDeviceGet   cuDevGet_   = (pfn_cuDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    pfn_cuCtxCreate   cuCtxNew_   = (pfn_cuCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    pfn_cuCtxDestroy  cuCtxDel_   = (pfn_cuCtxDestroy)GetProcAddress(cu, "cuCtxDestroy_v2");
    pfn_cuImportExternalMemory cuImport_ =
        (pfn_cuImportExternalMemory)GetProcAddress(cu, "cuImportExternalMemory");
    pfn_cuExternalMemoryGetMappedBuffer cuGetBuf_ =
        (pfn_cuExternalMemoryGetMappedBuffer)GetProcAddress(cu, "cuExternalMemoryGetMappedBuffer");
    pfn_cuDestroyExternalMemory cuDestroyExt_ =
        (pfn_cuDestroyExternalMemory)GetProcAddress(cu, "cuDestroyExternalMemory");
    pfn_cuMemcpyHtoD  cuHtoD_     = (pfn_cuMemcpyHtoD)GetProcAddress(cu, "cuMemcpyHtoD_v2");
    pfn_cuCtxSynchronize cuSync_  = (pfn_cuCtxSynchronize)GetProcAddress(cu, "cuCtxSynchronize");
    cuGetErrorName_ = (pfn_cuGetErrorName)GetProcAddress(cu, "cuGetErrorName");
    if (!cuImport_ || !cuGetBuf_) {
        printf("FAIL: cuImportExternalMemory / cuExternalMemoryGetMappedBuffer missing\n");
        return 1;
    }

    CUresult_t cr = cuInit_(0);
    if (cr) { printf("FAIL cuInit -> %s\n", cu_name(cr)); return 1; }
    CUdevice_t cudev = 0;
    cuDevGet_(&cudev, 0);
    CUcontext_t ctx = NULL;
    cr = cuCtxNew_(&ctx, 0, cudev);
    if (cr) { printf("FAIL cuCtxCreate -> %s\n", cu_name(cr)); return 1; }
    printf("CUDA context OK\n");

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC_t hd;
    memset(&hd, 0, sizeof(hd));
    hd.type = CU_EXT_MEM_HANDLE_TYPE_D3D12_RESOURCE_T;
    hd.handle.win32.handle = nt;
    hd.handle.win32.name = NULL;
    hd.size = ai.SizeInBytes;
    hd.flags = CUDA_EXTERNAL_MEMORY_DEDICATED_T;
    CUextMem_t ext = NULL;
    cr = cuImport_(&ext, &hd);
    printf("cuImportExternalMemory(D3D12_RESOURCE, size=%llu, DEDICATED) -> %s (%d)\n",
           (unsigned long long)hd.size, cu_name(cr), cr);
    if (cr) {
        // Retry without DEDICATED: the flag is required for textures but its
        // necessity for buffers is exactly the sort of thing worth measuring
        // rather than assuming.
        hd.flags = 0;
        cr = cuImport_(&ext, &hd);
        printf("  retry without DEDICATED -> %s (%d)\n", cu_name(cr), cr);
        if (cr) { printf("FAIL: buffer import rejected\n"); return 2; }
    }

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC_t bd;
    memset(&bd, 0, sizeof(bd));
    bd.offset = 0;
    bd.size   = total;
    bd.flags  = 0;
    CUdeviceptr_t dptr = 0;
    cr = cuGetBuf_(&dptr, ext, &bd);
    printf("cuExternalMemoryGetMappedBuffer(size=%zu) -> %s (%d)  dptr=0x%llx\n",
           total, cu_name(cr), cr, (unsigned long long)dptr);
    if (cr) { printf("FAIL: mapped buffer rejected\n"); return 3; }

    // ---------------- 4. Fill it with the pattern ----------------
    uint8_t *host = (uint8_t *)malloc(total);
    build_nv12_pitched(host, PROBE_W, PROBE_H, (int)pitch);
    cr = cuHtoD_(dptr, host, total);
    printf("cuMemcpyHtoD %zu bytes -> %s (%d)\n", total, cu_name(cr), cr);
    if (cuSync_) cuSync_();
    if (cr) return 4;

    // ---------------- 5. NVENC ----------------
    HMODULE nv = LoadLibraryA("nvEncodeAPI64.dll");
    typedef NVENCSTATUS(NVENCAPI *pfn_create)(NV_ENCODE_API_FUNCTION_LIST *);
    pfn_create create = (pfn_create)GetProcAddress(nv, "NvEncodeAPICreateInstance");
    NV_ENCODE_API_FUNCTION_LIST fl;
    memset(&fl, 0, sizeof(fl));
    fl.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS st = create(&fl);
    if (st) { printf("FAIL CreateInstance %s\n", nv_name(st)); return 1; }

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op;
    memset(&op, 0, sizeof(op));
    op.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    op.device = ctx;
    op.apiVersion = NVENCAPI_VERSION;
    void *session = NULL;
    st = fl.nvEncOpenEncodeSessionEx(&op, &session);
    printf("nvEncOpenEncodeSessionEx -> %s\n", nv_name(st));
    if (st) return 1;

    NV_ENC_INITIALIZE_PARAMS init;
    memset(&init, 0, sizeof(init));
    init.version = NV_ENC_INITIALIZE_PARAMS_VER;
    init.encodeGUID = CODEC_H264;
    init.presetGUID = PRESET_P4;
    init.encodeWidth = PROBE_W;  init.encodeHeight = PROBE_H;
    init.darWidth = PROBE_W;     init.darHeight = PROBE_H;
    init.frameRateNum = 30;      init.frameRateDen = 1;
    init.enablePTD = 1;
    init.maxEncodeWidth = PROBE_W; init.maxEncodeHeight = PROBE_H;
    init.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    st = fl.nvEncInitializeEncoder(session, &init);
    printf("nvEncInitializeEncoder -> %s\n", nv_name(st));
    if (st) return 1;

    NV_ENC_REGISTER_RESOURCE reg;
    memset(&reg, 0, sizeof(reg));
    reg.version = NV_ENC_REGISTER_RESOURCE_VER;
    reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
    reg.width = PROBE_W; reg.height = PROBE_H;
    reg.pitch = pitch;
    reg.resourceToRegister = (void *)(uintptr_t)dptr;
    reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12;
    reg.bufferUsage = NV_ENC_INPUT_IMAGE;
    st = fl.nvEncRegisterResource(session, &reg);
    printf("nvEncRegisterResource(CUDADEVICEPTR, pitch=%u) -> %s\n", pitch, nv_name(st));
    if (st) { printf("VERDICT: register rejected\n"); return 5; }

    NV_ENC_CREATE_BITSTREAM_BUFFER bs;
    memset(&bs, 0, sizeof(bs));
    bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    fl.nvEncCreateBitstreamBuffer(session, &bs);

    NV_ENC_MAP_INPUT_RESOURCE map;
    memset(&map, 0, sizeof(map));
    map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map.registeredResource = reg.registeredResource;
    st = fl.nvEncMapInputResource(session, &map);
    printf("nvEncMapInputResource -> %s  mappedBufferFmt=0x%08X\n",
           nv_name(st), (unsigned)map.mappedBufferFmt);
    if (st) return 6;

    NV_ENC_PIC_PARAMS pic;
    memset(&pic, 0, sizeof(pic));
    pic.version = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth = PROBE_W; pic.inputHeight = PROBE_H;
    pic.inputPitch = pitch;
    pic.inputBuffer = map.mappedResource;
    pic.outputBitstream = bs.bitstreamBuffer;
    pic.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
    pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
    printf("about to call nvEncEncodePicture (missing next line == CRASH)\n");
    st = fl.nvEncEncodePicture(session, &pic);
    printf("nvEncEncodePicture -> %s\n", nv_name(st));

    char path[64];
    snprintf(path, sizeof(path), "d3dbuf_pitch%u.h264", pitch);
    if (st == NV_ENC_SUCCESS) {
        NV_ENC_LOCK_BITSTREAM lock;
        memset(&lock, 0, sizeof(lock));
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = bs.bitstreamBuffer;
        st = fl.nvEncLockBitstream(session, &lock);
        printf("nvEncLockBitstream -> %s  %u bytes\n",
               nv_name(st), (unsigned)lock.bitstreamSizeInBytes);
        if (st == NV_ENC_SUCCESS) {
            FILE *f = fopen(path, "wb");
            fwrite(lock.bitstreamBufferPtr, 1, lock.bitstreamSizeInBytes, f);
            fclose(f);
            printf("wrote %s  <-- decode this and check the PIXELS\n", path);
            fl.nvEncUnlockBitstream(session, bs.bitstreamBuffer);
        }
    }

    printf("about to unmap (missing next line == UNMAP CRASHED)\n");
    fl.nvEncUnmapInputResource(session, map.mappedResource);
    printf("unmapped OK\n");
    fl.nvEncDestroyBitstreamBuffer(session, bs.bitstreamBuffer);
    fl.nvEncUnregisterResource(session, reg.registeredResource);
    fl.nvEncDestroyEncoder(session);
    if (cuDestroyExt_) cuDestroyExt_(ext);
    if (cuCtxDel_) cuCtxDel_(ctx);
    CloseHandle(nt);
    res->lpVtbl->Release(res);
    dev12->lpVtbl->Release(dev12);
    free(host);
    printf("\nVERDICT: D3D12 shared buffer -> CUdeviceptr -> NVENC survived, pitch=%u\n", pitch);
    return 0;
}
