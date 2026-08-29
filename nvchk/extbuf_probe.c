// extbuf_probe.c — layout + handle-type + teardown measurements for the
// SharedBuffer binding (the CUdeviceptr sibling to SharedTexture's CUarray).
//
// FOUR QUESTIONS, each of which the Rust binding is about to hard-code:
//
//  1. CUDA_EXTERNAL_MEMORY_BUFFER_DESC's size and field offsets, so
//     `CudaExternalMemoryBufferDesc` in src/interop/ffi/cuda_gl_vk_interop.rs is
//     a measurement rather than a transcription.   -> mode `layout`
//
//  2. The value of CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE.  The repo's
//     cuda_gl_vk_interop.rs says 4, and every existing texture import uses 4 and
//     works.  But the vendor header says 4 is D3D12_*HEAP* and 5 is
//     D3D12_*RESOURCE*.  Both cannot be right, and the Rust constant is the one
//     with the wrong NAME regardless of which value the driver tolerates.
//     -> modes `buf 4` / `buf 5` / `tex 4` / `tex 5`
//
//     A type the driver rejects fails at cuImportExternalMemory.  A type it
//     accepts but MISREADS fails later, or silently returns memory that is not
//     the resource.  So no rung stops at "the call returned SUCCESS": each one
//     round-trips a byte pattern through the mapping and compares.
//
//  3. How a mapped BUFFER is released.  cuExternalMemoryGetMappedMipmappedArray's
//     result is freed with cuMipmappedArrayDestroy (what SharedTexture does).
//     The buffer sibling's documented counterpart is cuMemFree, which is a
//     surprising thing to call on memory CUDA did not allocate — and calling the
//     wrong destroy on a driver object is an access violation, not an error
//     return.  -> the `buf` rung frees with cuMemFree and prints before/after.
//
//  4. Whether the mapped size may be the LOGICAL size while the import carries
//     the padded GetResourceAllocationInfo size (65536 vs 61440 here).  Both
//     rungs map the logical size over a padded import.
//
// Build: gcc -O1 -o extbuf_probe.exe extbuf_probe.c -ld3d12
// Run:   ./extbuf_probe.exe layout
//        ./extbuf_probe.exe buf 4     (one process per rung: a bad type can crash)
//        ./extbuf_probe.exe buf 5
//        ./extbuf_probe.exe tex 4
//        ./extbuf_probe.exe tex 5

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <initguid.h>
#include <d3d12.h>
#include "ffnvcodec/dynlink_cuda.h"

#define PROBE_BYTES 61440u
#define TEX_W  256u
#define TEX_H  128u
#define TEX_BPP 4u          /* R32_UINT, matching SharedTexture's ABGR10 path */

typedef int   CUresult_t;
typedef void *CUcontext_t;
typedef void *CUextMem_t;
typedef void *CUarray_t;
typedef void *CUmipmap_t;
typedef int   CUdevice_t;
typedef unsigned long long CUdeviceptr_t;

/* CUDA_MEMCPY2D, for the texture rung's pixel round-trip. */
typedef struct {
    size_t srcXInBytes, srcY;
    unsigned srcMemoryType;
    const void *srcHost;
    CUdeviceptr_t srcDevice;
    CUarray_t srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    unsigned dstMemoryType;
    void *dstHost;
    CUdeviceptr_t dstDevice;
    CUarray_t dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
} CUDA_MEMCPY2D_t;

#define CU_MEMORYTYPE_HOST_T  1u
#define CU_MEMORYTYPE_ARRAY_T 3u

typedef CUresult_t(__stdcall *pfn_cuInit)(unsigned);
typedef CUresult_t(__stdcall *pfn_cuDeviceGet)(CUdevice_t *, int);
typedef CUresult_t(__stdcall *pfn_cuCtxCreate)(CUcontext_t *, unsigned, CUdevice_t);
typedef CUresult_t(__stdcall *pfn_cuCtxDestroy)(CUcontext_t);
typedef CUresult_t(__stdcall *pfn_cuImportExternalMemory)(CUextMem_t *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
typedef CUresult_t(__stdcall *pfn_cuGetMappedBuffer)(CUdeviceptr_t *, CUextMem_t, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
typedef CUresult_t(__stdcall *pfn_cuGetMappedMipmap)(CUmipmap_t *, CUextMem_t, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
typedef CUresult_t(__stdcall *pfn_cuMipmapGetLevel)(CUarray_t *, CUmipmap_t, unsigned);
typedef CUresult_t(__stdcall *pfn_cuMipmapDestroy)(CUmipmap_t);
typedef CUresult_t(__stdcall *pfn_cuDestroyExternalMemory)(CUextMem_t);
typedef CUresult_t(__stdcall *pfn_cuMemcpyHtoD)(CUdeviceptr_t, const void *, size_t);
typedef CUresult_t(__stdcall *pfn_cuMemcpyDtoH)(void *, CUdeviceptr_t, size_t);
typedef CUresult_t(__stdcall *pfn_cuMemcpy2D)(const CUDA_MEMCPY2D_t *);
typedef CUresult_t(__stdcall *pfn_cuMemFree)(CUdeviceptr_t);
typedef CUresult_t(__stdcall *pfn_cuCtxSynchronize)(void);
typedef CUresult_t(__stdcall *pfn_cuGetErrorName)(CUresult_t, const char **);

static pfn_cuGetErrorName cuGetErrorName_;
static const char *cu_name(CUresult_t r) {
    const char *n = NULL;
    if (cuGetErrorName_ && cuGetErrorName_(r, &n) == 0 && n) return n;
    return "(unknown CUresult)";
}

static pfn_cuInit                  cuInit_;
static pfn_cuDeviceGet             cuDevGet_;
static pfn_cuCtxCreate             cuCtxNew_;
static pfn_cuCtxDestroy            cuCtxDel_;
static pfn_cuImportExternalMemory  cuImport_;
static pfn_cuGetMappedBuffer       cuGetBuf_;
static pfn_cuGetMappedMipmap       cuGetMip_;
static pfn_cuMipmapGetLevel        cuMipLevel_;
static pfn_cuMipmapDestroy         cuMipDestroy_;
static pfn_cuDestroyExternalMemory cuDestroyExt_;
static pfn_cuMemcpyHtoD            cuHtoD_;
static pfn_cuMemcpyDtoH            cuDtoH_;
static pfn_cuMemcpy2D              cuMemcpy2D_;
static pfn_cuMemFree               cuMemFree_;
static pfn_cuCtxSynchronize        cuSync_;

static int load_cuda(CUcontext_t *ctx) {
    HMODULE cu = LoadLibraryA("nvcuda.dll");
    if (!cu) { printf("FAIL LoadLibrary(nvcuda.dll)\n"); return 0; }
#define G(v, n) v = (void *)GetProcAddress(cu, n); if (!v) printf("  WARN missing %s\n", n)
    G(cuInit_,          "cuInit");
    G(cuDevGet_,        "cuDeviceGet");
    G(cuCtxNew_,        "cuCtxCreate_v2");
    G(cuCtxDel_,        "cuCtxDestroy_v2");
    G(cuImport_,        "cuImportExternalMemory");
    G(cuGetBuf_,        "cuExternalMemoryGetMappedBuffer");
    G(cuGetMip_,        "cuExternalMemoryGetMappedMipmappedArray");
    G(cuMipLevel_,      "cuMipmappedArrayGetLevel");
    G(cuMipDestroy_,    "cuMipmappedArrayDestroy");
    G(cuDestroyExt_,    "cuDestroyExternalMemory");
    G(cuHtoD_,          "cuMemcpyHtoD_v2");
    G(cuDtoH_,          "cuMemcpyDtoH_v2");
    G(cuMemcpy2D_,      "cuMemcpy2D_v2");
    G(cuMemFree_,       "cuMemFree_v2");
#undef G
    cuSync_ = (pfn_cuCtxSynchronize)GetProcAddress(cu, "cuCtxSynchronize");
    cuGetErrorName_ = (pfn_cuGetErrorName)GetProcAddress(cu, "cuGetErrorName");

    CUresult_t cr = cuInit_(0);
    if (cr) { printf("FAIL cuInit -> %s\n", cu_name(cr)); return 0; }
    CUdevice_t dev = 0;
    cuDevGet_(&dev, 0);
    cr = cuCtxNew_(ctx, 0, dev);
    if (cr) { printf("FAIL cuCtxCreate -> %s\n", cu_name(cr)); return 0; }
    printf("CUDA context OK\n");
    return 1;
}

#define S(t)    printf("SIZE %-44s %zu\n", #t, sizeof(t))
#define O(t,f)  printf("OFF  %-38s %-14s %zu\n", #t, #f, offsetof(t,f))
#define V(n)    printf("VAL  %-46s %u\n", #n, (unsigned)(n))

static int layout(void) {
    puts("--- CUDA external-memory struct layout, from the vendor header ---");
    S(CUDA_EXTERNAL_MEMORY_HANDLE_DESC);
    O(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, type);
    O(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, handle);
    O(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, size);
    O(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, flags);
    O(CUDA_EXTERNAL_MEMORY_HANDLE_DESC, reserved);
    puts("");
    S(CUDA_EXTERNAL_MEMORY_BUFFER_DESC);
    O(CUDA_EXTERNAL_MEMORY_BUFFER_DESC, offset);
    O(CUDA_EXTERNAL_MEMORY_BUFFER_DESC, size);
    O(CUDA_EXTERNAL_MEMORY_BUFFER_DESC, flags);
    O(CUDA_EXTERNAL_MEMORY_BUFFER_DESC, reserved);
    puts("");
    S(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC);
    O(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, offset);
    O(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, arrayDesc);
    O(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, numLevels);
    O(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, reserved);
    puts("");
    puts("--- handle-type enum values ---");
    V(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD);
    V(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32);
    V(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT);
    V(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP);
    V(CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE);
    return 0;
}

/// A shared committed resource plus its NT handle and padded allocation size.
typedef struct {
    ID3D12Device   *dev;
    ID3D12Resource *res;
    HANDLE          nt;
    UINT64          alloc_size;
} SharedRes;

static int make_shared(SharedRes *out, int as_buffer) {
    HMODULE d3d = LoadLibraryA("d3d12.dll");
    typedef HRESULT (WINAPI *pfn_create_dev)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
    pfn_create_dev D3D12CreateDevice_ = (pfn_create_dev)GetProcAddress(d3d, "D3D12CreateDevice");
    memset(out, 0, sizeof(*out));
    HRESULT hr = D3D12CreateDevice_(NULL, D3D_FEATURE_LEVEL_11_0,
                                    &IID_ID3D12Device, (void **)&out->dev);
    if (hr < 0) { printf("FAIL D3D12CreateDevice 0x%08lX\n", hr); return 0; }

    D3D12_HEAP_PROPERTIES hp; memset(&hp, 0, sizeof(hp));
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    hp.CreationNodeMask = 1; hp.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rd; memset(&rd, 0, sizeof(rd));
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (as_buffer) {
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = PROBE_BYTES; rd.Height = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    } else {
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = TEX_W; rd.Height = TEX_H;
        rd.Format = DXGI_FORMAT_R32_UINT;
        rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }

    D3D12_RESOURCE_ALLOCATION_INFO ai;
    out->dev->lpVtbl->GetResourceAllocationInfo(out->dev, &ai, 0, 1, &rd);
    printf("GetResourceAllocationInfo(%s) -> SizeInBytes=%llu Alignment=%llu\n",
           as_buffer ? "BUFFER" : "TEXTURE2D",
           (unsigned long long)ai.SizeInBytes, (unsigned long long)ai.Alignment);
    if (ai.SizeInBytes == 0 || ai.SizeInBytes == (UINT64)-1) {
        printf("FAIL: D3D12 rejected the descriptor\n"); return 0;
    }
    out->alloc_size = ai.SizeInBytes;

    hr = out->dev->lpVtbl->CreateCommittedResource(out->dev, &hp, D3D12_HEAP_FLAG_SHARED, &rd,
            D3D12_RESOURCE_STATE_COMMON, NULL, &IID_ID3D12Resource, (void **)&out->res);
    if (hr < 0) { printf("FAIL CreateCommittedResource 0x%08lX\n", hr); return 0; }
    hr = out->dev->lpVtbl->CreateSharedHandle(out->dev, (ID3D12DeviceChild *)out->res, NULL,
                                              GENERIC_ALL, NULL, &out->nt);
    if (hr < 0 || !out->nt) { printf("FAIL CreateSharedHandle 0x%08lX\n", hr); return 0; }
    printf("shared committed %s + NT handle OK (%p)\n",
           as_buffer ? "BUFFER" : "TEXTURE2D", out->nt);
    return 1;
}

static int import(CUextMem_t *ext, const SharedRes *sr, unsigned htype, unsigned long long size) {
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC hd; memset(&hd, 0, sizeof(hd));
    hd.type = (CUexternalMemoryHandleType)htype;
    hd.handle.win32.handle = sr->nt;
    hd.handle.win32.name = NULL;
    hd.size = size;
    hd.flags = 1u; /* CUDA_EXTERNAL_MEMORY_DEDICATED */
    printf("about to cuImportExternalMemory(type=%u, size=%llu) (missing next line == CRASH)\n",
           htype, size);
    CUresult_t cr = cuImport_(ext, &hd);
    printf("cuImportExternalMemory(type=%u, size=%llu, DEDICATED) -> %s (%d)\n",
           htype, size, cu_name(cr), cr);
    return cr == 0;
}

static int rung_buffer(unsigned htype, int import_logical_size) {
    printf("=== rung: D3D12 shared BUFFER, handle type %u, import size = %s ===\n",
           htype, import_logical_size ? "LOGICAL (61440)" : "PADDED (GetResourceAllocationInfo)");
    SharedRes sr;
    if (!make_shared(&sr, 1)) return 1;
    CUcontext_t ctx = NULL;
    if (!load_cuda(&ctx)) return 1;

    unsigned long long import_size = import_logical_size ? PROBE_BYTES : sr.alloc_size;
    CUextMem_t ext = NULL;
    if (!import(&ext, &sr, htype, import_size)) {
        printf("VERDICT type=%u import_size=%llu: import REJECTED\n", htype, import_size);
        return 2;
    }

    /* Map the LOGICAL size over the PADDED import (61440 of 65536). */
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd; memset(&bd, 0, sizeof(bd));
    bd.offset = 0; bd.size = PROBE_BYTES; bd.flags = 0;
    CUdeviceptr_t dptr = 0;
    printf("about to cuExternalMemoryGetMappedBuffer (missing next line == CRASH)\n");
    CUresult_t cr = cuGetBuf_(&dptr, ext, &bd);
    printf("cuExternalMemoryGetMappedBuffer(offset=0, size=%u of %llu imported) -> %s (%d) dptr=0x%llx\n",
           PROBE_BYTES, import_size, cu_name(cr), cr,
           (unsigned long long)dptr);
    if (cr) { printf("VERDICT type=%u: mapped-buffer REJECTED\n", htype); return 3; }

    uint8_t *tx = malloc(PROBE_BYTES), *rx = malloc(PROBE_BYTES);
    for (unsigned i = 0; i < PROBE_BYTES; i++) tx[i] = (uint8_t)(i * 31u + 7u);
    memset(rx, 0, PROBE_BYTES);
    cr = cuHtoD_(dptr, tx, PROBE_BYTES);
    printf("cuMemcpyHtoD %u -> %s\n", PROBE_BYTES, cu_name(cr));
    if (cr) return 4;
    if (cuSync_) cuSync_();
    cr = cuDtoH_(rx, dptr, PROBE_BYTES);
    printf("cuMemcpyDtoH %u -> %s\n", PROBE_BYTES, cu_name(cr));
    if (cr) return 5;
    if (cuSync_) cuSync_();

    unsigned bad = 0, first = 0;
    for (unsigned i = 0; i < PROBE_BYTES; i++)
        if (rx[i] != tx[i]) { if (!bad) first = i; bad++; }
    if (bad) printf("ROUNDTRIP type=%u: %u/%u bytes differ, first at %u (%02X != %02X)\n",
                    htype, bad, PROBE_BYTES, first, rx[first], tx[first]);
    else     printf("ROUNDTRIP type=%u: all %u bytes match\n", htype, PROBE_BYTES);

    /* Question 3: is cuMemFree the right release for a mapped buffer? */
    printf("about to cuMemFree(mapped dptr) (missing next line == CRASH)\n");
    cr = cuMemFree_ ? cuMemFree_(dptr) : -1;
    printf("cuMemFree(mapped dptr) -> %s (%d)\n", cu_name(cr), cr);
    printf("about to cuDestroyExternalMemory (missing next line == CRASH)\n");
    CUresult_t cr2 = cuDestroyExt_(ext);
    printf("cuDestroyExternalMemory -> %s (%d)\n", cu_name(cr2), cr2);

    if (cuCtxDel_) cuCtxDel_(ctx);
    CloseHandle(sr.nt);
    sr.res->lpVtbl->Release(sr.res);
    sr.dev->lpVtbl->Release(sr.dev);
    free(tx); free(rx);
    printf("VERDICT type=%u BUFFER: import+map+roundtrip+teardown %s\n",
           htype, bad ? "MISREAD DATA" : "all clean");
    return bad ? 6 : 0;
}

static int rung_texture(unsigned htype) {
    printf("=== rung: D3D12 shared TEXTURE2D R32_UINT %ux%u, handle type %u ===\n",
           TEX_W, TEX_H, htype);
    printf("    (this is exactly what SharedTexture::new allocates today)\n");
    SharedRes sr;
    if (!make_shared(&sr, 0)) return 1;
    CUcontext_t ctx = NULL;
    if (!load_cuda(&ctx)) return 1;

    CUextMem_t ext = NULL;
    if (!import(&ext, &sr, htype, sr.alloc_size)) {
        printf("VERDICT type=%u: import REJECTED\n", htype); return 2;
    }

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC md; memset(&md, 0, sizeof(md));
    md.offset = 0;
    md.arrayDesc.Width = TEX_W;
    md.arrayDesc.Height = TEX_H;
    md.arrayDesc.Depth = 0;
    md.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
    md.arrayDesc.NumChannels = 1;
    md.arrayDesc.Flags = 2u; /* CUDA_ARRAY3D_SURFACE_LDST */
    md.numLevels = 1;
    CUmipmap_t mip = NULL;
    printf("about to cuExternalMemoryGetMappedMipmappedArray (missing next line == CRASH)\n");
    CUresult_t cr = cuGetMip_(&mip, ext, &md);
    printf("cuExternalMemoryGetMappedMipmappedArray -> %s (%d)\n", cu_name(cr), cr);
    if (cr) { printf("VERDICT type=%u TEXTURE: mipmap map REJECTED\n", htype); return 3; }
    CUarray_t arr = NULL;
    cr = cuMipLevel_(&arr, mip, 0);
    printf("cuMipmappedArrayGetLevel(0) -> %s (%d) array=%p\n", cu_name(cr), cr, arr);
    if (cr) return 4;

    size_t row = (size_t)TEX_W * TEX_BPP, total = row * TEX_H;
    uint8_t *tx = malloc(total), *rx = malloc(total);
    for (size_t i = 0; i < total; i++) tx[i] = (uint8_t)(i * 17u + 3u);
    memset(rx, 0, total);

    CUDA_MEMCPY2D_t c; memset(&c, 0, sizeof(c));
    c.srcMemoryType = CU_MEMORYTYPE_HOST_T; c.srcHost = tx; c.srcPitch = row;
    c.dstMemoryType = CU_MEMORYTYPE_ARRAY_T; c.dstArray = arr;
    c.WidthInBytes = row; c.Height = TEX_H;
    cr = cuMemcpy2D_(&c);
    printf("cuMemcpy2D host->array -> %s (%d)\n", cu_name(cr), cr);
    if (cr) return 5;
    if (cuSync_) cuSync_();

    memset(&c, 0, sizeof(c));
    c.srcMemoryType = CU_MEMORYTYPE_ARRAY_T; c.srcArray = arr;
    c.dstMemoryType = CU_MEMORYTYPE_HOST_T; c.dstHost = rx; c.dstPitch = row;
    c.WidthInBytes = row; c.Height = TEX_H;
    cr = cuMemcpy2D_(&c);
    printf("cuMemcpy2D array->host -> %s (%d)\n", cu_name(cr), cr);
    if (cr) return 6;
    if (cuSync_) cuSync_();

    size_t bad = 0, first = 0;
    for (size_t i = 0; i < total; i++)
        if (rx[i] != tx[i]) { if (!bad) first = i; bad++; }
    if (bad) printf("ROUNDTRIP type=%u: %zu/%zu bytes differ, first at %zu (%02X != %02X)\n",
                    htype, bad, total, first, rx[first], tx[first]);
    else     printf("ROUNDTRIP type=%u: all %zu bytes match\n", htype, total);

    printf("about to cuMipmappedArrayDestroy (missing next line == CRASH)\n");
    cr = cuMipDestroy_(mip);
    printf("cuMipmappedArrayDestroy -> %s (%d)\n", cu_name(cr), cr);
    cr = cuDestroyExt_(ext);
    printf("cuDestroyExternalMemory -> %s (%d)\n", cu_name(cr), cr);

    if (cuCtxDel_) cuCtxDel_(ctx);
    CloseHandle(sr.nt);
    sr.res->lpVtbl->Release(sr.res);
    sr.dev->lpVtbl->Release(sr.dev);
    free(tx); free(rx);
    printf("VERDICT type=%u TEXTURE: import+map+roundtrip+teardown %s\n",
           htype, bad ? "MISREAD DATA" : "all clean");
    return bad ? 7 : 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        printf("usage: %s layout | buf <type> [logical] | tex <type>\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "layout") == 0) return layout();
    unsigned htype = (argc > 2) ? (unsigned)strtoul(argv[2], NULL, 10) : 4u;
    if (strcmp(argv[1], "buf") == 0) {
        int logical = (argc > 3) && strcmp(argv[3], "logical") == 0;
        return rung_buffer(htype, logical);
    }
    if (strcmp(argv[1], "tex") == 0) return rung_texture(htype);
    printf("unknown mode '%s'\n", argv[1]);
    return 1;
}
