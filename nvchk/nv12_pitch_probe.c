// nv12_pitch_probe.c — for an NV12 CUarray input, is NV_ENC_REGISTER_RESOURCE::pitch
// actually ignored?
//
// WHY THIS EXISTS.  encode_interop.rs registers its CUDAARRAY input with
// `pitch: 0`, justified by a P1.5 C-probe that swept pitch ∈ {0, width, width*4}
// against a PACKED SINGLE-PLANE R32Uint array and got byte-identical bitstreams.
// The conclusion recorded in the comment was "for CUDAARRAY the driver reads the
// stride from the array descriptor and ignores this field".
//
// nv12_probe.c encoded the same NV12 CUarray twice, once with pitch=width and
// once with pitch=0.  pitch=width produced correct pixels.  pitch=0 killed the
// process immediately after nvEncMapInputResource returned SUCCESS.
//
// This probe isolates that: one rung per pitch value, each in a FRESH session so
// a crash in one cannot be blamed on state left by another, and with the encode
// call bracketed by prints so the last line before a death names the call that
// died.  Run it once per pitch via argv so a crash still leaves the other
// results on record.
//
// Build: gcc -O1 -o nv12_pitch_probe.exe nv12_pitch_probe.c
// Run:   ./nv12_pitch_probe.exe <pitch>      e.g. 0, 256, 384, 1024

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "ffnvcodec/nvEncodeAPI_n12.2.72.0.h"

#define PROBE_W 256
#define PROBE_H 128

typedef int   CUresult_t;
typedef void *CUcontext_t;
typedef void *CUarray_t;
typedef int   CUdevice_t;

typedef struct {
    size_t   Width;
    size_t   Height;
    size_t   Depth;
    unsigned Format;
    unsigned NumChannels;
    unsigned Flags;
} CUDA_ARRAY3D_DESCRIPTOR_t;

typedef struct {
    size_t      srcXInBytes, srcY;
    unsigned    srcMemoryType;
    const void *srcHost;
    unsigned long long srcDevice;
    CUarray_t   srcArray;
    size_t      srcPitch;
    size_t      dstXInBytes, dstY;
    unsigned    dstMemoryType;
    void       *dstHost;
    unsigned long long dstDevice;
    CUarray_t   dstArray;
    size_t      dstPitch;
    size_t      WidthInBytes, Height;
} CUDA_MEMCPY2D_t;

typedef CUresult_t(__stdcall *pfn_cuInit)(unsigned);
typedef CUresult_t(__stdcall *pfn_cuDeviceGet)(CUdevice_t *, int);
typedef CUresult_t(__stdcall *pfn_cuCtxCreate)(CUcontext_t *, unsigned, CUdevice_t);
typedef CUresult_t(__stdcall *pfn_cuArray3DCreate)(CUarray_t *, const CUDA_ARRAY3D_DESCRIPTOR_t *);
typedef CUresult_t(__stdcall *pfn_cuMemcpy2D)(const CUDA_MEMCPY2D_t *);
typedef CUresult_t(__stdcall *pfn_cuCtxSynchronize)(void);

static const GUID CODEC_H264 = { 0x6bc82762, 0x4e63, 0x4ca4,
    { 0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf } };
static const GUID PRESET_P4  = { 0x90a7b826, 0xdf06, 0x4862,
    { 0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81 } };

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

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    uint32_t pitch = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : 0;
    // argv[2] selects the buffer format, so the same ladder can ask whether
    // pitch=0 is fatal for the PACKED single-plane format production uses today
    // (ABGR10) as well as for NV12.  P1.5 recorded pitch=0 as safe; that was
    // measured for ABGR10 only, and this is how that scope gets checked rather
    // than assumed.
    int use_abgr10 = (argc > 2 && strcmp(argv[2], "abgr10") == 0);
    const NV_ENC_BUFFER_FORMAT fmt = use_abgr10
        ? NV_ENC_BUFFER_FORMAT_ABGR10
        : NV_ENC_BUFFER_FORMAT_NV12;
    printf("nv12_pitch_probe: %dx%d %s CUarray, REGISTER_RESOURCE::pitch=%u\n\n",
           PROBE_W, PROBE_H, use_abgr10 ? "ABGR10" : "NV12", pitch);

    HMODULE cu = LoadLibraryA("nvcuda.dll");
    pfn_cuInit           cuInit_    = (pfn_cuInit)GetProcAddress(cu, "cuInit");
    pfn_cuDeviceGet      cuDevGet_  = (pfn_cuDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    pfn_cuCtxCreate      cuCtxNew_  = (pfn_cuCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    pfn_cuArray3DCreate  cuArrNew_  = (pfn_cuArray3DCreate)GetProcAddress(cu, "cuArray3DCreate_v2");
    pfn_cuMemcpy2D       cuCpy2D_   = (pfn_cuMemcpy2D)GetProcAddress(cu, "cuMemcpy2D_v2");
    pfn_cuCtxSynchronize cuSync_    = (pfn_cuCtxSynchronize)GetProcAddress(cu, "cuCtxSynchronize");

    if (cuInit_(0)) { printf("FAIL cuInit\n"); return 1; }
    CUdevice_t dev = 0;
    if (cuDevGet_(&dev, 0)) { printf("FAIL cuDeviceGet\n"); return 1; }
    CUcontext_t ctx = NULL;
    if (cuCtxNew_(&ctx, 0, dev)) { printf("FAIL cuCtxCreate\n"); return 1; }

    // Host source pattern.  Sized for the larger of the two layouts so the
    // ABGR10 rung has W*H*4 bytes to copy from.
    size_t nv12_sz   = (size_t)PROBE_W * PROBE_H * 3 / 2;
    size_t abgr10_sz = (size_t)PROBE_W * PROBE_H * 4;
    uint8_t *host = (uint8_t *)malloc(nv12_sz > abgr10_sz ? nv12_sz : abgr10_sz);
    memset(host, 0, nv12_sz > abgr10_sz ? nv12_sz : abgr10_sz);
    for (int y = 0; y < PROBE_H; y++)
        for (int x = 0; x < PROBE_W; x++) {
            int Y, U, V, bar = (x * 4) / PROBE_W;
            bt709_limited(BARS[bar][0], BARS[bar][1], BARS[bar][2], &Y, &U, &V);
            host[(size_t)y * PROBE_W + x] = (uint8_t)Y;
        }
    for (int y = 0; y < PROBE_H / 2; y++)
        for (int x = 0; x < PROBE_W / 2; x++) {
            int Y, U, V, bar = (x * 2 * 4) / PROBE_W;
            bt709_limited(BARS[bar][0], BARS[bar][1], BARS[bar][2], &Y, &U, &V);
            host[(size_t)PROBE_W * PROBE_H + (size_t)y * PROBE_W + x * 2 + 0] = (uint8_t)U;
            host[(size_t)PROBE_W * PROBE_H + (size_t)y * PROBE_W + x * 2 + 1] = (uint8_t)V;
        }

    CUarray_t arr = NULL;
    CUDA_ARRAY3D_DESCRIPTOR_t ad;
    memset(&ad, 0, sizeof(ad));
    if (use_abgr10) {
        // Packed 32-bit-per-pixel single plane, exactly what SharedTexture
        // allocates today for R32Uint.
        ad.Width = PROBE_W; ad.Height = PROBE_H; ad.Depth = 0;
        ad.Format = 0x03 /* CU_AD_FORMAT_UNSIGNED_INT32 */; ad.NumChannels = 1;
    } else {
        ad.Width = PROBE_W; ad.Height = (size_t)PROBE_H * 3 / 2; ad.Depth = 0;
        ad.Format = 0x01 /* CU_AD_FORMAT_UNSIGNED_INT8 */; ad.NumChannels = 1;
    }
    ad.Flags = 2 /* CUDA_ARRAY3D_SURFACE_LDST */;
    CUresult_t cr = cuArrNew_(&arr, &ad);
    printf("cuArray3DCreate(%zux%zu fmt=0x%x ch=%u SURFACE_LDST) -> %d\n",
           ad.Width, ad.Height, ad.Format, ad.NumChannels, cr);
    if (cr) return 1;

    CUDA_MEMCPY2D_t cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcMemoryType = 1; cp.srcHost = host;
    cp.srcPitch = use_abgr10 ? (size_t)PROBE_W * 4 : (size_t)PROBE_W;
    cp.dstMemoryType = 3; cp.dstArray = arr;
    cp.WidthInBytes = use_abgr10 ? (size_t)PROBE_W * 4 : (size_t)PROBE_W;
    cp.Height = use_abgr10 ? (size_t)PROBE_H : (size_t)PROBE_H * 3 / 2;
    cr = cuCpy2D_(&cp);
    printf("cuMemcpy2D HtoA -> %d\n", cr);
    if (cuSync_) cuSync_();

    // Fresh NVENC session, so nothing this probe measures can be contamination
    // from an earlier rung.
    HMODULE nv = LoadLibraryA("nvEncodeAPI64.dll");
    typedef NVENCSTATUS(NVENCAPI *pfn_create)(NV_ENCODE_API_FUNCTION_LIST *);
    pfn_create create = (pfn_create)GetProcAddress(nv, "NvEncodeAPICreateInstance");
    NV_ENCODE_API_FUNCTION_LIST fl;
    memset(&fl, 0, sizeof(fl));
    fl.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS st = create(&fl);
    if (st) { printf("FAIL create instance %d\n", st); return 1; }

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
    printf("nvEncInitializeEncoder   -> %s\n", nv_name(st));
    if (st) return 1;

    NV_ENC_REGISTER_RESOURCE reg;
    memset(&reg, 0, sizeof(reg));
    reg.version = NV_ENC_REGISTER_RESOURCE_VER;
    reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY;
    reg.width = PROBE_W; reg.height = PROBE_H;
    reg.pitch = pitch;
    reg.resourceToRegister = arr;
    reg.bufferFormat = fmt;
    reg.bufferUsage = NV_ENC_INPUT_IMAGE;
    st = fl.nvEncRegisterResource(session, &reg);
    printf("nvEncRegisterResource(pitch=%u) -> %s\n", pitch, nv_name(st));
    if (st) { printf("VERDICT: pitch=%u REJECTED at register\n", pitch); return 2; }

    NV_ENC_CREATE_BITSTREAM_BUFFER bs;
    memset(&bs, 0, sizeof(bs));
    bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    st = fl.nvEncCreateBitstreamBuffer(session, &bs);
    printf("nvEncCreateBitstreamBuffer -> %s\n", nv_name(st));

    NV_ENC_MAP_INPUT_RESOURCE map;
    memset(&map, 0, sizeof(map));
    map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map.registeredResource = reg.registeredResource;
    st = fl.nvEncMapInputResource(session, &map);
    printf("nvEncMapInputResource -> %s  mappedBufferFmt=0x%08X\n",
           nv_name(st), (unsigned)map.mappedBufferFmt);
    if (st) { printf("VERDICT: pitch=%u REJECTED at map\n", pitch); return 3; }

    NV_ENC_PIC_PARAMS pic;
    memset(&pic, 0, sizeof(pic));
    pic.version = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth = PROBE_W; pic.inputHeight = PROBE_H;
    pic.inputPitch = use_abgr10 ? (uint32_t)PROBE_W * 4 : (uint32_t)PROBE_W;
    pic.inputBuffer = map.mappedResource;
    pic.outputBitstream = bs.bitstreamBuffer;
    pic.bufferFmt = fmt;
    pic.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
    pic.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;

    printf("about to call nvEncEncodePicture (if the next line is missing, IT CRASHED)\n");
    st = fl.nvEncEncodePicture(session, &pic);
    printf("nvEncEncodePicture -> %s\n", nv_name(st));

    if (st == NV_ENC_SUCCESS) {
        NV_ENC_LOCK_BITSTREAM lock;
        memset(&lock, 0, sizeof(lock));
        lock.version = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = bs.bitstreamBuffer;
        st = fl.nvEncLockBitstream(session, &lock);
        printf("nvEncLockBitstream -> %s  %u bytes\n",
               nv_name(st), (unsigned)lock.bitstreamSizeInBytes);
        if (st == NV_ENC_SUCCESS) {
            char path[64];
            snprintf(path, sizeof(path), "pitch_%s_%u.h264", use_abgr10 ? "abgr10" : "nv12", pitch);
            FILE *f = fopen(path, "wb");
            fwrite(lock.bitstreamBufferPtr, 1, lock.bitstreamSizeInBytes, f);
            fclose(f);
            printf("wrote %s\n", path);
            fl.nvEncUnlockBitstream(session, bs.bitstreamBuffer);
        }
    }

    printf("about to unmap (if the next line is missing, UNMAP CRASHED)\n");
    fl.nvEncUnmapInputResource(session, map.mappedResource);
    printf("unmapped OK\n");
    fl.nvEncDestroyBitstreamBuffer(session, bs.bitstreamBuffer);
    fl.nvEncUnregisterResource(session, reg.registeredResource);
    fl.nvEncDestroyEncoder(session);
    printf("VERDICT: pitch=%u survived the full cycle\n", pitch);
    return 0;
}
