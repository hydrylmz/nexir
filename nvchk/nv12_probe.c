// nv12_probe.c — which NV12 allocation shape does NVENC accept as a zero-copy input?
//
// The zero-copy export path currently feeds NVENC packed ABGR10 (one 32-bit word
// per pixel, one plane) as a CUarray imported from a D3D12 texture.  Moving to
// NV12 means TWO planes in one allocation, and the two candidate shapes are not
// equally well documented:
//
//   Rung A  linear device buffer, NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
//           pitch = row stride in bytes, chroma implicitly at pitch*height.
//           This is what libavcodec's h264_nvenc does (nvenc.c:2269-2288).
//
//   Rung B  a 2D CUarray of width x (height*3/2) bytes, CU_AD_FORMAT_UNSIGNED_INT8
//           1 channel, NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY, pitch = width.
//           This is the shape that would drop into the existing SharedTexture
//           path with only a format change.  Nothing documents that NVENC reads
//           the chroma plane from row `height` of an over-tall array.
//
//   Rung B0 same as B but pitch = 0, i.e. what encode_interop.rs passes today.
//
// Each rung encodes ONE known pattern frame and writes a .h264 file.  Decoding
// those files and checking the pixels is what proves the plane interpretation —
// a rung that merely returns NV_ENC_SUCCESS proves nothing, because NVENC will
// happily encode garbage.
//
// SECOND QUESTION, added for P1.2: is the CODEC GUID in
// `src/interop/encode_interop.rs` byte-for-byte what the vendor header declares?
//
// `--codec hevc` initialises an HEVC session instead of H.264 and, either way,
// dumps the selected GUID's RAW MEMORY BYTES next to the byte array the Rust
// side hard-codes.  That comparison is the whole point: a GUID's first four
// bytes are `Data1` in little-endian order, so {790CDC88-...} is `88 dc 0c 79`,
// and the Rust constant read `88 cd 0c 79` — one transposed nibble.  The only
// symptom was nvEncInitializeEncoder returning NV_ENC_ERR_UNSUPPORTED_PARAM (12)
// for every HEVC job at every resolution, which is indistinguishable from "this
// GPU has no HEVC encoder" and silently routed every H.265 export to the FFmpeg
// path.  A wrong GUID cannot be caught by any struct-layout probe, and the driver
// never names the field it rejected — hence this.
//
// Build: gcc -O1 -o nv12_probe.exe nv12_probe.c
// Run:   ./nv12_probe.exe [--codec h264|hevc]

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include "ffnvcodec/nvEncodeAPI_n12.2.72.0.h"

#define PROBE_W 256
#define PROBE_H 128

// ---------------------------------------------------------------------------
// CUDA driver API, loaded dynamically (no import library, no build entanglement)
// ---------------------------------------------------------------------------
typedef int   CUresult_t;
typedef void *CUcontext_t;
typedef void *CUarray_t;
typedef int   CUdevice_t;
typedef unsigned long long CUdeviceptr_t;

typedef struct {
    size_t   Width;
    size_t   Height;
    size_t   Depth;
    unsigned Format;
    unsigned NumChannels;
    unsigned Flags;
} CUDA_ARRAY3D_DESCRIPTOR_t;

typedef struct {
    size_t        srcXInBytes;
    size_t        srcY;
    unsigned      srcMemoryType;
    const void   *srcHost;
    CUdeviceptr_t srcDevice;
    CUarray_t     srcArray;
    size_t        srcPitch;
    size_t        dstXInBytes;
    size_t        dstY;
    unsigned      dstMemoryType;
    void         *dstHost;
    CUdeviceptr_t dstDevice;
    CUarray_t     dstArray;
    size_t        dstPitch;
    size_t        WidthInBytes;
    size_t        Height;
} CUDA_MEMCPY2D_t;

#define CU_MEMORYTYPE_HOST_T   1u
#define CU_MEMORYTYPE_ARRAY_T  3u
#define CU_AD_FORMAT_U8_T      0x01u
#define CUDA_ARRAY3D_SURFACE_LDST_T 2u

typedef CUresult_t(__stdcall *pfn_cuInit)(unsigned);
typedef CUresult_t(__stdcall *pfn_cuDeviceGet)(CUdevice_t *, int);
typedef CUresult_t(__stdcall *pfn_cuCtxCreate)(CUcontext_t *, unsigned, CUdevice_t);
typedef CUresult_t(__stdcall *pfn_cuCtxDestroy)(CUcontext_t);
typedef CUresult_t(__stdcall *pfn_cuMemAlloc)(CUdeviceptr_t *, size_t);
typedef CUresult_t(__stdcall *pfn_cuMemFree)(CUdeviceptr_t);
typedef CUresult_t(__stdcall *pfn_cuMemcpyHtoD)(CUdeviceptr_t, const void *, size_t);
typedef CUresult_t(__stdcall *pfn_cuArray3DCreate)(CUarray_t *, const CUDA_ARRAY3D_DESCRIPTOR_t *);
typedef CUresult_t(__stdcall *pfn_cuArrayDestroy)(CUarray_t);
typedef CUresult_t(__stdcall *pfn_cuMemcpy2D)(const CUDA_MEMCPY2D_t *);
typedef CUresult_t(__stdcall *pfn_cuCtxSynchronize)(void);
typedef CUresult_t(__stdcall *pfn_cuGetErrorName)(CUresult_t, const char **);

static pfn_cuInit          cuInit_;
static pfn_cuDeviceGet     cuDeviceGet_;
static pfn_cuCtxCreate     cuCtxCreate_;
static pfn_cuCtxDestroy    cuCtxDestroy_;
static pfn_cuMemAlloc      cuMemAlloc_;
static pfn_cuMemFree       cuMemFree_;
static pfn_cuMemcpyHtoD    cuMemcpyHtoD_;
static pfn_cuArray3DCreate cuArray3DCreate_;
static pfn_cuArrayDestroy  cuArrayDestroy_;
static pfn_cuMemcpy2D      cuMemcpy2D_;
static pfn_cuCtxSynchronize cuCtxSynchronize_;
static pfn_cuGetErrorName  cuGetErrorName_;

static const char *cu_name(CUresult_t r) {
    const char *n = NULL;
    if (cuGetErrorName_ && cuGetErrorName_(r, &n) == 0 && n) return n;
    return "(unknown CUresult)";
}

// ---------------------------------------------------------------------------
// NVENC status names.  A bare integer is what makes these probes waste a day.
// ---------------------------------------------------------------------------
static const char *nv_name(NVENCSTATUS s) {
    switch (s) {
    case NV_ENC_SUCCESS:                      return "NV_ENC_SUCCESS";
    case NV_ENC_ERR_NO_ENCODE_DEVICE:         return "NV_ENC_ERR_NO_ENCODE_DEVICE";
    case NV_ENC_ERR_UNSUPPORTED_DEVICE:       return "NV_ENC_ERR_UNSUPPORTED_DEVICE";
    case NV_ENC_ERR_INVALID_ENCODERDEVICE:    return "NV_ENC_ERR_INVALID_ENCODERDEVICE";
    case NV_ENC_ERR_INVALID_DEVICE:           return "NV_ENC_ERR_INVALID_DEVICE";
    case NV_ENC_ERR_DEVICE_NOT_EXIST:         return "NV_ENC_ERR_DEVICE_NOT_EXIST";
    case NV_ENC_ERR_INVALID_PTR:              return "NV_ENC_ERR_INVALID_PTR";
    case NV_ENC_ERR_INVALID_EVENT:            return "NV_ENC_ERR_INVALID_EVENT";
    case NV_ENC_ERR_INVALID_PARAM:            return "NV_ENC_ERR_INVALID_PARAM";
    case NV_ENC_ERR_INVALID_CALL:             return "NV_ENC_ERR_INVALID_CALL";
    case NV_ENC_ERR_OUT_OF_MEMORY:            return "NV_ENC_ERR_OUT_OF_MEMORY";
    case NV_ENC_ERR_ENCODER_NOT_INITIALIZED:  return "NV_ENC_ERR_ENCODER_NOT_INITIALIZED";
    case NV_ENC_ERR_UNSUPPORTED_PARAM:        return "NV_ENC_ERR_UNSUPPORTED_PARAM";
    case NV_ENC_ERR_LOCK_BUSY:                return "NV_ENC_ERR_LOCK_BUSY";
    case NV_ENC_ERR_NOT_ENOUGH_BUFFER:        return "NV_ENC_ERR_NOT_ENOUGH_BUFFER";
    case NV_ENC_ERR_INVALID_VERSION:          return "NV_ENC_ERR_INVALID_VERSION";
    case NV_ENC_ERR_MAP_FAILED:               return "NV_ENC_ERR_MAP_FAILED";
    case NV_ENC_ERR_NEED_MORE_INPUT:          return "NV_ENC_ERR_NEED_MORE_INPUT";
    case NV_ENC_ERR_ENCODER_BUSY:             return "NV_ENC_ERR_ENCODER_BUSY";
    case NV_ENC_ERR_EVENT_NOT_REGISTERD:      return "NV_ENC_ERR_EVENT_NOT_REGISTERD";
    case NV_ENC_ERR_GENERIC:                  return "NV_ENC_ERR_GENERIC";
    case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY:  return "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY";
    case NV_ENC_ERR_UNIMPLEMENTED:            return "NV_ENC_ERR_UNIMPLEMENTED";
    case NV_ENC_ERR_RESOURCE_REGISTER_FAILED: return "NV_ENC_ERR_RESOURCE_REGISTER_FAILED";
    case NV_ENC_ERR_RESOURCE_NOT_REGISTERED:  return "NV_ENC_ERR_RESOURCE_NOT_REGISTERED";
    case NV_ENC_ERR_RESOURCE_NOT_MAPPED:      return "NV_ENC_ERR_RESOURCE_NOT_MAPPED";
    default:                                  return "(unnamed NVENCSTATUS)";
    }
}

// Codec GUIDs are taken from the VENDOR HEADER's own `static const GUID`
// definitions rather than transcribed here.  That is deliberate: a hand-copied
// GUID in the probe could carry the same typo as the one in the Rust FFI and the
// two would agree while both being wrong, which is exactly the failure mode this
// is meant to detect.  `NV_ENC_CODEC_H264_GUID` / `NV_ENC_CODEC_HEVC_GUID` come
// straight from ffnvcodec/nvEncodeAPI_n12.2.72.0.h:144-149.
static const GUID PRESET_P4  = { 0x90a7b826, 0xdf06, 0x4862,
    { 0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81 } };

// What `src/interop/encode_interop.rs` hard-codes, byte for byte, so this probe
// fails loudly if the two ever diverge again.  Keep in sync with the two
// `NV_ENC_CODEC_*_GUID` arrays there.
static const uint8_t RUST_H264_GUID[16] = {
    0x62, 0x27, 0xC8, 0x6B, 0x63, 0x4E, 0xa4, 0x4c,
    0xAA, 0x85, 0x1E, 0x50, 0xF3, 0x21, 0xF6, 0xBF,
};
static const uint8_t RUST_HEVC_GUID[16] = {
    0x88, 0xDC, 0x0C, 0x79, 0x22, 0x45, 0x7B, 0x4d,
    0x94, 0x25, 0xBD, 0xA9, 0x97, 0x5F, 0x76, 0x03,
};

/// Print a GUID's raw memory bytes beside what the Rust side declares, and
/// return 1 when they match.
///
/// The comparison is on RAW BYTES, not on the `{Data1-Data2-...}` text form:
/// `Data1` is a `unsigned long` and therefore little-endian in memory, so
/// {790CDC88-...} is `88 dc 0c 79` and a byte array that "looks like" the
/// printed GUID is wrong.  That is the mistake this catches.
static int compare_guid(const char *label, const GUID *header_guid,
                        const uint8_t *rust_bytes) {
    const uint8_t *hb = (const uint8_t *)header_guid;
    int same = memcmp(hb, rust_bytes, 16) == 0;

    printf("%s GUID:\n", label);
    printf("   vendor header bytes :");
    for (int i = 0; i < 16; i++) printf(" %02x", hb[i]);
    printf("\n   Rust FFI bytes      :");
    for (int i = 0; i < 16; i++) printf(" %02x", rust_bytes[i]);
    printf("\n   -> %s\n", same ? "MATCH" : "*** MISMATCH ***");

    if (!same) {
        printf("   first differing byte:");
        for (int i = 0; i < 16; i++) {
            if (hb[i] != rust_bytes[i]) {
                printf(" index %d, header 0x%02x vs Rust 0x%02x\n", i, hb[i], rust_bytes[i]);
                break;
            }
        }
        printf("   A wrong codec GUID makes nvEncInitializeEncoder return\n"
               "   NV_ENC_ERR_UNSUPPORTED_PARAM (12) with no indication of which\n"
               "   field was rejected — it looks like missing hardware support.\n");
    }
    return same;
}

// ---------------------------------------------------------------------------
// The test pattern, in NV12, BT.709 limited range, computed here so the
// expectation is independent of anything under test.
// ---------------------------------------------------------------------------
static void bt709_limited(double r, double g, double b, int *Y, int *U, int *V) {
    const double kr = 0.2126, kb = 0.0722, kg = 1.0 - kr - kb;
    double y  = kr * r + kg * g + kb * b;
    double cb = (b - y) / (2.0 * (1.0 - kb));
    double cr = (r - y) / (2.0 * (1.0 - kr));
    *Y = (int)(16.0  + 219.0 * y  + 0.5);
    *U = (int)(128.0 + 224.0 * cb + 0.5);
    *V = (int)(128.0 + 224.0 * cr + 0.5);
    if (*Y < 0) *Y = 0; if (*Y > 255) *Y = 255;
    if (*U < 0) *U = 0; if (*U > 255) *U = 255;
    if (*V < 0) *V = 0; if (*V > 255) *V = 255;
}

// Four vertical bars: red, green, blue, white.
static const double BARS[4][3] = {
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 },
    { 1.0, 1.0, 1.0 },
};

/// Fill `out` (w*h*3/2 bytes, pitch == w) with the NV12 pattern and print the
/// codes, so the expected decode is on the record before anything runs.
static void build_nv12(uint8_t *out, int w, int h) {
    uint8_t *luma   = out;
    uint8_t *chroma = out + (size_t)w * h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int bar = (x * 4) / w;
            int Y, U, V;
            bt709_limited(BARS[bar][0], BARS[bar][1], BARS[bar][2], &Y, &U, &V);
            luma[(size_t)y * w + x] = (uint8_t)Y;
        }
    }
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < w / 2; x++) {
            int bar = (x * 2 * 4) / w;
            int Y, U, V;
            bt709_limited(BARS[bar][0], BARS[bar][1], BARS[bar][2], &Y, &U, &V);
            chroma[(size_t)y * w + x * 2 + 0] = (uint8_t)U;
            chroma[(size_t)y * w + x * 2 + 1] = (uint8_t)V;
        }
    }
    printf("pattern (BT.709 limited, NV12):\n");
    for (int i = 0; i < 4; i++) {
        int Y, U, V;
        bt709_limited(BARS[i][0], BARS[i][1], BARS[i][2], &Y, &U, &V);
        printf("   bar %d  rgb(%.0f,%.0f,%.0f)  ->  Y=%3d U=%3d V=%3d\n",
               i, BARS[i][0] * 255, BARS[i][1] * 255, BARS[i][2] * 255, Y, U, V);
    }
}

// ---------------------------------------------------------------------------
// One rung: register `resource` with the given type/pitch, encode one frame.
// ---------------------------------------------------------------------------
static int run_rung(NV_ENCODE_API_FUNCTION_LIST *fl,
                    void *session,
                    const char *label,
                    NV_ENC_INPUT_RESOURCE_TYPE res_type,
                    void *resource,
                    uint32_t pitch,
                    const char *out_path) {
    printf("\n== rung %s: resourceType=%d pitch=%u ==\n", label, (int)res_type, pitch);

    NV_ENC_REGISTER_RESOURCE reg;
    memset(&reg, 0, sizeof(reg));
    reg.version            = NV_ENC_REGISTER_RESOURCE_VER;
    reg.resourceType       = res_type;
    reg.width              = PROBE_W;
    reg.height             = PROBE_H;
    reg.pitch              = pitch;
    reg.resourceToRegister = resource;
    reg.bufferFormat       = NV_ENC_BUFFER_FORMAT_NV12;
    reg.bufferUsage        = NV_ENC_INPUT_IMAGE;

    NVENCSTATUS st = fl->nvEncRegisterResource(session, &reg);
    printf("   nvEncRegisterResource      -> %s (%d)\n", nv_name(st), st);
    if (st != NV_ENC_SUCCESS) return 0;

    NV_ENC_CREATE_BITSTREAM_BUFFER bs;
    memset(&bs, 0, sizeof(bs));
    bs.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    st = fl->nvEncCreateBitstreamBuffer(session, &bs);
    printf("   nvEncCreateBitstreamBuffer -> %s (%d)\n", nv_name(st), st);
    if (st != NV_ENC_SUCCESS) { fl->nvEncUnregisterResource(session, reg.registeredResource); return 0; }

    NV_ENC_MAP_INPUT_RESOURCE map;
    memset(&map, 0, sizeof(map));
    map.version            = NV_ENC_MAP_INPUT_RESOURCE_VER;
    map.registeredResource = reg.registeredResource;
    st = fl->nvEncMapInputResource(session, &map);
    printf("   nvEncMapInputResource      -> %s (%d)  mappedBufferFmt=0x%08X\n",
           nv_name(st), st, (unsigned)map.mappedBufferFmt);
    if (st != NV_ENC_SUCCESS) {
        fl->nvEncDestroyBitstreamBuffer(session, bs.bitstreamBuffer);
        fl->nvEncUnregisterResource(session, reg.registeredResource);
        return 0;
    }

    NV_ENC_PIC_PARAMS pic;
    memset(&pic, 0, sizeof(pic));
    pic.version         = NV_ENC_PIC_PARAMS_VER;
    pic.inputWidth      = PROBE_W;
    pic.inputHeight     = PROBE_H;
    pic.inputPitch      = pitch ? pitch : PROBE_W;
    pic.inputBuffer     = map.mappedResource;
    pic.outputBitstream = bs.bitstreamBuffer;
    pic.bufferFmt       = NV_ENC_BUFFER_FORMAT_NV12;
    pic.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;
    pic.encodePicFlags  = NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
    pic.inputTimeStamp  = 0;
    st = fl->nvEncEncodePicture(session, &pic);
    printf("   nvEncEncodePicture         -> %s (%d)\n", nv_name(st), st);

    if (st == NV_ENC_ERR_NEED_MORE_INPUT) {
        NV_ENC_PIC_PARAMS eos;
        memset(&eos, 0, sizeof(eos));
        eos.version        = NV_ENC_PIC_PARAMS_VER;
        eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
        st = fl->nvEncEncodePicture(session, &eos);
        printf("   nvEncEncodePicture(EOS)    -> %s (%d)\n", nv_name(st), st);
    }

    int ok = 0;
    if (st == NV_ENC_SUCCESS) {
        NV_ENC_LOCK_BITSTREAM lock;
        memset(&lock, 0, sizeof(lock));
        lock.version         = NV_ENC_LOCK_BITSTREAM_VER;
        lock.outputBitstream = bs.bitstreamBuffer;
        st = fl->nvEncLockBitstream(session, &lock);
        printf("   nvEncLockBitstream         -> %s (%d)  %u bytes  picType=%u\n",
               nv_name(st), st, (unsigned)lock.bitstreamSizeInBytes, (unsigned)lock.pictureType);
        if (st == NV_ENC_SUCCESS) {
            FILE *f = fopen(out_path, "wb");
            if (f) {
                fwrite(lock.bitstreamBufferPtr, 1, lock.bitstreamSizeInBytes, f);
                fclose(f);
                printf("   wrote %s\n", out_path);
                ok = 1;
            } else {
                printf("   FAILED to open %s for writing\n", out_path);
            }
            fl->nvEncUnlockBitstream(session, bs.bitstreamBuffer);
        }
    }

    fl->nvEncUnmapInputResource(session, map.mappedResource);
    fl->nvEncDestroyBitstreamBuffer(session, bs.bitstreamBuffer);
    fl->nvEncUnregisterResource(session, reg.registeredResource);
    return ok;
}

int main(int argc, char **argv) {
    // Unbuffered: a crash mid-probe must not swallow the lines that say how far
    // it got, which is the only diagnostic a driver access violation leaves.
    setvbuf(stdout, NULL, _IONBF, 0);

    // ---- which codec ----
    int want_hevc = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
            i++;
            if      (strcmp(argv[i], "hevc") == 0 || strcmp(argv[i], "h265") == 0) want_hevc = 1;
            else if (strcmp(argv[i], "h264") == 0)                                 want_hevc = 0;
            else { printf("unknown --codec '%s' (want h264 or hevc)\n", argv[i]); return 2; }
        } else {
            printf("usage: %s [--codec h264|hevc]\n", argv[0]);
            return 2;
        }
    }
    const char *codec_label = want_hevc ? "HEVC" : "H264";
    const GUID *codec_guid  = want_hevc ? &NV_ENC_CODEC_HEVC_GUID : &NV_ENC_CODEC_H264_GUID;
    const uint8_t *rust_guid = want_hevc ? RUST_HEVC_GUID : RUST_H264_GUID;
    const char *ext = want_hevc ? "hevc" : "h264";

    printf("nv12_probe: NVENC NV12 zero-copy input shape, %dx%d, codec %s\n",
           PROBE_W, PROBE_H, codec_label);
    printf("compiled against header API %u.%u\n\n",
           NVENCAPI_MAJOR_VERSION, NVENCAPI_MINOR_VERSION);

    // ---- the GUID question, answered before any driver call ----
    // Deliberately first: it needs no CUDA context and no session, so it still
    // reports even on a machine where everything below is unavailable.
    int guid_ok = compare_guid(codec_label, codec_guid, rust_guid);
    printf("\n");

    // ---- CUDA ----
    HMODULE cu = LoadLibraryA("nvcuda.dll");
    if (!cu) { printf("FAIL LoadLibrary(nvcuda.dll) err=%lu\n", GetLastError()); return 1; }
    cuInit_          = (pfn_cuInit)GetProcAddress(cu, "cuInit");
    cuDeviceGet_     = (pfn_cuDeviceGet)GetProcAddress(cu, "cuDeviceGet");
    cuCtxCreate_     = (pfn_cuCtxCreate)GetProcAddress(cu, "cuCtxCreate_v2");
    cuCtxDestroy_    = (pfn_cuCtxDestroy)GetProcAddress(cu, "cuCtxDestroy_v2");
    cuMemAlloc_      = (pfn_cuMemAlloc)GetProcAddress(cu, "cuMemAlloc_v2");
    cuMemFree_       = (pfn_cuMemFree)GetProcAddress(cu, "cuMemFree_v2");
    cuMemcpyHtoD_    = (pfn_cuMemcpyHtoD)GetProcAddress(cu, "cuMemcpyHtoD_v2");
    cuArray3DCreate_ = (pfn_cuArray3DCreate)GetProcAddress(cu, "cuArray3DCreate_v2");
    cuArrayDestroy_  = (pfn_cuArrayDestroy)GetProcAddress(cu, "cuArrayDestroy");
    cuMemcpy2D_      = (pfn_cuMemcpy2D)GetProcAddress(cu, "cuMemcpy2D_v2");
    cuCtxSynchronize_= (pfn_cuCtxSynchronize)GetProcAddress(cu, "cuCtxSynchronize");
    cuGetErrorName_  = (pfn_cuGetErrorName)GetProcAddress(cu, "cuGetErrorName");
    if (!cuInit_ || !cuCtxCreate_ || !cuMemAlloc_ || !cuArray3DCreate_ || !cuMemcpy2D_) {
        printf("FAIL: a required cuda entry point is missing\n"); return 1;
    }

    CUresult_t cr = cuInit_(0);
    if (cr) { printf("FAIL cuInit -> %s (%d)\n", cu_name(cr), cr); return 1; }
    CUdevice_t dev = 0;
    cr = cuDeviceGet_(&dev, 0);
    if (cr) { printf("FAIL cuDeviceGet -> %s (%d)\n", cu_name(cr), cr); return 1; }
    CUcontext_t ctx = NULL;
    cr = cuCtxCreate_(&ctx, 0, dev);
    if (cr) { printf("FAIL cuCtxCreate -> %s (%d)\n", cu_name(cr), cr); return 1; }
    printf("CUDA context OK\n");

    // ---- NVENC ----
    HMODULE nv = LoadLibraryA("nvEncodeAPI64.dll");
    if (!nv) { printf("FAIL LoadLibrary(nvEncodeAPI64.dll) err=%lu\n", GetLastError()); return 1; }
    typedef NVENCSTATUS(NVENCAPI *pfn_create)(NV_ENCODE_API_FUNCTION_LIST *);
    pfn_create create = (pfn_create)GetProcAddress(nv, "NvEncodeAPICreateInstance");
    if (!create) { printf("FAIL GetProcAddress(NvEncodeAPICreateInstance)\n"); return 1; }

    NV_ENCODE_API_FUNCTION_LIST fl;
    memset(&fl, 0, sizeof(fl));
    fl.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NVENCSTATUS st = create(&fl);
    printf("NvEncodeAPICreateInstance -> %s (%d)\n", nv_name(st), st);
    if (st != NV_ENC_SUCCESS) return 1;

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op;
    memset(&op, 0, sizeof(op));
    op.version    = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    op.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    op.device     = ctx;
    op.apiVersion = NVENCAPI_VERSION;
    void *session = NULL;
    st = fl.nvEncOpenEncodeSessionEx(&op, &session);
    printf("nvEncOpenEncodeSessionEx  -> %s (%d)\n", nv_name(st), st);
    if (st != NV_ENC_SUCCESS) return 1;

    NV_ENC_INITIALIZE_PARAMS init;
    memset(&init, 0, sizeof(init));
    init.version         = NV_ENC_INITIALIZE_PARAMS_VER;
    // Initialise with the RUST FFI's bytes, not the header's.
    //
    // This is the load-bearing choice in the probe.  Using the header GUID here
    // would make the session succeed even when the Rust constant is wrong — the
    // byte diff above would report the mismatch, but the driver half of the check
    // would pass regardless and prove nothing about the shipping code.  Feeding
    // the driver exactly what `EncodeInterop::open` feeds it means a typo in
    // `src/interop/encode_interop.rs` REPRODUCES the real failure here:
    // NV_ENC_ERR_UNSUPPORTED_PARAM (12).  Verified by flipping one nibble of
    // RUST_HEVC_GUID and re-running.
    memcpy(&init.encodeGUID, rust_guid, 16);
    init.presetGUID      = PRESET_P4;
    init.encodeWidth     = PROBE_W;
    init.encodeHeight    = PROBE_H;
    init.darWidth        = PROBE_W;
    init.darHeight       = PROBE_H;
    init.frameRateNum    = 30;
    init.frameRateDen    = 1;
    init.enableEncodeAsync = 0;
    init.enablePTD       = 1;
    init.maxEncodeWidth  = PROBE_W;
    init.maxEncodeHeight = PROBE_H;
    init.tuningInfo      = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    st = fl.nvEncInitializeEncoder(session, &init);
    printf("nvEncInitializeEncoder(%s, Rust GUID bytes) -> %s (%d)\n",
           codec_label, nv_name(st), st);
    if (st != NV_ENC_SUCCESS) {
        // The GUID is the field this probe exists to rule out, so say so rather
        // than leaving the reader to conclude the GPU lacks the codec.
        if (st == NV_ENC_ERR_UNSUPPORTED_PARAM) {
            printf("   NV_ENC_ERR_UNSUPPORTED_PARAM names no field.  The GUID\n"
                   "   comparison above %s, so %s.\n",
                   guid_ok ? "MATCHED" : "FAILED",
                   guid_ok ? "look elsewhere (preset, tuningInfo, or the GPU "
                             "genuinely lacks this codec)"
                           : "that mismatch IS the cause — this is the exact "
                             "failure the shipping code hit");
        }
        return 1;
    }

    // The positive half of the GUID check, and it is only meaningful because the
    // bytes above came from the Rust constant: the driver accepts what our FFI
    // sends as a codec it actually implements.
    printf("   session initialised for %s — the driver accepts the Rust FFI's\n"
           "   codec GUID bytes as a codec it implements\n", codec_label);

    // ---- the pattern ----
    size_t nv12_size = (size_t)PROBE_W * PROBE_H * 3 / 2;
    uint8_t *host = (uint8_t *)malloc(nv12_size);
    if (!host) { printf("FAIL malloc\n"); return 1; }
    build_nv12(host, PROBE_W, PROBE_H);

    int a_ok = 0, b_ok = 0, b0_ok = 0;

    // Bitstream filenames carry the codec so an HEVC run cannot silently
    // overwrite an H.264 one and be decoded with the wrong -f later.
    char path_a[64], path_b[64], path_b0[64];
    snprintf(path_a,  sizeof(path_a),  "rung_a.%s",  ext);
    snprintf(path_b,  sizeof(path_b),  "rung_b.%s",  ext);
    snprintf(path_b0, sizeof(path_b0), "rung_b0.%s", ext);

    // ---- rung A: linear device buffer, CUDADEVICEPTR ----
    CUdeviceptr_t dptr = 0;
    cr = cuMemAlloc_(&dptr, nv12_size);
    printf("\ncuMemAlloc(%zu) -> %s (%d)\n", nv12_size, cu_name(cr), cr);
    if (cr == 0) {
        cr = cuMemcpyHtoD_(dptr, host, nv12_size);
        printf("cuMemcpyHtoD   -> %s (%d)\n", cu_name(cr), cr);
        if (cuCtxSynchronize_) cuCtxSynchronize_();
        if (cr == 0) {
            a_ok = run_rung(&fl, session, "A (linear buffer, CUDADEVICEPTR, pitch=W)",
                            NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                            (void *)(uintptr_t)dptr, PROBE_W, path_a);
        }
    }

    // ---- rung B: over-tall 2D CUarray, CUDAARRAY ----
    CUarray_t arr = NULL;
    CUDA_ARRAY3D_DESCRIPTOR_t ad;
    memset(&ad, 0, sizeof(ad));
    ad.Width       = PROBE_W;
    ad.Height      = (size_t)PROBE_H * 3 / 2;
    ad.Depth       = 0;
    ad.Format      = CU_AD_FORMAT_U8_T;
    ad.NumChannels = 1;
    ad.Flags       = CUDA_ARRAY3D_SURFACE_LDST_T;
    cr = cuArray3DCreate_(&arr, &ad);
    printf("\ncuArray3DCreate(%dx%zu, U8x1, SURFACE_LDST) -> %s (%d)\n",
           PROBE_W, ad.Height, cu_name(cr), cr);
    if (cr == 0) {
        CUDA_MEMCPY2D_t cp;
        memset(&cp, 0, sizeof(cp));
        cp.srcMemoryType = CU_MEMORYTYPE_HOST_T;
        cp.srcHost       = host;
        cp.srcPitch      = PROBE_W;
        cp.dstMemoryType = CU_MEMORYTYPE_ARRAY_T;
        cp.dstArray      = arr;
        cp.WidthInBytes  = PROBE_W;
        cp.Height        = (size_t)PROBE_H * 3 / 2;
        cr = cuMemcpy2D_(&cp);
        printf("cuMemcpy2D HtoA -> %s (%d)\n", cu_name(cr), cr);
        if (cuCtxSynchronize_) cuCtxSynchronize_();
        if (cr == 0) {
            b_ok = run_rung(&fl, session, "B (CUarray Wx1.5H, CUDAARRAY, pitch=W)",
                            NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
                            arr, PROBE_W, path_b);
            b0_ok = run_rung(&fl, session, "B0 (same array, CUDAARRAY, pitch=0 as today)",
                             NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY,
                             arr, 0, path_b0);
        }
    }

    printf("\n---- summary ----\n");
    printf("codec %s GUID vs Rust FFI                          : %s\n",
           codec_label, guid_ok ? "MATCH" : "*** MISMATCH ***");
    printf("rung A  (linear buffer / CUDADEVICEPTR / pitch=W) : %s\n", a_ok ? "bitstream written" : "NO OUTPUT");
    printf("rung B  (CUarray Wx1.5H / CUDAARRAY / pitch=W)    : %s\n", b_ok ? "bitstream written" : "NO OUTPUT");
    printf("rung B0 (CUarray Wx1.5H / CUDAARRAY / pitch=0)    : %s\n", b0_ok ? "bitstream written" : "NO OUTPUT");
    printf("A written bitstream still proves nothing about PIXELS — decode and compare.\n");

    if (arr)  cuArrayDestroy_(arr);
    if (dptr) cuMemFree_(dptr);
    free(host);
    fl.nvEncDestroyEncoder(session);
    if (cuCtxDestroy_) cuCtxDestroy_(ctx);
    printf("\ndone.\n");
    return 0;
}
