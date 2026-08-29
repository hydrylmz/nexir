// P3.2 ground-truth probe: compile against a real ffnvcodec nvEncodeAPI.h and
// print the ABGR10 enumerant plus every struct size Nexir's FFI asserts.
// Build: gcc -o probe_X.exe probe.c -DNVHDR="\"ffnvcodec/nvEncodeAPI_nX.h\""
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <windows.h>
#include NVHDR

int main(void) {
    printf("header                        = %s\n", NVHDR);
    printf("NVENCAPI_MAJOR_VERSION        = %u\n", (unsigned)NVENCAPI_MAJOR_VERSION);
    printf("NVENCAPI_MINOR_VERSION        = %u\n", (unsigned)NVENCAPI_MINOR_VERSION);
    printf("NVENCAPI_VERSION              = 0x%08X\n", (unsigned)NVENCAPI_VERSION);
    printf("\n-- enumerants --\n");
    printf("NV_ENC_BUFFER_FORMAT_ABGR10   = 0x%08X (%u)\n",
           (unsigned)NV_ENC_BUFFER_FORMAT_ABGR10, (unsigned)NV_ENC_BUFFER_FORMAT_ABGR10);
    printf("NV_ENC_BUFFER_FORMAT_ARGB10   = 0x%08X\n", (unsigned)NV_ENC_BUFFER_FORMAT_ARGB10);
    printf("NV_ENC_BUFFER_FORMAT_ABGR     = 0x%08X\n", (unsigned)NV_ENC_BUFFER_FORMAT_ABGR);
    printf("NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY = 0x%08X\n",
           (unsigned)NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY);
    printf("NV_ENC_PIC_STRUCT_FRAME       = 0x%08X\n", (unsigned)NV_ENC_PIC_STRUCT_FRAME);
    printf("NV_ENC_DEVICE_TYPE_CUDA       = 0x%08X\n", (unsigned)NV_ENC_DEVICE_TYPE_CUDA);
    printf("NV_ENC_SUCCESS                = %d\n", (int)NV_ENC_SUCCESS);

    printf("\n-- struct sizes --\n");
    printf("NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = %zu\n", sizeof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS));
    printf("NV_ENC_INITIALIZE_PARAMS             = %zu\n", sizeof(NV_ENC_INITIALIZE_PARAMS));
    printf("NV_ENC_REGISTER_RESOURCE             = %zu\n", sizeof(NV_ENC_REGISTER_RESOURCE));
    printf("NV_ENC_MAP_INPUT_RESOURCE            = %zu\n", sizeof(NV_ENC_MAP_INPUT_RESOURCE));
    printf("NV_ENC_PIC_PARAMS                    = %zu\n", sizeof(NV_ENC_PIC_PARAMS));
    printf("NV_ENC_LOCK_BITSTREAM                = %zu\n", sizeof(NV_ENC_LOCK_BITSTREAM));
    printf("NV_ENC_CREATE_BITSTREAM_BUFFER       = %zu\n", sizeof(NV_ENC_CREATE_BITSTREAM_BUFFER));
    printf("NV_ENC_EVENT_PARAMS                  = %zu\n", sizeof(NV_ENC_EVENT_PARAMS));
    printf("NV_ENCODE_API_FUNCTION_LIST          = %zu\n", sizeof(NV_ENCODE_API_FUNCTION_LIST));

    printf("\n-- struct version constants (_VER) --\n");
    printf("NV_ENC_INITIALIZE_PARAMS_VER         = 0x%08X\n", (unsigned)NV_ENC_INITIALIZE_PARAMS_VER);
    printf("NV_ENC_REGISTER_RESOURCE_VER         = 0x%08X\n", (unsigned)NV_ENC_REGISTER_RESOURCE_VER);
    printf("NV_ENC_MAP_INPUT_RESOURCE_VER        = 0x%08X\n", (unsigned)NV_ENC_MAP_INPUT_RESOURCE_VER);
    printf("NV_ENC_PIC_PARAMS_VER                = 0x%08X\n", (unsigned)NV_ENC_PIC_PARAMS_VER);
    printf("NV_ENC_LOCK_BITSTREAM_VER            = 0x%08X\n", (unsigned)NV_ENC_LOCK_BITSTREAM_VER);
    printf("NV_ENC_CREATE_BITSTREAM_BUFFER_VER   = 0x%08X\n", (unsigned)NV_ENC_CREATE_BITSTREAM_BUFFER_VER);
    printf("NV_ENC_EVENT_PARAMS_VER              = 0x%08X\n", (unsigned)NV_ENC_EVENT_PARAMS_VER);
    printf("NV_ENCODE_API_FUNCTION_LIST_VER      = 0x%08X\n", (unsigned)NV_ENCODE_API_FUNCTION_LIST_VER);

    printf("\n-- NV_ENC_REGISTER_RESOURCE offsets --\n");
#define O(s, f) printf("  %-24s = %zu\n", #f, offsetof(s, f))
    O(NV_ENC_REGISTER_RESOURCE, version);
    O(NV_ENC_REGISTER_RESOURCE, resourceType);
    O(NV_ENC_REGISTER_RESOURCE, width);
    O(NV_ENC_REGISTER_RESOURCE, height);
    O(NV_ENC_REGISTER_RESOURCE, pitch);
    O(NV_ENC_REGISTER_RESOURCE, subResourceIndex);
    O(NV_ENC_REGISTER_RESOURCE, resourceToRegister);
    O(NV_ENC_REGISTER_RESOURCE, registeredResource);
    O(NV_ENC_REGISTER_RESOURCE, bufferFormat);
    O(NV_ENC_REGISTER_RESOURCE, bufferUsage);
    O(NV_ENC_REGISTER_RESOURCE, pInputFencePoint);

    printf("\n-- NV_ENC_MAP_INPUT_RESOURCE offsets --\n");
    O(NV_ENC_MAP_INPUT_RESOURCE, version);
    O(NV_ENC_MAP_INPUT_RESOURCE, subResourceIndex);
    O(NV_ENC_MAP_INPUT_RESOURCE, inputResource);
    O(NV_ENC_MAP_INPUT_RESOURCE, registeredResource);
    O(NV_ENC_MAP_INPUT_RESOURCE, mappedResource);
    O(NV_ENC_MAP_INPUT_RESOURCE, mappedBufferFmt);

    printf("\n-- NV_ENC_PIC_PARAMS offsets --\n");
    O(NV_ENC_PIC_PARAMS, version);
    O(NV_ENC_PIC_PARAMS, inputWidth);
    O(NV_ENC_PIC_PARAMS, inputHeight);
    O(NV_ENC_PIC_PARAMS, inputPitch);
    O(NV_ENC_PIC_PARAMS, encodePicFlags);
    O(NV_ENC_PIC_PARAMS, frameIdx);
    O(NV_ENC_PIC_PARAMS, inputTimeStamp);
    O(NV_ENC_PIC_PARAMS, inputDuration);
    O(NV_ENC_PIC_PARAMS, inputBuffer);
    O(NV_ENC_PIC_PARAMS, outputBitstream);
    O(NV_ENC_PIC_PARAMS, completionEvent);
    O(NV_ENC_PIC_PARAMS, bufferFmt);
    O(NV_ENC_PIC_PARAMS, pictureStruct);
    O(NV_ENC_PIC_PARAMS, pictureType);

    printf("\n-- NV_ENC_INITIALIZE_PARAMS offsets --\n");
    O(NV_ENC_INITIALIZE_PARAMS, version);
    O(NV_ENC_INITIALIZE_PARAMS, encodeGUID);
    O(NV_ENC_INITIALIZE_PARAMS, presetGUID);
    O(NV_ENC_INITIALIZE_PARAMS, encodeWidth);
    O(NV_ENC_INITIALIZE_PARAMS, encodeHeight);
    O(NV_ENC_INITIALIZE_PARAMS, darWidth);
    O(NV_ENC_INITIALIZE_PARAMS, darHeight);
    O(NV_ENC_INITIALIZE_PARAMS, frameRateNum);
    O(NV_ENC_INITIALIZE_PARAMS, frameRateDen);
    O(NV_ENC_INITIALIZE_PARAMS, enableEncodeAsync);
    O(NV_ENC_INITIALIZE_PARAMS, enablePTD);
    O(NV_ENC_INITIALIZE_PARAMS, privDataSize);
    O(NV_ENC_INITIALIZE_PARAMS, privData);
    O(NV_ENC_INITIALIZE_PARAMS, encodeConfig);
    O(NV_ENC_INITIALIZE_PARAMS, maxEncodeWidth);
    O(NV_ENC_INITIALIZE_PARAMS, maxEncodeHeight);
    O(NV_ENC_INITIALIZE_PARAMS, maxMEHintCountsPerBlock);
    O(NV_ENC_INITIALIZE_PARAMS, tuningInfo);
    O(NV_ENC_INITIALIZE_PARAMS, bufferFormat);

    printf("\n-- NV_ENC_LOCK_BITSTREAM offsets --\n");
    O(NV_ENC_LOCK_BITSTREAM, version);
    O(NV_ENC_LOCK_BITSTREAM, bitstreamSizeInBytes);
    O(NV_ENC_LOCK_BITSTREAM, outputTimeStamp);
    O(NV_ENC_LOCK_BITSTREAM, bitstreamBufferPtr);
    O(NV_ENC_LOCK_BITSTREAM, pictureType);

    printf("\n-- NV_ENC_CREATE_BITSTREAM_BUFFER offsets --\n");
    O(NV_ENC_CREATE_BITSTREAM_BUFFER, version);
    O(NV_ENC_CREATE_BITSTREAM_BUFFER, bitstreamBuffer);
    O(NV_ENC_CREATE_BITSTREAM_BUFFER, bitstreamBufferPtr);

    printf("\n-- NV_ENC_EVENT_PARAMS offsets --\n");
    O(NV_ENC_EVENT_PARAMS, version);
    O(NV_ENC_EVENT_PARAMS, completionEvent);

    printf("\n-- function-list pointer slot indices (base as usize*) --\n");
    {
        size_t p = sizeof(void*);
#define SLOT(f) printf("  %-32s = %zu\n", #f, offsetof(NV_ENCODE_API_FUNCTION_LIST, f) / p)
        SLOT(nvEncOpenEncodeSession);
        SLOT(nvEncGetEncodePresetConfig);
        SLOT(nvEncInitializeEncoder);
        SLOT(nvEncCreateBitstreamBuffer);
        SLOT(nvEncDestroyBitstreamBuffer);
        SLOT(nvEncEncodePicture);
        SLOT(nvEncLockBitstream);
        SLOT(nvEncUnlockBitstream);
        SLOT(nvEncRegisterAsyncEvent);
        SLOT(nvEncUnregisterAsyncEvent);
        SLOT(nvEncMapInputResource);
        SLOT(nvEncUnmapInputResource);
        SLOT(nvEncDestroyEncoder);
        SLOT(nvEncOpenEncodeSessionEx);
        SLOT(nvEncRegisterResource);
    }
    return 0;
}
