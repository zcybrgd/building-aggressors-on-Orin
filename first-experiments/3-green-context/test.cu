#include <cuda.h>
#include <stdio.h>

//for error checking
#define CHECK_CU(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* str; \
        cuGetErrorString(err, &str); \
        printf("ERROR: %s (code %d) at %s:%d\n", str, err, __FILE__, __LINE__); \
        exit(1); \
    } \
}

int main() {
    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    int smCount;
    CHECK_CU(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    printf("total SMs in my orin: %d\n", smCount);
    printf("\nTest 1: Regular Context\n");
    CUcontext regularCtx;
    CUresult res1 = cuCtxCreate(&regularCtx, 0, device);
    if (res1 == CUDA_SUCCESS) {
        printf("Regular context works\n");
        cuCtxDestroy(regularCtx);
    } else {
        printf("Regular context failed\n");
        return 1;
    }
    // Test 2: Green context with 1 SM given to the context
    printf("\nTest 2: Green Context\n");
    CUexecAffinityParam param;
    param.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;//telling the driver i want to constrain this context by number of SMs
    param.param.smCount.val = 1;
    CUcontext greenCtx;
    CUresult res2 = cuCtxCreate_v3(&greenCtx, &param, 1, 0, device);
    if (res2 == CUDA_SUCCESS) {
        printf("Green context WORKS!\n");
        cuCtxDestroy(greenCtx);
        return 0;
    } else {
        const char* errStr;
        cuGetErrorString(res2, &errStr);
        printf("Green context failed: %s\n", errStr);
    }
    printf("\nTest 3: Green Context with CU_CTX_SCHED_AUTO\n");
    CUresult res3 = cuCtxCreate_v3(&greenCtx, &param, 1, CU_CTX_SCHED_AUTO, device);
    if (res3 == CUDA_SUCCESS) {
        printf("Green context with flags WORKS!\n");
        cuCtxDestroy(greenCtx);
        return 0;
    } else {
        const char* errStr;
        cuGetErrorString(res3, &errStr);
        printf("Still failed: %s\n", errStr);
    }
    return 1;
}
