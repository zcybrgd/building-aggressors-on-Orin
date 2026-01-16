#include <cuda.h>
#include <stdio.h>

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errName, *errStr; \
        cuGetErrorName(err, &errName); \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA Driver Error: %s: %s at %s:%d\n", \
                errName, errStr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


int main() {
    CHECK_CU(cuInit(0));    
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    int smCount;
    CHECK_CU(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    printf("Total SMs in Orin: %d\n", smCount);
    
    //get initial SM resources
    CUdevResource fullSMs;
    CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));
    //split SM resources
    CUdevResource smGroupA;
    CUdevResource smGroupB;
    unsigned int nbGroups = 1;  // Number of equal groups to create
    unsigned int minCount = 1;   // Minimum SMs per group
    CHECK_CU(cuDevSmResourceSplitByCount(&smGroupA, &nbGroups, &fullSMs, &smGroupB, 0, minCount));
    printf("Green context A gets: %u SMs\n", smGroupA.sm.smCount);
    printf("Green context B gets: %u SMs\n", smGroupB.sm.smCount);
    // generate resource descriptors
    CUdevResourceDesc descA, descB;
    CHECK_CU(cuDevResourceGenerateDesc(&descA, &smGroupA, 1));
    CHECK_CU(cuDevResourceGenerateDesc(&descB, &smGroupB, 1));
    // create green contexts
    CUgreenCtx ctxA, ctxB;
    CHECK_CU(cuGreenCtxCreate(&ctxA, descA, device, CU_GREEN_CTX_DEFAULT_STREAM));
    printf("Green context A created!\n");
    CHECK_CU(cuGreenCtxCreate(&ctxB, descB, device, CU_GREEN_CTX_DEFAULT_STREAM));
    printf("Green context B created!\n");
    // create streams bound to green contexts
    CUstream streamA, streamB;
    CHECK_CU(cuGreenCtxStreamCreate(&streamA, ctxA, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CU(cuGreenCtxStreamCreate(&streamB, ctxB, CU_STREAM_NON_BLOCKING, 0));
    printf("Streams created successfully\n");
    // Verify resources
    CUdevResource verifiedA;
    CHECK_CU(cuGreenCtxGetDevResource(ctxA, &verifiedA, CU_DEV_RESOURCE_TYPE_SM));
    printf("Verified: Context A has %u SMs\n", verifiedA.sm.smCount);

    CHECK_CU(cuStreamDestroy(streamA));
    CHECK_CU(cuStreamDestroy(streamB));
    CHECK_CU(cuGreenCtxDestroy(ctxA));
    CHECK_CU(cuGreenCtxDestroy(ctxB));
    return 0;
}

