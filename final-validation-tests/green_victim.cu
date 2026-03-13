#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errName, *errStr; \
        cuGetErrorName(err, &errName); \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "[CUDA Driver Error] %s: %s at %s:%d\n", errName, errStr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA Runtime Error] %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void victimKernel(float* d_data, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            sum += d_data[i] * 1.1f;
            sum += d_data[(i + 64)  % N] * 0.9f;
            sum += d_data[(i + 128) % N] * 0.8f;
        }
        if ((iter & 0xFF) == 0) d_data[tid % N] = sum * 0.001f;
    }
    if (tid < N) d_data[tid] = sum;
}

int main(int argc, char** argv) {
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 50000;
    // mode: 0=alone (all 8 SMs), 1=concurrent (6 SMs green context, enemy has 2 SMs)
    int concurrent = (argc > 2) ? atoi(argv[2]) : 0;

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext primaryCtx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&primaryCtx, device));
    CHECK_CU(cuCtxSetCurrent(primaryCtx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, 0));

    int N = 256 * 1024;
    float* d_data;
    CHECK_RT(cudaMalloc(&d_data, N * sizeof(float)));
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)i * 0.01f;
    CHECK_RT(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data);

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    if (!concurrent) {
        // ALONE: no green context, victim uses all 8 SMs
        printf("[VICTIM] ALONE — all %d SMs\n", totalSMs);
        printf("[VICTIM] Working set: %.2f MB | Iterations: %llu\n", N*sizeof(float)/1e6, iters);
        CHECK_RT(cudaEventRecord(start, 0));
        victimKernel<<<128, 256>>>(d_data, N, iters);
        CHECK_RT(cudaEventRecord(stop, 0));
        CHECK_RT(cudaStreamSynchronize(0));

    } else {
        // CONCURRENT: green context gives victim 6 SMs
        // enemy process independently holds 2 SMs green context
        // both share the same L2 cache -> contention
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

        // minCount=5 -> group=6 SMs (victim), remainder=2 SMs (enemy)
        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        CUdevResource verify;
        // create victim green context with 6 SMs
        CUdevResourceDesc descVictim;
        CHECK_CU(cuDevResourceGenerateDesc(&descVictim, &victimSlice, 1));
        CUgreenCtx victimGCtx;
        CHECK_CU(cuGreenCtxCreate(&victimGCtx, descVictim, device, CU_GREEN_CTX_DEFAULT_STREAM));
        CUstream victimStream;
        CHECK_CU(cuGreenCtxStreamCreate(&victimStream, victimGCtx, CU_STREAM_NON_BLOCKING, 0));

        CHECK_CU(cuGreenCtxGetDevResource(victimGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[VICTIM] CONCURRENT — %u SMs (enemy has 2 SMs, shared L2)\n", verify.sm.smCount);
        printf("[VICTIM] Working set: %.2f MB | Iterations: %llu\n", N*sizeof(float)/1e6, iters);

        CHECK_RT(cudaEventRecord(start, victimStream));
        victimKernel<<<128, 256, 0, victimStream>>>(d_data, N, iters);
        CHECK_RT(cudaEventRecord(stop, victimStream));
        CHECK_RT(cudaStreamSynchronize(victimStream));

        CHECK_CU(cuStreamDestroy(victimStream));
        CHECK_CU(cuGreenCtxDestroy(victimGCtx));
    }

    float ms;
    CHECK_RT(cudaEventElapsedTime(&ms, start, stop));
    printf("[VICTIM] Completed in %.2f ms\n", ms);
    printf("[VICTIM] Performance: %.2f GFLOPS\n", (3.0 * N * iters / (ms * 1e6)));

    CHECK_RT(cudaEventDestroy(start));
    CHECK_RT(cudaEventDestroy(stop));
    CHECK_RT(cudaFree(d_data));
    CHECK_CU(cuDevicePrimaryCtxRelease(device));
    return 0;
}
