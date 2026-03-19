#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

__global__ void GPUMultiplyMatrix(long *matrix1, long *matrix2, int paths, int count) {
    int element = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("[VICTIM] block %d -> SM %u\n", blockIdx.x, smid);
    }
    while (paths > 0) {
        long sum = 0;
        int col = element % count;
        int row = element / count;
        for (i = 0; i < count; i++) {
            sum += matrix1[count * i + col] * matrix2[row * count + i];
        }
        __syncthreads();
        matrix2[element] = sum;
        paths--;
    }
}

// Usage: ./matrix_victim <matrix_size> <block_x> <block_y> <paths> <mode>
//   mode: 0 = alone (all SMs), 1 = concurrent (6 SMs green context)
int main(int argc, char** argv) {
    int count     = (argc > 1) ? atoi(argv[1]) : 240;
    int block_x   = (argc > 2) ? atoi(argv[2]) : 32;
    int block_y   = (argc > 3) ? atoi(argv[3]) : 32;
    int paths     = (argc > 4) ? atoi(argv[4]) : 15;
    int concurrent = (argc > 5) ? atoi(argv[5]) : 0;

    int threadsPerBlock = block_x * block_y;
    int numBlocks = (count * count + threadsPerBlock - 1) / threadsPerBlock;

    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, devIdx));

    int totalElements = count * count;
    size_t matSize = (size_t)totalElements * sizeof(long);
    long *d_matrix1, *d_matrix2;
    CHECK_RT(cudaMalloc(&d_matrix1, matSize));
    CHECK_RT(cudaMalloc(&d_matrix2, matSize));

    long *h_matrix1 = (long*)malloc(matSize);
    long *h_matrix2 = (long*)malloc(matSize);
    for (int i = 0; i < totalElements; i++) {
        h_matrix1[i] = (i % 17) + 1;
        h_matrix2[i] = (i % 13) + 1;
    }
    CHECK_RT(cudaMemcpy(d_matrix1, h_matrix1, matSize, cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_matrix2, h_matrix2, matSize, cudaMemcpyHostToDevice));
    free(h_matrix1);
    free(h_matrix2);

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("[VICTIM] Matrix: %dx%d | Block: (%d,%d)=%d threads | Grid: %d blocks | Paths: %d | Memory: %.2f MB\n",
           count, count, block_x, block_y, threadsPerBlock, numBlocks, paths, 2.0 * matSize / 1e6);

    if (!concurrent) {
        printf("[VICTIM] ALONE — all %d SMs\n", totalSMs);
        CHECK_RT(cudaEventRecord(start, 0));
        GPUMultiplyMatrix<<<numBlocks, threadsPerBlock>>>(d_matrix1, d_matrix2, paths, count);
        CHECK_RT(cudaEventRecord(stop, 0));
        CHECK_RT(cudaStreamSynchronize(0));
    } else {
        // step 1: query the full SM resource pool of the device.
        // CU_DEV_RESOURCE_TYPE_SM is the only resource type relevant here;
        // it describes how many SMs are available for partitioning.
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

        // step 2: split SMs into a group + remainder.
        // cuDevSmResourceSplit is the recommended API but is not available in
        // CUDA 12.6 (missing from headers and libcuda.so on this Orin).
        // cuDevSmResourceSplitByCount is the only split API present here.
        // minCount=5 on 8 SMs with alignment=2 gives group=6 SMs (victim)
        // and remainder=2 SMs (enemy). Both processes call the same split
        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));
        // Step 3: pack the SM resource into an opaque descriptor.
        // cuDevResourceGenerateDesc bundles one or more CUdevResource objects
        // into a CUdevResourceDesc that cuGreenCtxCreate can consume.
        CUdevResourceDesc descVictim;
        CHECK_CU(cuDevResourceGenerateDesc(&descVictim, &victimSlice, 1));
        // Step 4: create the green context restricted to victimSlice (6 SMs).
        // CU_GREEN_CTX_DEFAULT_STREAM requests a default stream be associated
        // with this green context (required flag in CUDA 12.x).
        CUgreenCtx victimGCtx;
        CHECK_CU(cuGreenCtxCreate(&victimGCtx, descVictim, device, CU_GREEN_CTX_DEFAULT_STREAM));
        // Step 5: create a stream bound to the green context.
        // CU_STREAM_NON_BLOCKING prevents implicit synchronisation with stream 0
        // (the default stream), ensuring the victim's work stays isolated and
        // does not accidentally wait on unrelated Runtime API operations.
        CUstream victimStream;
        CHECK_CU(cuGreenCtxStreamCreate(&victimStream, victimGCtx, CU_STREAM_NON_BLOCKING, 0));

        // cuGreenCtxGetDevResource reads back the resource from the live context,
        // so this catches any silent rounding or rejection by the driver.
        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(victimGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[VICTIM] CONCURRENT — %u SMs\n", verify.sm.smCount);

        CHECK_RT(cudaEventRecord(start, victimStream));
        GPUMultiplyMatrix<<<numBlocks, threadsPerBlock, 0, victimStream>>>(d_matrix1, d_matrix2, paths, count);
        CHECK_RT(cudaEventRecord(stop, victimStream));
        CHECK_RT(cudaStreamSynchronize(victimStream));

        // cuStreamDestroy / cuGreenCtxDestroy must be used (not their Runtime
        // equivalents) because the stream and context were created via Driver API.
        CHECK_CU(cuStreamDestroy(victimStream));
        CHECK_CU(cuGreenCtxDestroy(victimGCtx));
    }

    float ms;
    CHECK_RT(cudaEventElapsedTime(&ms, start, stop));
    printf("[VICTIM] Completed in %.2f ms\n", ms);

    CHECK_RT(cudaEventDestroy(start));
    CHECK_RT(cudaEventDestroy(stop));
    CHECK_RT(cudaFree(d_matrix1));
    CHECK_RT(cudaFree(d_matrix2));
    return 0;
}
