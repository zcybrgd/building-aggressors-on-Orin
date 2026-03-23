// mul_host.cu — fixed version

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
        fprintf(stderr, "[CUDA Driver Error] %s: %s at %s:%d\n", \
                errName, errStr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA Runtime Error] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

const int N = 32;

// ── Renamed to match --kernel-name GPUMultiplyMatrix in the bash script ──
__global__ void GPUMultiplyMatrix(int* A, int* B, int* C) {
    int col   = blockIdx.x * blockDim.x + threadIdx.x;
    int lig   = blockIdx.y * blockDim.y + threadIdx.y;
    int index = lig * N + col;
    if (col < N && lig < N) {
        int inter = 0;
        for (int i = 0; i < N; ++i)
            inter += A[lig * N + i] * B[i * N + col];
        C[index] = inter;
    }
}

// ── Same argv contract as matrix_victim.cu ──
// Usage: ./matrix_victim <matrix_size> <block_x> <block_y> <paths> <mode>
//   matrix_size : accepted but IGNORED (kernel is hardcoded to N=32)
//   block_x/y   : thread block dimensions
//   paths       : number of repeated launches
//   mode        : 0 = alone, 1 = concurrent (6-SM green context)
int main(int argc, char** argv) {
    /* argv[1] = matrix_size — consumed for argv parity, not used */
    int block_x    = (argc > 2) ? atoi(argv[2]) : 32;
    int block_y    = (argc > 3) ? atoi(argv[3]) : 1;
    int paths      = (argc > 4) ? atoi(argv[4]) : 15;
    int concurrent = (argc > 5) ? atoi(argv[5]) : 0;

    int gridX = (N + block_x - 1) / block_x;
    int gridY = (N + block_y - 1) / block_y;
    dim3 grid(gridX, gridY);
    dim3 block(block_x, block_y);

    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs,
             cudaDevAttrMultiProcessorCount, devIdx));

    size_t matSize = (size_t)N * N * sizeof(int);
    int *d_A, *d_B, *d_C;
    CHECK_RT(cudaMalloc(&d_A, matSize));
    CHECK_RT(cudaMalloc(&d_B, matSize));
    CHECK_RT(cudaMalloc(&d_C, matSize));

    int *h_A = (int*)malloc(matSize);
    int *h_B = (int*)malloc(matSize);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (i % 17) + 1;
        h_B[i] = (i % 13) + 1;
    }
    CHECK_RT(cudaMemcpy(d_A, h_A, matSize, cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_B, h_B, matSize, cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemset(d_C, 0, matSize));
    free(h_A);
    free(h_B);

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("[VICTIM] Kernel: GPUMultiplyMatrix | Matrix: %dx%d (int) | "
           "Block: (%d,%d) | Grid: (%d,%d) | Paths: %d | Memory: %.2f KB\n",
           N, N, block_x, block_y, gridX, gridY, paths,
           3.0 * matSize / 1024.0);

    if (!concurrent) {
        printf("[VICTIM] ALONE — all %d SMs\n", totalSMs);
        CHECK_RT(cudaEventRecord(start, 0));
        for (int p = 0; p < paths; p++)
            GPUMultiplyMatrix<<<grid, block>>>(d_A, d_B, d_C);
        CHECK_RT(cudaEventRecord(stop, 0));
        CHECK_RT(cudaStreamSynchronize(0));

    } else {
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs,
                 CU_DEV_RESOURCE_TYPE_SM));

        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(
            &victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        CUdevResourceDesc descVictim;
        CHECK_CU(cuDevResourceGenerateDesc(&descVictim, &victimSlice, 1));

        CUgreenCtx victimGCtx;
        CHECK_CU(cuGreenCtxCreate(
            &victimGCtx, descVictim, device, CU_GREEN_CTX_DEFAULT_STREAM));

        CUstream victimStream;
        CHECK_CU(cuGreenCtxStreamCreate(
            &victimStream, victimGCtx, CU_STREAM_NON_BLOCKING, 0));

        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(
            victimGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[VICTIM] CONCURRENT — %u SMs assigned\n",
               verify.sm.smCount);

        CHECK_RT(cudaEventRecord(start, victimStream));
        for (int p = 0; p < paths; p++)
            GPUMultiplyMatrix<<<grid, block, 0, victimStream>>>(
                d_A, d_B, d_C);
        CHECK_RT(cudaEventRecord(stop, victimStream));
        CHECK_RT(cudaStreamSynchronize(victimStream));

        CHECK_CU(cuStreamDestroy(victimStream));
        CHECK_CU(cuGreenCtxDestroy(victimGCtx));
    }

    float ms;
    CHECK_RT(cudaEventElapsedTime(&ms, start, stop));
    printf("[VICTIM] Completed in %.2f ms\n", ms);

    CHECK_RT(cudaEventDestroy(start));
    CHECK_RT(cudaEventDestroy(stop));
    CHECK_RT(cudaFree(d_A));
    CHECK_RT(cudaFree(d_B));
    CHECK_RT(cudaFree(d_C));
    return 0;
}