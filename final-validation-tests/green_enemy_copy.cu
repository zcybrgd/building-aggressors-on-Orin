#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

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

volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    keep_running = 0;
}

__global__ void printSMIDs() {
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("[ENEMY] block %d -> SM %u\n", blockIdx.x, smid);
    }
}

__global__ void enemyKernel(unsigned int* d_chase, unsigned int* d_writeback, int size,
                             unsigned long long cycles, int* d_stop_flag) {
    unsigned long long start = clock64();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_words = size / sizeof(unsigned int);

    volatile unsigned int* v = (volatile unsigned int*)d_chase;
    volatile unsigned int* w = (volatile unsigned int*)d_writeback;
    int idx = tid % num_words;
    unsigned int sum = 0;

    if (cycles == 0) {
        while (d_stop_flag[0] == 0) {
            unsigned int next = v[idx];
            sum += next;
            int write_idx = (idx + tid * 131) % num_words;
            unsigned int write_val = sum ^ next ^ write_idx;
            w[write_idx] = write_val;
            w[(write_idx + 97) % num_words] = write_val + tid;
            idx = next;
        }
    } else {
        while ((clock64() - start) < cycles) {
            unsigned int next = v[idx];
            sum += next;
            int write_idx = (idx + tid * 131) % num_words;
            unsigned int write_val = sum ^ next ^ write_idx;
            w[write_idx] = write_val;
            w[(write_idx + 97) % num_words] = write_val + tid;
            idx = next;
        }
    }

    if (tid == 0) d_writeback[0] = sum;
}

int main(int argc, char** argv) {
    unsigned long long cycles = (argc > 1) ? atoll(argv[1]) : 0;
    int use_green              = (argc > 2) ? atoi(argv[2])  : 0;

    if (cycles == 0)
        printf("[ENEMY] Starting in INFINITE mode (run until killed)\n");
    else
        printf("[ENEMY] Starting for %llu cycles\n", cycles);

    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);

    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));
    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, devIdx));

    // Two independent buffers — each kernel chases its own, doubling L2 pressure
    int size = 16 * 1024 * 1024;
    int num_words = size / sizeof(unsigned int);

    unsigned int *d_chase1, *d_chase2;
    unsigned int *d_writeback1, *d_writeback2;
    int *d_stop_flag;

    CHECK_RT(cudaMalloc(&d_chase1,     size));
    CHECK_RT(cudaMalloc(&d_chase2,     size));
    CHECK_RT(cudaMalloc(&d_writeback1, size));
    CHECK_RT(cudaMalloc(&d_writeback2, size));
    CHECK_RT(cudaMalloc(&d_stop_flag,  sizeof(int)));

    // Initialize chase buffers on host, copy to both device buffers
    unsigned int* h_chase = (unsigned int*)malloc(size);
    for (int i = 0; i < num_words; i++)
        h_chase[i] = ((unsigned long long)i * 1234567) % num_words;

    CHECK_RT(cudaMemcpy(d_chase1, h_chase, size, cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(d_chase2, h_chase, size, cudaMemcpyHostToDevice));
    free(h_chase);

    CHECK_RT(cudaMemset(d_writeback1, 0, size));
    CHECK_RT(cudaMemset(d_writeback2, 0, size));

    int h_stop_flag = 0;
    CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));

    if (!use_green) {
        // ----------------------------------------------------------------
        // Non-green path: both kernels on default stream, all SMs
        // ----------------------------------------------------------------
        printf("[ENEMY] No green context — using all %d SMs\n", totalSMs);
        printf("[ENEMY] Running... (PID: %d)\n", getpid());

        printSMIDs<<<16, 1>>>();
        CHECK_RT(cudaDeviceSynchronize());

        // Launch two kernels on the default stream — they run sequentially
        // (default stream is in-order). For concurrent execution without a
        // green context you would need two separate non-blocking streams,
        // but that is not the focus of this path.
        enemyKernel<<<8, 1024>>>(d_chase1, d_writeback1, size, cycles, d_stop_flag);
        enemyKernel<<<8, 1024>>>(d_chase2, d_writeback2, size, cycles, d_stop_flag);

        if (cycles == 0) {
            while (keep_running) { sleep(1); }
            printf("[ENEMY] Stop signal received, terminating kernel...\n");
            h_stop_flag = 1;
            CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
        }

        CHECK_RT(cudaDeviceSynchronize());

    } else {
        // ----------------------------------------------------------------
        // Green context path: 2 SMs, 2 concurrent streams, 2 kernels
        // ----------------------------------------------------------------

        // Step 1: query full SM pool
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

        // Step 2: split — victim gets 6 SMs, enemy gets 2 (remainder)
        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        // Step 3: pack enemy slice into descriptor
        CUdevResourceDesc descEnemy;
        CHECK_CU(cuDevResourceGenerateDesc(&descEnemy, &enemySlice, 1));

        // Step 4: create green context restricted to the 2 enemy SMs
        CUgreenCtx enemyGCtx;
        CHECK_CU(cuGreenCtxCreate(&enemyGCtx, descEnemy, device, CU_GREEN_CTX_DEFAULT_STREAM));

        // Step 5: two non-blocking streams on the same green context
        // so the driver can schedule them concurrently across the 2 SMs
        CUstream enemyStream1, enemyStream2;
        CHECK_CU(cuGreenCtxStreamCreate(&enemyStream1, enemyGCtx, CU_STREAM_NON_BLOCKING, 0));
        CHECK_CU(cuGreenCtxStreamCreate(&enemyStream2, enemyGCtx, CU_STREAM_NON_BLOCKING, 0));

        // Sanity check
        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(enemyGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[ENEMY] Green context — %u SMs | 2 kernels | 2 independent buffers\n", verify.sm.smCount);
        printf("[ENEMY] Enemy working set: %.2f MB per buffer, %.2f MB total\n",
               size / 1e6, 4.0 * size / 1e6);
        printf("[ENEMY] Running... (PID: %d)\n", getpid());

        // Print SM assignments — 8 blocks per stream so each SM gets ~4 blocks
        printSMIDs<<<16, 1, 0, enemyStream1>>>();
        printSMIDs<<<16, 1, 0, enemyStream2>>>();
        CHECK_CU(cuStreamSynchronize(enemyStream1));
        CHECK_CU(cuStreamSynchronize(enemyStream2));

        // Launch: 8 blocks × 1024 threads per kernel
        // Each SM gets ~4 blocks from each kernel = 8 blocks total per SM
        enemyKernel<<<16, 1024, 0, enemyStream1>>>(d_chase1, d_writeback1, size, cycles, d_stop_flag);
        enemyKernel<<<16, 1024, 0, enemyStream2>>>(d_chase2, d_writeback2, size, cycles, d_stop_flag);

        if (cycles == 0) {
            while (keep_running) { sleep(1); }
            printf("[ENEMY] Stop signal received, terminating kernels...\n");
            h_stop_flag = 1;
            CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
        }

        CHECK_CU(cuStreamSynchronize(enemyStream1));
        CHECK_CU(cuStreamSynchronize(enemyStream2));
        CHECK_CU(cuStreamDestroy(enemyStream1));
        CHECK_CU(cuStreamDestroy(enemyStream2));
        CHECK_CU(cuGreenCtxDestroy(enemyGCtx));
    }

    printf("[ENEMY] Completed\n");

    CHECK_RT(cudaFree(d_chase1));
    CHECK_RT(cudaFree(d_chase2));
    CHECK_RT(cudaFree(d_writeback1));
    CHECK_RT(cudaFree(d_writeback2));
    CHECK_RT(cudaFree(d_stop_flag));
    return 0;
}
