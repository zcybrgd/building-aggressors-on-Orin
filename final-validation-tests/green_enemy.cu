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

// Same L2-thrashing kernel from enemy_process_enhanced.cu
// 16 MB footprint, pointer chase + scatter writes
__global__ void enemyKernel(unsigned int* d_chase, unsigned int* d_writeback, int size, unsigned long long cycles, int* d_stop_flag) {
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
    unsigned long long cycles = (argc > 1) ? atoll(argv[1]) : 0; // 0 = infinite
    // mode: 0=no green context, 1=use green context with remainder SMs
    int use_green = (argc > 2) ? atoi(argv[2]) : 0;

    if (cycles == 0) {
        printf("[ENEMY] Starting in INFINITE mode (run until killed)\n");
    } else {
        printf("[ENEMY] Starting for %llu cycles\n", cycles);
    }

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext primaryCtx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&primaryCtx, device));
    CHECK_CU(cuCtxSetCurrent(primaryCtx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, 0));

    int size = 16 * 1024 * 1024; // 16 MB
    unsigned int* d_chase;
    unsigned int* d_writeback;
    int* d_stop_flag;
    CHECK_RT(cudaMalloc(&d_chase, size));
    CHECK_RT(cudaMalloc(&d_writeback, size));
    CHECK_RT(cudaMalloc(&d_stop_flag, sizeof(int)));

    int h_stop_flag = 0;
    CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));

    // Build permutation for pointer chase
    unsigned int* h_chase = (unsigned int*)malloc(size);
    int num_words = size / sizeof(unsigned int);
    for (int i = 0; i < num_words; i++) {
        h_chase[i] = ((unsigned long long)i * 1234567) % num_words;
    }
    CHECK_RT(cudaMemcpy(d_chase, h_chase, size, cudaMemcpyHostToDevice));
    free(h_chase);
    CHECK_RT(cudaMemset(d_writeback, 0, size));

    if (!use_green) {
        // No green context — enemy uses default context (all SMs)
        printf("[ENEMY] No green context — using all %d SMs\n", totalSMs);
        printf("[ENEMY] Running... (PID: %d)\n", getpid());
        enemyKernel<<<16, 1024>>>(d_chase, d_writeback, size, cycles, d_stop_flag);

    } else {
        // Green context: enemy gets the remainder 2 SMs
        // Split: minCount=5 -> group=6 SMs (victim), remainder=2 SMs (enemy)
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        CUdevResourceDesc descEnemy;
        CHECK_CU(cuDevResourceGenerateDesc(&descEnemy, &enemySlice, 1));
        CUgreenCtx enemyGCtx;
        CHECK_CU(cuGreenCtxCreate(&enemyGCtx, descEnemy, device, CU_GREEN_CTX_DEFAULT_STREAM));
        CUstream enemyStream;
        CHECK_CU(cuGreenCtxStreamCreate(&enemyStream, enemyGCtx, CU_STREAM_NON_BLOCKING, 0));

        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(enemyGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[ENEMY] Green context — %u SMs (victim has 6 SMs, shared L2)\n", verify.sm.smCount);
        printf("[ENEMY] Running... (PID: %d)\n", getpid());

        enemyKernel<<<16, 1024, 0, enemyStream>>>(d_chase, d_writeback, size, cycles, d_stop_flag);

        if (cycles == 0) {
            while (keep_running) {
                sleep(1);
            }
            printf("[ENEMY] Stop signal received, terminating kernel...\n");
            h_stop_flag = 1;
            CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
        }

        CHECK_RT(cudaStreamSynchronize(enemyStream));
        CHECK_CU(cuStreamDestroy(enemyStream));
        CHECK_CU(cuGreenCtxDestroy(enemyGCtx));

        printf("[ENEMY] Completed\n");
        CHECK_RT(cudaFree(d_chase));
        CHECK_RT(cudaFree(d_writeback));
        CHECK_RT(cudaFree(d_stop_flag));
        CHECK_CU(cuDevicePrimaryCtxRelease(device));
        return 0;
    }

    // Non-green infinite mode wait
    if (cycles == 0) {
        while (keep_running) {
            sleep(1);
        }
        printf("[ENEMY] Stop signal received, terminating kernel...\n");
        h_stop_flag = 1;
        CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
    }

    cudaDeviceSynchronize();
    printf("[ENEMY] Completed\n");

    CHECK_RT(cudaFree(d_chase));
    CHECK_RT(cudaFree(d_writeback));
    CHECK_RT(cudaFree(d_stop_flag));
    CHECK_CU(cuDevicePrimaryCtxRelease(device));
    return 0;
}
