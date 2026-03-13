#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

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
void signal_handler(int s) { keep_running = 0; }

__global__ void enemyKernel(unsigned int* d_chase, int num_words, unsigned long long cycles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    volatile unsigned int* v = (volatile unsigned int*)d_chase;
    int idx = (tid * 7 + 1) % num_words;
    unsigned int sum = 0;
    unsigned long long start = clock64();
    while ((clock64() - start) < cycles) {
        unsigned int next = v[idx];
        sum += next;
        idx = (int)((next + tid + 1) % num_words);
    }
    if (tid == 0) d_chase[0] = sum;
}

int main(int argc, char** argv) {
    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext primaryCtx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&primaryCtx, device));
    CHECK_CU(cuCtxSetCurrent(primaryCtx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, 0));

    CUdevResource fullSMs;
    CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

    // minCount=5 -> group=6 SMs (victim gets this), remainder=2 SMs (enemy gets this)
    // Enemy takes the REMAINDER (2 SMs), same split as victim process
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
    printf("[ENEMY] Running on %u SMs | PID: %d\n", verify.sm.smCount, getpid());
    printf("[ENEMY] Sharing L2 with victim — contention active\n");
    fflush(stdout);

    int chase_size = 16 * 1024 * 1024;
    int num_words  = chase_size / sizeof(unsigned int);
    unsigned int* d_chase;
    CHECK_RT(cudaMalloc(&d_chase, chase_size));
    unsigned int* h_chase = (unsigned int*)malloc(chase_size);
    for (int i = 0; i < num_words; i++)
        h_chase[i] = (unsigned int)(((unsigned long long)i * 1234567 + 1) % num_words);
    CHECK_RT(cudaMemcpy(d_chase, h_chase, chase_size, cudaMemcpyHostToDevice));
    free(h_chase);

    unsigned long long cycles_per_launch = 1000000000ULL;
    printf("[ENEMY] Running until killed...\n");
    fflush(stdout);

    while (keep_running) {
        enemyKernel<<<16, 256, 0, enemyStream>>>(d_chase, num_words, cycles_per_launch);
        cudaStreamSynchronize(enemyStream);
    }

    printf("[ENEMY] Stopped.\n");
    CHECK_RT(cudaFree(d_chase));
    CHECK_CU(cuStreamDestroy(enemyStream));
    CHECK_CU(cuGreenCtxDestroy(enemyGCtx));
    CHECK_CU(cuDevicePrimaryCtxRelease(device));
    return 0;
}

