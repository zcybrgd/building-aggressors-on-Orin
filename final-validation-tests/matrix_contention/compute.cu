#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

// Compute-intensive kernel — minimal L2 usage
// Performs heavy FP32 arithmetic on a small dataset loaded once
__global__ void GPUComputeIntensive(float *input, float *output, int iterations, int dataSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= dataSize) return;
    
    // Load data ONCE from global memory (single L2 access per thread)
    float value = input[tid];
    
    // Heavy compute loop — all arithmetic, NO memory accesses
    for (int iter = 0; iter < iterations; iter++) {
        // Transcendental functions (expensive FP operations)
        value = sinf(value) * cosf(value);
        value = expf(value * 0.001f);
        value = logf(fabsf(value) + 1.0f);
        value = sqrtf(fabsf(value));
        
        // FMA operations (fused multiply-add)
        value = fmaf(value, 1.001f, 0.0001f);
        value = fmaf(value, value, -0.5f);
        
        // Polynomial evaluation (pure arithmetic)
        float x = value;
        value = x*x*x*x - 2.0f*x*x*x + 3.0f*x*x - 4.0f*x + 5.0f;
        
        // More transcendental ops
        value = tanhf(value * 0.01f);
        value = atanf(value);
        
        // Keep value in reasonable range to avoid overflow
        if (fabsf(value) > 10.0f) {
            value = fmodf(value, 10.0f);
        }
    }
    
    // Store result ONCE to global memory (single L2 access per thread)
    output[tid] = value;
}

// Usage: ./compute_victim <data_size> <block_x> <block_y> <iterations> <mode>
//   mode: 0 = alone (all SMs), 1 = concurrent (6 SMs green context)
int main(int argc, char** argv) {
    int dataSize    = (argc > 1) ? atoi(argv[1]) : 1024*1024;  // 1M elements by default
    int block_x     = (argc > 2) ? atoi(argv[2]) : 32;
    int block_y     = (argc > 3) ? atoi(argv[3]) : 32;
    int iterations  = (argc > 4) ? atoi(argv[4]) : 1000;       // Compute intensity
    int concurrent  = (argc > 5) ? atoi(argv[5]) : 0;

    int threadsPerBlock = block_x * block_y;
    int numBlocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext primaryCtx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&primaryCtx, device));
    CHECK_CU(cuCtxSetCurrent(primaryCtx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, 0));

    size_t memSize = (size_t)dataSize * sizeof(float);
    float *d_input, *d_output;
    CHECK_RT(cudaMalloc(&d_input, memSize));
    CHECK_RT(cudaMalloc(&d_output, memSize));

    // Initialize input on host
    float *h_input = (float*)malloc(memSize);
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = ((float)(i % 1000)) / 1000.0f;  // Values in [0, 1)
    }
    CHECK_RT(cudaMemcpy(d_input, h_input, memSize, cudaMemcpyHostToDevice));
    free(h_input);

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("[VICTIM-COMPUTE] Data: %d elements | Block: (%d,%d)=%d threads | Grid: %d blocks | Iterations: %d | Memory: %.2f MB\n",
           dataSize, block_x, block_y, threadsPerBlock, numBlocks, iterations, 2.0 * memSize / 1e6);
    printf("[VICTIM-COMPUTE] Kernel type: COMPUTE-BOUND (minimal L2 usage)\n");

    if (!concurrent) {
        printf("[VICTIM-COMPUTE] ALONE — all %d SMs\n", totalSMs);
        CHECK_RT(cudaEventRecord(start, 0));
        GPUComputeIntensive<<<numBlocks, threadsPerBlock>>>(d_input, d_output, iterations, dataSize);
        CHECK_RT(cudaEventRecord(stop, 0));
        CHECK_RT(cudaStreamSynchronize(0));
    } else {
        // Green context: victim gets 6 SMs
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        CUdevResourceDesc descVictim;
        CHECK_CU(cuDevResourceGenerateDesc(&descVictim, &victimSlice, 1));
        CUgreenCtx victimGCtx;
        CHECK_CU(cuGreenCtxCreate(&victimGCtx, descVictim, device, CU_GREEN_CTX_DEFAULT_STREAM));
        CUstream victimStream;
        CHECK_CU(cuGreenCtxStreamCreate(&victimStream, victimGCtx, CU_STREAM_NON_BLOCKING, 0));

        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(victimGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[VICTIM-COMPUTE] CONCURRENT — %u SMs (enemy has 2 SMs, shared L2)\n", verify.sm.smCount);

        CHECK_RT(cudaEventRecord(start, victimStream));
        GPUComputeIntensive<<<numBlocks, threadsPerBlock, 0, victimStream>>>(d_input, d_output, iterations, dataSize);
        CHECK_RT(cudaEventRecord(stop, victimStream));
        CHECK_RT(cudaStreamSynchronize(victimStream));

        CHECK_CU(cuStreamDestroy(victimStream));
        CHECK_CU(cuGreenCtxDestroy(victimGCtx));
    }

    float ms;
    CHECK_RT(cudaEventElapsedTime(&ms, start, stop));
    printf("[VICTIM-COMPUTE] Completed in %.2f ms\n", ms);

    // Optional: verify result integrity
    float *h_output = (float*)malloc(memSize);
    CHECK_RT(cudaMemcpy(h_output, d_output, memSize, cudaMemcpyDeviceToHost));
    printf("[VICTIM-COMPUTE] Sample output[0] = %f (sanity check)\n", h_output[0]);
    free(h_output);

    CHECK_RT(cudaEventDestroy(start));
    CHECK_RT(cudaEventDestroy(stop));
    CHECK_RT(cudaFree(d_input));
    CHECK_RT(cudaFree(d_output));
    CHECK_CU(cuDevicePrimaryCtxRelease(device));
    return 0;
}
