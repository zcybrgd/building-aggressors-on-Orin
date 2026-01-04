#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void victimKernel(float* d_data, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f; 
    // L2-intensive: repeated reads from same working set
    for (unsigned long long iter = 0; iter < iters; iter++) {
        // Access patterns that should benefit from L2 cache
        for (int i = tid; i < N; i += stride) {
            sum += d_data[i] * 1.1f;
            sum += d_data[(i + 64) % N] * 0.9f;  //reuse nearby data
            sum += d_data[(i + 128) % N] * 0.8f;
        }
        //occasional write to force cache coherence
        if ((iter & 0xFF) == 0) {
            int write_idx = tid % N;
            d_data[write_idx] = sum * 0.001f;
        }
    }
    // final write to prevent optimization
    if (tid < N) {
        d_data[tid] = sum;
    }
}

int main(int argc, char* argv[]) {
    //config
    int N = 256 * 1024;  //1 MB working set (fits in 2 MB L2) i shall increase it maybe
    
    //iterations: enough for NCU to profile properly
    //NCU needs multiple kernel invocations for accurate metrics
    unsigned long long iters;
    if (argc > 1) {
        iters = atoll(argv[1]);
    } else {
        // Default: enough iterations for ~2-3 seconds of execution
        // This gives NCU time to collect all metrics
        iters = 50000;
    }
    
    printf("[VICTIM] Configuration:\n");
    printf("  Working set: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Iterations: %llu\n", iters);
    printf("  Grid: 256 blocks x 256 threads = %d total threads\n", 256 * 256);

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    //init with some data
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i * 0.01f;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);
    
    //warm up (important for consistent profiling)
    //printf("[VICTIM] Warming up...\n");
    //victimKernel<<<256, 256>>>(d_data, N, 100);
    //cudaDeviceSynchronize();
    //printf("[VICTIM] Starting main execution...\n");
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Main execution
    cudaEventRecord(start);
    victimKernel<<<256, 256>>>(d_data, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[VICTIM] Completed in %.2f ms\n", ms);
    printf("[VICTIM] Performance: %.2f GFLOPS\n",  (3.0 * N * iters / (ms * 1e6)));  // 3 FLOPs per element per iter
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
