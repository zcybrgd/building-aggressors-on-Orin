#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// HIGH REUSE: Same data accessed many times (should have excellent L2 hit rate)
// Expected behavior: Very high L2 hit rate in isolation, drops dramatically under contention
// Best case for showing contention impact - aggressor evicts hot data
__global__ void highReuseKernel(float* d_data, float* d_result, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    
    // Working set is small (64KB = 16K floats) - fits entirely in L2
    int working_set = 16 * 1024;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        // Repeatedly access same small working set
        for (int i = tid; i < working_set; i += stride) {
            sum += d_data[i];
            sum += d_data[(i + 32) % working_set];
            sum += d_data[(i + 64) % working_set];
            sum += d_data[(i + 128) % working_set];
        }
    }
    
    atomicAdd(d_result, sum);
}

int main(int argc, char* argv[]) {
    int N = 16 * 1024;  // 64KB - small working set
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 100000;
    
    printf("[HIGH_REUSE] N=%d (%.1fKB), iters=%llu\n", N, N*4.0/1e3, iters);
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100) * 0.01f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    highReuseKernel<<<256, 256>>>(d_data, d_result, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[HIGH_REUSE] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
