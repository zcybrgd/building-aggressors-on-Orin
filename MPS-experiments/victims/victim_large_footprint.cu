#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// LARGE FOOTPRINT: Working set exceeds L2 (8MB vs 4MB L2)
// Expected behavior: Already low L2 hit rate, contention is additive
// Shows contention on bandwidth-limited workloads
__global__ void largeFootprintKernel(float* d_data, float* d_result, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            sum += d_data[i];
            sum += d_data[(i + N/4) % N];  // Jump around to defeat prefetch
        }
    }
    
    atomicAdd(d_result, sum);
}

int main(int argc, char* argv[]) {
    int N = 2 * 1024 * 1024;  // 8MB - 2x L2 size
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 5000;
    
    printf("[LARGE] N=%d (%.1fMB), iters=%llu\n", N, N*4.0/1e6, iters);
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 10000) * 0.0001f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    largeFootprintKernel<<<256, 256>>>(d_data, d_result, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[LARGE] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
