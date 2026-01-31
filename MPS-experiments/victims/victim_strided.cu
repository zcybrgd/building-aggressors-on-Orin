#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// STRIDED ACCESS: Non-coalesced memory access pattern
// Expected behavior: Poor L1 hit rate, moderate L2 hit rate
// Shows cache line utilization issues + contention
__global__ void stridedKernel(float* d_data, float* d_result, int N, int stride_factor, 
                               unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    float sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        // Strided access - each thread accesses elements stride_factor apart
        // This causes poor cache line utilization
        for (int i = 0; i < N / total_threads; i++) {
            int idx = (tid * stride_factor + i * total_threads * stride_factor) % N;
            sum += d_data[idx];
        }
    }
    
    atomicAdd(d_result, sum);
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    int stride_factor = (argc > 2) ? atoi(argv[2]) : 16;  // Stride in elements
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 50000;
    
    printf("[STRIDED] N=%d (%.1fMB), stride=%d, iters=%llu\n", 
           N, N*4.0/1e6, stride_factor, iters);
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    stridedKernel<<<256, 256>>>(d_data, d_result, N, stride_factor, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[STRIDED] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
