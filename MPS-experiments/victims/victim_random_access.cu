#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// RANDOM ACCESS: Poor spatial locality, defeats prefetching
// Expected behavior: Already low L1/L2 hit rates, contention makes it worse
// Shows contention impact on already-stressed cache
__global__ void randomAccessKernel(float* d_data, unsigned int* d_indices, float* d_result, 
                                    int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            // Random read based on precomputed indices
            unsigned int idx = d_indices[i];
            sum += d_data[idx];
            
            // Random write
            unsigned int write_idx = d_indices[(i + 1024) % N];
            d_data[write_idx] = sum * 0.0001f;
        }
    }
    
    atomicAdd(d_result, sum);
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 10000;  // Fewer iters - slower kernel
    
    printf("[RANDOM] N=%d (%.1fMB), iters=%llu\n", N, N*4.0/1e6, iters);
    
    float *d_data, *d_result;
    unsigned int *d_indices;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_indices, N * sizeof(unsigned int));
    cudaMalloc(&d_result, sizeof(float));
    
    // Generate random indices on host
    float* h_data = (float*)malloc(N * sizeof(float));
    unsigned int* h_indices = (unsigned int*)malloc(N * sizeof(unsigned int));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 1000) * 0.001f;
        // LCG for pseudo-random indices
        h_indices[i] = ((unsigned long long)i * 1103515245 + 12345) % N;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    free(h_data);
    free(h_indices);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    randomAccessKernel<<<256, 256>>>(d_data, d_indices, d_result, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[RANDOM] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
