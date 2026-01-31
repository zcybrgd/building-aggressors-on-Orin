#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// STREAMING: Sequential access, no reuse (STREAM benchmark style)
// Expected behavior: Low L2 hit rate baseline, bandwidth-limited
// Contention mainly affects DRAM bandwidth, less cache pollution impact
__global__ void streamingKernel(float* d_a, float* d_b, float* d_c, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float scalar = 3.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        // STREAM Triad: c[i] = a[i] + scalar * b[i]
        for (int i = tid; i < N; i += stride) {
            d_c[i] = d_a[i] + scalar * d_b[i];
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 512 * 1024;  // 2MB per array = 6MB total (exceeds L2)
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 10000;
    
    printf("[STREAM] N=%d (%.1fMB per array, %.1fMB total), iters=%llu\n", 
           N, N*4.0/1e6, N*12.0/1e6, iters);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)i * 0.001f;
    cudaMemcpy(d_a, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    streamingKernel<<<256, 256>>>(d_a, d_b, d_c, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Calculate achieved bandwidth
    double bytes = 3.0 * N * sizeof(float) * iters;  // 2 reads + 1 write
    double gbps = bytes / (ms * 1e6);
    
    printf("[STREAM] Time: %.2f ms, Bandwidth: %.2f GB/s\n", ms, gbps);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
