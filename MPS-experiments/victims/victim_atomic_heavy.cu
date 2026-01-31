#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ATOMIC HEAVY: Many atomic operations causing serialization
// Expected behavior: L2 atomics traffic, different contention pattern
// Shows contention impact on atomic-heavy workloads (histograms, etc.)
__global__ void atomicHeavyKernel(float* d_data, float* d_histogram, int N, int num_bins,
                                   unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            // Read data and compute bin
            float val = d_data[i];
            int bin = (int)(val * num_bins) % num_bins;
            
            // Atomic add to histogram
            atomicAdd(&d_histogram[bin], 1.0f);
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    int num_bins = 256;  // Histogram bins
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 10000;
    
    printf("[ATOMIC] N=%d (%.1fMB), bins=%d, iters=%llu\n", N, N*4.0/1e6, num_bins, iters);
    
    float *d_data, *d_histogram;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_histogram, num_bins * sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 1000) / 1000.0f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, num_bins * sizeof(float));
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    atomicHeavyKernel<<<256, 256>>>(d_data, d_histogram, N, num_bins, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[ATOMIC] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
