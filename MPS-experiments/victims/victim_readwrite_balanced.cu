#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// READ-WRITE BALANCED: Equal reads and writes (stencil-like pattern)
// Expected behavior: Both read and write hit rates degrade under contention
__global__ void stencilKernel(float* d_data, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            // Read neighbors (3-point stencil)
            float left = d_data[(i - 1 + N) % N];
            float center = d_data[i];
            float right = d_data[(i + 1) % N];
            
            // Compute and write back
            float result = 0.25f * left + 0.5f * center + 0.25f * right;
            d_data[i] = result;
        }
        __syncthreads();  // Ensure all writes complete before next iteration
    }
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 50000;
    
    printf("[STENCIL] N=%d (%.1fMB), iters=%llu\n", N, N*4.0/1e6, iters);
    
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100) * 0.01f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    stencilKernel<<<256, 256>>>(d_data, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[STENCIL] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
