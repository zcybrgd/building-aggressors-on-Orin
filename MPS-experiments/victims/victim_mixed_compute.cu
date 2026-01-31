#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// MIXED COMPUTE: Compute-heavy with occasional memory access
// Expected behavior: Less affected by contention (compute-bound)
// Shows that L2 contention impact depends on memory intensity
__global__ void mixedComputeKernel(float* d_data, float* d_result, int N, 
                                    int compute_intensity, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            float val = d_data[i];
            
            // Heavy compute between memory accesses
            for (int c = 0; c < compute_intensity; c++) {
                val = sinf(val) * cosf(val) + sqrtf(fabsf(val) + 0.001f);
                val = val * val - val * 0.5f + 0.25f;
            }
            
            sum += val;
        }
    }
    
    atomicAdd(d_result, sum);
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    int compute_intensity = (argc > 2) ? atoi(argv[2]) : 50;  // Compute ops per load
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 10000;
    
    printf("[MIXED] N=%d (%.1fMB), compute_intensity=%d, iters=%llu\n", 
           N, N*4.0/1e6, compute_intensity, iters);
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100) * 0.01f + 0.1f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    mixedComputeKernel<<<256, 256>>>(d_data, d_result, N, compute_intensity, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[MIXED] Time: %.2f ms\n", ms);
    
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
