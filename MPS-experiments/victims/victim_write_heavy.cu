#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void writeHeavyKernel(float* d_out, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float val = (float)tid * 0.001f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            d_out[i] = val;
            d_out[(i + 64) % N] = val * 1.1f;
            d_out[(i + 128) % N] = val * 0.9f;
            val = val * 1.0001f + 0.0001f;  // Evolve value to prevent optimization
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;  // 1MB
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 50000;
    
    printf("[WRITE_HEAVY] N=%d (%.1fMB), iters=%llu\n", N, N*4.0/1e6, iters);
    
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_out, 0, N * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    writeHeavyKernel<<<256, 256>>>(d_out, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("[WRITE_HEAVY] Time: %.2f ms\n", ms);
    
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
