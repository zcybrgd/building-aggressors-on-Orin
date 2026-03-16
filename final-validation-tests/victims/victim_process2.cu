#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void victimKernel(float* d_data, float* d_result, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            float val = d_data[i];
            local_sum += val;
            local_sum += d_data[(i + 64) % N] * 0.5f;
            local_sum += d_data[(i + 128) % N] * 0.25f;
        }
    }
    
    atomicAdd(d_result, local_sum);
}
int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 50000;
    printf("[VICTIM] N=%d (%.1fMB), iters=%llu\n", N, N*4.0/1e6, iters);
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 1000) * 0.0000000000001f;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    victimKernel<<<256, 256>>>(d_data, d_result, N, iters);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float gpu_result;
    cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("[VICTIM] Time: %.2f ms\n", ms);
    printf("[VICTIM] RESULT=%.6e\n", gpu_result);
    
    cudaFree(d_data); cudaFree(d_result);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}