#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void victimKernel(unsigned long long iters, unsigned long long* d_result) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long sum = tid;
    
    // Pure compute: integer multiply-add chain (no memory reads)
    for (unsigned long long iter = 0; iter < iters; iter++) {
        sum = (sum * 1103515245ULL + 12345ULL) * 48271ULL;  // Linear congruential generator
        sum ^= (sum >> 31);  // XOR for determinism
    }
    
    //atomicAdd(d_result, sum);
}

int main(int argc, char* argv[]) {
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 1000000;
    
    unsigned long long *d_result;
    cudaMalloc(&d_result, sizeof(unsigned long long));
    cudaMemset(d_result, 0, sizeof(unsigned long long));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    victimKernel<<<256, 256>>>(iters, d_result);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    
    float ms;
    unsigned long long result;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(&result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    printf("%.2f ms  RESULT=%llu\n", ms, result);
    
    cudaFree(d_result);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

