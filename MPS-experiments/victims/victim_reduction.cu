#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// REDUCTION: Tree-based reduction with shared memory
// Expected behavior: Shared memory heavy, less L2 pressure
// Shows contention impact on kernels with good shared memory usage
__global__ void reductionKernel(float* d_data, float* d_result, int N, unsigned long long iters) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float block_sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iters; iter++) {
        // Each thread loads and sums multiple elements
        float local_sum = 0.0f;
        for (int i = gid; i < N; i += stride) {
            local_sum += d_data[i];
        }
        
        // Store in shared memory
        sdata[tid] = local_sum;
        __syncthreads();
        
        // Tree reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        block_sum += sdata[0];
    }
    
    // Only thread 0 writes result
    if (tid == 0) {
        atomicAdd(d_result, block_sum);
    }
}

int main(int argc, char* argv[]) {
    int N = 256 * 1024;
    unsigned long long iters = (argc > 1) ? atoll(argv[1]) : 50000;
    
    printf("[REDUCTION] N=%d (%.1fMB), iters=%llu\n", N, N*4.0/1e6, iters);
    
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));
    
    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100) * 0.0001f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));
    free(h_data);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int blockSize = 256;
    int sharedMemSize = blockSize * sizeof(float);
    
    cudaEventRecord(start);
    reductionKernel<<<256, blockSize, sharedMemSize>>>(d_data, d_result, N, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms, result;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("[REDUCTION] Time: %.2f ms, Result: %.6e\n", ms, result);
    
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
