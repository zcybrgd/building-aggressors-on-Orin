#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

template <int BLOCK_SIZE> 
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;
    
    float Csub = 0;
    
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

int main(int argc, char **argv) {
    int size = 1024;
    if (argc > 1) size = atoi(argv[1]);    
    size_t mem_size = size * size * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    CHECK_CUDA(cudaMallocHost(&h_A, mem_size));
    CHECK_CUDA(cudaMallocHost(&h_B, mem_size)); 
    CHECK_CUDA(cudaMallocHost(&h_C, mem_size));
    
    CHECK_CUDA(cudaMalloc(&d_A, mem_size));
    CHECK_CUDA(cudaMalloc(&d_B, mem_size));
    CHECK_CUDA(cudaMalloc(&d_C, mem_size));
    
    for (int i = 0; i < size * size; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 0.01f;
    }
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice));
    
    dim3 threads(32, 32);
    dim3 grid(size / threads.x, size / threads.y);
    

  
    MatrixMulCUDA<32><<<grid, threads>>>(d_C, d_A, d_B, size, size);
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
    
    return 0;
}