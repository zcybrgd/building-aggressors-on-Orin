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

//RANDOM ACCESS PATTERN
__global__ void MatrixMulUnoptimized(float *C, float *A, float *B, int wA, int wB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < wA && col < wB) {
        float sum = 0.0f;
        volatile float temp; // Prevent optimization
        
        // RANDOM ACCESS PATTERN - destroys cache coherence
        for (int k = 0; k < wA; k++) {
            int a_index = (row * wA + ((k * 17) % wA)); // Non-sequential
            int b_index = (((k * 13) % wA) * wB + col); // Non-sequential
            
            temp = A[a_index] * B[b_index]; // Force memory access
            sum += temp;
        }
        
        C[row * wB + col] = sum;
    }
}


__global__ void MatrixMulLargeStride(float *C, float *A, float *B, int wA, int wB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < wA && col < wB) {
        float sum = 0.0f;
        
        //LARGE STRIDE
        for (int k = 0; k < wA; k++) {
            //Stride by 1024 elements guarantees cache misses
            int a_index = row * wA + ((k * 1024) % wA);
            int b_index = ((k * 1024) % wA) * wB + col;
            
            sum += A[a_index] * B[b_index];
        }
        
        C[row * wB + col] = sum;
    }
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
    
    dim3 threads(16, 16);
    dim3 grid((size + threads.x - 1) / threads.x, (size + threads.y - 1) / threads.y);
    //MatrixMulUnoptimized<<<grid, threads>>>(d_C, d_A, d_B, size, size);
    MatrixMulLargeStride<<<grid, threads>>>(d_C, d_A, d_B, size, size);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
    
    return 0;
}