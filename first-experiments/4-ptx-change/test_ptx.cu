//this is just a test to verify if PTX changes when sweeping matrix/block configurations
// Compile: nvcc -arch=sm_87 -ptx ptx_change_test.cu -o test_matrix240_block8x8.ptx
//nvcc -arch=sm_87 -ptx ptx_change_test.cu -o test_matrix2024_block32x32.ptx -DMATRIX_SIZE=2024 -DBLOCK_X=32 -DBLOCK_Y=32

#include <cuda_runtime.h>
#include <stdio.h>

//config just to test not like what LS-CAT uses
#ifndef MATRIX_SIZE
#define MATRIX_SIZE 240
#endif

#ifndef BLOCK_X
#define BLOCK_X 8
#endif

#ifndef BLOCK_Y  
#define BLOCK_Y 8
#endif

// test kernel
__global__ void test_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * n + idx;
    if (idx < n && idy < n) {
        c[id] = a[id] * 2.5f + b[id];
    }
}

// Alt: Runtime parameter version (what LS-CAT does)
__global__ void test_kernel_runtime(float* a, float* b, float* c, int xsize, int ysize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * xsize + idx;
    if (idx < xsize && idy < ysize) {
        c[id] = a[id] * 2.5f + b[id];
    }
}

int main() {
    printf("=== PTX Change Test ===\n");
    printf("Compile-time config:\n");
    printf("  MATRIX_SIZE: %d\n", MATRIX_SIZE);
    printf("  BLOCK_X: %d, BLOCK_Y: %d\n", BLOCK_X, BLOCK_Y);
    int size = MATRIX_SIZE * MATRIX_SIZE;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    int gridX = (MATRIX_SIZE + BLOCK_X - 1) / BLOCK_X;
    int gridY = (MATRIX_SIZE + BLOCK_Y - 1) / BLOCK_Y;
    dim3 gridDim(gridX, gridY);
    printf("\nLaunch config:\n");
    printf("  Grid: (%d, %d)\n", gridX, gridY);
    printf("  Block: (%d, %d)\n", BLOCK_X, BLOCK_Y);
    //Launch compile-time parameterized kernel
    test_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, MATRIX_SIZE);
    cudaDeviceSynchronize();
    //Launch runtime parameterized kernel (LS-CAT style)
    test_kernel_runtime<<<gridDim, blockDim>>>(d_a, d_b, d_c, MATRIX_SIZE, MATRIX_SIZE);
    cudaDeviceSynchronize();
    printf("\nALL DONEE!!\n");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
