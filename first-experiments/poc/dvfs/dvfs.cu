#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cmath>

__global__ void flopsKernel(float *data, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = data[idx];
    // light arithmetic, controllable by iters
    for (int i = 0; i < iters; ++i) {
        x = x * 1.0001f + 0.1234f;
        x = sqrtf(x);
        x = x + __sinf(x);
    }
    data[idx] = x;
}

int main(int argc, char** argv) {
    int numElems = 1 << 20; // 1M
    int iters    = (argc > 1) ? atoi(argv[1]) : 4000;
    int blocks   = 256;
    int threads  = 256;

    float *d_data;
    cudaMalloc(&d_data, numElems * sizeof(float));
    cudaMemset(d_data, 0, numElems * sizeof(float));

    // Warm-up
    flopsKernel<<<blocks, threads>>>(d_data, iters);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    flopsKernel<<<blocks, threads>>>(d_data, iters);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%.4f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    return 0;
}
