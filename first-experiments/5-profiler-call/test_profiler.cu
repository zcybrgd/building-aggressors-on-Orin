#include <stdio.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

__global__ void simple_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    const int N = 1024;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    printf("Starting warmup phase\n");
    cudaFree(0);  // Context warmup
    simple_kernel<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; i++) {
        simple_kernel<<<4, 256>>>(d_data, N);
    }
    cudaDeviceSynchronize();
    printf("Starting profiled measurements\n");
    cudaProfilerStart();  
    //MEASUREMENT PHASE 
    for (int i = 0; i < 3; i++) {
        printf("Measurement run %d\n", i + 1);
        simple_kernel<<<4, 256>>>(d_data, N);
    }
    cudaDeviceSynchronize();
    cudaProfilerStop();
    cudaFree(d_data);
    return 0;
}
