#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//simple compute-bound kernel (lightweight math operations)
__global__ void computeKernel(float *data, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];
    //lightweight computation to avoid overheating of my gpu :))
    for (int i = 0; i < iterations; i++) {
        value = sqrtf(value * 1.001f + 0.5f);
        value = value * 0.99f + sinf(value);
    }
    data[idx] = value;
}

//simple memory-bound kernel (strided access pattern to create some RAM // l2 pressure)
__global__ void memoryKernel(float *input, float *output, int stride, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int access_idx = (idx * stride) % size;
        output[idx] = input[access_idx] * 2.0f + 1.0f;
    }
}

void runComputeBoundTest(int blocks, int threads, int iterations) {
    int size = blocks * threads;
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    //initialize data
    float *h_data = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        //should be deterministic for the results to be reproducible
        h_data[i] = 1.0f + (float)i / size;
    }

    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    //warmup because there is CUDA context initialization, driver overhead, code loading to GPU etc..
    //so our measurement shouldnt be polluted with all of these
    computeKernel<<<blocks, threads>>>(d_data, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //timing 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    computeKernel<<<blocks, threads>>>(d_data, iterations);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float cuda_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_time_ms, start, stop));
    printf("COMPUTE_KERNEL_TIMING,%.4f\n", cuda_time_ms);
    
    //cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
}

void runMemoryBoundTest(int blocks, int threads, int stride) {
    int size = blocks * threads;
    float *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * sizeof(float)));
    
    //initialize data
    float *h_input = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    //warmup same thing
    memoryKernel<<<blocks, threads>>>(d_input, d_output, stride, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    //timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    memoryKernel<<<blocks, threads>>>(d_input, d_output, stride, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float cuda_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cuda_time_ms, start, stop));
    
    printf("MEMORY_KERNEL_TIMING,%.4f\n", cuda_time_ms);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
}

void printUsage(const char* progName) {
    printf("Usage: %s <kernel_type>\n", progName);
    printf("  kernel_type: 'compute' or 'memory'\n");
    printf("Example:\n");
    printf("  %s compute\n", progName);
    printf("  %s memory\n", progName);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printUsage(argv[0]);
        return 1;
    }

    const char* kernel_type = argv[1];

    // config for lightweight execution
    int blocks = 64; // 64/16Sms de mon gpu = 4 blocks per SM , we can replace it by 32 for a lighter test
    int threads = 256; // 256 * 4 = 1024 threads per SM
    int compute_iterations = 1000000;  // lightweight to avoid overheating
    int memory_stride = 16;
    printf("Configuration: %d blocks x %d threads\n", blocks, threads);

    if (strcmp(kernel_type, "compute") == 0) {
        printf("Running: COMPUTE-BOUND KERNEL\n");
        runComputeBoundTest(blocks, threads, compute_iterations);
    } else if (strcmp(kernel_type, "memory") == 0) {
        printf("Running: MEMORY-BOUND KERNEL\n");
        runMemoryBoundTest(blocks, threads, memory_stride);
    } else {
        fprintf(stderr, "Error: Invalid kernel type '%s'\n", kernel_type);
        printUsage(argv[0]);
        return 1;
    }
    return 0;
}