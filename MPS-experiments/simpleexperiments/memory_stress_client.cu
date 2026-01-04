#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                       \
        cudaError_t _err = (call);                                                             \
        if (_err != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,               \
                        cudaGetErrorString(_err));                                             \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)

__global__ void streaming_kernel(float *buffer, int elements, int iterations) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }

    float val = buffer[idx];
    for (int iter = 0; iter < iterations; ++iter) {
        val = fmaf(val, 1.00001f, 0.000001f);
        val = fmaf(val, 0.99999f, 0.0000003f);
    }
    buffer[idx] = val;
}

int main(int argc, char **argv) {
    const int iterations = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 50000;
    const int elements = (argc > 2) ? std::max(1024, std::atoi(argv[2])) : (1 << 20);
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    std::printf("[memory-client] iterations=%d elements=%d (%.2f MB)\n", iterations,
                elements, bytes / (1024.0 * 1024.0));

    std::vector<float> host(elements);
    for (int i = 0; i < elements; ++i) {
        host[i] = static_cast<float>((i % 97) * 0.013f);
    }

    float *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, bytes));
    CUDA_CHECK(cudaMemcpy(d_buffer, host.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid((elements + block.x - 1) / block.x);

    const auto start = std::chrono::high_resolution_clock::now();
    streaming_kernel<<<grid, block>>>(d_buffer, elements, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(host.data(), d_buffer, bytes, cudaMemcpyDeviceToHost));

    const double elapsed = std::chrono::duration<double>(end - start).count();
    std::printf("[memory-client] elapsed %.2f s, total iters %d\n", elapsed, iterations);

    CUDA_CHECK(cudaFree(d_buffer));

    return 0;
}
