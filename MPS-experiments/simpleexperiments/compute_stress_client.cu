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

__global__ void compute_kernel(float *buffer, int elements, int iterations) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }

    float val0 = buffer[idx];
    float val1 = val0 * 0.5f + 0.1f;
    for (int iter = 0; iter < iterations; ++iter) {
        val0 = fmaf(val0, 0.99991f, 0.00003f);
        val1 = fmaf(val1, 1.00004f, -0.00002f);
        val0 = val0 + val1 * 0.25f;
        val1 = val1 - val0 * 0.15f;
    }
    buffer[idx] = val0 + val1;
}

int main(int argc, char **argv) {
    const int iterations = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 50000;
    const int elements = (argc > 2) ? std::max(8192, std::atoi(argv[2])) : (1 << 19);
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    std::printf("[compute-client] iterations=%d elements=%d\n", iterations, elements);

    std::vector<float> host(elements, 0.5f);

    float *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, bytes));
    CUDA_CHECK(cudaMemcpy(d_buffer, host.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid((elements + block.x - 1) / block.x);

    const auto start = std::chrono::high_resolution_clock::now();
    compute_kernel<<<grid, block>>>(d_buffer, elements, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(host.data(), d_buffer, bytes, cudaMemcpyDeviceToHost));
    const double elapsed = std::chrono::duration<double>(end - start).count();
    std::printf("[compute-client] elapsed %.2f s, total iters %d\n", elapsed, iterations);
    std::printf("[compute-client] checksum %.6f\n", host[host.size() / 2]);

    CUDA_CHECK(cudaFree(d_buffer));
    return 0;
}
