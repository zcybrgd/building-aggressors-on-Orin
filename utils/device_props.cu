#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }
    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        error_id = cudaGetDeviceProperties(&prop, i);
        if (error_id != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << i << ": "
                      << cudaGetErrorString(error_id) << std::endl;
            continue;
        }
        std::cout << "\n--- Device " << i << " ---" << std::endl;
        std::cout << "  Name: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
 	std::cout << "L2 Cache Size: " << prop.l2CacheSize << " bytes ("
                  << prop.l2CacheSize / (1024.0 * 1024.0) << " MB)" << std::endl;
    }

    return 0;
}
