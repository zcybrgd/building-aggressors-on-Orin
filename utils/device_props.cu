#include <iostream>
#include <cuda_runtime.h>
//idem as props.cu but with more details
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

        // --- L2 Cache ---
        std::cout << "\n  --- L2 Cache ---" << std::endl;
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes ("
                  << prop.l2CacheSize / (1024.0 * 1024.0) << " MB)" << std::endl;
        std::cout << "  Persisting L2 Cache Max Size: " << prop.persistingL2CacheMaxSize / 1024 << " KB" << std::endl;

        // --- L1 Cache / Shared Memory Configuration ---
        std::cout << "\n  --- L1 / Shared Memory ---" << std::endl;
        std::cout << "  Global L1 Cache Supported: " << (prop.globalL1CacheSupported ? "Yes" : "No") << std::endl;
        std::cout << "  Local L1 Cache Supported: " << (prop.localL1CacheSupported ? "Yes" : "No") << std::endl;
        std::cout << "  Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Shared Memory Per Multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  Reserved Shared Memory Per Block: " << prop.reservedSharedMemPerBlock / 1024 << " KB" << std::endl;


        // --- Registers and Warps ---
        std::cout << "\n  --- Register File / Execution Model ---" << std::endl;
        std::cout << "  Registers Per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Registers Per Multiprocessor: " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Blocks Per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;

        // occupancy-related limits
        std::cout << "  Max Threads Dimension: ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max Grid Size: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << std::endl;

        // --- Mem Hierarchy ---
        std::cout << "\n  --- Memory Hierarchy ---" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        double peakBW = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        std::cout << "  Theoretical Peak Memory Bandwidth: " << peakBW << " GB/s" << std::endl;

        // Misc Execution Capabilities 
        std::cout << "\n  --- Execution & Concurrency ---" << std::endl;
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  Compute Preemption Supported: " << (prop.computePreemptionSupported ? "Yes" : "No") << std::endl;
        std::cout << "  Cooperative Launch: " << (prop.cooperativeLaunch ? "Yes" : "No") << std::endl;
        std::cout << "  Cooperative Multi-Device Launch: " << (prop.cooperativeMultiDeviceLaunch ? "Yes" : "No") << std::endl;

    }

    return 0;
}

/*
cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
cudaFuncSetAttribute
*/
