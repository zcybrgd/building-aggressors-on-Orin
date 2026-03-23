
// ============================================================================
// 1. SIMPLE KERNEL - Minimal resources (baseline)
// ============================================================================
__global__ void simpleKernel(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = idx * 2.0f;
    }
}

// ============================================================================
// 2. SHARED MEMORY KERNEL - Shows shared memory usage
// ============================================================================
__global__ void sharedMemoryKernel(float *input, float *output, int n) {
    __shared__ float sharedData[256];  // Static shared memory

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n) {
        sharedData[tid] = input[idx];
    }
    __syncthreads();  // Barrier synchronization

    // Process with shared memory
    if (idx < n && tid < 255) {
        output[idx] = (sharedData[tid] + sharedData[tid + 1]) * 0.5f;
    }
}

// ============================================================================
// 3. DYNAMIC SHARED MEMORY KERNEL - Runtime-sized shared memory
// ============================================================================
__global__ void dynamicSharedKernel(float *input, float *output, int n) {
    extern __shared__ float dynShared[];  // Dynamic shared memory

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        dynShared[tid] = input[idx] * 2.0f;
    }
    __syncthreads();

    if (idx < n) {
        output[idx] = dynShared[tid];
    }
}

// ============================================================================
// 4. HIGH REGISTER USAGE KERNEL - Forces register pressure
// ============================================================================
__global__ void highRegisterKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Many local variables to consume registers
    float a0 = input[idx] * 1.1f;
    float a1 = input[idx] * 2.2f;
    float a2 = input[idx] * 3.3f;
    float a3 = input[idx] * 4.4f;
    float a4 = input[idx] * 5.5f;
    float a5 = input[idx] * 6.6f;
    float a6 = input[idx] * 7.7f;
    float a7 = input[idx] * 8.8f;

    // Complex computation using all variables
    float result = a0 + a1 + a2 + a3;
    result = result * a4 + a5;
    result = sqrtf(result) + a6;
    result = result * a7 + expf(a0);

    output[idx] = result;
}

// ============================================================================
// 5. CONSTANT MEMORY KERNEL - Shows cmem usage
// ============================================================================
__constant__ float constCoeffs[64];  // Constant memory array

__global__ void constantMemoryKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 64; i++) {
            sum += input[idx] * constCoeffs[i];
        }
        output[idx] = sum;
    }
}

// ============================================================================
// 6. LOCAL MEMORY SPILL KERNEL - Forces register spills
// ============================================================================
__global__ void spillKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    //large local array that won't fit in registers
    float localArray[128];

    //init array
    for (int i = 0; i < 128; i++) {
        localArray[i] = input[idx] * (i + 1);
    }
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
        sum += localArray[i];
    }

    output[idx] = sum;
}

// ============================================================================
// 7. MULTIPLE BARRIERS KERNEL - Shows barrier usage
// ============================================================================
__global__ void multipleBarriersKernel(float *input, float *output, int n) {
    __shared__ float buffer1[256];
    __shared__ float buffer2[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n) {
        buffer1[tid] = input[idx];
    }
    __syncthreads();  // Barrier 1

    if (idx < n) {
        buffer2[tid] = buffer1[tid] * 2.0f;
    }
    __syncthreads();  // Barrier 2

    if (idx < n && tid > 0) {
        output[idx] = (buffer2[tid] + buffer2[tid - 1]) * 0.5f;
    }
    __syncthreads();  // Barrier 3
}

// ============================================================================
// 8. STATIC GLOBAL MEMORY KERNEL - Shows gmem usage
// ============================================================================
__device__ float globalBuffer[1024];  // Static global memory

__global__ void staticGlobalKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx < 1024) {
        globalBuffer[idx] = input[idx] * 2.0f;
        __threadfence();  // Ensure visibility
        output[idx] = globalBuffer[idx];
    }
}

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // Launch kernels
    simpleKernel<<<grid, block>>>(d_output, N);
    sharedMemoryKernel<<<grid, block>>>(d_input, d_output, N);
    dynamicSharedKernel<<<grid, block, 256 * sizeof(float)>>>(d_input, d_output, N);
    highRegisterKernel<<<grid, block>>>(d_input, d_output, N);
    constantMemoryKernel<<<grid, block>>>(d_input, d_output, N);
    spillKernel<<<grid, block>>>(d_input, d_output, N);
    multipleBarriersKernel<<<grid, block>>>(d_input, d_output, N);
    staticGlobalKernel<<<grid, block>>>(d_input, d_output, N);

    cudaDeviceSynchronize();
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

