#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CU(call) do { CUresult err = call; if (err != CUDA_SUCCESS) { const char *n, *s; cuGetErrorName(err, &n); cuGetErrorString(err, &s); fprintf(stderr, "[CU] %s: %s at %s:%d\n", n, s, __FILE__, __LINE__); exit(1); } } while(0)
#define CHECK_RT(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "[RT] %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } } while(0)

__global__ void computeKernel(float *out, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) { unsigned int smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid)); printf("[KERNEL] block %2d -> SM %u\n", blockIdx.x, smid); }
    float v = (float)tid * 0.001f + 1.0f;
    for (int i = 0; i < iters; i++) v = v * 1.0001f + 0.0001f;
    if (tid < 4096) out[tid] = v;
}

__global__ void victimKernel(float* d_data, int N, unsigned long long iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("[VICTIM] block %d -> SM %u\n", blockIdx.x, smid);
    }
    for (unsigned long long iter = 0; iter < iters; iter++) {
        for (int i = tid; i < N; i += stride) {
            sum += d_data[i] * 1.1f;
            sum += d_data[(i + 64)  % N] * 0.9f;
            sum += d_data[(i + 128) % N] * 0.8f;
        }
        if ((iter & 0xFF) == 0) d_data[tid % N] = sum * 0.001f;
    }
    if (tid < N) d_data[tid] = sum;
}



int main(int argc, char** argv) {
    int scenario = (argc > 1) ? atoi(argv[1]) : 0;
    int run_num  = (argc > 2) ? atoi(argv[2]) : 1;
    int N = 1 << 20;  // 1M elements
    unsigned long long iters = 500;
    CHECK_RT(cudaSetDevice(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    float *d_data; CHECK_RT(cudaMalloc(&d_data, N * sizeof(float)));
    // Initialize d_data so reads aren't all zeros
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 1000) * 0.001f;
    CHECK_RT(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data);
    cudaEvent_t t0, t1; CHECK_RT(cudaEventCreate(&t0)); CHECK_RT(cudaEventCreate(&t1));
    if (scenario == 0) {
        printf("[RUN %d][8 SMs] 16 blocks x 256 threads (no green ctx)\n", run_num);
        CHECK_RT(cudaEventRecord(t0,0));
        victimKernel<<<16,256>>>(d_data, N, iters);
        CHECK_RT(cudaEventRecord(t1,0)); CHECK_RT(cudaStreamSynchronize(0));
    } else {
        CUdevResource full; CHECK_CU(cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM));
        CUdevResource vSlice, eSlice; unsigned int nb=1;
        CHECK_CU(cuDevSmResourceSplitByCount(&vSlice, &nb, &full, &eSlice, 0, 5));
        CUdevResource *chosen = (scenario==1) ? &vSlice : &eSlice;
        CUdevResourceDesc desc; CHECK_CU(cuDevResourceGenerateDesc(&desc, chosen, 1));
        CUgreenCtx gCtx; CHECK_CU(cuGreenCtxCreate(&gCtx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        CUstream gStream; CHECK_CU(cuGreenCtxStreamCreate(&gStream, gCtx, CU_STREAM_NON_BLOCKING, 0));
        CUdevResource ver; CHECK_CU(cuGreenCtxGetDevResource(gCtx, &ver, CU_DEV_RESOURCE_TYPE_SM));
        printf("[RUN %d][%u SMs] 16 blocks x 256 threads (green ctx scenario %d)\n", run_num, ver.sm.smCount, scenario);
        CHECK_RT(cudaEventRecord(t0, gStream));
        victimKernel<<<16,256,0,gStream>>>(d_data, N, iters);
        CHECK_RT(cudaEventRecord(t1, gStream)); CHECK_RT(cudaStreamSynchronize(gStream));
        CHECK_CU(cuStreamDestroy(gStream)); CHECK_CU(cuGreenCtxDestroy(gCtx));
    }
    float ms; CHECK_RT(cudaEventElapsedTime(&ms, t0, t1));
    printf("[RUN %d] Elapsed: %.3f ms\n", run_num, ms);
    CHECK_RT(cudaEventDestroy(t0)); CHECK_RT(cudaEventDestroy(t1)); CHECK_RT(cudaFree(d_data));
    return 0;
}

