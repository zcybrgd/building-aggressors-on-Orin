
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errName, *errStr; \
        cuGetErrorName(err, &errName); \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "[CUDA Driver Error] %s: %s at %s:%d\n", \
                errName, errStr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA Runtime Error] %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int matrices_[7][2] = {{240,240},{496,496},{784,784},{1016,1016},{1232,1232},{1680,1680},{2024,2024}};
int blocks_[13][2] = {{1,1024}, {2,512}, {4,256}, {8,128}, {16,64},{32,32}, {64,16}, {128,8}, {256,4}, {512,2},{1024,1}, {4,128}, {8,64}};

__global__ void ConvolutionRowGPU(double *d_Dst, double *d_Src, double *d_Filter, int imageW, int imageH, int filterR){
    int k;
    double sum=0;
    int row=blockDim.y*blockIdx.y+threadIdx.y;
    int col=blockDim.x*blockIdx.x+threadIdx.x;

    for (k = -filterR; k <= filterR; k++) {
        int d = col+ k;
        if (d >= 0 && d < imageW) {
            sum += d_Src[row * imageW + d] * d_Filter[filterR - k];
        }
    }
    d_Dst[row * imageW + col] = sum;
}

// Usage: ./conv <matrix_size> <block_x> <block_y> <paths> <mode>
//   matrix_size : size from matrices_ array (240, 496, 784, 1016, 1232, 1680, 2024)
//   block_x/y   : block dimensions from blocks_ array
//   paths       : number of repeated launches
//   mode        : 0 = alone (all SMs), 1 = concurrent (6 SMs green context)
int main(int argc, char** argv) {
    char* p;
    int matrix_size = (argc > 1) ? strtol(argv[1], &p, 10) : 240;
    int block_x = (argc > 2) ? strtol(argv[2], &p, 10) : 32;
    int block_y = (argc > 3) ? strtol(argv[3], &p, 10) : 32;
    int paths = (argc > 4) ? strtol(argv[4], &p, 10) : 15;
    int concurrent = (argc > 5) ? strtol(argv[5], &p, 10) : 0;

    // Find matrix_idx from matrices_ array
    int matrix_idx = -1;
    for (int i = 0; i < 7; i++) {
        if (matrices_[i][0] == matrix_size) {
            matrix_idx = i;
            break;
        }
    }
    if (matrix_idx == -1) {
        fprintf(stderr, "[ERROR] Invalid matrix size %d\n", matrix_size);
        exit(EXIT_FAILURE);
    }

    // Find block_idx from blocks_ array
    int block_idx = -1;
    for (int i = 0; i < 13; i++) {
        if (blocks_[i][0] == block_x && blocks_[i][1] == block_y) {
            block_idx = i;
            break;
        }
    }
    if (block_idx == -1) {
        fprintf(stderr, "[ERROR] Invalid block config (%d,%d)\n", block_x, block_y);
        exit(EXIT_FAILURE);
    }

    int XSIZE = matrices_[matrix_idx][0];
    int YSIZE = matrices_[matrix_idx][1];
    int BLOCKX = blocks_[block_idx][0];
    int BLOCKY = blocks_[block_idx][1];

    double *d_Dst = NULL;
    CHECK_RT(cudaMalloc(&d_Dst, XSIZE*YSIZE*sizeof(double)));
    double *d_Src = NULL;
    CHECK_RT(cudaMalloc(&d_Src, XSIZE*YSIZE*sizeof(double)));
    double *d_Filter = NULL;
    CHECK_RT(cudaMalloc(&d_Filter, XSIZE*YSIZE*sizeof(double)));

    int imageW = 1;
    int imageH = 1;
    int filterR = 2;
    int iXSIZE = XSIZE;
    int iYSIZE = YSIZE;
    
    while(iXSIZE % BLOCKX != 0) {
        iXSIZE++;
    }
    while(iYSIZE % BLOCKY != 0) {
        iYSIZE++;
    }

    dim3 gridBlock(iXSIZE/BLOCKX, iYSIZE/BLOCKY);
    dim3 threadBlock(BLOCKX, BLOCKY);

    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs,
             cudaDevAttrMultiProcessorCount, devIdx));

    cudaEvent_t start, stop;
    CHECK_RT(cudaEventCreate(&start));
    CHECK_RT(cudaEventCreate(&stop));

    printf("[CONV] Matrix: %dx%d | Block: (%d,%d) | Grid: (%d,%d) | FilterR: %d\n",
           XSIZE, YSIZE, BLOCKX, BLOCKY, iXSIZE/BLOCKX, iYSIZE/BLOCKY, filterR);

    if (!concurrent) {
        printf("[CONV] ALONE — all %d SMs\n", totalSMs);
        CHECK_RT(cudaEventRecord(start, 0));
        ConvolutionRowGPU<<<gridBlock,threadBlock>>>(d_Dst,d_Src,d_Filter,imageW,imageH,filterR);
        CHECK_RT(cudaDeviceSynchronize());
        for (int loop_counter = 0; loop_counter < 2; ++loop_counter) {
            ConvolutionRowGPU<<<gridBlock,threadBlock>>>(d_Dst,d_Src,d_Filter,imageW,imageH,filterR);
            CHECK_RT(cudaDeviceSynchronize());
        }
        for (int loop_counter = 0; loop_counter < 15; loop_counter++) {
            ConvolutionRowGPU<<<gridBlock,threadBlock>>>(d_Dst,d_Src,d_Filter,imageW,imageH,filterR);
            CHECK_RT(cudaDeviceSynchronize());
        }
        CHECK_RT(cudaEventRecord(stop, 0));
        CHECK_RT(cudaStreamSynchronize(0));

    } else {
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs,
                 CU_DEV_RESOURCE_TYPE_SM));

        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(
            &victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        CUdevResourceDesc descVictim;
        CHECK_CU(cuDevResourceGenerateDesc(&descVictim, &victimSlice, 1));

        CUgreenCtx victimGCtx;
        CHECK_CU(cuGreenCtxCreate(
            &victimGCtx, descVictim, device, CU_GREEN_CTX_DEFAULT_STREAM));

        CUstream victimStream;
        CHECK_CU(cuGreenCtxStreamCreate(
            &victimStream, victimGCtx, CU_STREAM_NON_BLOCKING, 0));

        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(
            victimGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[CONV] CONCURRENT — %u SMs assigned\n",
               verify.sm.smCount);

        CHECK_RT(cudaEventRecord(start, victimStream));
        ConvolutionRowGPU<<<gridBlock,threadBlock,0,victimStream>>>(d_Dst,d_Src,d_Filter,imageW,imageH,filterR);
        CHECK_RT(cudaStreamSynchronize(victimStream));
        for (int loop_counter = 0; loop_counter < 2; ++loop_counter) {
            ConvolutionRowGPU<<<gridBlock,threadBlock,0,victimStream>>>(d_Dst,d_Src,d_Filter,imageW,imageH,filterR);
            CHECK_RT(cudaStreamSynchronize(victimStream));
        }
        for (int loop_counter = 0; loop_counter < 15; loop_counter++) {
            ConvolutionRowGPU<<<gridBlock,threadBlock,0,victimStream>>>(d_Dst,d_Src,d_Filter,imageW,imageH,filterR);
            CHECK_RT(cudaStreamSynchronize(victimStream));
        }
        CHECK_RT(cudaEventRecord(stop, victimStream));
        CHECK_RT(cudaStreamSynchronize(victimStream));

        CHECK_CU(cuStreamDestroy(victimStream));
        CHECK_CU(cuGreenCtxDestroy(victimGCtx));
    }

    float ms;
    CHECK_RT(cudaEventElapsedTime(&ms, start, stop));
    printf("[CONV] Completed in %.2f ms\n", ms);

    CHECK_RT(cudaEventDestroy(start));
    CHECK_RT(cudaEventDestroy(stop));
    CHECK_RT(cudaFree(d_Dst));
    CHECK_RT(cudaFree(d_Src));
    CHECK_RT(cudaFree(d_Filter));
    
    return 0;
}