#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Jetson Orin Nano  —  L2 Cache: 2 097 152 bytes (2 MB)
 *
 * Memory footprint of two N×N long matrices:
 *   F(N) = 2 * N² * sizeof(long)  =  2 * N² * 8  bytes
 *
 *   TILE_L2  = 32  |  N_L2_FIT  = 352  →  F = 1.97 MB  ≈ L2
 *   TILE_SM  = 16  |  N_SMALL   = 128  →  F = 0.26 MB  << L2
 *
 * Shared memory per block:
 *   TILE_L2: 2 × 32×32 × 8 B = 16 384 B  (16 KB)
 *   TILE_SM: 2 × 16×16 × 8 B =  4 096 B  ( 4 KB)
 * ----------------------------------------------------------------------- */

#define TILE_L2   32   /* tile side for the "fits-L2" kernel              */
#define TILE_SM   16   /* tile side for the "smaller-than-L2" kernel      */

#define N_L2_FIT  352  /* default matrix side: 2×352²×8 = 1.97 MB ≈ L2   */
#define N_SMALL   128  /* default matrix side: 2×128²×8 = 0.26 MB << L2  */

/* -----------------------------------------------------------------------
 * Error-check macros
 * ----------------------------------------------------------------------- */
#define CHECK_CU(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        const char *n, *s; \
        cuGetErrorName(_e, &n); cuGetErrorString(_e, &s); \
        fprintf(stderr, "[CU ERR] %s: %s  (%s:%d)\n", n, s, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[RT ERR] %s  (%s:%d)\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* -----------------------------------------------------------------------
 * ORIGINAL kernel (1-D indexing, L2-sensitive — kept for reference)
 * ----------------------------------------------------------------------- */
__global__ void GPUMultiplyMatrix(long *matrix1, long *matrix2,
                                  int paths, int count)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;
    while (paths > 0) {
        long sum = 0;
        int col = element % count;
        int row = element / count;
        for (int i = 0; i < count; i++)
            sum += matrix1[count * i + col] * matrix2[row * count + i];
        __syncthreads();
        matrix2[element] = sum;
        paths--;
    }
}

/* -----------------------------------------------------------------------
 * KERNEL 1 — "fits L2"
 *   Tiled GEMM with TILE_L2×TILE_L2 (32×32) shared-memory tiles.
 *   When used with N=N_L2_FIT=352, the two input matrices together occupy
 *   ~1.97 MB, which just fills the Orin Nano's 2 MB L2.  After the first
 *   pass the entire working set lives in L2 and subsequent tile loads hit
 *   the cache rather than DRAM.
 *
 *   C = A × B  (separate output, no in-place aliasing).
 *   Launch: dim3 grid((N+TILE_L2-1)/TILE_L2, (N+TILE_L2-1)/TILE_L2)
 *           dim3 block(TILE_L2, TILE_L2)
 * ----------------------------------------------------------------------- */
__global__ void GPUMultiplyMatrixTiled_L2Fit(const long * __restrict__ A,
                                             const long * __restrict__ B,
                                             long       * __restrict__ C,
                                             int N)
{
    /* Two shared-memory tiles: 2 × 32×32 × 8 B = 16 KB per block */
    __shared__ long As[TILE_L2][TILE_L2];
    __shared__ long Bs[TILE_L2][TILE_L2];

    int row = blockIdx.y * TILE_L2 + threadIdx.y;   /* output row   */
    int col = blockIdx.x * TILE_L2 + threadIdx.x;   /* output col   */
    long acc = 0;

    /* Sweep tiles along the shared k-dimension */
    for (int t = 0; t < (N + TILE_L2 - 1) / TILE_L2; t++) {

        /* Load A tile — guard out-of-bound elements */
        int aCol = t * TILE_L2 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0L;

        /* Load B tile */
        int bRow = t * TILE_L2 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0L;

        __syncthreads();

        /* Accumulate dot product for this tile */
        #pragma unroll
        for (int k = 0; k < TILE_L2; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

/* -----------------------------------------------------------------------
 * KERNEL 2 — "smaller than L2"
 *   Tiled GEMM with TILE_SM×TILE_SM (16×16) shared-memory tiles.
 *   When used with N=N_SMALL=128, the two matrices occupy only ~0.26 MB —
 *   well below the 2 MB L2.  The entire dataset fits in L2 from the very
 *   first iteration; every tile load after that is a cache hit.
 *   Shared memory per block is only 4 KB, useful when occupancy matters.
 *
 *   C = A × B  (separate output).
 *   Launch: dim3 grid((N+TILE_SM-1)/TILE_SM, (N+TILE_SM-1)/TILE_SM)
 *           dim3 block(TILE_SM, TILE_SM)
 * ----------------------------------------------------------------------- */
__global__ void GPUMultiplyMatrixTiled_Small(const long * __restrict__ A,
                                             const long * __restrict__ B,
                                             long       * __restrict__ C,
                                             int N)
{
    /* Two shared-memory tiles: 2 × 16×16 × 8 B = 4 KB per block */
    __shared__ long As[TILE_SM][TILE_SM];
    __shared__ long Bs[TILE_SM][TILE_SM];

    int row = blockIdx.y * TILE_SM + threadIdx.y;
    int col = blockIdx.x * TILE_SM + threadIdx.x;
    long acc = 0;

    for (int t = 0; t < (N + TILE_SM - 1) / TILE_SM; t++) {

        int aCol = t * TILE_SM + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < N && aCol < N) ? A[row * N + aCol] : 0L;

        int bRow = t * TILE_SM + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && col < N) ? B[bRow * N + col] : 0L;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SM; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

/* -----------------------------------------------------------------------
 * Helper: run a kernel on a given stream and report timing
 * ----------------------------------------------------------------------- */
static void run_kernel(int kernel_id,
                       long *dA, long *dB, long *dC,
                       int N, int paths,
                       cudaStream_t stream)
{
    cudaEvent_t t0, t1;
    CHECK_RT(cudaEventCreate(&t0));
    CHECK_RT(cudaEventCreate(&t1));

    if (kernel_id == 0) {
        /* Original: 1-D launch */
        int tpb = 256;
        int blocks = (N * N + tpb - 1) / tpb;
        CHECK_RT(cudaEventRecord(t0, stream));
        for (int p = 0; p < paths; p++)
            GPUMultiplyMatrix<<<blocks, tpb, 0, stream>>>(dA, dB, paths, N);
        CHECK_RT(cudaEventRecord(t1, stream));

    } else if (kernel_id == 1) {
        /* Tiled L2-fit */
        dim3 block(TILE_L2, TILE_L2);
        dim3 grid((N + TILE_L2 - 1) / TILE_L2,
                  (N + TILE_L2 - 1) / TILE_L2);
        CHECK_RT(cudaEventRecord(t0, stream));
        for (int p = 0; p < paths; p++)
            GPUMultiplyMatrixTiled_L2Fit<<<grid, block, 0, stream>>>(dA, dB, dC, N);
        CHECK_RT(cudaEventRecord(t1, stream));

    } else {
        /* Tiled small */
        dim3 block(TILE_SM, TILE_SM);
        dim3 grid((N + TILE_SM - 1) / TILE_SM,
                  (N + TILE_SM - 1) / TILE_SM);
        CHECK_RT(cudaEventRecord(t0, stream));
        for (int p = 0; p < paths; p++)
            GPUMultiplyMatrixTiled_Small<<<grid, block, 0, stream>>>(dA, dB, dC, N);
        CHECK_RT(cudaEventRecord(t1, stream));
    }

    CHECK_RT(cudaStreamSynchronize(stream));
    float ms;
    CHECK_RT(cudaEventElapsedTime(&ms, t0, t1));
    printf("[VICTIM] kernel %d completed in %.2f ms\n", kernel_id, ms);
    CHECK_RT(cudaEventDestroy(t0));
    CHECK_RT(cudaEventDestroy(t1));
}

/* -----------------------------------------------------------------------
 * Usage:
 *   ./matrix_victim_sm <kernel> <matrix_size> <paths> <concurrent>
 *
 *   kernel:     0 = original (no shared mem)
 *               1 = tiled L2-fit  (TILE=32, default N=352, ~1.97 MB)
 *               2 = tiled small   (TILE=16, default N=128, ~0.26 MB)
 *   matrix_size: override N (0 = use kernel default)
 *   paths:      iterations of the kernel (default 15)
 *   concurrent: 0 = alone (all SMs)
 *               1 = green-context split (victim 6 SMs / enemy 2 SMs)
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    int kernel_id  = (argc > 1) ? atoi(argv[1]) : 1;
    int N_override = (argc > 2) ? atoi(argv[2]) : 0;
    int paths      = (argc > 3) ? atoi(argv[3]) : 15;
    int concurrent = (argc > 4) ? atoi(argv[4]) : 0;

    /* Pick default N based on kernel */
    int N;
    if (N_override > 0) {
        N = N_override;
    } else {
        N = (kernel_id == 2) ? N_SMALL : N_L2_FIT;
    }

    size_t matSize = (size_t)N * N * sizeof(long);
    double footprintMB = 2.0 * matSize / 1e6;

    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));

    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, devIdx));

    printf("[VICTIM] kernel=%d  N=%d  paths=%d  concurrent=%d\n",
           kernel_id, N, paths, concurrent);
    printf("[VICTIM] Matrix footprint (A+B): %.2f MB  |  L2: 2.00 MB  |  ratio: %.2fx\n",
           footprintMB, footprintMB / 2.0);

    if (kernel_id == 1)
        printf("[VICTIM] Tile %dx%d  |  smem/block = %zu B  (fits L2)\n",
               TILE_L2, TILE_L2, 2 * TILE_L2 * TILE_L2 * sizeof(long));
    else if (kernel_id == 2)
        printf("[VICTIM] Tile %dx%d  |  smem/block = %zu B  (smaller than L2)\n",
               TILE_SM, TILE_SM, 2 * TILE_SM * TILE_SM * sizeof(long));

    /* Allocate device memory */
    long *dA, *dB, *dC;
    CHECK_RT(cudaMalloc(&dA, matSize));
    CHECK_RT(cudaMalloc(&dB, matSize));
    CHECK_RT(cudaMalloc(&dC, matSize));   /* separate output for tiled kernels */

    /* Initialise host matrices and copy */
    long *hA = (long*)malloc(matSize);
    long *hB = (long*)malloc(matSize);
    for (int i = 0; i < N * N; i++) {
        hA[i] = (i % 17) + 1;
        hB[i] = (i % 13) + 1;
    }
    CHECK_RT(cudaMemcpy(dA, hA, matSize, cudaMemcpyHostToDevice));
    CHECK_RT(cudaMemcpy(dB, hB, matSize, cudaMemcpyHostToDevice));
    free(hA); free(hB);

    if (!concurrent) {
        /* ---- ALONE: use all SMs ---------------------------------------- */
        printf("[VICTIM] ALONE — %d SMs\n", totalSMs);
        run_kernel(kernel_id, dA, dB, dC, N, paths, 0);

    } else {
        /* ---- CONCURRENT: green-context SM split ------------------------- */
        /* Step 1: query the full SM resource pool */
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

        /* Step 2: split — victim gets 6 SMs, remainder (enemy) gets 2 SMs */
        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups,
                                             &fullSMs, &enemySlice, 0, 5));

        /* Step 3: pack into a descriptor */
        CUdevResourceDesc descVictim;
        CHECK_CU(cuDevResourceGenerateDesc(&descVictim, &victimSlice, 1));

        /* Step 4: green context restricted to victim SMs */
        CUgreenCtx victimGCtx;
        CHECK_CU(cuGreenCtxCreate(&victimGCtx, descVictim, device,
                                  CU_GREEN_CTX_DEFAULT_STREAM));

        /* Step 5: stream bound to green context */
        CUstream victimStream;
        CHECK_CU(cuGreenCtxStreamCreate(&victimStream, victimGCtx,
                                        CU_STREAM_NON_BLOCKING, 0));

        /* Verify actual SM count assigned by driver */
        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(victimGCtx, &verify,
                                          CU_DEV_RESOURCE_TYPE_SM));
        printf("[VICTIM] CONCURRENT — %u SMs (enemy has 2 SMs, shared L2)\n",
               verify.sm.smCount);

        run_kernel(kernel_id, dA, dB, dC, N, paths,
                   (cudaStream_t)victimStream);

        CHECK_CU(cuStreamDestroy(victimStream));
        CHECK_CU(cuGreenCtxDestroy(victimGCtx));
    }

    CHECK_RT(cudaFree(dA));
    CHECK_RT(cudaFree(dB));
    CHECK_RT(cudaFree(dC));
    return 0;
}
