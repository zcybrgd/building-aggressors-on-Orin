#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* =========================================================================
 * GREEN CONTEXT ISOLATION VERIFICATION
 *
 * Implements the two verification methods:
 *
 * METHOD 1 — SM ID print
 *   Each block prints its physical SM ID via PTX %smid register.
 *   Expected result: victim blocks on SM {0..5}, enemy blocks on SM {6,7}
 *   → proves the driver enforced the partition at the hardware level.
 *
 * METHOD 2 — Timing test (professor's empirical method)
 *   Step A: kernel_compute alone on 6 SMs   → time_A
 *   Step B: kernel_thrash  alone on 2 SMs   → time_B
 *   Step C: both together  (A on 6, B on 2) → time_AB  (wall-clock)
 *
 *   If isolation is real:   time_AB ≈ max(time_A, time_B)  [parallel]
 *   If isolation is broken: time_AB ≈ time_A + time_B      [sequential]
 *
 * Both kernels are COMPUTE-INTENSIVE (SM-bound, not memory-bound) so that
 * execution time is dominated by SM occupancy, making the isolation visible.
 * ========================================================================= */

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

/* -------------------------------------------------------------------------
 * KERNEL A — compute-intensive (victim)
 * Does heavy FMA work per thread, very low memory footprint (fits registers).
 * SM-bound by design: time scales with SM count, not memory bandwidth.
 * print_smid=1 → each block prints its physical SM ID (METHOD 1).
 * ------------------------------------------------------------------------- */
__global__ void kernel_compute(float *out, long iterations, int print_smid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* METHOD 1: print SM ID for the first thread of each block */
    if (print_smid && threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("[VICTIM] block %3d -> SM %u\n", blockIdx.x, smid);
    }

    /* Compute-intensive loop: chained FMA, no memory dependency */
    float a = (float)tid * 0.0001f;
    float b = (float)(tid + 1) * 0.0001f;
    for (long i = 0; i < iterations; i++) {
        a = a * b + 0.999f;   /* FMA */
        b = b * a + 0.001f;   /* FMA — depends on previous a, forces serial */
    }

    /* Write result to prevent dead-code elimination by the compiler */
    if (tid == 0) out[0] = a + b;
}

/* -------------------------------------------------------------------------
 * KERNEL B — compute-intensive (enemy)
 * Same structure, slightly different constants so the compiler can't merge
 * the two kernels. print_smid=1 → prints SM IDs.
 * ------------------------------------------------------------------------- */
__global__ void kernel_thrash(float *out, long iterations, int print_smid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* METHOD 1: print SM ID */
    if (print_smid && threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("[ENEMY]  block %3d -> SM %u\n", blockIdx.x, smid);
    }

    float a = (float)tid * 0.0002f;
    float b = (float)(tid + 3) * 0.0002f;
    for (long i = 0; i < iterations; i++) {
        a = a * b + 0.998f;
        b = b * a + 0.002f;
    }

    if (tid == 0) out[0] = a + b;
}

/* -------------------------------------------------------------------------
 * Helper: create a green context + stream for a given SM slice
 * ------------------------------------------------------------------------- */
static void make_green_stream(CUdevice device,
                              CUdevResource slice,
                              CUgreenCtx   *gctx,
                              CUstream     *stream)
{
    CUdevResourceDesc desc;
    CHECK_CU(cuDevResourceGenerateDesc(&desc, &slice, 1));
    CHECK_CU(cuGreenCtxCreate(gctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuGreenCtxStreamCreate(stream, *gctx, CU_STREAM_NON_BLOCKING, 0));
}

/* -------------------------------------------------------------------------
 * Helper: split SMs into victim (6) + enemy (2) and verify
 * ------------------------------------------------------------------------- */
static void split_sms(CUdevice device,
                      CUdevResource *victim_slice,
                      CUdevResource *enemy_slice)
{
    CUdevResource fullSMs;
    CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));

    /* Hard guard: this split is tuned for Orin Nano (8 SMs) */
    if (fullSMs.sm.smCount != 8) {
        fprintf(stderr,
            "[FATAL] Expected 8 SMs, got %u. "
            "Split parameters are hardcoded for Orin Nano.\n",
            fullSMs.sm.smCount);
        exit(EXIT_FAILURE);
    }

    unsigned int nbGroups = 1;
    CHECK_CU(cuDevSmResourceSplitByCount(victim_slice, &nbGroups,
                                         &fullSMs, enemy_slice, 0, 5));

    /* Hard guard: verify the split result before creating anything */
    if (victim_slice->sm.smCount != 6 || enemy_slice->sm.smCount != 2) {
        fprintf(stderr,
            "[FATAL] Unexpected split: victim=%u SMs, enemy=%u SMs. "
            "Expected 6+2.\n",
            victim_slice->sm.smCount, enemy_slice->sm.smCount);
        exit(EXIT_FAILURE);
    }

    printf("[SPLIT]  victim=%u SMs  |  enemy=%u SMs  ✓\n",
           victim_slice->sm.smCount, enemy_slice->sm.smCount);
}

/* -------------------------------------------------------------------------
 * Helper: wall-clock time in milliseconds
 * ------------------------------------------------------------------------- */
static double wall_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* =========================================================================
 * MAIN
 * Usage: ./verify_isolation [iterations] [blocks_victim] [blocks_enemy]
 *   iterations:    FMA iterations per thread (default 5 000 000)
 *                  Increase until each kernel takes ~200-500 ms alone.
 *   blocks_victim: grid size for kernel_compute (default 6*16 = 96)
 *   blocks_enemy:  grid size for kernel_thrash  (default 2*16 = 32)
 * ========================================================================= */
int main(int argc, char **argv)
{
    long  iterations    = (argc > 1) ? atol(argv[1]) : 5000000L;
    int   blocks_victim = (argc > 2) ? atoi(argv[2]) : 96;   /* 6 SMs × 16 blocks */
    int   blocks_enemy  = (argc > 3) ? atoi(argv[3]) : 32;   /* 2 SMs × 16 blocks */
    int   threads       = 256;

    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));

    printf("=============================================================\n");
    printf(" GREEN CONTEXT ISOLATION VERIFICATION\n");
    printf(" iterations=%ld  blocks_victim=%d  blocks_enemy=%d\n",
           iterations, blocks_victim, blocks_enemy);
    printf("=============================================================\n\n");

    /* Allocate tiny output buffers (just to prevent dead-code elimination) */
    float *d_out_victim, *d_out_enemy;
    CHECK_RT(cudaMalloc(&d_out_victim, sizeof(float)));
    CHECK_RT(cudaMalloc(&d_out_enemy,  sizeof(float)));

    /* Split SMs */
    CUdevResource victim_slice, enemy_slice;
    split_sms(device, &victim_slice, &enemy_slice);

    /* =====================================================================
     * METHOD 1 — SM ID PRINT
     * Run both kernels concurrently and print which SM each block lands on.
     * ===================================================================== */
    printf("\n-------------------------------------------------------------\n");
    printf(" METHOD 1 — SM ID verification (PTX %%smid)\n");
    printf("-------------------------------------------------------------\n");
    printf(" Expected: VICTIM on SM {0..5}, ENEMY on SM {6,7}\n\n");

    CUgreenCtx  gctx_v1, gctx_e1;
    CUstream    stream_v1, stream_e1;
    make_green_stream(device, victim_slice, &gctx_v1, &stream_v1);
    make_green_stream(device, enemy_slice,  &gctx_e1, &stream_e1);

    /* Launch both concurrently with SM ID printing enabled (print_smid=1) */
    kernel_compute<<<blocks_victim, threads, 0, stream_v1>>>(d_out_victim, 1000L, 1);
    kernel_thrash <<<blocks_enemy,  threads, 0, stream_e1>>>(d_out_enemy,  1000L, 1);
    CHECK_RT(cudaStreamSynchronize(stream_v1));
    CHECK_RT(cudaStreamSynchronize(stream_e1));

    CHECK_CU(cuStreamDestroy(stream_v1)); CHECK_CU(cuGreenCtxDestroy(gctx_v1));
    CHECK_CU(cuStreamDestroy(stream_e1)); CHECK_CU(cuGreenCtxDestroy(gctx_e1));

    /* =====================================================================
     * METHOD 2 — TIMING TEST
     *
     * All three runs use green contexts so the SM count is the same in
     * every case — the only variable is whether A and B run in parallel.
     *
     * Step A: kernel_compute alone on 6 SMs  → time_A
     * Step B: kernel_thrash  alone on 2 SMs  → time_B
     * Step C: both together               → time_AB  (wall-clock)
     *
     * Prediction if isolation works:   time_AB ≈ max(time_A, time_B)
     * Prediction if isolation broken:  time_AB ≈ time_A + time_B
     * ===================================================================== */
    printf("\n-------------------------------------------------------------\n");
    printf(" METHOD 2 — Timing test (professor's empirical method)\n");
    printf("-------------------------------------------------------------\n");

    cudaEvent_t ev_start, ev_stop;
    CHECK_RT(cudaEventCreate(&ev_start));
    CHECK_RT(cudaEventCreate(&ev_stop));
    float ms;
    double wall_start, wall_end;

    /* --- Step A: victim alone on 6 SMs ---------------------------------- */
    {
        CUgreenCtx gctx; CUstream stream;
        make_green_stream(device, victim_slice, &gctx, &stream);

        CHECK_RT(cudaEventRecord(ev_start, stream));
        kernel_compute<<<blocks_victim, threads, 0, stream>>>(d_out_victim, iterations, 0);
        CHECK_RT(cudaEventRecord(ev_stop, stream));
        CHECK_RT(cudaStreamSynchronize(stream));
        CHECK_RT(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        printf("\n [A] kernel_compute ALONE on 6 SMs : %.2f ms\n", ms);
        float time_A = ms;
        (void)time_A;

        CHECK_CU(cuStreamDestroy(stream)); CHECK_CU(cuGreenCtxDestroy(gctx));
    }

    /* --- Step B: enemy alone on 2 SMs ----------------------------------- */
    {
        CUgreenCtx gctx; CUstream stream;
        make_green_stream(device, enemy_slice, &gctx, &stream);

        CHECK_RT(cudaEventRecord(ev_start, stream));
        kernel_thrash<<<blocks_enemy, threads, 0, stream>>>(d_out_enemy, iterations, 0);
        CHECK_RT(cudaEventRecord(ev_stop, stream));
        CHECK_RT(cudaStreamSynchronize(stream));
        CHECK_RT(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        printf(" [B] kernel_thrash  ALONE on 2 SMs : %.2f ms\n", ms);
        float time_B = ms;
        (void)time_B;

        CHECK_CU(cuStreamDestroy(stream)); CHECK_CU(cuGreenCtxDestroy(gctx));
    }

    /* --- Step C: both together ------------------------------------------ */
    {
        CUgreenCtx gctx_v, gctx_e;
        CUstream   stream_v, stream_e;
        make_green_stream(device, victim_slice, &gctx_v, &stream_v);
        make_green_stream(device, enemy_slice,  &gctx_e, &stream_e);

        /* Use host wall-clock to capture true parallel duration */
        wall_start = wall_ms();

        kernel_compute<<<blocks_victim, threads, 0, stream_v>>>(d_out_victim, iterations, 0);
        kernel_thrash <<<blocks_enemy,  threads, 0, stream_e>>>(d_out_enemy,  iterations, 0);

        /* Wait for BOTH streams to complete */
        CHECK_RT(cudaStreamSynchronize(stream_v));
        CHECK_RT(cudaStreamSynchronize(stream_e));

        wall_end = wall_ms();
        double time_AB = wall_end - wall_start;

        printf(" [C] A + B TOGETHER (wall-clock)   : %.2f ms\n\n", time_AB);

        CHECK_CU(cuStreamDestroy(stream_v)); CHECK_CU(cuGreenCtxDestroy(gctx_v));
        CHECK_CU(cuStreamDestroy(stream_e)); CHECK_CU(cuGreenCtxDestroy(gctx_e));
    }

    /* --- Interpretation -------------------------------------------------- */
    printf("-------------------------------------------------------------\n");
    printf(" INTERPRETATION\n");
    printf("-------------------------------------------------------------\n");
    printf(" If isolation works   →  time_AB  ≈  max(time_A, time_B)\n");
    printf("                         A and B run in TRUE PARALLEL\n");
    printf("                         wall-clock = duration of the longest\n\n");
    printf(" If isolation broken  →  time_AB  ≈  time_A + time_B\n");
    printf("                         SMs shared → quasi-sequential execution\n");
    printf("=============================================================\n\n");

    CHECK_RT(cudaEventDestroy(ev_start));
    CHECK_RT(cudaEventDestroy(ev_stop));
    CHECK_RT(cudaFree(d_out_victim));
    CHECK_RT(cudaFree(d_out_enemy));
    return 0;
}
