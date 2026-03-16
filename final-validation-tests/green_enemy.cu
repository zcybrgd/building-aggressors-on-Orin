#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errName, *errStr; \
        cuGetErrorName(err, &errName); \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "[CUDA Driver Error] %s: %s at %s:%d\n", errName, errStr, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_RT(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA Runtime Error] %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// keep_running is set to 0 by the signal handler so the host loop in infinite
// mode can exit cleanly and then write the stop flag to GPU memory
volatile sig_atomic_t keep_running = 1;

// Signal handler for SIGTERM / SIGINT (sent by the run script via kill) // only flips the flag
void signal_handler(int signum) {
    keep_running = 0;
}


// 16 MB footprint, pointer chase + scatter writes
// Two run modes controlled by `cycles`:
//   cycles==0 : run until d_stop_flag[0] is set to 1 by the host (infinite mode,
//               used when the enemy must outlive an NCU profiling session of the victim).
//   cycles >0 : run for exactly `cycles` GPU clock cycles then exit.
// d_stop_flag lives in device memory so the host can poke it at any time without
// needing a kernel re-launch or a CUDA IPC mechanism.

__global__ void printSMIDs() {
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        printf("[ENEMY] block %d -> SM %u\n", blockIdx.x, smid);
    }
}

__global__ void enemyKernel(unsigned int* d_chase, unsigned int* d_writeback, int size, unsigned long long cycles, int* d_stop_flag) {
    unsigned long long start = clock64();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_words = size / sizeof(unsigned int);

    volatile unsigned int* v = (volatile unsigned int*)d_chase;
    volatile unsigned int* w = (volatile unsigned int*)d_writeback;
    int idx = tid % num_words;
    unsigned int sum = 0;

    if (cycles == 0) {
        while (d_stop_flag[0] == 0) {
            unsigned int next = v[idx];
            sum += next;
            int write_idx = (idx + tid * 131) % num_words;
            unsigned int write_val = sum ^ next ^ write_idx;
            w[write_idx] = write_val;
            w[(write_idx + 97) % num_words] = write_val + tid;
            idx = next;
        }
    } else {
        while ((clock64() - start) < cycles) {
            unsigned int next = v[idx];
            sum += next;
            int write_idx = (idx + tid * 131) % num_words;
            unsigned int write_val = sum ^ next ^ write_idx;
            w[write_idx] = write_val;
            w[(write_idx + 97) % num_words] = write_val + tid;
            idx = next;
        }
    }

    if (tid == 0) d_writeback[0] = sum;
}

int main(int argc, char** argv) {
    unsigned long long cycles = (argc > 1) ? atoll(argv[1]) : 0; // 0 = infinite
    // mode: 0=no green context, 1=use green context with remainder SMs
    int use_green = (argc > 2) ? atoi(argv[2]) : 0;

    if (cycles == 0) {
        printf("[ENEMY] Starting in INFINITE mode (run until killed)\n");
    } else {
        printf("[ENEMY] Starting for %llu cycles\n", cycles);
    }

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    //the Runtime API automatically initialises CUDA and creates/activates the primary context on the first Runtime call
    // cuDeviceGet() is still required to obtain the CUdevice handle used by the Driver API
    // green context calls below (cuDeviceGetDevResource, cuGreenCtxCreate, etc.)
    int devIdx = 0;
    CHECK_RT(cudaSetDevice(devIdx));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, devIdx));
    int totalSMs;
    CHECK_RT(cudaDeviceGetAttribute(&totalSMs, cudaDevAttrMultiProcessorCount, devIdx));
    int size = 16 * 1024 * 1024; 
    unsigned int* d_chase;
    unsigned int* d_writeback;
    int* d_stop_flag;
    CHECK_RT(cudaMalloc(&d_chase, size));
    CHECK_RT(cudaMalloc(&d_writeback, size));
    CHECK_RT(cudaMalloc(&d_stop_flag, sizeof(int)));
    int h_stop_flag = 0;
    CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
    unsigned int* h_chase = (unsigned int*)malloc(size);
    int num_words = size / sizeof(unsigned int);
    for (int i = 0; i < num_words; i++) {
        h_chase[i] = ((unsigned long long)i * 1234567) % num_words;
    }
    CHECK_RT(cudaMemcpy(d_chase, h_chase, size, cudaMemcpyHostToDevice));
    free(h_chase);
    CHECK_RT(cudaMemset(d_writeback, 0, size));

    if (!use_green) {
        printf("[ENEMY] No green context — using all %d SMs\n", totalSMs);
        printf("[ENEMY] Running... (PID: %d)\n", getpid());
        printSMIDs<<<16, 1>>>();
        CHECK_RT(cudaDeviceSynchronize());
        enemyKernel<<<16, 1024>>>(d_chase, d_writeback, size, cycles, d_stop_flag);

    } else {
        //from the documentation
        //step 1: query the full SM resource pool of the device
        CUdevResource fullSMs;
        CHECK_CU(cuDeviceGetDevResource(device, &fullSMs, CU_DEV_RESOURCE_TYPE_SM));
        //step 2: split SMs into a group + remainder.
        // cuDevSmResourceSplit is the recommended API but is not available in
        // CUDA 12.6 (missing from headers and libcuda.so on this Orin).
        // cuDevSmResourceSplitByCount is the only split API present here.
        // minCount=5 on 8 SMs with alignment=2 gives group=6 SMs (victim)
        // and remainder=2 SMs. Enemy takes the *remainder* (enemySlice).
        CUdevResource victimSlice, enemySlice;
        unsigned int nbGroups = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&victimSlice, &nbGroups, &fullSMs, &enemySlice, 0, 5));

        //step 3: pack enemySlice into an opaque descriptor for cuGreenCtxCreate.
        CUdevResourceDesc descEnemy;
        CHECK_CU(cuDevResourceGenerateDesc(&descEnemy, &enemySlice, 1));

        //step 4: create green context restricted to enemySlice (2 SMs)
        CUgreenCtx enemyGCtx;
        CHECK_CU(cuGreenCtxCreate(&enemyGCtx, descEnemy, device, CU_GREEN_CTX_DEFAULT_STREAM));

        //step 5: stream bound to the green context
        // CU_STREAM_NON_BLOCKING prevents implicit sync with stream 0, so the
        // long-running enemy kernel never accidentally blocks other Runtime calls.
        CUstream enemyStream;
        CHECK_CU(cuGreenCtxStreamCreate(&enemyStream, enemyGCtx, CU_STREAM_NON_BLOCKING, 0));

        //sanity check: confirm the driver assigned the expected 2 SMs
        CUdevResource verify;
        CHECK_CU(cuGreenCtxGetDevResource(enemyGCtx, &verify, CU_DEV_RESOURCE_TYPE_SM));
        printf("[ENEMY] Green context — %u SMs (victim has 6 SMs, shared L2)\n", verify.sm.smCount);
        printf("[ENEMY] Running... (PID: %d)\n", getpid());
        printSMIDs<<<16, 1, 0, enemyStream>>>();
        CHECK_RT(cudaStreamSynchronize(enemyStream));
        enemyKernel<<<16, 1024, 0, enemyStream>>>(d_chase, d_writeback, size, cycles, d_stop_flag);

        if (cycles == 0) {
            // Infinite mode: block the host until a SIGTERM/SIGINT arrives
            // sleep(1) keeps the CPU idle so we don't busy-spin while the GPU
            // kernel runs autonomously on the green context stream
            while (keep_running) {
                sleep(1);
            }
            // Write stop flag to device memory so the GPU kernel exits its loop
            // on the next iteration. Using a device-visible flag is safer than
            // killing a running kernel mid-flight
            printf("[ENEMY] Stop signal received, terminating kernel...\n");
            h_stop_flag = 1;
            CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
        }


        CHECK_RT(cudaStreamSynchronize(enemyStream));
        CHECK_CU(cuStreamDestroy(enemyStream));
        CHECK_CU(cuGreenCtxDestroy(enemyGCtx));

        printf("[ENEMY] Completed\n");
        CHECK_RT(cudaFree(d_chase));
        CHECK_RT(cudaFree(d_writeback));
        CHECK_RT(cudaFree(d_stop_flag));
        return 0;
    }

    // Non-green infinite mode: same stop-flag mechanism as the green path above.
    // The kernel was launched on the default stream, so cudaDeviceSynchronize()
    // is sufficient to wait for it after the flag has been written.
    if (cycles == 0) {
        while (keep_running) {
            sleep(1);
        }
        printf("[ENEMY] Stop signal received, terminating kernel...\n");
        h_stop_flag = 1;
        CHECK_RT(cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice));
    }

    cudaDeviceSynchronize();
    printf("[ENEMY] Completed\n");

    CHECK_RT(cudaFree(d_chase));
    CHECK_RT(cudaFree(d_writeback));
    CHECK_RT(cudaFree(d_stop_flag));
    return 0;
}