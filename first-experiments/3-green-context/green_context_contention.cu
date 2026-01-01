/*
 * L2 Cache Contention with CUDA Green Contexts
 * 
 * Victim: L2-intensive streaming reads (moderate speed, cache-friendly)
 * Enemy:  L2-thrashing pointer chasing (aggressive, evicts victim's lines)
 * 
 * Uses Green Contexts to:
 * - Victim gets 30% of SMs (less resources)
 * - Enemy gets 70% of SMs (more aggressive)
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#define CHECK_CUDA(err) { \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CU(err) { \
    if ((err) != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA Driver ERROR at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
}

// ============================================================================
// VICTIM KERNEL: L2-intensive streaming access (cache-friendly pattern)
// ============================================================================
__global__ void victimKernel(float* d_input, float* d_output, 
                             int N, unsigned long long iterations) {
    /*
     * Victim kernel repeatedly reads from arrays in a PREDICTABLE pattern
     * - Reads sequentially with stride (cache-friendly)
     * - Re-uses data from L2
     * - Should have high L2 hit rate when alone
     * - Gets destroyed when enemy thrashes L2
     */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    for (unsigned long long iter = 0; iter < iterations; ++iter) {
        // Read multiple elements to stress L2
        // Working set should fit in L2 for good reuse
        for (int i = tid; i < N; i += stride) {
            // Streaming reads - victim wants these in L2!
            sum += d_input[i] * 1.1f;
            sum += d_input[(i + 64) % N] * 0.9f;  // Some reuse
        }
        
        // Occasional write (forces cache coherence)
        if ((iter & 0xFF) == 0 && tid < N) {
            d_output[tid] = sum;
        }
    }
    
    // Final write to prevent optimization
    if (tid < N) {
        d_output[tid] = sum;
    }
}

// ============================================================================
// ENEMY KERNEL: Aggressive L2 thrashing with pointer chasing
// ============================================================================
__global__ void enemyKernel(unsigned int* d_chase_array, 
                            unsigned int* d_result,
                            int array_size_bytes, 
                            unsigned long long run_cycles) {
    /*
     * Enemy kernel does RANDOM pointer chasing to evict victim's cache lines
     * - Accesses scattered memory locations (cache-hostile)
     * - Large working set (doesn't fit in L2)
     * - High L2 miss rate by design
     * - Runs FASTER than victim (more SMs allocated)
     * - Goal: evict victim's data from L2
     */
    unsigned long long start = clock64();
    unsigned long long now = start;
    
    const int line_size = 128;  // L2 cache line size on Ampere
    int num_lines = array_size_bytes / line_size;
    
    if (num_lines <= 1) return;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Start at a random line based on thread ID
    unsigned int idx_line = (unsigned int)(tid % num_lines);
    volatile unsigned int* v = (volatile unsigned int*)d_chase_array;
    
    unsigned long long iterations = 0;
    unsigned int local_sum = 0;
    
    // Pointer chasing: jump randomly through memory
    int word_index = idx_line * (line_size / sizeof(unsigned int));
    
    do {
        // Read next pointer (cache miss likely!)
        unsigned int next = v[word_index];
        local_sum += next;
        
        // Jump to next random location
        word_index = next;
        
        iterations++;
        now = clock64();
    } while ((now - start) < run_cycles);
    
    // Write result
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_result[0] = (unsigned int)iterations;
        d_result[1] = local_sum;
    }
}

// ============================================================================
// Helper: Build pointer chase array for enemy
// ============================================================================
void buildPointerChaseArray(unsigned int* h_array, int array_size_bytes) {
    const int line_size = 128;
    int num_lines = array_size_bytes / line_size;
    
    if (num_lines <= 0) return;
    
    // Create permutation of lines
    unsigned int* lines = (unsigned int*)malloc(num_lines * sizeof(unsigned int));
    for (int i = 0; i < num_lines; ++i) lines[i] = i;
    
    // Shuffle using LCG
    unsigned int seed = 0xDEADBEEF;
    for (int i = num_lines - 1; i > 0; --i) {
        seed = 1103515245u * seed + 12345u;
        unsigned int j = seed % (i + 1);
        unsigned int tmp = lines[i];
        lines[i] = lines[j];
        lines[j] = tmp;
    }
    
    // Build chase pointers
    int words_per_line = line_size / sizeof(unsigned int);
    int num_words = array_size_bytes / sizeof(unsigned int);
    
    for (int w = 0; w < num_words; ++w) h_array[w] = 0;
    
    for (int p = 0; p < num_lines; ++p) {
        int cur_line = lines[p];
        int next_line = lines[(p + 1) % num_lines];
        int cur_word_idx = cur_line * words_per_line;
        int next_word_idx = next_line * words_per_line;
        
        for (int w = 0; w < words_per_line; ++w) {
            h_array[cur_word_idx + w] = (unsigned int)next_word_idx;
        }
    }
    
    free(lines);
}

// ============================================================================
// Green Context Setup
// ============================================================================

CUcontext createGreenContext(int smCount, int numQueues) {
    CUcontext greenCtx;
    
    // Step 1: Create SM resource descriptor
    CUexecAffinityParam smParam;
    smParam.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    smParam.param.smCount.val = smCount;
    
    // Step 2: Create work queue resource descriptor
    CUexecAffinityParam queueParam;
    queueParam.type = CU_EXEC_AFFINITY_TYPE_QUEUE_COUNT;
    queueParam.param.queueCount.val = numQueues;
    
    // Step 3: Combine into resource descriptor
    CUexecAffinityParam params[2] = {smParam, queueParam};
    
    // Step 4: Create green context
    CHECK_CU(cuCtxCreate_v3(&greenCtx, params, 2, 0, 0));
    
    return greenCtx;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    // Parameters
    int victim_iters = 1000;
    int enemy_run_seconds = 10;
    int victim_working_set_mb = 4;    // Small enough to fit in L2 (2 MB)
    int enemy_array_mb = 16;          // Large enough to thrash L2
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-vi") == 0 && i + 1 < argc) 
            victim_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-et") == 0 && i + 1 < argc) 
            enemy_run_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vw") == 0 && i + 1 < argc) 
            victim_working_set_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ea") == 0 && i + 1 < argc) 
            enemy_array_mb = atoi(argv[++i]);
    }
    
    // Initialize CUDA
    CHECK_CU(cuInit(0));
    
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    printf("\n=== GPU INFO ===\n");
    printf("Device: %s\n", prop.name);
    printf("Total SMs: %d\n", prop.multiProcessorCount);
    printf("L2 Cache: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Clock Rate: %d MHz\n", prop.clockRate / 1000);
    printf("================\n\n");
    
    // Calculate SM allocation (victim gets fewer SMs)
    int victim_sm_count = (int)(prop.multiProcessorCount * 0.3);  // 30% SMs
    int enemy_sm_count = prop.multiProcessorCount - victim_sm_count;  // 70% SMs
    
    if (victim_sm_count < 1) victim_sm_count = 1;
    if (enemy_sm_count < 1) enemy_sm_count = 1;
    
    printf("=== GREEN CONTEXT ALLOCATION ===\n");
    printf("Victim SMs: %d (%.0f%%)\n", victim_sm_count, 
           100.0 * victim_sm_count / prop.multiProcessorCount);
    printf("Enemy SMs:  %d (%.0f%%)\n", enemy_sm_count,
           100.0 * enemy_sm_count / prop.multiProcessorCount);
    printf("================================\n\n");
    
    // Create primary context first (required)
    CHECK_CUDA(cudaSetDevice(0));
    
    // Create green contexts
    CUcontext victimCtx = createGreenContext(victim_sm_count, 4);
    CUcontext enemyCtx = createGreenContext(enemy_sm_count, 8);
    
    printf("✓ Green contexts created\n\n");
    
    // Allocate memory
    int victim_size = victim_working_set_mb * 1024 * 1024;
    int enemy_size = enemy_array_mb * 1024 * 1024;
    
    float *d_victim_input, *d_victim_output;
    unsigned int *d_enemy_array, *d_enemy_result;
    
    CHECK_CUDA(cudaMalloc(&d_victim_input, victim_size));
    CHECK_CUDA(cudaMalloc(&d_victim_output, victim_size));
    CHECK_CUDA(cudaMalloc(&d_enemy_array, enemy_size));
    CHECK_CUDA(cudaMalloc(&d_enemy_result, 2 * sizeof(unsigned int)));
    
    // Initialize victim data
    int victim_N = victim_size / sizeof(float);
    float* h_victim = (float*)malloc(victim_size);
    for (int i = 0; i < victim_N; ++i) h_victim[i] = (float)i * 0.1f;
    CHECK_CUDA(cudaMemcpy(d_victim_input, h_victim, victim_size, cudaMemcpyHostToDevice));
    free(h_victim);
    
    // Initialize enemy pointer chase array
    unsigned int* h_enemy = (unsigned int*)malloc(enemy_size);
    buildPointerChaseArray(h_enemy, enemy_size);
    CHECK_CUDA(cudaMemcpy(d_enemy_array, h_enemy, enemy_size, cudaMemcpyHostToDevice));
    free(h_enemy);
    
    // Create streams for each context
    cudaStream_t victimStream, enemyStream;
    
    CHECK_CU(cuCtxSetCurrent(victimCtx));
    CHECK_CUDA(cudaStreamCreate(&victimStream));
    
    CHECK_CU(cuCtxSetCurrent(enemyCtx));
    CHECK_CUDA(cudaStreamCreate(&enemyStream));
    
    printf("✓ Streams created\n\n");
    
    // Calculate enemy run cycles
    unsigned long long enemy_cycles = 
        (unsigned long long)enemy_run_seconds * prop.clockRate * 1000ULL;
    
    // Create events
    cudaEvent_t start_victim, stop_victim, start_enemy, stop_enemy;
    CHECK_CUDA(cudaEventCreate(&start_victim));
    CHECK_CUDA(cudaEventCreate(&stop_victim));
    CHECK_CUDA(cudaEventCreate(&start_enemy));
    CHECK_CUDA(cudaEventCreate(&stop_enemy));
    
    // Kernel launch configs
    int victim_blocks = victim_sm_count * 4;  // 4 blocks per SM
    int victim_threads = 256;
    
    int enemy_blocks = enemy_sm_count * 8;    // 8 blocks per SM (more aggressive)
    int enemy_threads = 256;
    
    printf("=== LAUNCH CONFIG ===\n");
    printf("Victim: %d blocks × %d threads = %d total threads\n", 
           victim_blocks, victim_threads, victim_blocks * victim_threads);
    printf("Enemy:  %d blocks × %d threads = %d total threads\n",
           enemy_blocks, enemy_threads, enemy_blocks * enemy_threads);
    printf("=====================\n\n");
    
    // ========================================================================
    // SCENARIO 1: Victim alone
    // ========================================================================
    printf("=== SCENARIO 1: Victim Alone ===\n");
    
    CHECK_CU(cuCtxSetCurrent(victimCtx));
    CHECK_CUDA(cudaEventRecord(start_victim, victimStream));
    
    victimKernel<<<victim_blocks, victim_threads, 0, victimStream>>>(
        d_victim_input, d_victim_output, victim_N, victim_iters);
    
    CHECK_CUDA(cudaEventRecord(stop_victim, victimStream));
    CHECK_CUDA(cudaEventSynchronize(stop_victim));
    
    float t_victim_alone = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_alone, start_victim, stop_victim));
    
    printf("✓ Victim alone: %.2f ms\n\n", t_victim_alone);
    
    // ========================================================================
    // SCENARIO 2: Victim + Enemy concurrent (with green contexts)
    // ========================================================================
    printf("=== SCENARIO 2: Victim + Enemy (Green Contexts) ===\n");
    
    // Launch enemy first (more aggressive)
    CHECK_CU(cuCtxSetCurrent(enemyCtx));
    CHECK_CUDA(cudaEventRecord(start_enemy, enemyStream));
    
    enemyKernel<<<enemy_blocks, enemy_threads, 0, enemyStream>>>(
        d_enemy_array, d_enemy_result, enemy_size, enemy_cycles);
    
    CHECK_CUDA(cudaEventRecord(stop_enemy, enemyStream));
    
    // Small delay to let enemy warm up
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Launch victim (victim context with fewer SMs)
    CHECK_CU(cuCtxSetCurrent(victimCtx));
    CHECK_CUDA(cudaEventRecord(start_victim, victimStream));
    
    victimKernel<<<victim_blocks, victim_threads, 0, victimStream>>>(
        d_victim_input, d_victim_output, victim_N, victim_iters);
    
    CHECK_CUDA(cudaEventRecord(stop_victim, victimStream));
    
    // Wait for both
    CHECK_CUDA(cudaEventSynchronize(stop_victim));
    CHECK_CUDA(cudaEventSynchronize(stop_enemy));
    
    float t_victim_concurrent = 0.0f, t_enemy = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_concurrent, start_victim, stop_victim));
    CHECK_CUDA(cudaEventElapsedTime(&t_enemy, start_enemy, stop_enemy));
    
    printf("Enemy:            %.2f ms\n", t_enemy);
    printf("Victim concurrent: %.2f ms\n\n", t_victim_concurrent);
    
    // ========================================================================
    // Results
    // ========================================================================
    printf("=== RESULTS ===\n");
    printf("Victim alone:      %.2f ms\n", t_victim_alone);
    printf("Victim concurrent: %.2f ms\n", t_victim_concurrent);
    printf("Slowdown:          %.2fx\n", t_victim_concurrent / t_victim_alone);
    printf("Performance loss:  %.1f%%\n", 
           100.0 * (t_victim_concurrent - t_victim_alone) / t_victim_alone);
    printf("===============\n\n");
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_victim_input));
    CHECK_CUDA(cudaFree(d_victim_output));
    CHECK_CUDA(cudaFree(d_enemy_array));
    CHECK_CUDA(cudaFree(d_enemy_result));
    
    CHECK_CUDA(cudaStreamDestroy(victimStream));
    CHECK_CUDA(cudaStreamDestroy(enemyStream));
    
    CHECK_CU(cuCtxDestroy(victimCtx));
    CHECK_CU(cuCtxDestroy(enemyCtx));
    
    CHECK_CUDA(cudaEventDestroy(start_victim));
    CHECK_CUDA(cudaEventDestroy(stop_victim));
    CHECK_CUDA(cudaEventDestroy(start_enemy));
    CHECK_CUDA(cudaEventDestroy(stop_enemy));
    
    return 0;
}