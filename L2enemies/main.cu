#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define CHECK_CUDA(err) { \
    if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

//the victim kernel performs a fixed number of iterations accessing the L2 (allocated a working set with a size greater than the L1 cache)
__global__ void victimKernel(unsigned int* d_victim_array, float* d_result,
                             int array_size_bytes, unsigned long long n_iters,
                             int k_param, int stride) {
    const int line_size = 128;
    int num_lines = array_size_bytes / line_size;
    if (num_lines <= 0) {
        if (threadIdx.x==0 && blockIdx.x==0) { d_result[0]=0.0f; d_result[1]=0.0f; }
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int line_idx = tid % num_lines;
    //to not use registers
    volatile unsigned int* v = (volatile unsigned int*) d_victim_array;
    unsigned int local_sum = 0;
    // fixed iteration count - deterministic execution time
    for (unsigned long long iter = 0; iter < n_iters; ++iter) {
        for (int k = 0; k < k_param; ++k) {
            int idx_line = (line_idx + k * stride) % num_lines;
            int idx = idx_line * (line_size / sizeof(unsigned int));
            local_sum += v[idx];
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_result[0] = (float)n_iters;
        d_result[1] = (float)local_sum;
    }
}

//runs as long as needed and performs pointer chasing on an array built to have line-aligned pointers
__global__ void enemyPointerChase(unsigned int* d_enemy_array, int* d_result,int array_size_bytes, volatile int* d_stop_signal) {
    const int line_size = 128;
    int num_lines = array_size_bytes / line_size;
    if (num_lines <= 1) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx_line = (unsigned int)(tid % num_lines);
    volatile unsigned int* v = (volatile unsigned int*) d_enemy_array;
    unsigned long long iterations = 0;
    unsigned int local_sum = 0;
    int word_index = idx_line * (line_size / sizeof(unsigned int));
    while (!(*d_stop_signal)) {
        unsigned int next = v[word_index];
        local_sum += next;
        word_index = next;
        iterations++;
        if ((iterations & 0xFF) == 0) {
            __threadfence(); 
        }
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_result[0] = (int)iterations;
        d_result[1] = (int)local_sum;
    }
}

void buildLineAlignedPointerChase(unsigned int* h_array, int array_size_bytes) {
    const int line_size = 128;
    int num_lines = array_size_bytes / line_size;
    if (num_lines <= 0) return;
    unsigned int* lines = (unsigned int*)malloc(num_lines * sizeof(unsigned int));

    for (int i = 0; i < num_lines; ++i) lines[i] = i;
    unsigned int seed = 123456789u;
    for (int i = num_lines - 1; i > 0; --i) {
        seed = 1103515245u * seed + 12345u;
        unsigned int j = (unsigned int)(seed % (i + 1));
        unsigned int tmp = lines[i];
        lines[i] = lines[j];
        lines[j] = tmp;
    }
    int words_per_line = line_size / sizeof(unsigned int);
    int num_words = array_size_bytes / sizeof(unsigned int);
    for (int w = 0; w < num_words; ++w) h_array[w] = 0u;
    for (int p = 0; p < num_lines; ++p) {
        int cur_line = lines[p];
        int next_line = lines[(p + 1) % num_lines];
        int cur_word_index = cur_line * words_per_line;
        int next_word_index = next_line * words_per_line;
        for (int w = 0; w < words_per_line; ++w)
            h_array[cur_word_index + w] = (unsigned int)next_word_index;
    }
    free(lines);
}

unsigned long long compute_run_cycles(int run_seconds, const cudaDeviceProp &prop) {
    unsigned long long cycles_per_sec = (unsigned long long)prop.clockRate * 1000ULL;
    return (unsigned long long)run_seconds * cycles_per_sec;
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -v  <KB>     Victim working set size in KB (default: 512)\n");
    printf("  -e  <MB>     Enemy array size in MB (default: 8)\n");
    printf("  -vi <iters>  Victim iterations (default: 1000000)\n");
    printf("  -et <sec>    Enemy runtime in seconds (default: 15)\n");
    printf("  -m  <num>    Number of SMs for enemy (default: 8)\n");
    printf("  -vg <blocks> Victim grid size (blocks) (default: 1)\n");
    printf("  -vb <thr>    Victim block size (threads) (default: 32)\n");
    printf("  -eg <blocks> Enemy grid size (blocks) (default: auto)\n");
    printf("  -eb <thr>    Enemy block size (threads) (default: 128)\n");
    printf("  -vk <num>    Victim inner loop iterations (k parameter) (default: 16)\n");
    printf("  -vs <num>    Victim stride for line access (default: 1)\n");
    printf("  -d  <ms>     Delay before victim launch in ms (default: 10)\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char* argv[]) {
    int victim_working_set_kb = 512;
    int enemy_array_mb = 8;
    int num_enemy_sms = 8;
    int enemy_run_seconds = 15;
    unsigned long long victim_iters = 1000000ULL;
    int victim_grid_blocks = 1;
    int victim_block_threads = 32;
    int enemy_grid_blocks = -1;  // -1 means auto-calculate
    int enemy_block_threads = 128;
    int victim_k_param = 16;
    int victim_stride = 1;
    int launch_delay_ms = 10;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) victim_working_set_kb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) enemy_array_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) num_enemy_sms = atoi(argv[++i]);
        else if (strcmp(argv[i], "-et") == 0 && i + 1 < argc) enemy_run_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vi") == 0 && i + 1 < argc) victim_iters = (unsigned long long)atoll(argv[++i]);
        else if (strcmp(argv[i], "-vg") == 0 && i + 1 < argc) victim_grid_blocks = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vb") == 0 && i + 1 < argc) victim_block_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-eg") == 0 && i + 1 < argc) enemy_grid_blocks = atoi(argv[++i]);
        else if (strcmp(argv[i], "-eb") == 0 && i + 1 < argc) enemy_block_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vk") == 0 && i + 1 < argc) victim_k_param = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vs") == 0 && i + 1 < argc) victim_stride = atoi(argv[++i]);
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) launch_delay_ms = atoi(argv[++i]);
        else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("SM Count: %d, Clock Rate: %d MHz\n", prop.multiProcessorCount, prop.clockRate / 1000);
    printf("\n=== Configuration ===\n");
    printf("Victim:\n");
    printf("  Working set: %d KB (%d cache lines)\n", 
           victim_working_set_kb, (victim_working_set_kb * 1024) / 128);
    printf("  Iterations: %llu\n", victim_iters);
    printf("  Grid: %d blocks, Block: %d threads (total: %d threads)\n", 
           victim_grid_blocks, victim_block_threads, victim_grid_blocks * victim_block_threads);
    printf("  Inner loop (k): %d, Stride: %d\n", victim_k_param, victim_stride);
    printf("Enemy:\n");
    printf("  Array: %d MB (%d cache lines)\n", 
           enemy_array_mb, (enemy_array_mb * 1024 * 1024) / 128);
    printf("  Runtime: %d seconds\n", enemy_run_seconds);
    if (enemy_grid_blocks == -1) {
        printf("  Grid: auto (based on %d SMs)\n", num_enemy_sms);
    } else {
        printf("  Grid: %d blocks\n", enemy_grid_blocks);
    }
    printf("  Block: %d threads\n", enemy_block_threads);
    printf("Other:\n");
    printf("  Launch delay: %d ms\n", launch_delay_ms);

    int victim_size = victim_working_set_kb * 1024;
    int enemy_size = enemy_array_mb * 1024 * 1024;

    unsigned int* d_victim_array = nullptr;
    unsigned int* d_enemy_array = nullptr;
    float* d_victim_result = nullptr;
    int* d_enemy_result = nullptr;
    int* d_stop_signal = nullptr; // added this for stop signal

    CHECK_CUDA(cudaMalloc(&d_victim_array, victim_size));
    CHECK_CUDA(cudaMalloc(&d_enemy_array, enemy_size));
    CHECK_CUDA(cudaMalloc(&d_victim_result, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enemy_result, 2 * sizeof(int)));
    //added this for stop signal
    CHECK_CUDA(cudaMalloc(&d_stop_signal, sizeof(int))); 
    CHECK_CUDA(cudaMemset(d_stop_signal, 0, sizeof(int)));

    // init enemy array with pointer chase pattern
    unsigned int* h_enemy = (unsigned int*)malloc(enemy_size);
    buildLineAlignedPointerChase(h_enemy, enemy_size);
    CHECK_CUDA(cudaMemcpy(d_enemy_array, h_enemy, enemy_size, cudaMemcpyHostToDevice));
    free(h_enemy);

    // init victim array
    unsigned int* h_victim = (unsigned int*)malloc(victim_size);
    int victim_words = victim_size / sizeof(unsigned int);
    for (int i = 0; i < victim_words; ++i) h_victim[i] = i;
    CHECK_CUDA(cudaMemcpy(d_victim_array, h_victim, victim_size, cudaMemcpyHostToDevice));
    free(h_victim);

    cudaStream_t enemy_stream, victim_stream;
    CHECK_CUDA(cudaStreamCreate(&enemy_stream));
    CHECK_CUDA(cudaStreamCreate(&victim_stream));

    // configure enemy kernel
    int maxActiveBlocksPerSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, enemyPointerChase, enemy_block_threads, 0));
    if (maxActiveBlocksPerSM < 1) maxActiveBlocksPerSM = 1;
    
    if (enemy_grid_blocks == -1) {
        enemy_grid_blocks = num_enemy_sms * maxActiveBlocksPerSM;
    }
    if (enemy_grid_blocks < 1) enemy_grid_blocks = 1;

    dim3 enemyGrid(enemy_grid_blocks);
    dim3 enemyBlock(enemy_block_threads);
    unsigned long long enemy_run_cycles = compute_run_cycles(enemy_run_seconds, prop);

    dim3 victimGrid(victim_grid_blocks);
    dim3 victimBlock(victim_block_threads);
    
    printf("\n=== Final Launch Configuration ===\n");
    printf("Victim: <<<%d, %d>>> (%d total threads)\n", 
           victimGrid.x, victimBlock.x, victimGrid.x * victimBlock.x);
    printf("Enemy:  <<<%d, %d>>> (%d total threads, max %d blocks/SM)\n", 
           enemyGrid.x, enemyBlock.x, enemyGrid.x * enemyBlock.x, maxActiveBlocksPerSM);

    //create CUDA events for precise timing
    cudaEvent_t start_victim, stop_victim, start_enemy, stop_enemy;
    CHECK_CUDA(cudaEventCreate(&start_victim));
    CHECK_CUDA(cudaEventCreate(&stop_victim));
    CHECK_CUDA(cudaEventCreate(&start_enemy));
    CHECK_CUDA(cudaEventCreate(&stop_enemy));

    printf("\n=== SCENARIO 1: Victim alone (baseline) ===\n");
    CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
        d_victim_array, d_victim_result, victim_size, victim_iters, victim_k_param, victim_stride);
    CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));
    CHECK_CUDA(cudaEventSynchronize(stop_victim));

    float t_victim_alone_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_alone_ms, start_victim, stop_victim));
    printf("Victim alone completed in %.2f ms (baseline)\n", t_victim_alone_ms);

    // ===== SCENARIO 2: Victim + Enemy concurrent =====
    printf("\n=== SCENARIO 2: Victim + Enemy concurrent ===\n");
    CHECK_CUDA(cudaMemset(d_stop_signal, 0, sizeof(int)));
    // Launch enemy first (it will run for enemy_run_seconds)
    CHECK_CUDA(cudaEventRecord(start_enemy, enemy_stream));
    enemyPointerChase<<<enemyGrid, enemyBlock, 0, enemy_stream>>>(
        d_enemy_array, d_enemy_result, enemy_size,  d_stop_signal);
    CHECK_CUDA(cudaEventRecord(stop_enemy, enemy_stream));

    // delay to let enemy establish cache pressure
    if (launch_delay_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(launch_delay_ms));
    }

    // launch victim (fixed iterations - independent of enemy)
    CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
        d_victim_array, d_victim_result, victim_size, victim_iters, victim_k_param, victim_stride);
    CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));

    //wait for both to complete
    CHECK_CUDA(cudaEventSynchronize(stop_victim));
    int stop = 1;
    CHECK_CUDA(cudaMemcpyAsync(d_stop_signal, &stop, sizeof(int), cudaMemcpyHostToDevice, enemy_stream));

    float t_victim_concurrent_ms = 0.0f, t_enemy_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_concurrent_ms, start_victim, stop_victim));
    printf("Victim completed in %.2f ms (with enemy)\n", t_victim_concurrent_ms);

    
    CHECK_CUDA(cudaEventSynchronize(stop_enemy));

    
    CHECK_CUDA(cudaEventElapsedTime(&t_enemy_ms, start_enemy, stop_enemy));

    
    printf("Enemy completed in %.2f ms\n", t_enemy_ms);
    
    //calculate slowdown
    float slowdown = t_victim_concurrent_ms / t_victim_alone_ms;
    printf("\n=== RESULTS ===\n");
    printf("Victim slowdown: %.2fx (%.2f ms → %.2f ms)\n", slowdown, t_victim_alone_ms, t_victim_concurrent_ms);

    CHECK_CUDA(cudaFree(d_victim_array));
    CHECK_CUDA(cudaFree(d_enemy_array));
    CHECK_CUDA(cudaFree(d_victim_result));
    CHECK_CUDA(cudaFree(d_enemy_result));
    CHECK_CUDA(cudaStreamDestroy(victim_stream));
    CHECK_CUDA(cudaStreamDestroy(enemy_stream));
    CHECK_CUDA(cudaEventDestroy(start_victim));
    CHECK_CUDA(cudaEventDestroy(stop_victim));
    CHECK_CUDA(cudaEventDestroy(start_enemy));
    CHECK_CUDA(cudaEventDestroy(stop_enemy));
    CHECK_CUDA(cudaDeviceReset());
    return 0;
}