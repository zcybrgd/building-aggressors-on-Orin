/*
here we have a cache contention experiment that demonstrates how concurrent GPU kernels can interfere with 
each other's performance by competing for L2 cache resources
victim : unoptimized GEMM kernel that repeatedly accesses the same data, causing high L2 cache pressure
Enemy kernel: An adversarial workload using pointer chasing to thrash the cache
*/
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

// Unoptimized GEMM victim kernel: C = A * B
// Performs naive matrix multiplication with no optimizations
// This creates predictable memory access patterns and L2 cache pressure
__global__ void victimKernel(float* d_A, float* d_B, float* d_C,
                             int M, int N, int K) {
    /*
     Unoptimized GEMM: each thread computes one element of C
     - No shared memory usage
     - No tiling
     - Poor memory coalescing
     - Repeatedly accesses same rows of A and columns of B from L2
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    float sum = 0.0f;
        sum = 0.0f;
        // Unoptimized inner loop: no blocking, reads A and B from global memory
        for (int k = 0; k < K; ++k) {
            sum += d_A[row * K + k] * d_B[k * N + col];
        }
    
    // Write final result
    d_C[row * N + col] = sum;
}

//le kernel enemy qui fait du pointer chasing the goal is to evict the victim's cache lines from L2
__global__ void enemyPointerChase(unsigned int* d_enemy_array, int* d_result,
                                  int array_size_bytes, unsigned long long run_time_cycles) {
    unsigned long long start = clock64(), now = start;
    const int line_size = 128;
    int num_lines = array_size_bytes / line_size;
    if (num_lines <= 1) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx_line = (unsigned int)(tid % num_lines);
    volatile unsigned int* v = (volatile unsigned int*) d_enemy_array;
    unsigned long long iterations = 0;
    unsigned int local_sum = 0;
    int word_index = idx_line * (line_size / sizeof(unsigned int));
    do {
        unsigned int next = v[word_index];
        local_sum += next;
        word_index = next;
        iterations++;
        now = clock64();
    } while ((now - start) < run_time_cycles);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_result[0] = (int)iterations;
        d_result[1] = (int)local_sum;
    }
}

//we construct an array where each element inthe array points to the index of the next element to be visited
/**
un tableau h_array tel que, si on lit h_array[word_index] on saute vers la prochaine ligne voulue
la permutation garantit qu’on visite chaque ligne exactement une fois avant de revenir au début (cycle sur toutes les lignes)
*/
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

unsigned long long estimateVictimIters(int run_seconds, const cudaDeviceProp &prop, int num_threads, int lines_per_iter=16, int cycles_per_load=256) {
    unsigned long long total_cycles = (unsigned long long)run_seconds * prop.clockRate * 1000ULL;
    unsigned long long cycles_per_iter = num_threads * lines_per_iter * cycles_per_load;
    return total_cycles / cycles_per_iter;
}



int main(int argc, char* argv[]) {
    //par défaut
    int run_seconds = 10;
    int num_enemy_sms = 8;
    int matrix_size = 512;  // Matrix dimension (creates 512x512 matrices)
    int enemy_array_mb = 8;

    int victim_block_dim = 16;  // Use 16x16 thread blocks for GEMM
    int enemy_threads_per_block = 128;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) run_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) num_enemy_sms = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) matrix_size = atoi(argv[++i]);  // Matrix size
        else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) enemy_array_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vb") == 0 && i + 1 < argc) victim_block_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "-eb") == 0 && i + 1 < argc) enemy_threads_per_block = atoi(argv[++i]);
    }

    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // GEMM matrices: A (M x K), B (K x N), C (M x N)
    int M = matrix_size, N = matrix_size, K = matrix_size;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    printf("Matrix sizes: A=%dx%d, B=%dx%d, C=%dx%d\n", M, K, K, N, M, N);
    printf("Memory: A=%.2f MB, B=%.2f MB, C=%.2f MB, Total=%.2f MB\n",
           size_A/1e6, size_B/1e6, size_C/1e6, (size_A+size_B+size_C)/1e6);

    //plus le tableau est très grand plus on augmente les chances de couvrir toute la L2
    int enemy_size = enemy_array_mb * 1024 * 1024;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    unsigned int* d_enemy_array = nullptr;
    int* d_enemy_result = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMalloc(&d_enemy_array, enemy_size));
    CHECK_CUDA(cudaMalloc(&d_enemy_result, 2 * sizeof(int)));

    unsigned int* h_enemy = (unsigned int*)malloc(enemy_size);
    memset(h_enemy, 0, enemy_size);
    //build pointer chase array
    buildLineAlignedPointerChase(h_enemy, enemy_size);
    CHECK_CUDA(cudaMemcpy(d_enemy_array, h_enemy, enemy_size, cudaMemcpyHostToDevice));
    free(h_enemy);

    // Initialize matrices A and B on host
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;  // Simple initialization
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    free(h_A);
    free(h_B);

    cudaStream_t enemy_stream, victim_stream;
    CHECK_CUDA(cudaStreamCreate(&enemy_stream));
    CHECK_CUDA(cudaStreamCreate(&victim_stream));

    //unsigned long long run_time_cycles = compute_run_cycles(run_seconds, prop);
    int maxActiveBlocksPerSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, enemyPointerChase, enemy_threads_per_block, 0));

    if (maxActiveBlocksPerSM < 1) maxActiveBlocksPerSM = 1;
    int enemy_blocks = num_enemy_sms * maxActiveBlocksPerSM;
    if (enemy_blocks < 1) enemy_blocks = 1;

    dim3 enemyGrid(enemy_blocks);
    dim3 enemyBlock(enemy_threads_per_block);
    // paramètre d'itérations 
    //unsigned long long n_iters = 100ULL; //number of GEMM iterations (reduced since GEMM is expensive)
    unsigned long long run_time_cycles = compute_run_cycles(run_seconds, prop);

    // 2D grid for GEMM
    dim3 victimBlock(victim_block_dim, victim_block_dim);
    dim3 victimGrid((N + victimBlock.x - 1) / victimBlock.x, 
                    (M + victimBlock.y - 1) / victimBlock.y);

    //create CUDA events for precise GPU timing
    cudaEvent_t start_victim, stop_victim, start_enemy, stop_enemy;
    CHECK_CUDA(cudaEventCreate(&start_victim));
    CHECK_CUDA(cudaEventCreate(&stop_victim));
    CHECK_CUDA(cudaEventCreate(&start_enemy));
    CHECK_CUDA(cudaEventCreate(&stop_enemy));

    printf("\n=== SCENARIO 1: Victim unoptimized GEMM alone ===\n");
    CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
        d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));
    CHECK_CUDA(cudaEventSynchronize(stop_victim));

    float t_victim_alone_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_alone_ms, start_victim, stop_victim));
    printf("Victim alone completed in %.2f ms (baseline)\n", t_victim_alone_ms);

    printf("\n=== SCENARIO 2: Victim + Enemy concurrent ===\n");
    CHECK_CUDA(cudaEventRecord(start_enemy, enemy_stream));
    enemyPointerChase<<<enemyGrid, enemyBlock, 0, enemy_stream>>>(
        d_enemy_array, d_enemy_result, enemy_size, run_time_cycles);
    CHECK_CUDA(cudaEventRecord(stop_enemy, enemy_stream));

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //launching the victim after some time; histoire de laisser l'enemy take over the lts a little bit
    CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
        d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));

    CHECK_CUDA(cudaEventSynchronize(stop_victim));
    CHECK_CUDA(cudaEventSynchronize(stop_enemy));

    float t_enemy_ms = 0.0f, t_victim_concurrent_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_enemy_ms, start_enemy, stop_enemy));
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_concurrent_ms, start_victim, stop_victim));

    printf("Enemy kernel finished in %.2f ms\n", t_enemy_ms);
    printf("Victim kernel finished in %.2f ms (concurrent scenario)\n", t_victim_concurrent_ms);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_enemy_array));
    CHECK_CUDA(cudaFree(d_enemy_result));
    CHECK_CUDA(cudaStreamDestroy(victim_stream));
    CHECK_CUDA(cudaStreamDestroy(enemy_stream));
    CHECK_CUDA(cudaEventDestroy(start_victim));
    CHECK_CUDA(cudaEventDestroy(stop_victim));
    CHECK_CUDA(cudaEventDestroy(start_enemy));
    CHECK_CUDA(cudaEventDestroy(stop_enemy));

    return 0;
}





