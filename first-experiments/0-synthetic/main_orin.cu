/*
here we have a cache contention experiment that demonstrates how concurrent GPU kernels can interfere with 
each other's performance by competing for L2 cache resources
Victim kernel: A workload that repeatedly accesses a small working set
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

//ce kernel victim 
//comme argument on a un tableau d'entiers non signés (d_victim_array) que la victime lit (une suite de mots de 32 bits)
//un tableau de flottants (d_result)        
//la taille du tableau en octets (array_size_bytes) pour calculer le nombre de lignes de cache couvertes.
__global__ void victimKernel(unsigned int* d_victim_array, float* d_result,
                             int array_size_bytes, unsigned long long n_iters) {
    /*
 it repeatedly reads from a small working set d_victim_array
 accesses the same cache lines over and over in each iteration
 runs for a fixed number of iterations 
    */
    const int line_size = 128;
    int num_lines = array_size_bytes / line_size;
    if (num_lines <= 0) {
        if (threadIdx.x==0 && blockIdx.x==0) { d_result[0]=0.0f; d_result[1]=0.0f; }
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 1;
    int line_idx = tid % num_lines;
    volatile unsigned int* v = (volatile unsigned int*) d_victim_array; //to force memory reads not from registers
    unsigned long long iterations = 0;
    unsigned int local_sum = 0;

    for (unsigned long long iter = 0; iter < n_iters; ++iter) {
        //lz victim réutilise les mm 16 lignes à chaque itération donc si on perd cces lignes on perd bcp de perf
        //they shouldnt tenir in registers psk on a mis volatile nor L1
        //le working set should be bigger ig 
        //we are here repeatedly loading data from L2
        for (int k = 0; k < 16; ++k) {               // k tunable 
            int idx_line = (line_idx + k * stride) % num_lines; //calcule le num de la ligne a lire
            int idx = idx_line * (line_size / sizeof(unsigned int));
            local_sum += v[idx];
            // occasional write: uncomment/tune if desired
            // if ((iter & 0x3F) == 0) ((unsigned int*)v)[idx] = local_sum;
        }
        iterations++;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_result[0] = (float)iterations;
        d_result[1] = (float)local_sum;
    }
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

// L2 cache flush kernel
__global__ void flushL2(unsigned int* flush_array, int array_size_bytes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int num_words = array_size_bytes / sizeof(unsigned int);
    
    volatile unsigned int* v = (volatile unsigned int*)flush_array;
    unsigned int sum = 0;
    
    // Read through the entire array to flush L2
    for (int i = tid; i < num_words; i += stride) {
        sum += v[i];
    }
    
    // Write the sum to prevent optimization
    if (tid == 0) {
        v[0] = sum;
    }
}



int main(int argc, char* argv[]) {
    //par défaut
    int run_seconds = 10;
    int num_enemy_sms = 2;
    int victim_working_set_kb = 256;
    int enemy_array_mb = 2;

    int victim_grid_x = 1;
    int victim_block_x = 32;
    int enemy_threads_per_block = 128;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) run_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) num_enemy_sms = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) victim_working_set_kb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) enemy_array_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vg") == 0 && i + 1 < argc) victim_grid_x = atoi(argv[++i]);
        else if (strcmp(argv[i], "-vb") == 0 && i + 1 < argc) victim_block_x = atoi(argv[++i]);
        else if (strcmp(argv[i], "-eb") == 0 && i + 1 < argc) enemy_threads_per_block = atoi(argv[++i]);
    }

    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    int victim_size = victim_working_set_kb * 1024;
    //plus le tableau est très grand plus on augmente les chances de couvrir toute la L2
    int enemy_size = enemy_array_mb * 1024 * 1024;

    // Allocate L2 flush array (large enough to flush L2 cache)
    // Typically L2 cache is several MB, allocating sufficient size
    int flush_size = 3 * 1024 * 1024; // 16 MB to ensure L2 flush
    unsigned int* d_flush_array = nullptr;
    CHECK_CUDA(cudaMalloc(&d_flush_array, flush_size));

    unsigned int* d_victim_array = nullptr;
    unsigned int* d_enemy_array = nullptr;
    float* d_victim_result = nullptr;
    int* d_enemy_result = nullptr;

    CHECK_CUDA(cudaMalloc(&d_victim_array, victim_size));
    CHECK_CUDA(cudaMalloc(&d_enemy_array, enemy_size));
    CHECK_CUDA(cudaMalloc(&d_victim_result, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enemy_result, 2 * sizeof(int)));

    unsigned int* h_enemy = (unsigned int*)malloc(enemy_size);
    memset(h_enemy, 0, enemy_size);
    //build pointer chase array
    buildLineAlignedPointerChase(h_enemy, enemy_size);
    CHECK_CUDA(cudaMemcpy(d_enemy_array, h_enemy, enemy_size, cudaMemcpyHostToDevice));
    free(h_enemy);

    unsigned int* h_victim = (unsigned int*)malloc(victim_size);
    int victim_words = victim_size / sizeof(unsigned int);
    for (int i = 0; i < victim_words; ++i) h_victim[i] = i;
    CHECK_CUDA(cudaMemcpy(d_victim_array, h_victim, victim_size, cudaMemcpyHostToDevice));
    free(h_victim);

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
    unsigned long long n_iters = 100000ULL; //number of iterations for victim kernel
    unsigned long long run_time_cycles = compute_run_cycles(run_seconds, prop);

    dim3 victimGrid(victim_grid_x);
    dim3 victimBlock(victim_block_x);

    //create CUDA events for precise GPU timing
    cudaEvent_t start_victim, stop_victim, start_enemy, stop_enemy;
    CHECK_CUDA(cudaEventCreate(&start_victim));
    CHECK_CUDA(cudaEventCreate(&stop_victim));
    CHECK_CUDA(cudaEventCreate(&start_enemy));
    CHECK_CUDA(cudaEventCreate(&stop_enemy));

    // Setup for L2 flush kernel
    int flush_blocks = prop.multiProcessorCount * 2;
    int flush_threads = 256;

    printf("\n=== SCENARIO 1: Victim alone ===\n");
    // Flush L2 cache before experiment 1
    printf("Flushing L2 cache...\n");
    flushL2<<<flush_blocks, flush_threads>>>(d_flush_array, flush_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
        d_victim_array, d_victim_result, victim_size, n_iters);
    CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));
    CHECK_CUDA(cudaEventSynchronize(stop_victim));

    float t_victim_alone_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_alone_ms, start_victim, stop_victim));
    printf("Victim alone completed in %.2f ms (baseline)\n", t_victim_alone_ms);
    //i added this just to test
    //CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    //victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
      //  d_victim_array, d_victim_result, victim_size, n_iters);
    //CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));
    //CHECK_CUDA(cudaEventSynchronize(stop_victim));

    //t_victim_alone_ms = 0.0f;
    //CHECK_CUDA(cudaEventElapsedTime(&t_victim_alone_ms, start_victim, stop_victim));


    //printf("Victim alone 2 completed in %.2f ms (baseline)\n", t_victim_alone_ms);
     // finish that test
    printf("\n=== SCENARIO 2: Victim + Enemy concurrent ===\n");
    //printf("Flushing L2 cache...\n");
    //flushL2<<<flush_blocks, flush_threads>>>(d_flush_array, flush_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start_enemy, enemy_stream));
    enemyPointerChase<<<enemyGrid, enemyBlock, 0, enemy_stream>>>(
        d_enemy_array, d_enemy_result, enemy_size, run_time_cycles);
    CHECK_CUDA(cudaEventRecord(stop_enemy, enemy_stream));

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //launching the victim after some time; histoire de laisser l'enemy take over the lts a little bit
    CHECK_CUDA(cudaEventRecord(start_victim, victim_stream));
    victimKernel<<<victimGrid, victimBlock, 0, victim_stream>>>(
        d_victim_array, d_victim_result, victim_size, n_iters);
    CHECK_CUDA(cudaEventRecord(stop_victim, victim_stream));

    CHECK_CUDA(cudaEventSynchronize(stop_victim));
    CHECK_CUDA(cudaEventSynchronize(stop_enemy));

    float t_enemy_ms = 0.0f, t_victim_concurrent_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t_enemy_ms, start_enemy, stop_enemy));
    CHECK_CUDA(cudaEventElapsedTime(&t_victim_concurrent_ms, start_victim, stop_victim));

    printf("Enemy kernel finished in %.2f ms\n", t_enemy_ms);
    printf("Victim kernel finished in %.2f ms (concurrent scenario)\n", t_victim_concurrent_ms);
    
    CHECK_CUDA(cudaFree(d_flush_array));
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

    return 0;
}

