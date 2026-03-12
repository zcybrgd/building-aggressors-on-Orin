#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <atomic>

// global flag for infinite mode
volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    keep_running = 0;
}

// design notes:
// 1) 16 mb footprint beats l1 and l2 so cache keeps thrashing
// 2) dependent pointer chain blocks prefetch and keeps sm waiting on memory
// 3) every load adds two random stores so both clean and dirty lines fight for l2 space
// 4) chase buffer stays read only, stores go to a second buffer so links never break
__global__ void enemyKernel(unsigned int* d_chase, unsigned int* d_writeback, int size, unsigned long long cycles, int* d_stop_flag) {
    unsigned long long start = clock64();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_words = size / sizeof(unsigned int);
    
    volatile unsigned int* v = (volatile unsigned int*)d_chase;
    volatile unsigned int* w = (volatile unsigned int*)d_writeback;
    int idx = tid % num_words;
    unsigned int sum = 0;
    
    // host flips d_stop_flag to stop the kernel without killing the context
    if (cycles == 0) {
        while (d_stop_flag[0] == 0) {
            unsigned int next = v[idx];
            sum += next;
            // two scatter stores dirty the cache each loop
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
            // same read write mix but limited by the cycles budget
            int write_idx = (idx + tid * 131) % num_words;
            unsigned int write_val = sum ^ next ^ write_idx;
            w[write_idx] = write_val;
            w[(write_idx + 97) % num_words] = write_val + tid;
            idx = next;
        }
    }
    
    if (tid == 0) d_writeback[0] = sum;
}

int main(int argc, char* argv[]) {
    int size = 16 * 1024 * 1024;  // 16 mb
    unsigned long long cycles = (argc > 1) ? atoll(argv[1]) : 10000000000ULL;
    
    if (cycles == 0) {
        printf("[ENEMY] Starting in INFINITE mode (run until killed)\n");
    } else {
        printf("[ENEMY] Starting for %llu cycles\n", cycles);
    }
    
    //setup signal handler
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    
    unsigned int* d_chase;
    unsigned int* d_writeback;
    int* d_stop_flag;
    cudaMalloc(&d_chase, size);
    cudaMalloc(&d_writeback, size);
    cudaMalloc(&d_stop_flag, sizeof(int));
    
    int h_stop_flag = 0;
    cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice);
    
    // build a permutation so every load jumps somewhere new
    unsigned int* h_chase = (unsigned int*)malloc(size);
    int num_words = size / sizeof(unsigned int);
    for (int i = 0; i < num_words; i++) {
        h_chase[i] = ((unsigned long long) i * 1234567) % num_words;
    }
    cudaMemcpy(d_chase, h_chase, size, cudaMemcpyHostToDevice);
    free(h_chase);
    cudaMemset(d_writeback, 0, size);
    
    // 16x256 launch keeps memory pipes hot but still leaves room for victims under mps
    enemyKernel<<<16, 256>>>(d_chase, d_writeback, size, cycles, d_stop_flag);
    
    //if infinite mode, wait for signal
    if (cycles == 0) {
        printf("[ENEMY] Running... (PID: %d)\n", getpid());
        while (keep_running) {
            sleep(1);
        }
        printf("[ENEMY] Stop signal received, terminating kernel...\n");
        h_stop_flag = 1;
        cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    cudaDeviceSynchronize();
    
    printf("[ENEMY] Completed\n");
    
    cudaFree(d_chase);
    cudaFree(d_writeback);
    cudaFree(d_stop_flag);
    return 0;
}
