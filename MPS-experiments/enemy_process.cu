#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <atomic>

// Global flag for infinite mode
volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    keep_running = 0;
}

__global__ void enemyKernel(unsigned int* d_chase, int size, unsigned long long cycles, int* d_stop_flag) {
    unsigned long long start = clock64();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    volatile unsigned int* v = (volatile unsigned int*)d_chase;
    int idx = tid % (size / sizeof(unsigned int));
    unsigned int sum = 0;
    
    //check stop flag
    if (cycles == 0) {
        while (d_stop_flag[0] == 0) {
            unsigned int next = v[idx];
            sum += next;
            idx = next;
        }
    } else {
        // Timed mode
        while ((clock64() - start) < cycles) {
            unsigned int next = v[idx];
            sum += next;
            idx = next;
        }
    }
    
    if (tid == 0) d_chase[0] = sum;
}

int main(int argc, char* argv[]) {
    int size = 16 * 1024 * 1024;  // 16 MB
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
    int* d_stop_flag;
    cudaMalloc(&d_chase, size);
    cudaMalloc(&d_stop_flag, sizeof(int));
    
    int h_stop_flag = 0;
    cudaMemcpy(d_stop_flag, &h_stop_flag, sizeof(int), cudaMemcpyHostToDevice);
    
    //build pointer chase
    unsigned int* h_chase = (unsigned int*)malloc(size);
    int num_words = size / sizeof(unsigned int);
    for (int i = 0; i < num_words; i++) {
        h_chase[i] = (i * 1234567) % num_words;
    }
    cudaMemcpy(d_chase, h_chase, size, cudaMemcpyHostToDevice);
    free(h_chase);
    
    // launch aggressive enemy
    enemyKernel<<<512, 256>>>(d_chase, size, cycles, d_stop_flag);
    
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
    cudaFree(d_stop_flag);
    return 0;
}
