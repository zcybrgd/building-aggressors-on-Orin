#!/bin/bash

# Setup environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "========================================="
echo "  L2 CACHE CONTENTION with MPS + NSYS"
echo "========================================="
echo ""

# Compile programs
echo "Compiling victim and enemy..."
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
nvcc -arch=sm_87 -O3 -o victim_process victim_process.cu
nvcc -arch=sm_87 -O3 -o enemy_process enemy_process.cu
echo "Compilation done"
echo ""

# Start MPS (no sudo inside setup_mps.sh)
./setup_mps.sh

if [ $? -ne 0 ]; then
    echo "MPS setup failed!"
    exit 1
fi

# Victim iterations
VICTIM_ITERS=50000

sleep 2

echo "========================================="
echo "  SCENARIO 2: Victim + Enemy (MPS + NSYS)"
echo "========================================="
echo ""

# Launch INFINITE enemy
echo "Launching enemy in background..."
./enemy_process 0 > enemy_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

# Wait for enemy to saturate L2
sleep 15 

# Profile victim with fork tracking (captures enemy too)
echo "Profiling ENTIRE SYSTEM with NSYS (enemy + victim)..."
sudo -E env "PATH=$PATH" nsys profile \
    --trace=cuda\
    --sample=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    -o nsys_victim_concurrent \
    ./victim_process $VICTIM_ITERS

echo " Victim concurrent profiling complete"

# Kill enemy
echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill -9 $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo " Enemy terminated"

echo ""
echo "Enemy output:"
cat enemy_log.txt

echo ""
echo "========================================="
echo "  RESULTS ANALYSIS"
echo "========================================="
echo ""

echo "=== Scenario 1: nsys_victim_alone.nsys-rep ==="
echo "=== Scenario 2: nsys_victim_concurrent.nsys-rep (BOTH PROCESSES) ==="
echo ""
echo "View concurrent execution:"
echo "nsys-ui nsys_victim_alone.nsys-rep nsys_victim_concurrent.nsys-rep"
echo ""
echo "Look for:"
echo "  - Enemy kernels running during victim execution"
echo "  - Timeline overlap proving MPS concurrency"
echo "  - CUDA context switches between processes"

# Stop MPS properly (run as sudo for cleanup)
sudo sh -c "echo quit | CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"
sudo killall nvidia-cuda-mps-control 2>/dev/null
sudo killall nvidia-cuda-mps-server 2>/dev/null

# Cleanup
sudo rm -rf $CUDA_MPS_PIPE_DIRECTORY
sudo rm -rf $CUDA_MPS_LOG_DIRECTORY

echo ""
echo " Done! Results saved:"
echo "  - nsys_victim_alone.nsys-rep"
echo "  - nsys_victim_concurrent.nsys-rep"
echo ""
echo "Run 'nsys-ui nsys_victim_concurrent.nsys-rep' to see MPS concurrency!"

