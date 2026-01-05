#!/bin/bash

# Setup environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "========================================"
echo "  L2 CACHE CONTENTION with MPS"
echo "========================================"
echo ""

# Compile programs
echo "Compiling victim and enemy..."
nvcc -arch=sm_87 -O3 -o victim_process victim_process.cu
nvcc -arch=sm_87 -O3 -o enemy_process enemy_process.cu
echo "Compilation done"
echo ""

# Start MPS
./setup_mps_nosudo.sh

if [ $? -ne 0 ]; then
    echo "MPS setup failed!"
    exit 1
fi

# Victim iterations
VICTIM_ITERS=50000

echo "========================================"
echo "  SCENARIO 1: Victim Alone"
echo "========================================"
echo ""

./victim_process $VICTIM_ITERS

echo " Victim alone complete"
echo ""

sleep 1 

echo "========================================"
echo "  SCENARIO 2: Victim + Enemy (MPS)"
echo "========================================"
echo ""

# Launch INFINITE enemy
echo "Launching enemy in background..."
./enemy_process 0 > enemy_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

# Wait for enemy to saturate L2
sleep 15
# Run victim
echo "Running victim (enemy running)..."
./victim_process $VICTIM_ITERS

echo " Victim concurrent complete"

# Kill enemy
echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill -9 $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo " Enemy terminated"

# Stop MPS properly
echo ""
echo "========================================"
echo "  CLEANUP"
echo "========================================"
echo ""
echo "Stopping MPS..."

echo quit | sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

# Cleanup
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY

echo ""
echo " Done!"
echo ""

