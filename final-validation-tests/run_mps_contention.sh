#!/bin/bash

# Setup environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "========================================"
echo "  L2 CACHE CONTENTION with MPS + NCU"
echo "========================================"
echo ""

# Compile programs
echo "Compiling victim and enemy..."
nvcc -arch=sm_87 -O3 -o victim_process ../MPS-experiments/victim_process.cu 
nvcc -arch=sm_87 -O3 -o enemy_process enemy_process_enhanced.cu
echo "Compilation done"
echo ""

# Start MPS
./setup_mps.sh

if [ $? -ne 0 ]; then
    echo "MPS setup failed!"
    exit 1
fi

# NCU metrics
NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,sm__cycles_elapsed.avg,gpu__time_active.sum" 
VICTIM_ITERS=50000

echo "========================================"
echo "  SCENARIO 1: Victim Alone (with NCU)"
echo "========================================"
echo ""

sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_alone.log \
    ./victim_process $VICTIM_ITERS \
    > ncu_victim_alone.csv 2>&1

echo " Victim alone profiling complete"
echo ""

sleep 2

echo "========================================"
echo "  SCENARIO 2: Victim + Enemy (MPS + NCU)"
echo "========================================"
echo ""

# Launch INFINITE enemy
echo "Launching enemy in background..."
./enemy_process 0 > enemy_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

# Wait for enemy to saturate L2
sleep 3

# Profile victim
echo "Profiling victim with NCU (enemy running)..."
sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel --csv \
    --log-file ncu_victim_concurrent.log \
    ./victim_process $VICTIM_ITERS \
    > ncu_victim_concurrent.csv 2>&1

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
echo "========================================"
echo "  RESULTS ANALYSIS"
echo "========================================"
echo ""

# Parse results
echo "=== VICTIM ALONE ==="
grep -A 5 "victimKernel" ncu_victim_alone.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

echo ""
echo "=== VICTIM + ENEMY ==="
grep -A 5 "victimKernel" ncu_victim_concurrent.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

# Stop MPS properly
echo ""
echo "========================================"
echo "  CLEANUP"
echo "========================================"
echo ""
echo "Stopping MPS..."

# Use explicit environment for sudo
echo quit | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

# Cleanup
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY

echo ""
echo " Done! Results saved:"
echo "  - ncu_victim_alone.csv"
echo "  - ncu_victim_concurrent.csv"
echo ""

