#!/bin/bash

# ================================================================
# L2 CACHE CONTENTION with MPS STATIC SM PARTITIONING + NCU
# ================================================================
# This experiment isolates victim and enemy onto separate SM
# partitions using MPS static partitioning (-S, Ampere+).
# Enemy gets 1 chunk (2 SMs on iGPU), victim gets the rest.
# This proves L2 contention persists even with SM isolation,
# because L2 cache is shared across all SMs.
# ================================================================

# Setup environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "========================================================"
echo "  L2 CACHE CONTENTION with STATIC SM PARTITIONING + NCU"
echo "========================================================"
echo ""

# Compile programs
echo "Compiling victim and enemy..."
nvcc -arch=sm_87 -O3 -o victim_process ../MPS-experiments/victim_process.cu
nvcc -arch=sm_87 -O3 -o enemy_process enemy_process_enhanced.cu
echo "Compilation done"
echo ""

# -----------------------------------------------------------
# Start MPS with static partitioning
# -----------------------------------------------------------
echo "Setting up MPS with static SM partitioning..."

# Clean old MPS
sudo killall nvidia-cuda-mps-control 2>/dev/null
sudo killall nvidia-cuda-mps-server 2>/dev/null
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY
sleep 1

mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
chmod 777 $CUDA_MPS_PIPE_DIRECTORY
chmod 777 $CUDA_MPS_LOG_DIRECTORY

# Start MPS daemon with -S (static partitioning, requires Ampere+)
sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
            CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
            nvidia-cuda-mps-control -d -S"
sleep 2

if ! ps aux | grep -v grep | grep nvidia-cuda-mps-control > /dev/null; then
    echo "MPS Control Daemon failed to start!"
    cat $CUDA_MPS_LOG_DIRECTORY/control.log 2>/dev/null
    exit 1
fi
echo "MPS Control Daemon (static partitioning) is running"

# Get GPU UUID
GPU_UUID=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
if [ -z "$GPU_UUID" ]; then
    echo "ERROR: Could not detect GPU UUID"
    exit 1
fi
echo "GPU UUID: $GPU_UUID"

# -----------------------------------------------------------
# Create partitions: 1 chunk for enemy, rest for victim
# On iGPU (Orin): 1 chunk = 2 SMs
# -----------------------------------------------------------
ENEMY_CHUNKS=1

echo ""
echo "Creating enemy partition ($ENEMY_CHUNKS chunk = $((ENEMY_CHUNKS * 2)) SMs)..."
ENEMY_PARTITION=$(echo "sm_partition add $GPU_UUID $ENEMY_CHUNKS" | \
    sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")

if echo "$ENEMY_PARTITION" | grep -qi "fail\|error"; then
    echo "ERROR creating enemy partition: $ENEMY_PARTITION"
    exit 1
fi
echo "Enemy partition: $ENEMY_PARTITION"

# Query remaining free chunks for victim
LSPART_OUTPUT=$(echo "lspart" | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")
echo ""
echo "Partition state after enemy allocation:"
echo "$LSPART_OUTPUT"

# Extract free chunks from the GPU line (not a partition line)
VICTIM_CHUNKS=$(echo "$LSPART_OUTPUT" | grep "^$GPU_UUID\|^GPU-" | grep -v "/" | head -n 1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) {print $i; exit}}')
if [ -z "$VICTIM_CHUNKS" ] || [ "$VICTIM_CHUNKS" -eq 0 ]; then
    echo "ERROR: No free chunks remaining for victim"
    exit 1
fi

echo ""
echo "Creating victim partition ($VICTIM_CHUNKS chunks = $((VICTIM_CHUNKS * 2)) SMs)..."
VICTIM_PARTITION=$(echo "sm_partition add $GPU_UUID $VICTIM_CHUNKS" | \
    sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")

if echo "$VICTIM_PARTITION" | grep -qi "fail\|error"; then
    echo "ERROR creating victim partition: $VICTIM_PARTITION"
    exit 1
fi
echo "Victim partition: $VICTIM_PARTITION"

# Show final layout
echo ""
echo "=== Final SM Partition Layout ==="
echo "lspart" | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"
echo ""

# NCU metrics
NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,sm__cycles_elapsed.avg,gpu__time_active.sum"
VICTIM_ITERS=50000

# -----------------------------------------------------------
# SCENARIO 1: Victim Alone (on its own partition, with NCU)
# -----------------------------------------------------------
echo "========================================================"
echo "  SCENARIO 1: Victim Alone (static partition, with NCU)"
echo "========================================================"
echo ""

sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
            CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
            CUDA_MPS_SM_PARTITION=$VICTIM_PARTITION \
            $(which ncu) --metrics $NCU_METRICS \
            --kernel-name victimKernel \
            --csv \
            --log-file ncu_static_victim_alone.log \
            ./victim_process $VICTIM_ITERS" \
    > ncu_static_victim_alone.csv 2>&1

echo " Victim alone profiling complete"
echo ""

sleep 2

# -----------------------------------------------------------
# SCENARIO 2: Victim + Enemy (each on separate partitions)
# -----------------------------------------------------------
echo "========================================================"
echo "  SCENARIO 2: Victim + Enemy (static partitioned + NCU)"
echo "========================================================"
echo ""

# Launch enemy on its own partition (1 chunk = 2 SMs)
echo "Launching enemy on partition: $ENEMY_PARTITION"
echo "  ($ENEMY_CHUNKS chunk = $((ENEMY_CHUNKS * 2)) SMs)"
CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
CUDA_MPS_SM_PARTITION=$ENEMY_PARTITION \
    ./enemy_process 0 > enemy_static_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

# Wait for enemy to warm up and start thrashing L2
sleep 3

# Profile victim on its own partition (rest of the SMs)
echo "Profiling victim on partition: $VICTIM_PARTITION"
echo "  ($VICTIM_CHUNKS chunks = $((VICTIM_CHUNKS * 2)) SMs)"
sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
            CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
            CUDA_MPS_SM_PARTITION=$VICTIM_PARTITION \
            $(which ncu) --metrics $NCU_METRICS \
            --kernel-name victimKernel --csv \
            --log-file ncu_static_victim_concurrent.log \
            ./victim_process $VICTIM_ITERS" \
    > ncu_static_victim_concurrent.csv 2>&1

echo " Victim concurrent profiling complete"

# Kill enemy
echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill -9 $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo " Enemy terminated"

echo ""
echo "Enemy output:"
cat enemy_static_log.txt

# -----------------------------------------------------------
# RESULTS
# -----------------------------------------------------------
echo ""
echo "========================================================"
echo "  RESULTS ANALYSIS (Static SM Partitioning)"
echo "========================================================"
echo ""
echo "SM Layout: Enemy=$((ENEMY_CHUNKS * 2)) SMs | Victim=$((VICTIM_CHUNKS * 2)) SMs"
echo "(SMs are isolated via static partitioning, but L2 cache is SHARED)"
echo ""

echo "=== VICTIM ALONE (on dedicated partition) ==="
grep -A 5 "victimKernel" ncu_static_victim_alone.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

echo ""
echo "=== VICTIM + ENEMY (separate static partitions) ==="
grep -A 5 "victimKernel" ncu_static_victim_concurrent.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

# -----------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------
echo ""
echo "========================================================"
echo "  CLEANUP"
echo "========================================================"
echo ""
echo "Stopping MPS..."

echo quit | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY

echo ""
echo " Done! Results saved:"
echo "  - ncu_static_victim_alone.csv"
echo "  - ncu_static_victim_concurrent.csv"
echo "  - enemy_static_log.txt"
echo ""
