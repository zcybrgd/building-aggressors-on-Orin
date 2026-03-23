#!/bin/bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

echo "========================================"
echo "  L2 CACHE CONTENTION with MPS + NCU"
echo "  Static SM Partitioning (Jetson Orin)"
echo "========================================"
echo ""

# Compile programs
echo "Compiling victim and enemy..."
nvcc -arch=sm_87 -O3 -o victim_process victim_process.cu
nvcc -arch=sm_87 -O3 -o enemy_process enemy_process.cu
echo "Compilation done"
echo ""

# -----------------------------------------------
# Start MPS with STATIC partitioning (-S flag)
# Modeled exactly on your working setup_mps.sh
# -----------------------------------------------
echo "=== Starting MPS Daemon with Static Partitioning ==="

# Kill any existing MPS processes first (same as setup_mps.sh)
sudo killall nvidia-cuda-mps-control 2>/dev/null
sudo killall nvidia-cuda-mps-server 2>/dev/null
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY
sleep 1

# Create directories with open permissions (same as setup_mps.sh)
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
chmod 777 $CUDA_MPS_PIPE_DIRECTORY
chmod 777 $CUDA_MPS_LOG_DIRECTORY

echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"

sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
            CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
            nvidia-cuda-mps-control -d -S"

sleep 2

if ps aux | grep -v grep | grep nvidia-cuda-mps-control > /dev/null; then
    echo "MPS Control Daemon is running (static partitioning mode)"
else
    echo "MPS Control Daemon failed to start"
    cat $CUDA_MPS_LOG_DIRECTORY/control.log 2>/dev/null
    exit 1
fi
echo ""

# -----------------------------------------------
# Discover GPU UUID
# -----------------------------------------------
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)
if [ -z "$GPU_UUID" ]; then
    echo "Could not detect GPU UUID!"
    exit 1
fi
echo "Detected GPU UUID: $GPU_UUID"
echo ""

# -----------------------------------------------
# Create SM partitions
#
# Jetson Orin Nano iGPU: chunk = 2 SMs
# Total SMs = 8, Total chunks = 4
#   enemy  = 1 chunk (2 SMs)
#   victim = 3 chunks (6 SMs)
# -----------------------------------------------
TOTAL_SM=$(nvidia-smi --query-gpu=multiprocessor.count --format=csv,noheader | head -n 1 | tr -d ' ')
echo "Total SMs on device: $TOTAL_SM"
CHUNK_SIZE=2
TOTAL_CHUNKS=$(( TOTAL_SM / CHUNK_SIZE ))
echo "Total chunks available: $TOTAL_CHUNKS (chunk size = $CHUNK_SIZE SMs)"
echo ""

ENEMY_CHUNKS=1
VICTIM_CHUNKS=$(( TOTAL_CHUNKS - ENEMY_CHUNKS ))

echo "Partition plan:"
echo "  Enemy  partition: $ENEMY_CHUNKS chunk(s) = $(( ENEMY_CHUNKS * CHUNK_SIZE )) SMs"
echo "  Victim partition: $VICTIM_CHUNKS chunk(s) = $(( VICTIM_CHUNKS * CHUNK_SIZE )) SMs"
echo ""

# Create enemy partition
echo "Creating enemy partition ($ENEMY_CHUNKS chunk)..."
ENEMY_PARTITION=$(echo "sm_partition add $GPU_UUID $ENEMY_CHUNKS" | \
    sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")
if [[ "$ENEMY_PARTITION" == *"Failed"* ]] || [ -z "$ENEMY_PARTITION" ]; then
    echo "Failed to create enemy partition: $ENEMY_PARTITION"
    echo quit | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"
    exit 1
fi
echo "Enemy partition ID: $ENEMY_PARTITION"

# Create victim partition
echo "Creating victim partition ($VICTIM_CHUNKS chunks)..."
VICTIM_PARTITION=$(echo "sm_partition add $GPU_UUID $VICTIM_CHUNKS" | \
    sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")
if [[ "$VICTIM_PARTITION" == *"Failed"* ]] || [ -z "$VICTIM_PARTITION" ]; then
    echo "Failed to create victim partition: $VICTIM_PARTITION"
    echo quit | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"
    exit 1
fi
echo "Victim partition ID: $VICTIM_PARTITION"
echo ""

# Show current partition layout
echo "Current partition layout:"
echo "lspart" | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"
echo ""

# NCU metrics
NCU_METRICS="lts__t_sectors_lookup_miss.sum,lts__t_sector_op_read_hit_rate.pct,sm__cycles_elapsed.avg,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct,gpu__time_duration"

VICTIM_ITERS=50000

echo "========================================"
echo "  SCENARIO 1: Victim Alone (with NCU)"
echo "========================================"
echo ""

CUDA_MPS_SM_PARTITION=$VICTIM_PARTITION \
sudo $(which ncu) --metrics $NCU_METRICS --launch-skip 1 \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_alone.log \
    ./victim_process $VICTIM_ITERS \
    > ncu_victim_alone.csv 2>&1

echo "Victim alone profiling complete"
echo ""
sleep 2

echo "========================================"
echo "  SCENARIO 2: Victim + Enemy (MPS + NCU)"
echo "  Enemy isolated in 1-chunk partition"
echo "========================================"
echo ""

# Launch enemy in its isolated partition
echo "Launching enemy in isolated partition..."
CUDA_MPS_SM_PARTITION=$ENEMY_PARTITION \
    ./enemy_process 0 > enemy_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

sleep 3

# Profile victim in its own partition
echo "Profiling victim with NCU (enemy running in isolated partition)..."
CUDA_MPS_SM_PARTITION=$VICTIM_PARTITION \
sudo $(which ncu) --metrics $NCU_METRICS --launch-skip 1 \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_concurrent.log \
    ./victim_process $VICTIM_ITERS \
    > ncu_victim_concurrent.csv 2>&1

echo "Victim concurrent profiling complete"

# Kill enemy
echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill -9 $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo "Enemy terminated"
echo ""
echo "Enemy output:"
cat enemy_log.txt
echo ""

echo "========================================"
echo "  RESULTS ANALYSIS"
echo "========================================"
echo ""

echo "=== VICTIM ALONE ==="
grep -A 5 "victimKernel" ncu_victim_alone.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"
echo ""
echo "=== VICTIM + ENEMY (enemy in isolated 2-SM partition) ==="
grep -A 5 "victimKernel" ncu_victim_concurrent.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

echo ""
echo "========================================"
echo "  CLEANUP"
echo "========================================"
echo ""

# Remove partitions (enemy already dead)
echo "Removing partitions..."
echo "sm_partition rm $ENEMY_PARTITION"  | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"
echo "sm_partition rm $VICTIM_PARTITION" | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

# Stop MPS
echo "Stopping MPS..."
echo quit | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

# Cleanup directories
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY

echo ""
echo "Done! Results saved:"
echo "  - ncu_victim_alone.csv"
echo "  - ncu_victim_concurrent.csv"
echo ""
