#!/bin/bash

# Setup MPS with STATIC SM PARTITIONING (-S flag)
# On iGPU (Orin, Ampere sm_87): each chunk = 2 SMs

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

sudo killall nvidia-cuda-mps-control 2>/dev/null
sudo killall nvidia-cuda-mps-server 2>/dev/null
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY
sleep 1

mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
chmod 777 $CUDA_MPS_PIPE_DIRECTORY
chmod 777 $CUDA_MPS_LOG_DIRECTORY

echo "=== Starting MPS Daemon (Static Partitioning) ==="
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"

# Start MPS with -S for static partitioning (requires Ampere+)
sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
            CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
            nvidia-cuda-mps-control -d --static-partitioning"

sleep 2

if ps aux | grep -v grep | grep nvidia-cuda-mps-control > /dev/null; then
    echo "MPS Control Daemon (Static Partitioning) is running"
else
    echo "MPS Control Daemon failed to start"
    cat $CUDA_MPS_LOG_DIRECTORY/control.log 2>/dev/null
    exit 1
fi

# Get GPU UUID
GPU_UUID=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | head -n 1 | tr -d '[:space:]')
if [ -z "$GPU_UUID" ]; then
    echo "ERROR: Could not detect GPU UUID"
    exit 1
fi
echo "GPU UUID: $GPU_UUID"

# Parse enemy and victim chunk counts from arguments (default: 1 enemy, rest victim)
ENEMY_CHUNKS=${1:-1}
VICTIM_CHUNKS=${2:-0}  # 0 = auto (use remaining)

# Create enemy partition (small: 1 chunk = 2 SMs on iGPU)
echo ""
echo "=== Creating SM Partitions ==="
echo "Enemy chunks: $ENEMY_CHUNKS (= $((ENEMY_CHUNKS * 2)) SMs on iGPU)"

ENEMY_PARTITION=$(echo "sm_partition add $GPU_UUID $ENEMY_CHUNKS" | \
    sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")

if echo "$ENEMY_PARTITION" | grep -q "Failed"; then
    echo "ERROR creating enemy partition: $ENEMY_PARTITION"
    exit 1
fi
echo "Enemy partition ID: $ENEMY_PARTITION"

# Determine remaining chunks for victim
if [ "$VICTIM_CHUNKS" -eq 0 ]; then
    LSPART_OUTPUT=$(echo "lspart" | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")
    echo ""
    echo "Current partition state:"
    echo "$LSPART_OUTPUT"
    VICTIM_CHUNKS=$(echo "$LSPART_OUTPUT" | grep "$GPU_UUID" | grep -v "/" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/) {print $i; exit}}')
    if [ -z "$VICTIM_CHUNKS" ] || [ "$VICTIM_CHUNKS" -eq 0 ]; then
        echo "ERROR: No free chunks remaining for victim"
        exit 1
    fi
fi

echo "Victim chunks: $VICTIM_CHUNKS (= $((VICTIM_CHUNKS * 2)) SMs on iGPU)"

VICTIM_PARTITION=$(echo "sm_partition add $GPU_UUID $VICTIM_CHUNKS" | \
    sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control")

if echo "$VICTIM_PARTITION" | grep -q "Failed"; then
    echo "ERROR creating victim partition: $VICTIM_PARTITION"
    exit 1
fi
echo "Victim partition ID: $VICTIM_PARTITION"

# Show final partition layout
echo ""
echo "=== Final Partition Layout ==="
echo "lspart" | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

# Export partition IDs for callers
export ENEMY_SM_PARTITION="$ENEMY_PARTITION"
export VICTIM_SM_PARTITION="$VICTIM_PARTITION"

echo ""
echo "MPS Static Partitioning ready"
echo "  ENEMY_SM_PARTITION=$ENEMY_SM_PARTITION"
echo "  VICTIM_SM_PARTITION=$VICTIM_SM_PARTITION"
echo ""
