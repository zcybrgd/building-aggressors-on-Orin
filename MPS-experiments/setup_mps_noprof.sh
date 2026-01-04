#!/bin/bash

# Setup MPS with proper variable handling

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Clean old MPS
killall nvidia-cuda-mps-control 2>/dev/null
killall nvidia-cuda-mps-server 2>/dev/null
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY
sleep 1

# Create directories
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
chmod 777 $CUDA_MPS_PIPE_DIRECTORY
chmod 777 $CUDA_MPS_LOG_DIRECTORY

echo "=== Starting MPS Daemon ==="
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"

# Start with explicit environment
sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
            CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY \
            nvidia-cuda-mps-control -d"

sleep 2

# Check control daemon
if ps aux | grep -v grep | grep nvidia-cuda-mps-control > /dev/null; then
    echo "MPS Control Daemon is running"
else
    echo "MPS Control Daemon failed to start"
    cat $CUDA_MPS_LOG_DIRECTORY/control.log 2>/dev/null
    exit 1
fi

# The MPS SERVER will start automatically when first CUDA app runs
# This is NORMAL behavior - don't check for server yet!

echo "MPS Control Daemon ready"
echo "  (MPS Server will start on first CUDA application)"
echo ""

