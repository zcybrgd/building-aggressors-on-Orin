#!/bin/bash

# setup MPS directories
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

#clean old MPS if exists
#sudo killall nvidia-cuda-mps-control 2>/dev/null
#sudo killall nvidia-cuda-mps-server 2>/dev/null
#rm -rf $CUDA_MPS_PIPE_DIRECTORY
#rm -rf $CUDA_MPS_LOG_DIRECTORY

#create directories
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
echo "=== Starting MPS Daemon ==="
sudo -E nvidia-cuda-mps-control -d
sleep 2
#check if MPS is running
if ps aux | grep -v grep | grep nvidia-cuda-mps-server > /dev/null; then
    echo "MPS Server is running"
    ps aux | grep nvidia-cuda-mps
else
    echo "MPS failed to start"
    exit 1
fi

#showing MPS info
echo ""
echo "=== MPS Configuration ==="
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
echo ""

#querying MPS
echo "=== MPS Status ==="
echo get_server_list | sudo nvidia-cuda-mps-control
echo ""
echo "MPS is ready!"
echo ""
echo "To stop MPS:"
echo "  echo quit | sudo nvidia-cuda-mps-control"
