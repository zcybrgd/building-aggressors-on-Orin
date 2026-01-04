#!/bin/bash
# Runs two long CUDA kernels under a single MPS daemon so they overlap on the GPU.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEM_SRC="$SCRIPT_DIR/memory_stress_client.cu"
COMP_SRC="$SCRIPT_DIR/compute_stress_client.cu"
MEM_BIN="$SCRIPT_DIR/memory_stress_client"
COMP_BIN="$SCRIPT_DIR/compute_stress_client"

# Tunables
: "${CUDA_ARCH:=sm_87}"
: "${MEM_ITERS:=50000}"
: "${MEM_ELEMS:=1048576}"
: "${COMP_ITERS:=50000}"
: "${COMP_ELEMS:=524288}"

: "${CUDA_MPS_PIPE_DIRECTORY:=/tmp/mps-simple-pipe}"
: "${CUDA_MPS_LOG_DIRECTORY:=/tmp/mps-simple-log}"

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

compile_client() {
    local src=$1
    local bin=$2
    if [[ ! -x "$bin" || "$src" -nt "$bin" ]]; then
        echo "[build] nvcc $(basename "$src") -> $(basename "$bin")"
        nvcc -std=c++14 -O3 -arch="$CUDA_ARCH" -lineinfo "$src" -o "$bin"
    fi
}

compile_client "$MEM_SRC" "$MEM_BIN"
compile_client "$COMP_SRC" "$COMP_BIN"

cleanup() {
    local status=$?
    echo "[cleanup] Stopping clients..."
    if [[ -n "${MEM_PID:-}" ]]; then kill -9 "$MEM_PID" 2>/dev/null || true; fi
    if [[ -n "${COMP_PID:-}" ]]; then kill -9 "$COMP_PID" 2>/dev/null || true; fi
    
    if [[ "${MPS_STARTED:-false}" == true ]]; then
        echo "[cleanup] Stopping MPS daemon..."
        echo "quit" | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            nvidia-cuda-mps-control >/dev/null 2>&1 || \
            echo "[warn] Failed to stop MPS; may need manual cleanup."
    fi
    
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    exit $status
}
trap cleanup EXIT INT TERM

# Kill any existing MPS as root first
sudo pkill -9 nvidia-cuda-mps-control 2>/dev/null || true
sudo pkill -9 nvidia-cuda-mps-server 2>/dev/null || true
sudo rm -rf /tmp/nvidia-mps /tmp/nvidia-log
sleep 1

echo "[info] Starting MPS control daemon as USER (no sudo)."
CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
nvidia-cuda-mps-control -d
MPS_STARTED=true

export CUDA_MPS_PIPE_DIRECTORY
export CUDA_MPS_LOG_DIRECTORY

sleep 2

echo "[run] launching memory client"
"$MEM_BIN" "$MEM_ITERS" "$MEM_ELEMS" &
MEM_PID=$!

echo "[run] launching compute client"
"$COMP_BIN" "$COMP_ITERS" "$COMP_ELEMS" &
COMP_PID=$!

wait "$MEM_PID" || echo "[warn] Memory client failed"
wait "$COMP_PID" || echo "[warn] Compute client failed"

echo "[done] both MPS clients exited."