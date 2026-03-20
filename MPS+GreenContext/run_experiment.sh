#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

PATHS=15
NUM_RUNS=1

MATRIX_SIZES=(240 496 784 1016 1232)          #  (240 496 784 1016 1232)
BLOCK_CONFIGS=("32,32" "1,1024" "64,16")     # ("1,1024" "32,32" "64,16")...

NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,\
lts__t_sector_op_write_hit_rate.pct,\
lts__t_sectors.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
lts__t_sectors_op_write_lookup_miss.sum,\
sm__cycles_elapsed.avg,\
gpu__time_active.sum,\
sm__inst_executed.sum,\
smsp__warps_active.avg,\
sm__warps_launched.sum"

RESULTS_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

RAW_CSV="$RESULTS_DIR/raw_all_runs.csv"
echo "matrix_size,block_x,block_y,threads_per_block,num_blocks,\
memory_MB,scenario,run,gpu_time_ns,l2_read_hit_rate_pct,\
l2_total_sectors,l2_read_miss_sectors,l2_write_miss_sectors,\
sm_cycles_avg,sm_inst_executed,smsp_warps_active_avg,sm_warps_launched" \
    > "$RAW_CSV"

echo "Compiling :"
VICTIM_SRC="$SCRIPT_DIR/../final-validation-tests/matrix_contention/matrix_victim.cu"
ENEMY_SRC="$SCRIPT_DIR/../final-validation-tests/green_enemy.cu"
nvcc -arch=sm_87 -O3 -o "$SCRIPT_DIR/matrix_victim" \
    "$VICTIM_SRC" -lcuda
nvcc -arch=sm_87 -O3 -o "$SCRIPT_DIR/green_enemy" \
    "$ENEMY_SRC" -lcuda
echo "Compilation done"
echo ""

#start mps
"$SCRIPT_DIR/../MPS-experiments/setup_mps.sh"
echo ""

extract_metric() {
    local logfile="$1" metric="$2"
    grep "\"$metric\"" "$logfile" 2>/dev/null \
        | tail -1 \
        | awk -F'"' '{print $(NF-1)}' \
        | tr -d ' '
}

kill_enemy() {
    local pid="$1"
    kill    "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    wait    "$pid" 2>/dev/null || true
}

TOTAL_COMBOS=$(( ${#MATRIX_SIZES[@]} * ${#BLOCK_CONFIGS[@]} ))
COMBO=0

for MSIZE in "${MATRIX_SIZES[@]}"; do
    for BCONFIG in "${BLOCK_CONFIGS[@]}"; do
        IFS=',' read -r BX BY <<< "$BCONFIG"
        COMBO=$(( COMBO + 1 ))
        THREADS=$(( BX * BY ))
        TOTAL_EL=$(( MSIZE * MSIZE ))
        NUM_BLOCKS=$(( (TOTAL_EL + THREADS - 1) / THREADS ))
        MEM_MB=$(echo "scale=2; 2 * $TOTAL_EL * 8 / 1000000" | bc)
        TAG="m${MSIZE}_b${BX}x${BY}"
        echo "========================================================"
        echo "  [$COMBO/$TOTAL_COMBOS] Matrix: ${MSIZE}x${MSIZE} | \
Block: (${BX},${BY})=${THREADS} | Grid: ${NUM_BLOCKS}"
        echo "  Memory: ${MEM_MB} MB | Paths: ${PATHS} | Runs: ${NUM_RUNS}"

        for RUN in $(seq 1 "$NUM_RUNS"); do
            echo "  --- Run $RUN/$NUM_RUNS ---"
            ALONE_LOG="$RESULTS_DIR/ncu_${TAG}_alone_r${RUN}.log"
            ALONE_OUT="$RESULTS_DIR/victim_${TAG}_alone_r${RUN}.stdout"
            sudo "$(which ncu)" \
                    --metrics "$NCU_METRICS" \
                     --kernel-name GPUMultiplyMatrix \
                     --csv \
                    --log-file "$ALONE_LOG" \
                     "$SCRIPT_DIR/matrix_victim" \
                         "$MSIZE" "$BX" "$BY" "$PATHS" 0 \
                 > "$ALONE_OUT" 2>/dev/null
            #"$SCRIPT_DIR/matrix_victim" "$MSIZE" "$BX" "$BY" "$PATHS" 0 \
            #> "$ALONE_OUT" 2>/dev/null

            A_TIME=$(extract_metric "$ALONE_LOG" "gpu__time_active.sum") || true
            A_RHIT=$(extract_metric "$ALONE_LOG" "lts__t_sector_op_read_hit_rate.pct") || true
            A_SECT=$(extract_metric "$ALONE_LOG" "lts__t_sectors.sum") || true
            A_RMISS=$(extract_metric "$ALONE_LOG" "lts__t_sectors_op_read_lookup_miss.sum") || true
            A_WMISS=$(extract_metric "$ALONE_LOG" "lts__t_sectors_op_write_lookup_miss.sum") || true
            A_CYCLES=$(extract_metric "$ALONE_LOG" "sm__cycles_elapsed.avg") || true
            A_INST=$(extract_metric "$ALONE_LOG" "sm__inst_executed.sum") || true
            A_WACT=$(extract_metric "$ALONE_LOG" "smsp__warps_active.avg") || true
            A_WLNCH=$(extract_metric "$ALONE_LOG" "sm__warps_launched.sum") || true

            echo "    [ALONE]  gpu_time: ${A_TIME} ns | L2 hit: ${A_RHIT}% | read_miss: ${A_RMISS}"
            echo "$MSIZE,$BX,$BY,$THREADS,$NUM_BLOCKS,$MEM_MB,alone,\
$RUN,$A_TIME,$A_RHIT,$A_SECT,$A_RMISS,$A_WMISS,$A_CYCLES,$A_INST,$A_WACT,$A_WLNCH" \
                >> "$RAW_CSV"

            sleep 1
            CONC_LOG="$RESULTS_DIR/ncu_${TAG}_concurrent_r${RUN}.log"
            CONC_OUT="$RESULTS_DIR/victim_${TAG}_concurrent_r${RUN}.stdout"
            ENEMY_LOG="$RESULTS_DIR/enemy_${TAG}_r${RUN}.log"
            CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
            CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
                "$SCRIPT_DIR/green_enemy" 0 1 \
                > "$ENEMY_LOG" 2>&1 &
            ENEMY_PID=$!
            echo "    [ENEMY]  PID=$ENEMY_PID — waiting for L2 saturation..."
            sleep 3
            sudo  "$(which ncu)" \
                     --metrics "$NCU_METRICS" \
                    --kernel-name GPUMultiplyMatrix \
                     --csv \
                     --log-file "$CONC_LOG" \
                     "$SCRIPT_DIR/matrix_victim" \
                         "$MSIZE" "$BX" "$BY" "$PATHS" 1 \
                 > "$CONC_OUT" 2>/dev/null
            #"$SCRIPT_DIR/matrix_victim" "$MSIZE" "$BX" "$BY" "$PATHS" 1 \
    #> "$CONC_OUT" 2>/dev/null

            kill_enemy "$ENEMY_PID"

            C_TIME=$(extract_metric "$CONC_LOG" "gpu__time_active.sum") || true
            C_RHIT=$(extract_metric "$CONC_LOG" "lts__t_sector_op_read_hit_rate.pct") || true
            C_SECT=$(extract_metric "$CONC_LOG" "lts__t_sectors.sum") || true
            C_RMISS=$(extract_metric "$CONC_LOG" "lts__t_sectors_op_read_lookup_miss.sum") || true
            C_WMISS=$(extract_metric "$CONC_LOG" "lts__t_sectors_op_write_lookup_miss.sum") || true
            C_CYCLES=$(extract_metric "$CONC_LOG" "sm__cycles_elapsed.avg") || true
            C_INST=$(extract_metric "$CONC_LOG" "sm__inst_executed.sum") || true
            C_WACT=$(extract_metric "$CONC_LOG" "smsp__warps_active.avg") || true
            C_WLNCH=$(extract_metric "$CONC_LOG" "sm__warps_launched.sum") || true

            echo "    [CONC]   gpu_time: ${C_TIME} ns | L2 hit: ${C_RHIT}% | read_miss: ${C_RMISS}"
            echo "$MSIZE,$BX,$BY,$THREADS,$NUM_BLOCKS,$MEM_MB,concurrent,\
$RUN,$C_TIME,$C_RHIT,$C_SECT,$C_RMISS,$C_WMISS,$C_CYCLES,$C_INST,$C_WACT,$C_WLNCH" \
                >> "$RAW_CSV"

            echo "    [ENEMY log excerpt]:"
            head -6 "$ENEMY_LOG" | sed 's/^/      /'

            sleep 1
        done

        echo ""
        sleep 1
    done
done

echo "Stopping MPS :"
echo quit | sudo sh -c \
    "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
     nvidia-cuda-mps-control" 2>/dev/null || true
sudo rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" || true

echo ""
echo "========================================================"
echo "  DONE"
echo "  Results directory : $RESULTS_DIR/"
echo "  Raw CSV           : $RAW_CSV"
echo "  Total runs        : $((TOTAL_COMBOS * NUM_RUNS * 2)) \
($TOTAL_COMBOS combos × $NUM_RUNS runs × 2 scenarios)"
echo "========================================================"