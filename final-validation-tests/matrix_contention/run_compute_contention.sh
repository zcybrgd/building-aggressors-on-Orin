#!/bin/bash

# ================================================================
# COMPUTE KERNEL CONTENTION EXPERIMENT 
# Green Context SM Isolation: Victim=6 SMs, Enemy=2 SMs
# L2 cache is SHARED — testing contention across all matrix/block combos
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
PATHS=15       # kernel internal iteration count
NUM_RUNS=5    # how many times to repeat each experiment

# NCU metrics
NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,sm__cycles_elapsed.avg,gpu__time_active.sum,sm__inst_executed.sum,smsp__warps_active.avg,sm__warps_launched.sum"

# Matrix sizes
MATRIX_SIZES=(784)

# Block configs: "BLOCK_X,BLOCK_Y"
BLOCK_CONFIGS=(
    "1,1024"
   # "2,512"
   # "4,256"
    "8,128"
    "16,64"
    "32,32"
   # "64,16"
   # "128,8"
   # "256,4"
   # "4,128"
   # "8,64"
   # "1024,1"
   # "512,2"
)

echo "========================================================"
echo "  COMPUTE KERNEL L2 CONTENTION EXPERIMENT"
echo "  Green Context: Victim=6 SMs | Enemy=2 SMs | L2=SHARED"
echo "  Runs per combo: $NUM_RUNS"
echo "========================================================"
echo ""

# --- Compile ---
echo "Compiling compute_kernel..."
nvcc -arch=sm_87 -O3 -o "$SCRIPT_DIR/matrix_victim" "$SCRIPT_DIR/compute.cu" -lcuda
echo "Compiling green_enemy (from parent dir)..."
if [ ! -f "$PARENT_DIR/green_enemy" ]; then
    nvcc -arch=sm_87 -O3 -o "$PARENT_DIR/green_enemy" "$PARENT_DIR/green_enemy.cu" -lcuda
fi
cp "$PARENT_DIR/green_enemy" "$SCRIPT_DIR/green_enemy"
echo "Compilation done"
echo ""

# --- Create results directory ---
mkdir -p "$RESULTS_DIR"

# --- Raw CSV (every single run) ---
RAW_CSV="$RESULTS_DIR/raw_all_runs.csv"
echo "matrix_size,block_x,block_y,threads_per_block,num_blocks,memory_MB,scenario,run,gpu_time_ns,l2_read_hit_rate_pct,l2_total_sectors,l2_read_miss_sectors,l2_write_miss_sectors,sm_cycles_avg,sm_inst_executed,smsp_warps_active_avg,sm_warps_launched" > "$RAW_CSV"

# --- Helper to extract metric value from NCU CSV log ---
extract_metric() {
    local logfile="$1"
    local metric="$2"
    grep "\"$metric\"" "$logfile" | tail -1 | awk -F'"' '{print $(NF-1)}' | tr -d ' '
}

# --- Run experiments ---
TOTAL_COMBOS=$(( ${#MATRIX_SIZES[@]} * ${#BLOCK_CONFIGS[@]} ))
COMBO=0

for MSIZE in "${MATRIX_SIZES[@]}"; do
    for BCONFIG in "${BLOCK_CONFIGS[@]}"; do
        IFS=',' read -r BX BY <<< "$BCONFIG"
        COMBO=$((COMBO + 1))
        TOTAL_ELEMENTS=$((MSIZE * MSIZE))
        THREADS=$((BX * BY))
        NUM_BLOCKS=$(( (TOTAL_ELEMENTS + THREADS - 1) / THREADS ))
        MEM_MB=$(echo "scale=2; 2 * $TOTAL_ELEMENTS * 8 / 1000000" | bc)

        TAG="m${MSIZE}_b${BX}x${BY}"

        echo "========================================================"
        echo "  [$COMBO/$TOTAL_COMBOS] Matrix: ${MSIZE}x${MSIZE} | Block: (${BX},${BY})=${THREADS} | Grid: ${NUM_BLOCKS}"
        echo "  Memory: ${MEM_MB} MB | Paths: ${PATHS} | Runs: ${NUM_RUNS}"
        echo "========================================================"

        for RUN in $(seq 1 $NUM_RUNS); do
            echo "  --- Run $RUN/$NUM_RUNS ---"

            # ---- ALONE ----
            ALONE_LOG="$RESULTS_DIR/ncu_${TAG}_alone_r${RUN}.log"
            sudo $(which ncu) --metrics $NCU_METRICS \
                --kernel-name GPUComputeIntensive \
                --csv \
                --log-file "$ALONE_LOG" \
                "$SCRIPT_DIR/matrix_victim" $MSIZE $BX $BY $PATHS 0 \
                > /dev/null 2>&1

            A_TIME=$(extract_metric "$ALONE_LOG" "gpu__time_active.sum")
            A_RHIT=$(extract_metric "$ALONE_LOG" "lts__t_sector_op_read_hit_rate.pct")
            A_SECT=$(extract_metric "$ALONE_LOG" "lts__t_sectors.sum")
            A_RMISS=$(extract_metric "$ALONE_LOG" "lts__t_sectors_op_read_lookup_miss.sum")
            A_WMISS=$(extract_metric "$ALONE_LOG" "lts__t_sectors_op_write_lookup_miss.sum")
            A_CYCLES=$(extract_metric "$ALONE_LOG" "sm__cycles_elapsed.avg")
            A_INST=$(extract_metric "$ALONE_LOG" "sm__inst_executed.sum")
            A_WACTIVE=$(extract_metric "$ALONE_LOG" "smsp__warps_active.avg")
            A_WLAUNCH=$(extract_metric "$ALONE_LOG" "sm__warps_launched.sum")

            echo "    [ALONE]  Time: ${A_TIME} ns | L2 hit: ${A_RHIT}% | Miss: ${A_RMISS}"
            echo "$MSIZE,$BX,$BY,$THREADS,$NUM_BLOCKS,$MEM_MB,alone,$RUN,$A_TIME,$A_RHIT,$A_SECT,$A_RMISS,$A_WMISS,$A_CYCLES,$A_INST,$A_WACTIVE,$A_WLAUNCH" >> "$RAW_CSV"

            sleep 1

            # ---- CONCURRENT (with enemy) ----
            CONC_LOG="$RESULTS_DIR/ncu_${TAG}_concurrent_r${RUN}.log"
            "$SCRIPT_DIR/green_enemy" 0 1 > "$RESULTS_DIR/enemy_${TAG}_r${RUN}.log" 2>&1 &
            ENEMY_PID=$!

            sleep 3

            sudo $(which ncu) --metrics $NCU_METRICS \
                --kernel-name GPUComputeIntensive \
                --csv \
                --log-file "$CONC_LOG" \
                "$SCRIPT_DIR/matrix_victim" $MSIZE $BX $BY $PATHS 1 \
                > /dev/null 2>&1

            # Kill enemy
            kill $ENEMY_PID 2>/dev/null || true
            sleep 1
            kill -9 $ENEMY_PID 2>/dev/null || true
            wait $ENEMY_PID 2>/dev/null || true

            C_TIME=$(extract_metric "$CONC_LOG" "gpu__time_active.sum")
            C_RHIT=$(extract_metric "$CONC_LOG" "lts__t_sector_op_read_hit_rate.pct")
            C_SECT=$(extract_metric "$CONC_LOG" "lts__t_sectors.sum")
            C_RMISS=$(extract_metric "$CONC_LOG" "lts__t_sectors_op_read_lookup_miss.sum")
            C_WMISS=$(extract_metric "$CONC_LOG" "lts__t_sectors_op_write_lookup_miss.sum")
            C_CYCLES=$(extract_metric "$CONC_LOG" "sm__cycles_elapsed.avg")
            C_INST=$(extract_metric "$CONC_LOG" "sm__inst_executed.sum")
            C_WACTIVE=$(extract_metric "$CONC_LOG" "smsp__warps_active.avg")
            C_WLAUNCH=$(extract_metric "$CONC_LOG" "sm__warps_launched.sum")

            echo "    [CONC]   Time: ${C_TIME} ns | L2 hit: ${C_RHIT}% | Miss: ${C_RMISS}"
            echo "$MSIZE,$BX,$BY,$THREADS,$NUM_BLOCKS,$MEM_MB,concurrent,$RUN,$C_TIME,$C_RHIT,$C_SECT,$C_RMISS,$C_WMISS,$C_CYCLES,$C_INST,$C_WACTIVE,$C_WLAUNCH" >> "$RAW_CSV"

            sleep 1
        done

        echo ""
        sleep 1
    done
done

echo "========================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "========================================================"
echo ""
echo "Results directory: $RESULTS_DIR/"
echo "Raw data CSV:      $RESULTS_DIR/raw_all_runs.csv"
echo "Individual logs:   $RESULTS_DIR/ncu_m*_r*.log"
echo ""
echo "Total: $TOTAL_COMBOS combos × $NUM_RUNS runs × 2 scenarios = $((TOTAL_COMBOS * NUM_RUNS * 2)) profiling runs"
