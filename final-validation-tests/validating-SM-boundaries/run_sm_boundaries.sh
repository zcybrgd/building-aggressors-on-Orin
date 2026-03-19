#!/bin/bash

# runs the same 16-block kernel under three SM allocations:
#   Scenario 0: all 8 SMs  (default context, baseline)
#   Scenario 1: 6 SMs only (green context, victim slice)
#   Scenario 2: 2 SMs only (green context, remainder slice)
# 5 runs per scenario → 15 NCU profiling sessions total.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
NUM_RUNS=10

NCU_METRICS="gpu__time_active.sum,sm__cycles_elapsed.avg,sm__warps_launched.sum,smsp__warps_active.avg,sm__inst_executed.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sectors_op_read_lookup_miss.sum"

echo "Compiling simple_kernel..."
nvcc -arch=sm_87 -O3 -lcuda -o "$SCRIPT_DIR/simple_kernel" "$SCRIPT_DIR/simple_kernel.cu"
if [ $? -ne 0 ]; then echo "Compilation failed!"; exit 1; fi
echo "Compilation done"
echo ""

mkdir -p "$RESULTS_DIR"


RAW_CSV="$RESULTS_DIR/raw_all_runs.csv"
echo "scenario,sm_count,run,gpu_time_ns,sm_cycles_avg,warps_launched,warps_active_avg,inst_executed,l2_read_hit_pct,l2_read_misses" > "$RAW_CSV"


extract_metric() {
    local logfile="$1"
    local metric="$2"
    grep "\"$metric\"" "$logfile" | tail -1 | awk -F'"' '{print $(NF-1)}' | tr -d ' ' | tr ',' '.'
}


for SCENARIO in 0 1 2; do

    case $SCENARIO in
        0) SM_COUNT=8;  LABEL="8_SMs_baseline" ;;
        1) SM_COUNT=6;  LABEL="6_SMs_green_ctx" ;;
        2) SM_COUNT=2;  LABEL="2_SMs_green_ctx" ;;
    esac

    echo "========================================================"
    echo "  Scenario $SCENARIO: $LABEL"
    echo "========================================================"

    for RUN in $(seq 1 $NUM_RUNS); do
        echo "  --- Run $RUN / $NUM_RUNS ---"

        LOG="$RESULTS_DIR/ncu_s${SCENARIO}_r${RUN}.log"
        sudo $(which ncu) \
            --metrics $NCU_METRICS \
            --kernel-name computeKernel \
            --csv \
            --log-file "$LOG" \
            "$SCRIPT_DIR/simple_kernel" $SCENARIO $RUN \
            2>/dev/null

        T=$(extract_metric   "$LOG" "gpu__time_active.sum")
        CY=$(extract_metric  "$LOG" "sm__cycles_elapsed.avg")
        WL=$(extract_metric  "$LOG" "sm__warps_launched.sum")
        WA=$(extract_metric  "$LOG" "smsp__warps_active.avg")
        IN=$(extract_metric  "$LOG" "sm__inst_executed.sum")
        HR=$(extract_metric  "$LOG" "lts__t_sector_op_read_hit_rate.pct")
        RM=$(extract_metric  "$LOG" "lts__t_sectors_op_read_lookup_miss.sum")

        echo "    Time: ${T} ns | SM cycles: ${CY} | Warps active: ${WA}"
        echo "$SCENARIO,$SM_COUNT,$RUN,$T,$CY,$WL,$WA,$IN,$HR,$RM" >> "$RAW_CSV"

        sleep 1
    done

    echo ""
done

echo "  ALL DONE\n"
echo "Results : $RESULTS_DIR/"
echo "CSV     : $RAW_CSV"
echo "Logs    : $RESULTS_DIR/ncu_s*_r*.log"
echo "Total   : 3 scenarios x $NUM_RUNS runs = $((3 * NUM_RUNS)) NCU runs"
