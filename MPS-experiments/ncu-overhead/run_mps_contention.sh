#!/bin/bash

# Setup environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log


echo "L2 CACHE CONTENTION with MPS + NCU"


if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found in PATH"
    exit 1
fi

NCU_BIN=$(command -v ncu)

# Compile programs
#echo "Compiling victim and enemy"
#nvcc -arch=sm_87 -O3 -o victim_process victim_process.cu
#nvcc -arch=sm_87 -O3 -o enemy_process enemy_process.cu
#echo "Compilation done"
#echo ""

#start MPS
./setup_mps.sh

if [ $? -ne 0 ]; then
    echo "MPS setup failed"
    exit 1
fi



NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_lookup_miss.sum,lts__t_requests_aperture_device_op_read_lookup_hit,lts__t_requests_aperture_device_op_read_lookup_miss,lts__t_requests_aperture_device_op_write_lookup_hit,lts__t_requests_aperture_device_op_write_lookup_miss,gpu__time_duration.avg,sm__cycles_elapsed.avg,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,sm__active_warps.avg"

# Victim iterations and profiling repetitions
VICTIM_ITERS=50000
RUNS=30

ALONE_TIMING_CSV="results0/timing_victim_alone_runs.csv"
ALONE_SUMMARY_CSV="results0/timing_victim_alone_summary.csv"
CONCURRENT_TIMING_CSV="results0/timing_victim_concurrent_runs.csv"
CONCURRENT_SUMMARY_CSV="results0/timing_victim_concurrent_summary.csv"

generate_timing_summary() {
    local timings_file="$1"
    local summary_file="$2"
    ./summarize_timings.py "$timings_file" "$summary_file"
}


echo "SCENARIO 1: Victim Alone (with NCU)"



echo "run,elapsed_ms" > "$ALONE_TIMING_CSV"
for ((i = 1; i <= RUNS; i++)); do
    printf "Run %02d/%02d: Profiling victim alone...\n" "$i" "$RUNS"
    csv_file=$(printf "results0/ncu_victim_alone_run_%02d.csv" "$i")
    log_file=$(printf "results0/ncu_victim_alone_run_%02d.log" "$i")
    start_ns=$(date +%s%N)
    if ! sudo "$NCU_BIN" --metrics "$NCU_METRICS" \
        --kernel-name victimKernel \
        --csv \
        --log-file "$log_file" \
        ./victim_process "$VICTIM_ITERS" \
        > "$csv_file" 2>&1; then
        echo "NCU profiling failed during victim-alone run $i"
        exit 1
    fi
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    printf "%d,%d\n" "$i" "$elapsed_ms" >> "$ALONE_TIMING_CSV"
done

generate_timing_summary "$ALONE_TIMING_CSV" "$ALONE_SUMMARY_CSV"

echo " Victim-alone profiling complete (30 runs)"
echo " Timing summary saved to $ALONE_SUMMARY_CSV"

sleep 2


echo "SCENARIO 2: Victim + Enemy (MPS + NCU)"

# Launch INFINITE enemy
echo "Launching enemy in background..."
./enemy_process 0 > enemy_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

# Wait for enemy to saturate L2
sleep 3

echo "run,elapsed_ms" > "$CONCURRENT_TIMING_CSV"
for ((i = 1; i <= RUNS; i++)); do
    printf "Run %02d/%02d: Profiling victim with enemy...\n" "$i" "$RUNS"
    csv_file=$(printf "results0/ncu_victim_concurrent_run_%02d.csv" "$i")
    log_file=$(printf "results0/ncu_victim_concurrent_run_%02d.log" "$i")
    start_ns=$(date +%s%N)
    if ! sudo "$NCU_BIN" --metrics "$NCU_METRICS" \
        --kernel-name victimKernel \
        --csv \
        --log-file "$log_file" \
        ./victim_process "$VICTIM_ITERS" \
        > "$csv_file" 2>&1; then
        echo "NCU profiling failed during victim+enemy run $i"
        kill -9 $ENEMY_PID 2>/dev/null
        wait $ENEMY_PID 2>/dev/null
        exit 1
    fi
    end_ns=$(date +%s%N)
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    printf "%d,%d\n" "$i" "$elapsed_ms" >> "$CONCURRENT_TIMING_CSV"
done

generate_timing_summary "$CONCURRENT_TIMING_CSV" "$CONCURRENT_SUMMARY_CSV"

echo " Victim concurrent profiling complete (30 runs)"
echo " Timing summary saved to $CONCURRENT_SUMMARY_CSV"

# Kill enemy
echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill -9 $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo " Enemy terminated"


echo "  CLEANUP"


echo quit | sudo sh -c "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY nvidia-cuda-mps-control"

# Cleanup
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY


