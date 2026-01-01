#!/bin/bash

# Configuration
EXECUTABLE="./ncuhead"
COMMAND="compute"
ITERATIONS=${1:-30}  # Default 30, but use first argument if provided
OUTPUT_FILE="benchmark_results_${ITERATIONS}iter.txt"
NCU_OUTPUT="ncu_metrics_${ITERATIONS}iter.csv"
TEMP_NCU_DIR="temp_ncu_runs"

# NCU metrics to collect from Masola et al., 2023 selection
NCU_METRICS="smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,\
smsp__warp_issue_stalled_misc_per_warp_active.pct,\
smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,\
lts__t_sectors.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors_aperture_sysmem_op_read.sum.per_second,\
lts__t_sectors_op_read.sum.per_second,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,\
lts__t_sectors_op_write.sum.per_second,\
smsp__inst_executed_op_global_ld.sum,\
lts__t_requests.sum,\
lts__t_requests.min,\
lts__t_requests.max,\
lts__t_sector_hit_rate.pct,\
lts__t_bytes.min,\
lts__t_request_hit_rate.ratio,\
smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"

# Validate iterations is a positive number
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
    echo "ERROR: Iterations must be a positive integer"
    echo "Usage: $0 [iterations]"
    echo "Example: $0 50"
    exit 1
fi

# Arrays to store results
declare -a timings=()
declare -a gpu_freqs=()
declare -a mem_freqs=()

# Create temp directory for NCU runs
mkdir -p "$TEMP_NCU_DIR"

echo "Starting benchmark: $ITERATIONS iterations with NCU profiling"
echo "Results will be saved to $OUTPUT_FILE"
echo "NCU metrics will be saved to $NCU_OUTPUT"
echo "========================================"

# Run iterations with NCU profiling
for i in $(seq 1 $ITERATIONS); do
    echo "Iteration $i/$ITERATIONS"
    
    # Get current GPU and memory frequencies before execution
    GPU_FREQ=$(sudo cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq 2>/dev/null || echo "N/A")
    MEM_FREQ=$(sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate 2>/dev/null || echo "N/A")
    
    # Run with NCU profiling for this iteration
    NCU_TEMP_FILE="${TEMP_NCU_DIR}/run_${i}.csv"
    
    echo "  Running NCU profiling..."
    ncu --metrics "$NCU_METRICS" --csv --log-file "$NCU_TEMP_FILE" $EXECUTABLE $COMMAND > /dev/null 2>&1
    
    # Also run without NCU to get clean timing
    output=$($EXECUTABLE $COMMAND 2>&1)
    
    # Extract timing from output
    timing=$(echo "$output" | grep "COMPUTE_KERNEL_TIMING" | cut -d',' -f2)
    
    if [ -n "$timing" ]; then
        timings+=($timing)
        gpu_freqs+=($GPU_FREQ)
        mem_freqs+=($MEM_FREQ)
        echo "  Timing: ${timing}ms | GPU: ${GPU_FREQ}Hz | MEM: ${MEM_FREQ}Hz"
    else
        echo "  ERROR: Could not extract timing"
    fi
done

# Calculate statistics
if [ ${#timings[@]} -eq 0 ]; then
    echo "ERROR: No valid timing data collected!"
    exit 1
fi

# Use awk for statistics calculation
stats=$(printf '%s\n' "${timings[@]}" | awk '
{
    sum += $1
    array[NR] = $1
    if (NR == 1 || $1 < min) min = $1
    if (NR == 1 || $1 > max) max = $1
}
END {
    avg = sum / NR
    for (i in array) {
        diff = array[i] - avg
        sumsq += diff * diff
    }
    stddev = sqrt(sumsq / NR)
    printf "%.4f %.4f %.4f %.4f", min, max, avg, stddev
}')

read MIN MAX AVG STD <<< "$stats"

# Save detailed timing results
{
    echo "Benchmark Results - $(date)"
    echo "========================================"
    echo "Executable: $EXECUTABLE $COMMAND"
    echo "Iterations: $ITERATIONS"
    echo ""
    echo "Statistics (ms):"
    echo "  Minimum:    $MIN"
    echo "  Maximum:    $MAX"
    echo "  Average:    $AVG"
    echo "  Std Dev:    $STD"
    echo ""
    echo "Detailed Results:"
    echo "Iteration,Timing(ms),GPU_Freq(Hz),MEM_Freq(Hz)"
    for i in "${!timings[@]}"; do
        echo "$((i+1)),${timings[$i]},${gpu_freqs[$i]},${mem_freqs[$i]}"
    done
} > "$OUTPUT_FILE"

echo ""
echo "========================================"
echo "Summary:"
echo "  Minimum:    $MIN ms"
echo "  Maximum:    $MAX ms"
echo "  Average:    $AVG ms"
echo "  Std Dev:    $STD ms"
echo ""
echo "Detailed results saved to: $OUTPUT_FILE"

# Merge all NCU results into one CSV
echo ""
echo "Merging NCU profiling results..."

# Create header with iteration column
first_file=$(ls ${TEMP_NCU_DIR}/run_1.csv 2>/dev/null)
if [ -f "$first_file" ]; then
    # Extract header and add Iteration column
    echo -n "Iteration," > "$NCU_OUTPUT"
    head -n 1 "$first_file" >> "$NCU_OUTPUT"
    
    # Append data from all runs
    for i in $(seq 1 $ITERATIONS); do
        NCU_FILE="${TEMP_NCU_DIR}/run_${i}.csv"
        if [ -f "$NCU_FILE" ]; then
            # Skip header and add iteration number
            tail -n +2 "$NCU_FILE" | sed "s/^/${i},/" >> "$NCU_OUTPUT"
        fi
    done
    
    echo "NCU metrics from all $ITERATIONS runs saved to: $NCU_OUTPUT"
    
    # Clean up temp files
    rm -rf "$TEMP_NCU_DIR"
else
    echo "WARNING: NCU profiling failed or not available"
fi

echo "Benchmark complete!"
