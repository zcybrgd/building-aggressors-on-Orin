param(
    [int]$Iterations = 30  # Default 30, can be overridden
)

# Configuration
$EXECUTABLE = "ncuhead.exe"
$COMMAND = "compute"
$OUTPUT_FILE = "benchmark_results_${Iterations}iter.txt"
$NCU_OUTPUT = "ncu_metrics_${Iterations}iter.csv"
$TEMP_NCU_DIR = "temp_ncu_runs"

# Validate iterations
if ($Iterations -lt 1) {
    Write-Host "ERROR: Iterations must be a positive integer"
    Write-Host "Usage: powershell -File run_benchmark.ps1 -Iterations <number>"
    Write-Host "Example: powershell -File run_benchmark.ps1 -Iterations 50"
    exit 1
}

# NCU metrics to collect
$NCU_METRICS = "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct," +
"smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct," +
"smsp__warp_issue_stalled_misc_per_warp_active.pct," +
"smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active," +
"lts__t_sectors.avg.pct_of_peak_sustained_elapsed," +
"lts__t_sectors_aperture_sysmem_op_read.sum.per_second," +
"lts__t_sectors_op_read.sum.per_second," +
"l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second," +
"lts__t_sectors_op_write.sum.per_second," +
"smsp__inst_executed_op_global_ld.sum," +
"lts__t_requests.sum," +
"lts__t_requests.min," +
"lts__t_requests.max," +
"lts__t_sector_hit_rate.pct," +
"lts__t_bytes.min," +
"lts__t_request_hit_rate.ratio," +
"smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"

# Arrays to store results
$timings = @()
$gpu_freqs = @()

# Create temp directory
New-Item -ItemType Directory -Force -Path $TEMP_NCU_DIR | Out-Null

Write-Host "Starting benchmark: $Iterations iterations with NCU profiling"
Write-Host "Results will be saved to $OUTPUT_FILE"
Write-Host "NCU metrics will be saved to $NCU_OUTPUT"
Write-Host "========================================"

# Run iterations with NCU profiling
for ($i = 1; $i -le $Iterations; $i++) {
    Write-Host "Iteration $i/$Iterations"
    
    # Get GPU frequency
    try {
        $gpu_freq = (nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits).Trim()
        $mem_freq = (nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits).Trim()
    } catch {
        $gpu_freq = "N/A"
        $mem_freq = "N/A"
    }
    
    # Run with NCU profiling
    $NCU_TEMP_FILE = "$TEMP_NCU_DIR\run_$i.csv"
    Write-Host "  Running NCU profiling..."
    & ncu --metrics $NCU_METRICS --csv --log-file $NCU_TEMP_FILE ".\$EXECUTABLE" $COMMAND 2>&1 | Out-Null
    
    # Run without NCU to get clean timing
    $output = & ".\$EXECUTABLE" $COMMAND 2>&1 | Out-String
    
    # Extract timing
    if ($output -match "COMPUTE_KERNEL_TIMING,(\d+\.?\d*)") {
        $timing = $matches[1]
        $timings += [double]$timing
        $gpu_freqs += "$gpu_freq MHz / $mem_freq MHz"
        Write-Host "  Timing: ${timing}ms | GPU: $gpu_freq MHz | MEM: $mem_freq MHz"
    } else {
        Write-Host "  ERROR: Could not extract timing"
    }
}

# Calculate statistics
if ($timings.Count -eq 0) {
    Write-Host "ERROR: No valid timing data collected!"
    exit 1
}

$min = ($timings | Measure-Object -Minimum).Minimum
$max = ($timings | Measure-Object -Maximum).Maximum
$avg = ($timings | Measure-Object -Average).Average
$stddev = [Math]::Sqrt((($timings | ForEach-Object { [Math]::Pow($_ - $avg, 2) } | Measure-Object -Sum).Sum) / $timings.Count)

# Save timing results
$results = @"
Benchmark Results - $(Get-Date)
========================================
Executable: $EXECUTABLE $COMMAND
Iterations: $Iterations

Statistics (ms):
  Minimum:    $($min.ToString("F4"))
  Maximum:    $($max.ToString("F4"))
  Average:    $($avg.ToString("F4"))
  Std Dev:    $($stddev.ToString("F4"))

Detailed Results:
Iteration,Timing(ms),GPU_Freq/MEM_Freq
"@

for ($i = 0; $i -lt $timings.Count; $i++) {
    $results += "`n$($i+1),$($timings[$i]),$($gpu_freqs[$i])"
}

$results | Out-File -FilePath $OUTPUT_FILE -Encoding UTF8

Write-Host ""
Write-Host "========================================"
Write-Host "Summary:"
Write-Host "  Minimum:    $($min.ToString('F4')) ms"
Write-Host "  Maximum:    $($max.ToString('F4')) ms"
Write-Host "  Average:    $($avg.ToString('F4')) ms"
Write-Host "  Std Dev:    $($stddev.ToString('F4')) ms"
Write-Host ""
Write-Host "Detailed results saved to: $OUTPUT_FILE"

# Merge NCU results
Write-Host ""
Write-Host "Merging NCU profiling results..."

$first_file = "$TEMP_NCU_DIR\run_1.csv"
if (Test-Path $first_file) {
    # Read header and add Iteration column
    $header = "Iteration," + (Get-Content $first_file -First 1)
    $header | Out-File -FilePath $NCU_OUTPUT -Encoding UTF8
    
    # Append data from all runs
    for ($i = 1; $i -le $Iterations; $i++) {
        $NCU_FILE = "$TEMP_NCU_DIR\run_$i.csv"
        if (Test-Path $NCU_FILE) {
            $data = Get-Content $NCU_FILE | Select-Object -Skip 1
            foreach ($line in $data) {
                "$i,$line" | Out-File -FilePath $NCU_OUTPUT -Append -Encoding UTF8
            }
        }
    }
    
    Write-Host "NCU metrics from all $Iterations runs saved to: $NCU_OUTPUT"
    
    # Clean up
    Remove-Item -Recurse -Force $TEMP_NCU_DIR
} else {
    Write-Host "WARNING: NCU profiling failed or not available"
}

Write-Host "Benchmark complete!"
