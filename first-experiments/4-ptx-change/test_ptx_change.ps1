# tests if ptx assembly changes with different runtime configurations
Write-Host "PTX Change Analysis: Does parameter sweep change compiled code?" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

# create output directory for ptx files
$OUTPUT_DIR = "ptx_analysis"
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

# test configs
$MATRICES = @(240, 496, 2024)
$BLOCKS = @("8,8", "16,16", "32,32", "1,256", "1,1024")

Write-Host ""
Write-Host "Test 1: COMPILE-TIME parameters (baked into PTX)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "simulates if matrix/block sizes were #define constants" -ForegroundColor Gray
Write-Host ""

# loop through all matrix x block combinations
foreach ($matrix in $MATRICES) {
    foreach ($block in $BLOCKS) {
        # split block size "8,8" into bx=8, by=8
        $bx, $by = $block -split ','
        # generate unique filename for this config's ptx
        $ptx_file = "$OUTPUT_DIR/compile_time_m${matrix}_b${bx}x${by}.ptx"
        Write-Host "  Compiling: MATRIX=$matrix, BLOCK=${bx}x${by}" -NoNewline
        # compile with -d flags: this bakes values into ptx at compile time
        # -dmatrix_size=240 means compiler replaces matrix_size with literal 240
        $compile_cmd = "nvcc -arch=sm_75 -ptx test_ptx.cu " +
                      "-DMATRIX_SIZE=$matrix -DBLOCK_X=$bx -DBLOCK_Y=$by " +
                      "-o `"$ptx_file`""
        
        Invoke-Expression $compile_cmd 2>$null
        # extract metrics from generated ptx
        if (Test-Path $ptx_file) {
            # count register usage (.maxnreg directive in ptx)
            $num_regs = (Select-String -Path $ptx_file -Pattern "\.maxnreg" | 
                        Select-Object -First 1).Line -replace '.*\.maxnreg\s+(\d+).*','$1'
            # count ptx instructions (lines starting with instruction mnemonics)
            $num_insts = (Select-String -Path $ptx_file -Pattern "^\s*[a-z]").Count
            Write-Host " - Regs: $num_regs, Instructions: $num_insts" -ForegroundColor Green
        } else {
            Write-Host " - FAILED" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "Test 2: RUNTIME parameters" -ForegroundColor Yellow
Write-Host "-------------------------------------------------------------------" -ForegroundColor Yellow

# compile once with no -d flags means sizes are runtime parameters
$ptx_runtime = "$OUTPUT_DIR/runtime_params.ptx"
Write-Host "  Compiling with runtime parameters (NO -D flags)..." -NoNewline
nvcc -arch=sm_75 -ptx test_ptx.cu -o "$ptx_runtime" 2>$null
if (Test-Path $ptx_runtime) {
    # same metrics extraction
    $num_regs = (Select-String -Path $ptx_runtime -Pattern "\.maxnreg" | 
                Select-Object -First 1).Line -replace '.*\.maxnreg\s+(\d+).*','$1'
    $num_insts = (Select-String -Path $ptx_runtime -Pattern "^\s*[a-z]").Count
    Write-Host " - Regs: $num_regs, Instructions: $num_insts" -ForegroundColor Green
    Write-Host "    (this ptx works for all matrix/block sizes)" -ForegroundColor Cyan
} else {
    Write-Host " - FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test 3: PTX Diff Analysis (Compile-time variants)" -ForegroundColor Yellow
Write-Host "-------------------------" -ForegroundColor Yellow
Write-Host "comparing small vs large matrix with compile-time constants" -ForegroundColor Gray
Write-Host ""

# compare ptx for smallest vs largest config
$ptx_small = "$OUTPUT_DIR/compile_time_m240_b8x8.ptx"
$ptx_large = "$OUTPUT_DIR/compile_time_m2024_b32x32.ptx"

if ((Test-Path $ptx_small) -and (Test-Path $ptx_large)) {
    Write-Host "  Comparing: 240x240 (8x8) vs 2024x2024 (32x32)" -NoNewline
    # compare files and count differences
    $diff = Compare-Object -ReferenceObject (Get-Content $ptx_small) `
                          -DifferenceObject (Get-Content $ptx_large)
    if ($diff.Count -eq 0) {
        Write-Host " - PTX IDENTICAL" -ForegroundColor Green
        Write-Host "    (compiler optimized constants away)" -ForegroundColor Cyan
    } else {
        Write-Host " - PTX DIFFERS: $($diff.Count) lines changed" -ForegroundColor Magenta
        Write-Host "    sample differences (first 10 lines):" -ForegroundColor Gray
        $diff | Select-Object -First 10 | ForEach-Object {
            $marker = if ($_.SideIndicator -eq "<=") { "SMALL" } else { "LARGE" }
            Write-Host "      [$marker] $($_.InputObject)" -ForegroundColor DarkGray
        }
    }
}

Write-Host ""
Write-Host "Test 4: Compile-time vs Runtime (KEY TEST)" -ForegroundColor Yellow
Write-Host "---------------------------------------" -ForegroundColor Yellow
Write-Host "does using runtime params change ptx structure?" -ForegroundColor Gray
Write-Host ""

$ptx_compile = "$OUTPUT_DIR/compile_time_m496_b16x16.ptx"

if ((Test-Path $ptx_compile) -and (Test-Path $ptx_runtime)) {
    Write-Host "  Comparing: Compile-time (496, 16x16) vs Runtime params" -NoNewline
    
    $diff = Compare-Object -ReferenceObject (Get-Content $ptx_compile) `
                          -DifferenceObject (Get-Content $ptx_runtime)
    
    if ($diff.Count -eq 0) {
        Write-Host " - PTX IDENTICAL" -ForegroundColor Green
        Write-Host "    runtime parameters don't change ptx" -ForegroundColor Cyan
    } else {
        Write-Host " - PTX DIFFERS: $($diff.Count) lines" -ForegroundColor Magenta
        Write-Host "    likely: runtime uses ld.param instead of constants" -ForegroundColor Yellow
        
        # show key differences (parameter loading)
        Write-Host "    key difference patterns:" -ForegroundColor Gray
        $diff | Where-Object { $_.InputObject -match "ld\.param|mov\.u32.*496|mov\.u32.*16" } |
                Select-Object -First 5 | ForEach-Object {
            Write-Host "      $($_.InputObject)" -ForegroundColor DarkGray
        }
    }
}
