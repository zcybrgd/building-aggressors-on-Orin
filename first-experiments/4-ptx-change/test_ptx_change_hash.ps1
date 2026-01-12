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

# store all hashes for later comparison
$compile_hashes = @{}

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
        $compile_cmd = "nvcc -arch=sm_75 -ptx test_ptx.cu " +
                      "-DMATRIX_SIZE=$matrix -DBLOCK_X=$bx -DBLOCK_Y=$by " +
                      "-o `"$ptx_file`""
        
        Invoke-Expression $compile_cmd 2>$null
        
        # calculate md5 hash instead of counting lines
        if (Test-Path $ptx_file) {
            $hash = (Get-FileHash -Path $ptx_file -Algorithm MD5).Hash
            $compile_hashes["$matrix-$bx-$by"] = $hash
            
            # show first 16 chars of hash
            $hash_short = $hash.Substring(0, 16)
            Write-Host " - MD5: $hash_short..." -ForegroundColor Green
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
    $runtime_hash = (Get-FileHash -Path $ptx_runtime -Algorithm MD5).Hash
    $hash_short = $runtime_hash.Substring(0, 16)
    Write-Host " - MD5: $hash_short..." -ForegroundColor Green
    Write-Host "    (this ptx works for all matrix/block sizes)" -ForegroundColor Cyan
} else {
    Write-Host " - FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test 3: Hash Uniqueness Check (Compile-time)" -ForegroundColor Yellow
Write-Host "-------------------------" -ForegroundColor Yellow
Write-Host "how many unique ptx variants were generated?" -ForegroundColor Gray
Write-Host ""

# get unique hash values
$unique_hashes = $compile_hashes.Values | Select-Object -Unique

Write-Host "  Total configs compiled: $($compile_hashes.Count)" -ForegroundColor White
Write-Host "  Unique PTX hashes: $($unique_hashes.Count)" -ForegroundColor Cyan

if ($unique_hashes.Count -eq 1) {
    Write-Host "  all identical (compiler optimized away constants)" -ForegroundColor Green
} elseif ($unique_hashes.Count -eq $compile_hashes.Count) {
    Write-Host "  all different (each config has unique ptx)" -ForegroundColor Magenta
} else {
    Write-Host "  mixed (some configs share ptx)" -ForegroundColor Yellow
}

# show which configs share the same hash
Write-Host ""
Write-Host "  grouping configs by hash:" -ForegroundColor Gray
$hash_groups = @{}
foreach ($key in $compile_hashes.Keys) {
    $hash = $compile_hashes[$key]
    if (-not $hash_groups.ContainsKey($hash)) {
        $hash_groups[$hash] = @()
    }
    $hash_groups[$hash] += $key
}

foreach ($hash in $hash_groups.Keys) {
    $configs = $hash_groups[$hash]
    $hash_short = $hash.Substring(0, 16)
    if ($configs.Count -gt 1) {
        Write-Host "    hash $hash_short shared by $($configs.Count) configs:" -ForegroundColor Cyan
        $configs | ForEach-Object { Write-Host "      - $_" -ForegroundColor DarkGray }
    }
}

Write-Host ""
Write-Host "Test 4: Compile-time vs Runtime Hash Match" -ForegroundColor Yellow
Write-Host "---------------------------------------" -ForegroundColor Yellow
Write-Host "does runtime ptx match any compile-time ptx?" -ForegroundColor Gray
Write-Host ""

# compare small matrix compile-time vs large matrix compile-time
$hash_small = $compile_hashes["240-8-8"]
$hash_large = $compile_hashes["2024-32-32"]

Write-Host "  compile-time 240x240 (8x8):    $($hash_small.Substring(0,32))..." -ForegroundColor White
Write-Host "  compile-time 2024x2024 (32x32): $($hash_large.Substring(0,32))..." -ForegroundColor White

if ($hash_small -eq $hash_large) {
    Write-Host "  hashes match - compile-time constants don't affect ptx" -ForegroundColor Green
} else {
    Write-Host "  hashes differ - compile-time constants do affect ptx" -ForegroundColor Magenta
}

Write-Host ""

# compare runtime vs one compile-time config
$compare_key = "496-16-16"
if ($compile_hashes.ContainsKey($compare_key) -and (Test-Path $ptx_runtime)) {
    $compile_hash = $compile_hashes[$compare_key]
    
    Write-Host "  compile-time 496x496 (16x16): $($compile_hash.Substring(0,32))..." -ForegroundColor White
    Write-Host "  runtime params:               $($runtime_hash.Substring(0,32))..." -ForegroundColor White
    
    if ($compile_hash -eq $runtime_hash) {
        Write-Host ""
        Write-Host "  hashes match - runtime produces identical ptx" -ForegroundColor Green
        Write-Host "  compiler behavior is consistent" -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "  hashes differ - runtime uses different code path" -ForegroundColor Magenta
        Write-Host "  expected: runtime uses ld.param instructions" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "CONCLUSION FOR LS-CAT:" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

if ($unique_hashes.Count -eq 1) {
    Write-Host "  all compile-time configs produced same ptx" -ForegroundColor Green
    Write-Host "  runtime params will also produce same ptx" -ForegroundColor Green
    Write-Host "  extract ptx once per kernel" -ForegroundColor Green
} else {
    Write-Host "  different configs produced different ptx" -ForegroundColor Magenta
    Write-Host "  this is expected with -d flags (compile-time constants)" -ForegroundColor Yellow
  }
Write-Host ""
