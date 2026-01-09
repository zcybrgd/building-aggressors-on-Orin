#!/bin/bash
# test_ptx_changes.sh
# Tests if PTX changes with different configurations

echo "==================================================================="
echo "PTX Change Analysis: Does parameter sweep change compiled code?"
echo "==================================================================="

OUTPUT_DIR="ptx_analysis"
mkdir -p $OUTPUT_DIR

#test configurations matching LS-CAT sweep
MATRICES=(240 496 2024)
BLOCKS=("8,8" "16,16" "32,32" "1,256" "1,1024")

echo ""
echo "Test 1: COMPILE-TIME parameters (constant propagation possible)"
echo "----------------------------------------------------------------"

for matrix in "${MATRICES[@]}"; do
    for block in "${BLOCKS[@]}"; do
        IFS=',' read -r bx by <<< "$block"
        ptx_file="${OUTPUT_DIR}/compile_time_m${matrix}_b${bx}x${by}.ptx"
        echo "  Compiling: MATRIX=$matrix, BLOCK=$bx×$by"
        nvcc -arch=sm_87 -ptx ptx_change_test.cu \
            -DMATRIX_SIZE=$matrix -DBLOCK_X=$bx -DBLOCK_Y=$by \
            -o "$ptx_file" 2>/dev/null
        #extract key metrics from PTX
        if [ -f "$ptx_file" ]; then
            num_regs=$(grep ".maxnreg" "$ptx_file" | head -1 | awk '{print $2}')
            num_insts=$(grep -c "^\s*[a-z]" "$ptx_file")
            echo "Registers: $num_regs, Instructions: $num_insts"
        fi
    done
done

echo ""
echo "Test 2: RUNTIME parameters like LS-CAT"
echo "-------------------------------------------------------------------"
#compile once with runtime parameters (no -D flags)
ptx_runtime="${OUTPUT_DIR}/runtime_params.ptx"
echo "  Compiling with runtime parameters..."
nvcc -arch=sm_87 -ptx ptx_change_test.cu -o "$ptx_runtime" 2>/dev/null
if [ -f "$ptx_runtime" ]; then
    num_regs=$(grep ".maxnreg" "$ptx_runtime" | head -1 | awk '{print $2}')
    num_insts=$(grep -c "^\s*[a-z]" "$ptx_runtime")
    echo "    → Registers: $num_regs, Instructions: $num_insts"
fi
echo ""
echo "Test 3: PTX Diff Analysis"
echo "-------------------------"
# compare first small matrix vs large matrix (compile-time)
ptx_small="${OUTPUT_DIR}/compile_time_m240_b8x8.ptx"
ptx_large="${OUTPUT_DIR}/compile_time_m2024_b32x32.ptx"

if [ -f "$ptx_small" ] && [ -f "$ptx_large" ]; then
    echo "  Comparing PTX: 240×240 (8×8) vs 2024×2024 (32×32)"
    diff_lines=$(diff "$ptx_small" "$ptx_large" | grep -c "^<\|^>")
    if [ $diff_lines -eq 0 ]; then
        echo "PTX IDENTICAL (constant propagation worked!)"
    else
        echo "PTX DIFFERS: $diff_lines lines changed"
        echo "Sample differences:"
        diff "$ptx_small" "$ptx_large" | head -20
    fi
fi

echo ""
echo "Test 4: Compare Compile-time vs Runtime"
echo "---------------------------------------"
ptx_compile="${OUTPUT_DIR}/compile_time_m496_b16x16.ptx"
if [ -f "$ptx_compile" ] && [ -f "$ptx_runtime" ]; then
    echo "  Comparing: Compile-time params vs Runtime params"
    diff_lines=$(diff "$ptx_compile" "$ptx_runtime" | grep -c "^<\|^>")
    if [ $diff_lines -eq 0 ]; then
        echo "PTX IDENTICAL"
    else
        echo "PTX DIFFERS: $diff_lines lines changed"
        echo "Runtime params prevent constant propagation"
    fi
fi
