#!/bin/bash
# MATRIX MULTIPLY CONTENTION EXPERIMENT — ALL KERNELS
# Scenarios:
#   original   — GPUMultiplyMatrix             (1-D, no shared mem, matrix_victim)
#   tile_l2fit — GPUMultiplyMatrixTiled_L2Fit  (tile=32, N~352, ≈ L2)
#   tile_small — GPUMultiplyMatrixTiled_Small  (tile=16, N~128, << L2)
#
# Layout:
#   results_YYYYMMDD_HHMMSS/
#     original/    raw_all_runs.csv  ncu_*.log  enemy_*.log
#     tile_l2fit/  raw_all_runs.csv  ncu_*.log  enemy_*.log
#     tile_small/  raw_all_runs.csv  ncu_*.log  enemy_*.log
#     all_kernels_merged.csv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
PATHS=15
NUM_RUNS=5

NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,sm__cycles_elapsed.avg,gpu__time_active.sum,sm__inst_executed.sum,smsp__warps_active.avg,sm__warps_launched.sum"

# ── Matrix sizes (shared across all kernels) ─────────────────────────────────
MATRIX_SIZES=(240 496 784 1016 1232 1680 2024)

# ── Block configs: only used by the original 1-D kernel ──────────────────────
BLOCK_CONFIGS=(
    "1,1024"
    #"2,512"
    "8,128"
   # "16,64"
    "32,32"
    "256,4"
    "4,128"
   # "8,64"
   # "1024,1"
)

# ── Kernel descriptors ────────────────────────────────────────────────────────
# Format: "kernel_id|subfolder|binary|ncu_kernel_name"
# Tiled kernels have a fixed 2-D block (tile×tile); no block sweep needed.
KERNEL_DESCS=(
    "0|original|matrix_victim|GPUMultiplyMatrix"
    "1|tile_l2fit|matrix_victim_sm|GPUMultiplyMatrixTiled_L2Fit"
    "2|tile_small|matrix_victim_sm|GPUMultiplyMatrixTiled_Small"
)

# ── Compilation ───────────────────────────────────────────────────────────────
echo "======================================================================"
echo " Compiling binaries"
echo "======================================================================"
nvcc -arch=sm_87 -O3 -o "$SCRIPT_DIR/matrix_victim"    "$SCRIPT_DIR/matrix_victim.cu"    -lcuda
nvcc -arch=sm_87 -O3 -o "$SCRIPT_DIR/matrix_victim_sm" "$SCRIPT_DIR/matrix_victim_sm.cu" -lcuda
nvcc -arch=sm_87 -O3 -o "$PARENT_DIR/green_enemy"      "$PARENT_DIR/green_enemy.cu"      -lcuda
cp "$PARENT_DIR/green_enemy" "$SCRIPT_DIR/green_enemy"
echo "Compilation done"
echo ""

mkdir -p "$RESULTS_DIR"

# ── Helper: extract one NCU CSV metric ───────────────────────────────────────
extract_metric() {
    local logfile="$1"
    local metric="$2"
    grep "\"$metric\"" "$logfile" | tail -1 | awk -F'"' '{print $(NF-1)}' | tr -d ' '
}

# ── Helper: write CSV header ──────────────────────────────────────────────────
write_csv_header() {
    local csv="$1"
    echo "kernel_id,matrix_size,block_x,block_y,threads_per_block,num_blocks,memory_MB,scenario,run,gpu_time_ns,l2_read_hit_rate_pct,l2_total_sectors,l2_read_miss_sectors,l2_write_miss_sectors,sm_cycles_avg,sm_inst_executed,smsp_warps_active_avg,sm_warps_launched" > "$csv"
}

# ── Helper: run NCU, extract metrics, append one row to CSV ──────────────────
# $1  csv path
# $2  ncu log path
# $3  full path to binary
# $4  ncu kernel name filter
# $5  run-time args string (passed unquoted — intentional)
# $6‥$14  metadata fields: kid msize bx by tpb nblk mem scen run
profile_and_record() {
    local csv="$1" logfile="$2" binary="$3" kname="$4" run_args="$5"
    local kid="$6" msize="$7" bx="$8" by="$9" tpb="${10}"
    local nblk="${11}" mem="${12}" scen="${13}" run="${14}"

    # shellcheck disable=SC2086  # run_args must split into separate tokens
    sudo $(which ncu) --metrics "$NCU_METRICS" \
        --kernel-name "$kname" \
        --csv \
        --log-file "$logfile" \
        "$binary" $run_args \
        2>/dev/null

    local T RH S RM WM CY IN WA WL
    T=$(extract_metric  "$logfile" "gpu__time_active.sum")
    RH=$(extract_metric "$logfile" "lts__t_sector_op_read_hit_rate.pct")
    S=$(extract_metric  "$logfile" "lts__t_sectors.sum")
    RM=$(extract_metric "$logfile" "lts__t_sectors_op_read_lookup_miss.sum")
    WM=$(extract_metric "$logfile" "lts__t_sectors_op_write_lookup_miss.sum")
    CY=$(extract_metric "$logfile" "sm__cycles_elapsed.avg")
    IN=$(extract_metric "$logfile" "sm__inst_executed.sum")
    WA=$(extract_metric "$logfile" "smsp__warps_active.avg")
    WL=$(extract_metric "$logfile" "sm__warps_launched.sum")

    echo "    [$scen]  Time: ${T} ns | L2 hit: ${RH}% | Miss: ${RM}"
    echo "$kid,$msize,$bx,$by,$tpb,$nblk,$mem,$scen,$run,$T,$RH,$S,$RM,$WM,$CY,$IN,$WA,$WL" >> "$csv"
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — one outer iteration per kernel scenario
# ─────────────────────────────────────────────────────────────────────────────
for KDESC in "${KERNEL_DESCS[@]}"; do
    IFS='|' read -r KID KFOLDER KBINARY KNAME <<< "$KDESC"

    KDIR="$RESULTS_DIR/$KFOLDER"
    mkdir -p "$KDIR"
    RAW_CSV="$KDIR/raw_all_runs.csv"
    write_csv_header "$RAW_CSV"

    KBIN="$SCRIPT_DIR/$KBINARY"

    echo "======================================================================"
    echo " KERNEL: $KFOLDER  (id=$KID)  binary=$KBINARY"
    echo " Output: $KDIR/"
    echo "======================================================================"

    # ── Build combo list for this kernel ─────────────────────────────────────
    # original   → MATRIX_SIZES × BLOCK_CONFIGS   (block sweep)
    # tile_l2fit → MATRIX_SIZES only, block fixed 32×32
    # tile_small → MATRIX_SIZES only, block fixed 16×16
    declare -a COMBOS=()
    if [ "$KID" -eq 0 ]; then
        for MSIZE in "${MATRIX_SIZES[@]}"; do
            for BCONFIG in "${BLOCK_CONFIGS[@]}"; do
                IFS=',' read -r BX BY <<< "$BCONFIG"
                COMBOS+=("$MSIZE,$BX,$BY")
            done
        done
    elif [ "$KID" -eq 1 ]; then
        for MSIZE in "${MATRIX_SIZES[@]}"; do
            COMBOS+=("$MSIZE,32,32")
        done
    else
        for MSIZE in "${MATRIX_SIZES[@]}"; do
            COMBOS+=("$MSIZE,16,16")
        done
    fi

    TOTAL_COMBOS=${#COMBOS[@]}
    COMBO=0

    for ENTRY in "${COMBOS[@]}"; do
        IFS=',' read -r MSIZE BX BY <<< "$ENTRY"
        COMBO=$((COMBO + 1))
        THREADS=$((BX * BY))
        TOTAL_ELEMENTS=$((MSIZE * MSIZE))
        NUM_BLOCKS=$(( (TOTAL_ELEMENTS + THREADS - 1) / THREADS ))
        MEM_MB=$(echo "scale=2; 2 * $TOTAL_ELEMENTS * 8 / 1000000" | bc)
        TAG="m${MSIZE}_b${BX}x${BY}"

        echo "  [$COMBO/$TOTAL_COMBOS] Matrix: ${MSIZE}x${MSIZE} | Block: (${BX},${BY})=${THREADS} | Grid: ${NUM_BLOCKS} | Mem: ${MEM_MB} MB"

        for RUN in $(seq 1 $NUM_RUNS); do
            echo "  --- Run $RUN/$NUM_RUNS ---"

            # ── Build runtime args ────────────────────────────────────────────
            # matrix_victim     args: <N> <BX> <BY> <paths> <concurrent>
            # matrix_victim_sm  args: <kernel_id> <N> <paths> <concurrent>
            if [ "$KID" -eq 0 ]; then
                ALONE_ARGS="$MSIZE $BX $BY $PATHS 0"
                CONC_ARGS="$MSIZE $BX $BY $PATHS 1"
            else
                ALONE_ARGS="$KID $MSIZE $PATHS 0"
                CONC_ARGS="$KID $MSIZE $PATHS 1"
            fi

            # ── ALONE ─────────────────────────────────────────────────────────
            ALONE_LOG="$KDIR/ncu_${TAG}_alone_r${RUN}.log"
            profile_and_record \
                "$RAW_CSV" "$ALONE_LOG" "$KBIN" "$KNAME" \
                "$ALONE_ARGS" \
                "$KID" "$MSIZE" "$BX" "$BY" "$THREADS" "$NUM_BLOCKS" "$MEM_MB" \
                "alone" "$RUN"

            sleep 1

            # ── CONCURRENT (enemy saturating L2 in background) ────────────────
            CONC_LOG="$KDIR/ncu_${TAG}_concurrent_r${RUN}.log"
            "$SCRIPT_DIR/green_enemy" 0 1 \
                > "$KDIR/enemy_${TAG}_r${RUN}.log" 2>&1 &
            ENEMY_PID=$!
            sleep 3   # let enemy fully initialise before victim starts

            profile_and_record \
                "$RAW_CSV" "$CONC_LOG" "$KBIN" "$KNAME" \
                "$CONC_ARGS" \
                "$KID" "$MSIZE" "$BX" "$BY" "$THREADS" "$NUM_BLOCKS" "$MEM_MB" \
                "concurrent" "$RUN"

            kill    $ENEMY_PID 2>/dev/null || true
            sleep 1
            kill -9 $ENEMY_PID 2>/dev/null || true
            wait    $ENEMY_PID 2>/dev/null || true

            sleep 1
        done

        echo ""
        sleep 1
    done

    echo "  → $KFOLDER done.  CSV: $RAW_CSV"
    echo ""
done

# ── Merge all per-kernel CSVs into one top-level file ────────────────────────
MERGED="$RESULTS_DIR/all_kernels_merged.csv"
HEADER_WRITTEN=0
for KDESC in "${KERNEL_DESCS[@]}"; do
    IFS='|' read -r KID KFOLDER _ _ <<< "$KDESC"
    SUB_CSV="$RESULTS_DIR/$KFOLDER/raw_all_runs.csv"
    if [ -f "$SUB_CSV" ]; then
        if [ $HEADER_WRITTEN -eq 0 ]; then
            cat "$SUB_CSV" > "$MERGED"
            HEADER_WRITTEN=1
        else
            tail -n +2 "$SUB_CSV" >> "$MERGED"   # skip repeated header
        fi
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
ORIG_COMBOS=$(( ${#MATRIX_SIZES[@]} * ${#BLOCK_CONFIGS[@]} ))
TILED_COMBOS=${#MATRIX_SIZES[@]}
TOTAL_PROFILE_RUNS=$(( (ORIG_COMBOS + TILED_COMBOS + TILED_COMBOS) * NUM_RUNS * 2 ))

echo "======================================================================"
echo " ALL KERNELS DONE"
echo "======================================================================"
printf " %-14s %s\n" "Results root:"  "$RESULTS_DIR/"
printf " %-14s %s\n" "Subfolders:"    "original/  tile_l2fit/  tile_small/"
printf " %-14s %s\n" "Merged CSV:"    "$MERGED"
printf " %-14s original=%d  tile_l2fit=%d  tile_small=%d\n" \
    "Combos:" "$ORIG_COMBOS" "$TILED_COMBOS" "$TILED_COMBOS"
printf " %-14s %d\n" "Total NCU runs:" "$TOTAL_PROFILE_RUNS"
echo "======================================================================"