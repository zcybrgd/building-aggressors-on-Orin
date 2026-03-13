#!/bin/bash

# ================================================================
# L2 CACHE CONTENTION with GREEN CONTEXT SM ISOLATION + NCU
# ================================================================
# Victim gets 6 SMs, enemy gets 2 SMs via CUDA Green Contexts.
# SMs are spatially isolated. L2 cache is SHARED.
# No MPS needed — green contexts work within a single process
# address space using the CUDA Driver API.
# ================================================================

echo "========================================================"
echo "  L2 CACHE CONTENTION with GREEN CONTEXT SM ISOLATION"
echo "========================================================"
echo ""

# Compile
echo "Compiling green_victim and green_enemy..."
nvcc -arch=sm_87 -O3 -o green_victim green_victim.cu -lcuda
nvcc -arch=sm_87 -O3 -o green_enemy green_enemy.cu -lcuda
echo "Compilation done"
echo ""

# NCU metrics (L2 contention + SM isolation verification)
NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors_op_read_lookup_miss.sum,lts__t_sectors_op_write_lookup_miss.sum,sm__cycles_elapsed.avg,gpu__time_active.sum,sm__inst_executed.sum,smsp__warps_active.avg,sm__warps_launched.sum"
VICTIM_ITERS=50000

# -----------------------------------------------------------
# SCENARIO 1: Victim Alone (all 8 SMs, no green context)
# -----------------------------------------------------------
echo "========================================================"
echo "  SCENARIO 1: Victim Alone (all SMs, with NCU)"
echo "========================================================"
echo ""

sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_green_victim_alone.log \
    ./green_victim $VICTIM_ITERS 0 \
    > ncu_green_victim_alone.csv 2>&1

echo " Victim alone profiling complete"
echo ""

sleep 2

# -----------------------------------------------------------
# SCENARIO 2: Victim + Enemy (green context SM isolation)
# Victim: 6 SMs (green context, concurrent=1)
# Enemy:  2 SMs (green context, use_green=1)
# -----------------------------------------------------------
echo "========================================================"
echo "  SCENARIO 2: Victim + Enemy (Green Context + NCU)"
echo "  Victim: 6 SMs | Enemy: 2 SMs | L2: SHARED"
echo "========================================================"
echo ""

# Launch enemy with green context (2 SMs), infinite mode
echo "Launching enemy (green context, 2 SMs) in background..."
./green_enemy 0 1 > enemy_green_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

# Wait for enemy to saturate L2
sleep 3

# Profile victim with green context (6 SMs)
echo "Profiling victim (green context, 6 SMs) with NCU..."
sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel --csv \
    --log-file ncu_green_victim_concurrent.log \
    ./green_victim $VICTIM_ITERS 1 \
    > ncu_green_victim_concurrent.csv 2>&1

echo " Victim concurrent profiling complete"

# Kill enemy
echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill $ENEMY_PID 2>/dev/null
sleep 1
kill -9 $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo " Enemy terminated"

echo ""
echo "Enemy output:"
cat enemy_green_log.txt

# -----------------------------------------------------------
# RESULTS
# -----------------------------------------------------------
echo ""
echo "========================================================"
echo "  RESULTS ANALYSIS (Green Context SM Isolation)"
echo "========================================================"
echo ""
echo "SM Layout: Victim=6 SMs | Enemy=2 SMs (green contexts)"
echo "(SMs are spatially isolated, but L2 cache is SHARED)"
echo ""

echo "=== VICTIM ALONE (all SMs) ==="
grep -A 5 "victimKernel" ncu_green_victim_alone.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

echo ""
echo "=== VICTIM + ENEMY (green context, separate SMs) ==="
grep -A 5 "victimKernel" ncu_green_victim_concurrent.csv | grep -E "lts__t_sectors_lookup_miss.sum|lts__t_sector_op_read_hit_rate.pct|sm__cycles_elapsed.avg"

echo ""
echo " Done! Results saved:"
echo "  - ncu_green_victim_alone.csv"
echo "  - ncu_green_victim_concurrent.csv"
echo "  - enemy_green_log.txt"
echo ""
