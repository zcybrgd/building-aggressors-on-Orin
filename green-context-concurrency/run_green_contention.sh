#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH

echo "========================================"
echo "  L2 CACHE CONTENTION - Green Contexts"
echo "  ALONE:      Victim = 8 SMs, no restriction"
echo "  CONCURRENT: Victim = 6 SMs, Enemy = 2 SMs"
echo "  SM isolated, L2 shared -> contention"
echo "========================================"
echo ""

echo "Compiling..."
nvcc -arch=sm_87 -O3 -lcuda -o green_victim green_victim.cu
nvcc -arch=sm_87 -O3 -lcuda -o green_enemy  green_enemy.cu
if [ $? -ne 0 ]; then echo "Compilation failed!"; exit 1; fi
echo "Compilation done"
echo ""

NCU_METRICS="lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sector_op_read_hit_rate.ratio,lts__t_sectors.sum,lts__t_sectors_lookup_miss.sum,gpu__time_duration,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct,sm__cycles_elapsed"
VICTIM_ITERS=50000

echo "========================================"
echo "  SCENARIO 1: Victim Alone (8 SMs)"
echo "========================================"
echo ""

sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_alone.log \
    ./green_victim $VICTIM_ITERS 0 \
    > ncu_victim_alone.csv 2>&1

echo " Victim alone profiling complete"
echo ""
sleep 2

echo "========================================"
echo "  SCENARIO 2: Victim(6 SMs) + Enemy(2 SMs)"
echo "  Enemy runs as separate process"
echo "  NCU attaches only to green_victim"
echo "========================================"
echo ""

# Enemy runs as completely separate process — NCU never touches it
echo "Launching enemy (separate process, 2 SMs green context)..."
./green_enemy > enemy_log.txt 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"
sleep 3

echo "Profiling victim with NCU..."
sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_concurrent.log \
    ./green_victim $VICTIM_ITERS 1 \
    > ncu_victim_concurrent.csv 2>&1

echo " Victim concurrent profiling complete"

echo ""
echo "Terminating enemy (PID: $ENEMY_PID)..."
kill $ENEMY_PID 2>/dev/null
wait $ENEMY_PID 2>/dev/null
echo " Enemy terminated"

echo ""
echo "Enemy output:"
cat enemy_log.txt

echo ""
echo "========================================"
echo "  RESULTS ANALYSIS"
echo "========================================"
echo ""
echo "=== VICTIM ALONE (8 SMs) ==="
grep "victimKernel" ncu_victim_alone.log | grep -v "^==" | head -5
echo ""
echo "=== VICTIM + ENEMY (6 SMs vs 2 SMs, shared L2) ==="
grep "victimKernel" ncu_victim_concurrent.log | grep -v "^==" | head -5

echo ""
echo " Done! Results saved:"
echo "  - ncu_victim_alone.csv / .log"
echo "  - ncu_victim_concurrent.csv / .log"
echo ""

