#!/bin/bash

#setup environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

#compile programs
echo "Compiling victim and enemy..."
nvcc -arch=sm_87 -O3 -o victim_process victim_process.cu
nvcc -arch=sm_87 -O3 -o enemy_process enemy_process.cu
echo ""

#start MPS
./setup_mps.sh

# NCU metrics to collect
NCU_METRICS="lts__t_sectors_lookup_miss.sum,lts__t_sector_op_read_hit_rate.pct,dram__bytes_read,sm__cycles_elapsed.avg,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct,gpu__time_duration"

echo "========================================"
echo "  SCENARIO 1: Victim Alone (with NCU)"
echo "========================================"
echo ""

sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_alone.log \
    ./victim_process 5000 \
    > ncu_victim_alone.csv 2>&1

echo "Victim alone profiling complete"
echo ""

sleep 2

echo "========================================"
echo "  SCENARIO 2: Victim + Enemy (MPS + NCU)"
echo "========================================"
echo ""

echo "Launching enemy in background..."
./enemy_process 0 > enemy_log.txt 2>&1 &  # 0 = infinite mode
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"

#give enemy time to warm up and saturate L2
sleep 3

#launch victim with NCU profiling
echo "Profiling victim with NCU (while enemy runs)..."

sudo $(which ncu) --metrics $NCU_METRICS \
    --kernel-name victimKernel \
    --csv \
    --log-file ncu_victim_concurrent.log \
    ./victim_process 5000 \
    > ncu_victim_concurrent.csv 2>&1

echo "Victim concurrent profiling complete"

# NOW we kill the enemy because victim is done
echo ""
echo "Killing enemy process (PID: $ENEMY_PID)..."
kill -9 $ENEMY_PID
wait $ENEMY_PID 2>/dev/null

echo "Enemy terminated"

echo ""
echo "Enemy output:"
cat enemy_log.txt

#to be removed later
echo ""
echo "========================================"
echo "  RESULTS COMPARISON"
echo "========================================"
echo ""

# Parse NCU results
echo "Victim Alone Metrics:"
grep "lts__t_sectors_lookup_miss.sum" ncu_victim_alone.csv | tail -1
grep "lts__t_sector_op_read_hit_rate.pct" ncu_victim_alone.csv | tail -1

echo ""
echo "Victim + Enemy Metrics:"
grep "lts__t_sectors_lookup_miss.sum" ncu_victim_concurrent.csv | tail -1
grep "lts__t_sector_op_read_hit_rate.pct" ncu_victim_concurrent.csv | tail -1

#stop MPS
echo ""
echo "Stopping MPS..."
echo quit | sudo nvidia-cuda-mps-control

#cleanup
rm -rf $CUDA_MPS_PIPE_DIRECTORY
rm -rf $CUDA_MPS_LOG_DIRECTORY
echo "Done!"