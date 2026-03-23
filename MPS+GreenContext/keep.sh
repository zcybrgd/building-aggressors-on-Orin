# Start MPS
sudo bash ~/building-aggressors-on-Orin/MPS-experiments/setup_mps.sh

# Then the full sequence:
./green_enemy 0 1 > enemy_nsys.log 2>&1 &
ENEMY_PID=$!
echo "Enemy PID: $ENEMY_PID"
sleep 3
echo "Enemy still running? $(kill -0 $ENEMY_PID 2>&1 || echo DEAD)"


env PATH="$PATH" nsys profile   --trace=cuda-sw,osrt   --output="./results_nsys/ncu_under_nsys"   --force-overwrite=true  sudo -S "$(which ncu)"     --metrics "gpu__time_active.sum,lts__t_sector_op_read_hit_rate.pct"     --kernel-name GPUMultiplyMatrix     --csv     ./matrix_victim 784 32 32 15 1

env PATH="$PATH" nsys profile \
  --trace=cuda-sw,osrt \
  --output="./results_nsys/ncu_under_nsys" \
  --force-overwrite=true \
  bash run_experiment.sh

echo "ORINpfe2026" | sudo env PATH="$PATH" nsys profile \
  --trace=cuda-sw,osrt \
  --output="./results_nsys/ncu_under_nsys" \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  bash run_experiment.sh


kill $ENEMY_PID

echo quit | sudo sh -c \
    "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY \
     nvidia-cuda-mps-control" 2>/dev/null || true
