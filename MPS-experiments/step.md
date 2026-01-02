# Steps to configure NVIDIA MPS on Orin Nano 
This guide provides step-by-step instructions to configure NVIDIA Multi-Process Service (MPS) on an NVIDIA Orin Nano device. MPS allows multiple CUDA applications to share a single GPU context, improving resource utilization and performance.

## Check if it's installed 
```bash
which nvidia-cuda-mps-control
```

## Next it may not be configured so run : 
```bash
#create the MPS folder
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
# launch the MPS control daemon
sudo -E nvidia-cuda-mps-control -d
# verify that it's running
ps aux | grep mps
```


## i have automated this in a bash script called setup_mps.sh which you can run to setup MPS quickly.

```bash
chmod +x setup_mps.sh run_mps_contention.sh
./run_mps_contention.sh
```