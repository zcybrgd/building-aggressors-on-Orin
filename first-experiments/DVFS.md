Modern NVIDIA GPUs dynamically adjust frequencies based on thermal conditions, power limits, and workload characteristics. This feature, known as Dynamic Voltage and Frequency Scaling (DVFS), can impact performance measurements and benchmarking results.

When conducting performance evaluations or benchmarking on NVIDIA GPUs, it's essential to consider the effects of DVFS. Here are some key points to keep in mind:
1. **Stable Environment**: To obtain consistent and reliable performance measurements, it's advisable to run benchmarks in a controlled environment where temperature and power conditions are stable. This can help minimize the impact of DVFS on performance results.
2. **Fixed Clock Speeds**: For benchmarking purposes, you may want to set fixed clock speeds for the GPU. This can be done using NVIDIA's System Management Interface (nvidia-smi) or other tools that allow you to lock the GPU's frequency and voltage settings.
The command to set a fixed clock speed using nvidia-smi is as follows:

```bash
nvidia-smi -lgc <min_clock>,<max_clock>
```
and to reset it back to default:

```bash
nvidia-smi -rgc
```

```bash
nvidia-smi -q -d CLOCK
```


3. **Profiling Tools**: Utilize NVIDIA's profiling tools, such as Nsight Systems and Nsight Compute, to monitor GPU performance metrics, including clock speeds, during benchmark runs. This can help you understand how DVFS affects performance under different workloads.
4. **Multiple Runs**: Conduct multiple runs of your benchmarks to account for variability introduced by DVFS. Analyze the results statistically to identify trends and outliers.
5. **Documentation**: Clearly document the GPU model, driver version, and any DVFS settings used during benchmarking. This information is crucial for reproducibility and comparison with other results.


```bash
# Query available frequencies (requires root/sudo)
nvidia-smi -q -d SUPPORTED_CLOCKS

# Lock to maximum stable frequencies
sudo nvidia-smi -lgc <max_graphics_clock>
sudo nvidia-smi -lmc <max_memory_clock>

# Example for common configurations:
sudo nvidia-smi -lgc 1410  # Lock graphics clock to 1410 MHz
sudo nvidia-smi -lmc 5001  # Lock memory clock to 5001 MHz

# Verify locked state
nvidia-smi -q -d CLOCK

# Reset to default after experiments
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```