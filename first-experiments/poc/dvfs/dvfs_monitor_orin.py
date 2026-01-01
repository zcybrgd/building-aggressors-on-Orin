import subprocess
import time
import sys
from statistics import mean
from pathlib import Path

def get_gpu_freq():
    """Get current GPU frequency on Jetson"""
    try:
        with open("/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq", 'r') as f:
            freq_hz = int(f.read().strip())
            return freq_hz // 1000000  # Convert to MHz
    except:
        return None

def get_mem_freq():
    """Get current memory frequency on Jetson"""
    try:
        result = subprocess.run(
            ['sudo', 'cat', '/sys/kernel/debug/bpmp/debug/clk/emc/rate'],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            freq_hz = int(result.stdout.strip())
            return freq_hz // 1000000  # Convert to MHz
    except:
        pass
    return None

def get_gpu_temp():
    """Get GPU temperature on Jetson"""
    try:
        with open("/sys/devices/virtual/thermal/thermal_zone0/temp", 'r') as f:
            temp_millidegrees = int(f.read().strip())
            return temp_millidegrees / 1000.0  # Convert to Celsius
    except:
        return None

def run_kernel(reps=10, iters=4000):
    times = []
    print(f"\nRunning {reps} iterations with {iters} compute iterations per kernel...\n")
    
    for i in range(reps):
        result = subprocess.run(
            ["./dvfs", str(iters)],
            capture_output=True, text=True, check=True
        )
        t_ms = float(result.stdout.strip())
        times.append(t_ms)
        
        # Show GPU freq during run
        gpu_freq = get_gpu_freq()
        freq_str = f" (GPU: {gpu_freq} MHz)" if gpu_freq else ""
        print(f"  Run {i+1:2d}: {t_ms:.4f} ms{freq_str}")
        
        time.sleep(0.1)
    
    avg = mean(times)
    std = (sum((t - avg)**2 for t in times) / len(times))**0.5 if len(times) > 1 else 0
    
    print(f"\n{'='*50}")
    print(f"KERNEL TIMING SUMMARY:")
    print(f"  Average: {avg:.4f} ms ± {std:.4f}")
    print(f"  Min:     {min(times):.4f} ms")
    print(f"  Max:     {max(times):.4f} ms")
    print(f"  CV:      {(std/avg*100):.2f}%")
    print(f"{'='*50}\n")

def monitor_gpu(duration_s=60, interval_s=2):
    print(f"\n{'='*60}")
    print(f"GPU MONITORING STARTED (Duration: {duration_s}s, Interval: {interval_s}s)")
    print(f"{'='*60}\n")
    
    t_end = time.time() + duration_s
    
    gpu_freqs, mem_freqs, temps = [], [], []
    
    print(f"{'Time':>8s} | {'GPU Freq':>10s} | {'MEM Freq':>10s} | {'Temp':>8s}")
    print("-" * 50)
    
    start_time = time.time()
    
    while time.time() < t_end:
        elapsed = time.time() - start_time
        
        gpu_freq = get_gpu_freq()
        mem_freq = get_mem_freq()
        temp = get_gpu_temp()
        
        # Collect data
        if gpu_freq:
            gpu_freqs.append(gpu_freq)
        if mem_freq:
            mem_freqs.append(mem_freq)
        if temp:
            temps.append(temp)
        
        # Display current values
        gpu_str = f"{gpu_freq:4d} MHz" if gpu_freq else "N/A"
        mem_str = f"{mem_freq:4d} MHz" if mem_freq else "N/A"
        temp_str = f"{temp:5.1f}°C" if temp else "N/A"
        
        print(f"{elapsed:7.1f}s | {gpu_str:>10s} | {mem_str:>10s} | {temp_str:>8s}")
        
        time.sleep(interval_s)
    
    print(f"\n{'='*60}")
    print("MONITORING SUMMARY:")
    if gpu_freqs:
        print(f"  GPU Freq:  {mean(gpu_freqs):7.1f} MHz (min: {min(gpu_freqs)}, max: {max(gpu_freqs)})")
    if mem_freqs:
        print(f"  MEM Freq:  {mean(mem_freqs):7.1f} MHz (min: {min(mem_freqs)}, max: {max(mem_freqs)})")
    if temps:
        print(f"  Temp:      {mean(temps):7.1f}°C (min: {min(temps):.1f}, max: {max(temps):.1f})")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3 dvfs_monitor_orin.py [monitor|benchmark]")
        print("\nExamples:")
        print("  python3 dvfs_monitor_orin.py monitor              # Monitor GPU for 60s")
        print("  python3 dvfs_monitor_orin.py monitor 120          # Monitor for 120s")
        print("  python3 dvfs_monitor_orin.py benchmark            # Run kernel 10 times")
        print("  python3 dvfs_monitor_orin.py benchmark 20 8000    # 20 runs, 8000 iters")
        print("\nNote: Memory frequency monitoring requires sudo")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "monitor":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        monitor_gpu(duration_s=duration, interval_s=2)
    elif mode == "benchmark":
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        iters = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
        run_kernel(reps=reps, iters=iters)
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'monitor' or 'benchmark'")
        sys.exit(1)
