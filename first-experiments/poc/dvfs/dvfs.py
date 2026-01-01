import subprocess
import time
import sys
from statistics import mean

def run_kernel(reps=10, iters=4000):
    times = []
    print(f"\nRunning {reps} iterations with {iters} compute iterations per kernel...\n")
    for i in range(reps):
        result = subprocess.run(
            ["dvfs.exe", str(iters)],
            capture_output=True, text=True, check=True
        )
        t_ms = float(result.stdout.strip())
        times.append(t_ms)
        print(f"  Run {i+1:2d}: {t_ms:.4f} ms")
        time.sleep(0.1)
    
    print(f"\n{'='*50}")
    print(f"KERNEL TIMING SUMMARY:")
    print(f"  Average: {mean(times):.4f} ms")
    print(f"  Min:     {min(times):.4f} ms")
    print(f"  Max:     {max(times):.4f} ms")
    print(f"{'='*50}\n")

def monitor_gpu(duration_s=60, interval_s=2):
    print(f"\n{'='*60}")
    print(f"GPU MONITORING STARTED (Duration: {duration_s}s, Interval: {interval_s}s)")
    print(f"{'='*60}\n")
    
    t_end = time.time() + duration_s
    
    sm_clocks, mem_clocks, temps, powers = [], [], [], []
    
    while time.time() < t_end:
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = [x.strip() for x in line.split(',')]
            
            try:
                sm = int(parts[0])
                mem = int(parts[1])
                temp = int(parts[2])
                try:
                    pwr = float(parts[3])
                except:
                    pwr = float('nan')
                
                sm_clocks.append(sm)
                mem_clocks.append(mem)
                temps.append(temp)
                powers.append(pwr)
                
                print(f"SM {sm:4d} MHz | MEM {mem:4d} MHz | TEMP {temp:2d}°C | PWR {pwr:5.1f}W")
            except (ValueError, IndexError) as e:
                print(f"Error parsing nvidia-smi output: {e}")
        else:
            print("nvidia-smi query failed")
        
        time.sleep(interval_s)
    
    print(f"\n{'='*60}")
    print("MONITORING SUMMARY:")
    if sm_clocks:
        print(f"  SM clock:  {mean(sm_clocks):7.1f} MHz")
        print(f"  MEM clock: {mean(mem_clocks):7.1f} MHz")
        print(f"  Temp:      {mean(temps):7.1f} °C")
        valid_powers = [p for p in powers if p == p]  
        if valid_powers:
            print(f"  Power:     {mean(valid_powers):7.1f} W")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python dvfs.py [monitor|benchmark]")
        print("\nExamples:")
        print("  python dvfs.py monitor      # Monitor GPU clocks/temp for 60s")
        print("  python dvfs.py benchmark    # Run kernel benchmark 10 times")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "monitor":
        monitor_gpu(duration_s=600, interval_s=2)
    elif mode == "benchmark":
        run_kernel(reps=int(sys.argv[2]) if len(sys.argv) > 2 else 10, iters= int(sys.argv[3]) if len(sys.argv) > 3 else 4000)
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'monitor' or 'benchmark'")
