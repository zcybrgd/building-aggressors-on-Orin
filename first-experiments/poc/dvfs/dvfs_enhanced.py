import subprocess
import time
import sys
import json
import csv
from statistics import mean, stdev
from pathlib import Path
import matplotlib.pyplot as plt


class DVFSExperiment:
    def __init__(self, output_dir="dvfs_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def set_gpu_clocks(self, sm_clock=None, mem_clock=None):
        """Set GPU clock frequencies using nvidia-smi"""
        print(f"\n{'='*60}")
        print(f"SETTING GPU CLOCKS:")
        
        #enable persistence mode (required for clock control) in linux but im on windows for now
        #subprocess.run(['nvidia-smi', '-pm', '1'], capture_output=True, check=False)
        
        if sm_clock:
            print(f"  SM Clock: {sm_clock} MHz")
            result = subprocess.run(
                ['nvidia-smi', '-lgc', str(sm_clock)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Warning: Failed to set SM clock")
                print(f"      {result.stderr}")
        
        if mem_clock:
            print(f"  MEM Clock: {mem_clock} MHz")
            result = subprocess.run(
                ['nvidia-smi', '-lmc', str(mem_clock)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Warning: Failed to set MEM clock")
                print(f"      {result.stderr}")
        
        time.sleep(2)  # Let clocks stabilize
        print(f"{'='*60}\n")
    
    def reset_gpu_clocks(self):
        """Reset GPU clocks to default"""
        print(f"\n{'='*60}")
        print(f"RESETTING GPU CLOCKS TO DEFAULT")
        print(f"{'='*60}\n")
        subprocess.run(['nvidia-smi', '-rgc'], capture_output=True)
        subprocess.run(['nvidia-smi', '-rmc'], capture_output=True)
        time.sleep(2)
    
    def get_current_clocks(self):
        """Get current GPU clock frequencies"""
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=clocks.sm,clocks.mem',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = [x.strip() for x in result.stdout.strip().split(',')]
            return int(parts[0]), int(parts[1])
        return None, None
    
    def run_benchmark(self, reps=10, iters=4000, label="default"):
        """Run kernel benchmark and return statistics"""
        times = []
        print(f"\nRunning benchmark: {label}")
        print(f"  {reps} iterations × {iters} compute ops\n")
        
        for i in range(reps):
            result = subprocess.run(
                ["dvfs.exe", str(iters)],
                capture_output=True, text=True, check=True
            )
            t_ms = float(result.stdout.strip())
            times.append(t_ms)
            print(f"  Run {i+1:2d}: {t_ms:.4f} ms")
            time.sleep(0.1)
        
        avg = mean(times)
        std = stdev(times) if len(times) > 1 else 0
        min_t = min(times)
        max_t = max(times)
        
        print(f"\n  Average: {avg:.4f} ms ± {std:.4f}")
        print(f"  Min:     {min_t:.4f} ms")
        print(f"  Max:     {max_t:.4f} ms\n")
        
        return {
            'label': label,
            'times': times,
            'avg': avg,
            'std': std,
            'min': min_t,
            'max': max_t
        }
    
    def run_clock_sweep(self, sm_clocks, mem_clock_fixed=None, 
                       reps=10, iters=4000):
        """Sweep through different SM clock rates"""
        print(f"\n{'#'*60}")
        print(f"STARTING CLOCK SWEEP EXPERIMENT")
        print(f"{'#'*60}\n")
        
        if mem_clock_fixed:
            print(f"Fixed MEM clock: {mem_clock_fixed} MHz")
        print(f"SM clock sweep: {sm_clocks}\n")
        
        for sm_clock in sm_clocks:
            # Set clocks
            self.set_gpu_clocks(sm_clock=sm_clock, 
                               mem_clock=mem_clock_fixed)
            
            # Verify actual clocks
            actual_sm, actual_mem = self.get_current_clocks()
            
            # Run benchmark
            label = f"SM_{sm_clock}MHz"
            if mem_clock_fixed:
                label += f"_MEM_{mem_clock_fixed}MHz"
            
            result = self.run_benchmark(reps=reps, iters=iters, label=label)
            result['requested_sm_clock'] = sm_clock
            result['actual_sm_clock'] = actual_sm
            result['requested_mem_clock'] = mem_clock_fixed
            result['actual_mem_clock'] = actual_mem
            
            self.results.append(result)
        
        # Reset to default
        self.reset_gpu_clocks()
    
    def save_results(self):
        """Save results to JSON and CSV"""
        # JSON (full data)
        json_path = self.output_dir / "dvfs_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved JSON: {json_path}")
        
        # CSV (summary)
        csv_path = self.output_dir / "dvfs_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Label', 'SM_Clock_Req', 'SM_Clock_Actual', 
                           'MEM_Clock_Req', 'MEM_Clock_Actual',
                           'Avg_Time_ms', 'Std_Dev_ms', 'Min_ms', 'Max_ms'])
            for r in self.results:
                writer.writerow([
                    r['label'],
                    r['requested_sm_clock'],
                    r['actual_sm_clock'],
                    r['requested_mem_clock'],
                    r['actual_mem_clock'],
                    f"{r['avg']:.4f}",
                    f"{r['std']:.4f}",
                    f"{r['min']:.4f}",
                    f"{r['max']:.4f}"
                ])
        print(f"Saved CSV: {csv_path}")
    
    def plot_results(self):
        """Generate plots"""
        if not self.results:
            print("No results to plot")
            return
        
        sm_clocks = [r['actual_sm_clock'] for r in self.results]
        avg_times = [r['avg'] for r in self.results]
        std_times = [r['std'] for r in self.results]
        
        # Plot 1: Clock vs Performance
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Execution time vs clock
        ax1.errorbar(sm_clocks, avg_times, yerr=std_times, 
                    fmt='o-', linewidth=2, markersize=8,
                    capsize=5, capthick=2, color='#2ecc71')
        ax1.set_xlabel('SM Clock Frequency (MHz)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Kernel Execution Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('DVFS Impact on Kernel Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Performance (inverted - higher is better)
        throughput = [1000.0 / t for t in avg_times]  # ops/sec (arbitrary unit)
        ax2.plot(sm_clocks, throughput, 'o-', linewidth=2, 
                markersize=8, color='#e74c3c')
        ax2.set_xlabel('SM Clock Frequency (MHz)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Throughput (ops/sec)', fontsize=12, fontweight='bold')
        ax2.set_title('Throughput vs Clock Frequency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "dvfs_performance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close()
        
        # Plot 2: Speedup analysis
        if len(self.results) > 1:
            baseline_time = self.results[0]['avg']
            speedups = [baseline_time / r['avg'] for r in self.results]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sm_clocks, speedups, 'o-', linewidth=2.5, 
                   markersize=10, color='#3498db')
            ax.axhline(y=1.0, color='red', linestyle='--', 
                      linewidth=2, label='Baseline')
            ax.set_xlabel('SM Clock Frequency (MHz)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Speedup vs Lowest Clock', fontsize=12, fontweight='bold')
            ax.set_title('Clock Frequency Scaling Efficiency', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            plot_path = self.output_dir / "dvfs_speedup.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_path}")
            plt.close()


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python dvfs_experiment.py <mode>")
        print("\nModes:")
        print("  quick    - Quick test (3 clock frequencies)")
        print("  full     - Full sweep (10+ frequencies)")
        print("  custom   - Custom clock list")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # GTX 1650 Ti typical clock ranges (check your GPU's specific range)
    if mode == "quick":
        sm_clocks = [300, 900, 1500]  # Low, Medium, High
    elif mode == "full":
        sm_clocks = list(range(300, 1800, 150))  # 300 to 1785 MHz in 150 MHz steps
    elif mode == "custom":
        # Example: python dvfs_experiment.py custom 600,900,1200,1500
        if len(sys.argv) < 3:
            print("Error: Provide comma-separated clock list")
            print("Example: python dvfs_experiment.py custom 600,900,1200,1500")
            sys.exit(1)
        sm_clocks = [int(c) for c in sys.argv[2].split(',')]
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
    
    print(f"\n{'#'*60}")
    print(f"  DVFS PERFORMANCE CHARACTERIZATION")
    print(f"  GTX 1650 Ti - SM Clock Sweep")
    print(f"{'#'*60}\n")
    
    exp = DVFSExperiment()
    
    try:
        exp.run_clock_sweep(
            sm_clocks=sm_clocks,
            mem_clock_fixed=None,  
            reps=10,
            iters=4000
        )
        
        exp.save_results()
        exp.plot_results()
        
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT COMPLETE")
        print(f"  Results saved in: {exp.output_dir}/")
        print(f"{'#'*60}\n")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        exp.reset_gpu_clocks()
    except Exception as e:
        print(f"\nError: {e}")
        exp.reset_gpu_clocks()
        raise


if __name__ == "__main__":
    main()
