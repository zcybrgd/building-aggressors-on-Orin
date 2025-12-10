#!/usr/bin/env python3
"""
NCU Overhead Analysis - Proof of Concept
Measures the overhead introduced by NCU profiling as metrics are added one-by-one
"""

import subprocess
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import time
import shutil

#comprehensive metric list (we add one by one to find threshold)
METRICS_LIST = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.sum",
    "sm__warps_active.avg",
    "sm__inst_executed.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_lookup_miss.sum",
    "lts__t_requests_op_read_lookup_miss.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "lts__t_sectors.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__warps_launched.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
]

KERNEL_MAP = {
    'compute': 'computeKernel',
    'memory': 'memoryKernel'
}

class NCUOverheadAnalyzer:
    def __init__(self, executable_path, kernel_type):
        self.executable = Path(executable_path)
        if not self.executable.exists():
            raise FileNotFoundError(f"Executable not found: {executable_path}")
        if kernel_type not in KERNEL_MAP:
            raise ValueError(f"Invalid kernel type: {kernel_type}. Choose 'compute' or 'memory'")
        self.kernel_type = kernel_type
        self.kernel_name = KERNEL_MAP[kernel_type]
        self.results = []
        self.baseline_time = None
        
    def run_baseline(self):
        """Run without NCU profiling to get baseline"""
        print(f"\n{'='*60}")
        print(f"RUNNING BASELINE (no NCU) - {self.kernel_type.upper()} KERNEL")
        result = subprocess.run([str(self.executable), self.kernel_type],capture_output=True,text=True,shell=True)
        timing_key = f"{self.kernel_type.upper()}_KERNEL_TIMING"
        for line in result.stdout.split('\n'):
            if timing_key in line:
                parts = line.split(',')
                self.baseline_time = {
                    'cuda_events': float(parts[1])
                }
                break
        if not self.baseline_time:
            raise RuntimeError("Failed to parse baseline timing")
        print(f"CUDA Events:  {self.baseline_time['cuda_events']:.4f} ms")
        return self.baseline_time
    
    
    def run_with_metrics(self, metrics_list):
        """Run NCU with specified metrics"""
        metrics_str = ','.join(metrics_list)
        num_metrics = len(metrics_list)
        print(f"  [{num_metrics:2d} metrics] Profiling...", end=' ', flush=True)
        
        # ensure ncu is available
        ncu_exe = shutil.which('ncu')
        if ncu_exe is None:
            print('\n[ERROR] Nsight Compute CLI `ncu` not found in PATH. Provide its full path or add it to PATH.')
            print('  set PATH=%PATH%;"C:\\Program Files\\NVIDIA Corporation\\Nsight Compute <version>"')
            return None

        # constructing the ncu command
        cmd = [
            ncu_exe, 
            "--launch-skip", "1",  #skip the warmup run
            "--kernel-name", self.kernel_name,
            "--metrics", metrics_str,
            str(self.executable),
            self.kernel_type
        ]

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            total_profiling_time = time.time() - start_time
            num_passes = self._parse_num_passes(result.stdout)

            #if num_metrics <= :
            #    print(f"\n{'='*80}")
            #    print(f"[DEBUG] FULL STDERR for {num_metrics} metrics:")
            #    print(result.stderr)
            #    print(f"\n[DEBUG] FULL STDOUT for {num_metrics} metrics:")
            #    print(result.stdout)
            #    print(f"{'='*80}\n")
            
            #extract kernel duration from the program's stdout
            kernel_time_with_ncu_ms = self._parse_kernel_duration_from_output(result.stdout)
            
            if kernel_time_with_ncu_ms is None:
                kernel_time_with_ncu_ms = self.baseline_time['cuda_events']
                print(f"(using baseline)", end=' ')
            
            #calculate overhead: kernel time WITH NCU - kernel time WITHOUT NCU (baseline)
            baseline_ms = self.baseline_time['cuda_events']
            ncu_overhead_ms = kernel_time_with_ncu_ms - baseline_ms
            overhead_percentage = (ncu_overhead_ms / baseline_ms) * 100 if baseline_ms > 0 else 0
            
            print(f"→ {num_passes} pass(es), Kernel: {kernel_time_with_ncu_ms:.4f}ms")
            
            return {
                'num_metrics': num_metrics,
                'num_passes': num_passes,
                'kernel_time_with_ncu_ms': kernel_time_with_ncu_ms,
                'baseline_ms': baseline_ms,
                'ncu_overhead_ms': ncu_overhead_ms,
                'overhead_percentage': overhead_percentage,
                'total_profiling_time': total_profiling_time,
                'metrics': metrics_list.copy()
            }
            
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT]")
            return None
        except FileNotFoundError as e:
            print(f"[ERROR] Executable not found when running ncu: {e}")
            print("Make sure `ncu` is installed and available in PATH, or provide full path.")
            return None
        except Exception as e:
            print(f"[ERROR: {str(e)}]")
            return None
    
    def _parse_num_passes(self, ncu_output):
        """Extract number of passes from NCU output (stdout)"""
        # Pattern: ==PROF== Profiling "computeKernel": 0%....50%....100% - 1 pass
        # or: ==PROF== Profiling "computeKernel": 0%....50%....100% - 3 passes
        match = re.search(r'-\s*(\d+)\s*pass', ncu_output)
        if match:
            return int(match.group(1))
        return 1
    
    def _parse_kernel_duration_from_output(self, program_stdout):
        """Extract kernel duration from the program's own output (CUDA events timing)"""
        # Look for: COMPUTE_KERNEL_TIMING,565.5785 or MEMORY_KERNEL_TIMING,12.345
        timing_key = f"{self.kernel_type.upper()}_KERNEL_TIMING"
        
        for line in program_stdout.split('\n'):
            if timing_key in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        pass
        return None
    
    def run_incremental_analysis(self, max_metrics=None):
        """Add metrics one by one and measure overhead"""
        if max_metrics is None:
            max_metrics = len(METRICS_LIST)
        
        print(f"\n{'='*60}")
        print(f"INCREMENTAL ANALYSIS: {self.kernel_type.upper()} KERNEL")
        print(f"{'='*60}\n")
        
        results = []
        
        for i in range(1, min(max_metrics + 1, len(METRICS_LIST) + 1)):
            metrics_subset = METRICS_LIST[:i]
            result = self.run_with_metrics(metrics_subset)
            
            if result:
                results.append(result)
                if i > 1 and result['num_passes'] > results[-2]['num_passes']:
                    print(f"PASS TRANSITION DETECTED at {i} metrics!")
            else:
                print(f"Stopping analysis due to failure")
                break
        
        self.results = results
        return results
    
    def generate_report(self, output_dir="."):
        """Generate comprehensive report with graphs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        report_data = {
            'kernel_type': self.kernel_type,
            'kernel_name': self.kernel_name,
            'baseline': self.baseline_time,
            'profiling_results': self.results
        }
        
        json_path = output_dir / f'ncu_overhead_{self.kernel_type}.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nJSON saved: {json_path}")
        self._plot_combined(output_dir)
        self._write_summary_report(output_dir)
        print(f"Report generated in: {output_dir}")
      
    
    def _plot_combined(self, output_dir):
        """Create combined plot: metrics vs passes AND metrics vs kernel execution time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        if not self.results:
            return
        
        num_metrics = [r['num_metrics'] for r in self.results]
        num_passes = [r['num_passes'] for r in self.results]
        kernel_times = [r['kernel_time_with_ncu_ms'] for r in self.results]
        baseline = self.baseline_time['cuda_events']
        
        # Plot 1: Metrics vs Passes (Staircase)
        ax1.step(num_metrics, num_passes, where='post', linewidth=2.5, 
                label='NCU Passes', color='#e74c3c')
        ax1.scatter(num_metrics, num_passes, s=80, color='#c0392b', 
                   zorder=5, alpha=0.7)
        
        ax1.set_xlabel('Number of Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of NCU Passes', fontsize=12, fontweight='bold')
        ax1.set_title(f'Metrics vs Passes\n{self.kernel_name}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Annotate pass transitions
        for i in range(1, len(num_passes)):
            if num_passes[i] > num_passes[i-1]:
                ax1.annotate(f'Pass ↑',
                           xy=(num_metrics[i], num_passes[i]),
                           xytext=(num_metrics[i] + 0.5, num_passes[i] + 0.2),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           fontsize=9, color='red', fontweight='bold')
        
        # Plot 2: Metrics vs Kernel Execution Time
        ax2.plot(num_metrics, kernel_times, 'o-', linewidth=2.5, markersize=8,
                color='#2ecc71', label='Kernel Time with NCU')
        
        # Add baseline reference line
        ax2.axhline(y=baseline, color='#3498db', linestyle='--', linewidth=2,
                   label=f'Baseline (no NCU): {baseline:.2f}ms')
        
        ax2.set_xlabel('Number of Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Kernel Execution Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Metrics vs Kernel Execution Time\n{self.kernel_name}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = output_dir / f'ncu_overhead_{self.kernel_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.close()
    
    def _write_summary_report(self, output_dir):
        """Write text summary report"""
        report_path = output_dir / f'ncu_overhead_{self.kernel_type}_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
      
            f.write(f"NCU OVERHEAD ANALYSIS - {self.kernel_type.upper()} KERNEL\n")
            f.write("BASELINE (No NCU Profiling)\n")
            f.write("-" * 70 + "\n")
            f.write(f"CUDA Events:  {self.baseline_time['cuda_events']:.4f} ms\n")
            f.write("\n\n")
            if not self.results:
                f.write("No profiling results recorded (profiling failed or ncu not found).\n")
                f.write("\n")
            else:
                f.write("INCREMENTAL PROFILING RESULTS\n")
                f.write("="*90 + "\n")
                f.write(f"{'Metrics':<10} {'Passes':<10} {'Kernel Time (ms)':<20}\n")
                f.write("-" * 90 + "\n")

                prev_passes = 0
                for r in self.results:
                    marker = " " if r['num_passes'] > prev_passes and prev_passes > 0 else "   "
                    f.write(f"{marker}{r['num_metrics']:<7} {r['num_passes']:<10} "
                           f"{r['kernel_time_with_ncu_ms']:<20.4f}\n")
                    prev_passes = r['num_passes']
            f.write("\n\nPASS THRESHOLDS DETECTED\n")
            f.write("="*70 + "\n")
            if not self.results or len(self.results) < 2:
                f.write("  No pass transitions detected (insufficient data).\n")
            else:
                prev_passes = self.results[0]['num_passes']
                transition_count = 0
                for r in self.results[1:]:
                    if r['num_passes'] > prev_passes:
                        transition_count += 1
                        f.write(f"  {transition_count}. At {r['num_metrics']} metrics: "
                               f"{prev_passes} -> {r['num_passes']} passes\n")
                        prev_passes = r['num_passes']
                if transition_count == 0:
                    f.write("  No pass transitions detected in tested range.\n")
            f.write("\n\nBASELINE REFERENCE\n")
            f.write("="*70 + "\n")
            f.write(f"Baseline (no NCU): {self.baseline_time['cuda_events']:.4f} ms\n")
        
        print(f"Summary saved: {report_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python ncuhead.py <executable> <kernel_type> [max_metrics]")
        print("  kernel_type: 'compute' or 'memory'")
        print("  max_metrics: optional, default 20") #just as an initial test and then i'll expand it per sections per options
        #i know for a fact that to collect all the metrics set you need 30 passes (atleast on my gpu : ) )
        print("\nExample:")
        #we are on windows so the executable is ncuhead.exe not ./ncuhead
        print("  python ncuhead.py ncuhead.exe compute 20")
        sys.exit(1)
    
    executable = sys.argv[1]
    kernel_type = sys.argv[2]
    max_metrics = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     NCU OVERHEAD PROOF OF CONCEPT ANALYZER                ║
    ╚═══════════════════════════════════════════════════════════╝""")
    
    try:
        analyzer = NCUOverheadAnalyzer(executable, kernel_type)
        # Step 1: we run the baseline with no nsight profiling
        analyzer.run_baseline()
        # Step 2: Incremental analysis
        analyzer.run_incremental_analysis(max_metrics=max_metrics)
        # Step 3: Generate report
        analyzer.generate_report(output_dir=f"ncu_overhead_{kernel_type}")
        print("\nAnalysis complete")
        print(f"Results in: ncu_overhead_{kernel_type}/")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":
    main()