#!/usr/bin/env python3
"""
NCU Overhead Analysis - Proof of Concept
Measures the overhead introduced by NCU profiling as metrics are added one-by-one progressively
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

# i cant profile over 184 metrics because past this, the windows command line length limit is exceeded and ncu fails to run
# so i will query ncu for available metrics and filter them
#and also all the set of metrics take 30 passes

def load_available_metrics():
    """Query NCU for all available metrics on the system"""
    ncu_exe = shutil.which('ncu')
    if ncu_exe is None:
        print('[ERROR] ncu not found in PATH')
        return None
    try:
        result = subprocess.run([ncu_exe, '--query-metrics'], capture_output=True, text=True,timeout=30)
        #parse metric names from the output
        # Format: "metric_name    Counter    unit    description"
        metrics = []
        for line in result.stdout.split('\n'):
            if not line.strip() or '---' in line or 'Metric Name' in line:
                continue
            parts = line.split()
            if parts:
                metric_name = parts[0]
                if '_' in metric_name and not metric_name.startswith('#'):
                    metrics.append(metric_name)
        
        print(f"[INFO] Loaded {len(metrics)} metrics from NCU")
        return metrics
    
    except Exception as e:
        print(f'[ERROR] Failed to query metrics: {e}')
        return None

SKIP_PREFIXES = {'fbpa__', #Frame Buffer Path Adapter (connects GPU to external framebuffer)
                  'fe__', #Frontend (graphics pipeline entry)
                  'gpc__' #Graphics Processing Cluster	
                 , 'gpu__'
                 , 'pcie__','gr__','idc__'
                 ,'l1tex__data_pipe','l1tex__texin','l1tex__t_set_conflicts','l1tex__t_set_accesses'
                 ,'l1tex__f_tex2sm_cycles','l1tex__f_wavefronts'
                 }
METRICS_LIST = ['gpu__time_duration'] + [m for m in load_available_metrics() if not any(m.startswith(p) for p in SKIP_PREFIXES) and m != 'gpu__time_duration']
'''
METRICS_LIST = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.sum",
    "sm__cycles_active.avg",
    "gpu__cycles_elapsed.sum",
    "gpu__cycles_active.sum",
    "gpu__time_duration.sum",
    "sm__warps_active.avg",
    "sm__warps_active.sum",
    "sm__warps_launched.sum",
    "smsp__cycles_active.sum",
    "smsp__cycles_elapsed.sum",
    "sm__inst_executed.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_issued.sum",
    "sm__sass_inst_executed.sum",
    "sm__pipe_alu_cycles_active.sum",
    "sm__pipe_fma_cycles_active.sum",
    "sm__pipe_fp64_cycles_active.sum",
    "sm__pipe_tensor_cycles_active.sum",
    "sm__sass_inst_executed_op_atom.sum",
    "smsp__inst_executed_op_branch.sum",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__sectors.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "dram__cycles_active.sum",
    "dram__cycles_active_read.sum",
    "dram__cycles_active_write.sum",
    "dram__cycles_elapsed.sum",
    "dram__throughput.avg",
    "l1tex__t_sectors.sum",
    "l1tex__t_sectors_lookup_hit.sum",
    "l1tex__t_sectors_lookup_miss.sum",
    "l1tex__t_bytes.sum",
    "l1tex__t_bytes_lookup_hit.sum",
    "l1tex__t_bytes_lookup_miss.sum",
    "l1tex__data_bank_conflicts.sum",
    "l1tex__data_bank_reads.sum",
    "l1tex__data_bank_writes.sum",
    "l1tex__throughput.avg",
    "l1tex__t_requests.sum",
    "l1tex__t_requests_pipe_lsu.sum",
    "l1tex__t_requests_pipe_tex.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "l1tex__t_sector_hit_rate.avg",
    "l1tex__t_sector_pipe_lsu_hit_rate.avg",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.avg",
    "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.avg",
    "l1tex__t_set_accesses.sum",
    "l1tex__t_set_conflicts.sum",
    "l1tex__t_set_conflicts_pipe_lsu.sum",
    "lts__t_requests.sum",
    "lts__t_requests_lookup_hit.sum",
    "lts__t_requests_lookup_miss.sum",
    "lts__t_request_hit_rate.avg",
    "lts__t_requests_op_read.sum",
    "lts__t_requests_op_read_lookup_hit.sum",
    "lts__t_requests_op_read_lookup_miss.sum",
    "lts__t_requests_op_write.sum",
    "lts__t_requests_op_write_lookup_hit.sum",
    "lts__t_requests_op_write_lookup_miss.sum",
    "lts__t_requests_op_atom.sum",
    "lts__t_requests_op_membar.sum",
    "lts__t_requests_aperture_device.sum",
    "lts__t_requests_aperture_device_lookup_hit.sum",
    "lts__t_requests_aperture_device_lookup_miss.sum",
    "lts__t_requests_aperture_peer.sum",
    "lts__t_requests_aperture_sysmem.sum",
    "lts__t_sectors.sum",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "lts__t_bytes.sum",
    "idc__requests.sum",
    "idc__requests_lookup_hit.sum",
    "idc__requests_lookup_miss.sum",
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.avg",
    "smsp__average_warp_latency_issue_stalled_short_scoreboard.avg",
    "smsp__average_warp_latency_issue_stalled_not_selected.avg",
    "smsp__average_warp_latency_issue_stalled_barrier.avg",
    "smsp__average_warp_latency_issue_stalled_no_instruction.avg",
]
'''





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
        self.failed_metrics = []  #track metrics that cause failures
        
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
    
    def _parse_ncu_metric_value(self, ncu_output, metric_name):
        """Parse a specific metric value from NCU output and return in milliseconds"""
        for line in ncu_output.split('\n'):
            if metric_name in line and not line.strip().startswith('#'):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == metric_name:
                        if i + 2 < len(parts):
                            try:
                                value_str = parts[i + 2].replace(',', '')
                                value = float(value_str)
                            
                                if i + 1 < len(parts):
                                    unit = parts[i + 1]
                                    if unit == 'us':
                                        return value / 1000  # us to ms
                                    elif unit == 'ms':
                                        return value
                                    elif unit == 'ns':
                                        return value / 1e6  # ns to ms
                                return value
                            except (ValueError, IndexError):
                                continue
        return None



    
    def run_with_metrics(self, metrics_list):
        """Run NCU with specified metrics"""
        metrics_str = ','.join(metrics_list)
        num_metrics = len(metrics_list)
        print(f"  [{num_metrics:2d} metrics] Profiling...", end=' ', flush=True)

        # ensure ncu is available
        ncu_exe = shutil.which('ncu')
        if ncu_exe is None:
            print('\n[ERROR] Nsight Compute CLI `ncu` not found in PATH. Provide its full path or add it to PATH.')
            return None

        def _exec(metrics_variant):
            """Execute NCU for the provided metrics list and return tuple ('ok', data) or ('err', info)"""
            metrics_str_local = ','.join(metrics_variant)
            cmd = [
                ncu_exe,
                "--launch-skip", "1", #we skip the warmup run
                "--kernel-name", self.kernel_name,
                "--metrics", metrics_str_local,
                str(self.executable),
                self.kernel_type
            ]
            try:
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                total_profiling_time = time.time() - start_time

                if result.returncode != 0:
                    #just for debugging purposes
                    #print(f"[DEBUG] NCU FAILED!")
                    #print(f"[DEBUG] STDERR (first 500 chars):\n{result.stderr[:500]}")
                    #print(f"[DEBUG] STDOUT (first 500 chars):\n{result.stdout[:500]}")
                    return ('err', {'type': 'ncu_returncode', 'code': result.returncode, 'stderr': result.stderr, 'stdout': result.stdout})

                num_passes = self._parse_num_passes(result.stdout)
                kernel_time_with_ncu_ms = self._parse_kernel_duration_from_output(result.stdout)
                ncu_reported_time_ms = self._parse_ncu_metric_value(result.stdout + result.stderr, 'gpu__time_duration.avg') 
                if kernel_time_with_ncu_ms is None:
                    return ('err', {'type': 'no_output', 'stdout': result.stdout, 'stderr': result.stderr})
                baseline_ms = self.baseline_time['cuda_events']
                ncu_overhead_ms = kernel_time_with_ncu_ms - baseline_ms
                overhead_percentage = (ncu_overhead_ms / baseline_ms) * 100 if baseline_ms > 0 else 0

                return ('ok', {
                    'num_metrics': len(metrics_variant),
                    'num_passes': num_passes,
                    'kernel_time_with_ncu_ms': kernel_time_with_ncu_ms,
                    'ncu_reported_time_ms': ncu_reported_time_ms,
                    'baseline_ms': baseline_ms,
                    'ncu_overhead_ms': ncu_overhead_ms,
                    'overhead_percentage': overhead_percentage,
                    'total_profiling_time': total_profiling_time,
                    'metrics': metrics_variant.copy()
                })

            except subprocess.TimeoutExpired:
                print(f"[DEBUG] NCU TIMEOUT after 120 seconds!")
                return ('err', {'type': 'timeout'})
            except FileNotFoundError as e:
                print(f"[DEBUG] FILE NOT FOUND: {e}")
                return ('err', {'type': 'file_not_found', 'exc': str(e)})
            except Exception as e:
                print(f"[DEBUG] EXCEPTION: {type(e).__name__}: {e}")
                return ('err', {'type': 'exception', 'exc': str(e)})

        #first attempt with the provided metrics
        first_attempt = _exec(metrics_list)
        if first_attempt[0] == 'ok':
            data = first_attempt[1]
            ncu_time_str = f", NCU: {data['ncu_reported_time_ms']:.4f}ms" if data.get('ncu_reported_time_ms') else ""
            print(f"→ {data['num_passes']} pass(es), Kernel: {data['kernel_time_with_ncu_ms']:.4f}ms{ncu_time_str}")

            return data

        #if it failed and we have at least one metric, we try appending suffixes to the last metric
        if num_metrics > 0:
            failing_metric = metrics_list[-1]
            # only try suffixes if metric doesn't already end with a suffix
            suffixes = ['.avg', '.sum','.pct']
            tried = []
            for suf in suffixes:
                if failing_metric.endswith(suf):
                    continue
                tried_metric = failing_metric + suf
                tried.append(tried_metric)
                test_metrics = metrics_list[:-1] + [tried_metric]
                print(f"\n  [Retry] Trying metric variant: {tried_metric}...", end=' ', flush=True)
                attempt = _exec(test_metrics)
                if attempt[0] == 'ok':
                    data = attempt[1]
                    print(f"→ {data['num_passes']} pass(es), Kernel: {data['kernel_time_with_ncu_ms']:.4f}ms (succeeded with suffix {suf})")
                    return data
                else:
                    # continue trying other suffix
                    print("failed")

            #if we reach here, all suffix attempts failed -> mark original metric as incompatible
            print(f"[NO OUTPUT / ERROR after retries] Marking '{failing_metric}' as incompatible")
            self.failed_metrics.append(failing_metric)
            return None
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
        # Look for stuff like: COMPUTE_KERNEL_TIMING,565.5785 or MEMORY_KERNEL_TIMING,12.345
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
        valid_metrics = []  #track which metrics actually work
        for i in range(1, min(max_metrics + 1, len(METRICS_LIST) + 1)):
            #build metric list, skipping known failures
            current_metric = METRICS_LIST[i-1]
            if current_metric in self.failed_metrics:
                print(f"  [Skipping known bad metric: {current_metric}]")
                continue
            test_metrics = valid_metrics + [current_metric]
            result = self.run_with_metrics(test_metrics)
            if result:
                #valid_metrics.append(current_metric)
                successful_metric = result['metrics'][-1]  #get the actual metric name that worked
                valid_metrics.append(successful_metric)
                results.append(result)
                if len(results) > 1 and result['num_passes'] > results[-2]['num_passes']:
                    print(f"PASS TRANSITION DETECTED at {len(valid_metrics)} metrics!")
            else:
                print(f"  [Skipping problematic metric, continuing...]")
                #print(f"  Previous {len(valid_metrics)} metrics: {valid_metrics}")
                continue
        
        self.results = results
        if self.failed_metrics:
            print(f"\n[INFO] Skipped {len(self.failed_metrics)} incompatible metrics")
        
        return results
    
    def generate_report(self, output_dir="."):
        """Generate comprehensive report with graphs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        report_data = {
            'kernel_type': self.kernel_type,
            'kernel_name': self.kernel_name,
            'baseline': self.baseline_time,
            'profiling_results': self.results,
            'failed_metrics': self.failed_metrics 
        }
        
        json_path = output_dir / f'ncu_overhead_{self.kernel_type}.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\nJSON saved: {json_path}")
        self._plot_combined(output_dir)
        self._write_summary_report(output_dir)
        print(f"Report generated in: {output_dir}")
      
    
    def _plot_combined(self, output_dir):
        """create 2 separate plots: metrics vs passes and metrics vs kernel execution time"""
        if not self.results:
            return
        num_metrics = [r['num_metrics'] for r in self.results]
        num_passes = [r['num_passes'] for r in self.results]
        kernel_times = [r['kernel_time_with_ncu_ms'] for r in self.results]
        baseline = self.baseline_time['cuda_events']

        # Plot 1: Metrics vs Passes 
        fig1, ax1 = plt.subplots(figsize=(30, 15))
        ax1.step(num_metrics, num_passes, where='post', linewidth=2.5, label='NCU Passes', color='#e74c3c')
        ax1.scatter(num_metrics, num_passes, s=80, color='#c0392b', zorder=5, alpha=0.7)
        ax1.set_xlabel('Number of Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of NCU Passes', fontsize=12, fontweight='bold')
        ax1.set_title(f'Metrics vs Passes\n{self.kernel_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10)
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for i in range(1, len(num_passes)):
            if num_passes[i] > num_passes[i-1]:
                ax1.annotate('Pass ↑', xy=(num_metrics[i], num_passes[i]), xytext=(num_metrics[i] + 0.5, num_passes[i] + 0.2), arrowprops=dict(arrowstyle='->', color='red', lw=1.5), fontsize=9, color='red', fontweight='bold')
        plot_path1 = output_dir / f'ncu_overhead_{self.kernel_type}_passes.png'
        fig1.tight_layout()
        fig1.savefig(plot_path1, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path1}")
        plt.close(fig1)

        # Plot 2: Metrics vs Kernel Execution Time
        fig2, ax2 = plt.subplots(figsize=(30, 15))
        ax2.plot(num_metrics, kernel_times, 'o-', linewidth=2.5, markersize=8, color='#2ecc71', label='Kernel Time with NCU')
        ncu_reported_times = [r.get('ncu_reported_time_ms') for r in self.results]
        valid_ncu_times = [(m, t) for m, t in zip(num_metrics, ncu_reported_times) if t is not None]
        if valid_ncu_times:
            ncu_metrics, ncu_times = zip(*valid_ncu_times)
            ax2.plot(ncu_metrics, ncu_times, 's-', linewidth=2.5, markersize=8, color='#9b59b6', label='NCU gputimeduration', alpha=0.8)
        ax2.axhline(y=baseline, color='#3498db', linestyle='--', linewidth=2, label=f'Baseline (no NCU): {baseline:.2f}ms')
        ax2.set_xlabel('Number of Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Kernel Execution Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Metrics vs Kernel Execution Time\n{self.kernel_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
        plot_path2 = output_dir / f'ncu_overhead_{self.kernel_type}_kernel_time.png'
        fig2.tight_layout()
        fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path2}")
        plt.close(fig2)
    
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
                f.write(f"{'Metrics':<10} {'Passes':<10} {'CUDA Events (ms)':<20} {'NCU Time (ms)':<20}\n")
                f.write("-" * 90 + "\n")

                prev_passes = 0
                for r in self.results:
                    marker = " " if r['num_passes'] > prev_passes and prev_passes > 0 else "   "
                    ncu_time_str = f"{r['ncu_reported_time_ms']:.4f}" if r.get('ncu_reported_time_ms') else "N/A"
                    f.write(f"{marker}{r['num_metrics']:<7} {r['num_passes']:<10} "
                      f"{r['kernel_time_with_ncu_ms']:<20.4f} {ncu_time_str:<20}\n")

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
            if self.failed_metrics:
                f.write("\n\nINCOMPATIBLE METRICS (SKIPPED)\n")
                f.write("="*70 + "\n")
                for metric in self.failed_metrics:
                    f.write(f"  - {metric}\n")
        
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
        #1. Y-axis: Nombre de passes vs X-axis: nombre de métriques
        # 2. Y-axis: Temps d'exécution du kernel vs X-axis: nombre de métriques
        analyzer.generate_report(output_dir=f"ncu_overhead_{kernel_type}")
        print("\nAnalysis complete")
        print(f"Results in: ncu_overhead_{kernel_type}/")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":
    main()