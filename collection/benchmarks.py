# so we have benchmarks that we wanna run on different types of interference; there are compute bound benchmarks that
#we want to put under compute contention, and there are memory bound benchmarks that we want to put under memory contention.

#ideal test case is we have NVIDIA MPS and 3 contention kernels: L2 cache Contention Kernel, L1 cache CK; compute CK
# there are 3 types of benchmarks those who we run against only one of those, those we run against 2 etc.. 

#lets not forget that the goal is to construct our transformer dataset
#the goal is at the end to have a csv that has : benchmark name, contention type (it can be none(baseline), l1,l2,compute),
# the combination of parameters grid size victim working set size etc.,the metrics for example if L2 contention 
# then wel'll have execution time + lts_sector lookup miss (execution time is always present)

# so this script for example for L2 benchmarks will launch all the benchmarks under L2 contention and collect the metrics
# i ll write that script that is dedicated to L2 in this file : interferenceL2.py it will take as an output one .exe of the victim



# with the csv we can do plots and analysis later on.



"""
Benchmark Orchestration System for GPU Interference Analysis
=============================================================
This script manages different benchmark workloads under various contention scenarios:
- L1 cache contention
- L2 cache contention  
- Compute contention
- Combined contention scenarios

Generates comprehensive CSV datasets for transformer training, including:
- Benchmark characteristics (compute-bound vs memory-bound)
- Contention type (none/baseline, L1, L2, compute, or combinations)
- Configuration parameters
- Performance metrics (execution time, cache misses, etc.)

Future: Will use NVIDIA MPS for true concurrent kernel execution on Linux.
"""

import subprocess, logging, re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime
from interferenceL2 import L2InterferenceRunner, ExperimentConfig as L2Config

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentionType(Enum):
    """Types of GPU resource contention"""
    NONE = "none"  # baseline, no contention
    L1_CACHE = "l1"
    L2_CACHE = "l2"
    COMPUTE = "compute"
    #hybrid ones
    L1_L2 = "l1_l2" 
    L1_COMPUTE = "l1_compute"
    L2_COMPUTE = "l2_compute"
    ALL = "all"  # L1/shared + L2 + compute


#is it a good idea to categorize//classify the benchmarks? its an exhaustive list we can never enumerate all their types
#will go back to this one
class BenchmarkType(Enum):
    """Benchmark workload characteristics"""
    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    HYBRID = "hybrid"


@dataclass
class BenchmarkInfo:
    """Metadata for a benchmark workload"""
    name: str
    executable_path: str
    benchmark_type: BenchmarkType
    description: str
    # which contention types to test this benchmark against
    test_contention_types: Set[ContentionType]
    #default parameters for this benchmark
    default_params: Dict[str, any]


@dataclass
class BenchmarkResult:
    """Results from running a benchmark under specific contention"""
    benchmark_name: str
    benchmark_type: str
    contention_type: str
    # config params (varies by benchmark)
    config_params: Dict[str, any]
    # performance metrics
    execution_time_ms: float
    baseline_time_ms: Optional[float] = None  # For comparison
    slowdown_factor: Optional[float] = None
    # contention-specific metrics
    metrics: Dict[str, any] = None
    # Metadata
    #timestamp: str = None
    #notes: str = ""


class BenchmarkOrchestrator:
    """Manages execution of multiple benchmarks under various contention scenarios"""
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.benchmarks: Dict[str, BenchmarkInfo] = {}
        self.results: List[BenchmarkResult] = []
        # init interference runners
        self.l2_runner: Optional[L2InterferenceRunner] = None
        # TODO: Add L1 and compute runners when implemented
        # self.l1_runner: Optional[L1InterferenceRunner] = None
        # self.compute_runner: Optional[ComputeInterferenceRunner] = None
        logger.info(f"Initialized BenchmarkOrchestrator, output: {self.output_dir}")
    
    def register_benchmark(self, benchmark: BenchmarkInfo):
        """Register a benchmark for testing"""
        if not Path(benchmark.executable_path).exists():
            logger.warning(f"Executable not found: {benchmark.executable_path}")
        self.benchmarks[benchmark.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.name} ({benchmark.benchmark_type.value})")
    
    def _run_baseline(self, benchmark: BenchmarkInfo) -> float:
        """
        Run benchmark without any contention to establish baseline
        
        Returns:
            Baseline execution time in milliseconds
        """
        logger.info(f"Running baseline for {benchmark.name}...")
        #build command with default params
        cmd = [benchmark.executable_path]
        for param, value in benchmark.default_params.items():
            cmd.extend([f"-{param}", str(value)])
        try:
            result = subprocess.run(cmd,capture_output=True, text=True,timeout=60, shell=True)
            # parse timing from output (assumes format: "completed in X.XX ms")
            match = re.search(r'completed in ([\d.]+) ms', result.stdout)
            if match:
                baseline_time = float(match.group(1))
                logger.info(f"Baseline: {baseline_time:.2f}ms")
                return baseline_time
            else:
                logger.error("Could not parse baseline timing")
                return 0.0 
        except Exception as e:
            logger.error(f"Baseline run failed: {e}")
            return 0.0
    
    #there is a big redondance here with interferenceL2.py ; shouldve made a common base class for all interference runners
    def _run_under_l2_contention(self, benchmark: BenchmarkInfo) -> List[BenchmarkResult]:
        """run benchmark under L2 cache contention with parameter sweeps"""
        logger.info(f"Running {benchmark.name} under L2 CACHE contention")
        if self.l2_runner is None:
            # Initialize L2 runner with this benchmark's executable
            self.l2_runner = L2InterferenceRunner(
                exe_path=benchmark.executable_path,
                output_csv=self.output_dir / f"{benchmark.name}_l2.csv"
            )
        
        results = []
        baseline_time = self._run_baseline(benchmark)
        base_config = L2Config(**benchmark.default_params)
        if benchmark.benchmark_type == BenchmarkType.MEMORY_BOUND:
            sweep_configs = [
                ('victim_ws_kb', [64, 128, 256, 512, 1024, 2048]),
                ('num_enemy_sms', [2, 4, 8, 12, 16]),
                ('enemy_array_mb', [4, 8, 16, 32]),
            ]
        else:  # for s
            sweep_configs = [
                ('victim_ws_kb', [256, 512, 1024]),
                ('num_enemy_sms', [4, 8, 16]),
            ]
        for param_name, param_values in sweep_configs:
            sweep_results = self.l2_runner.run_parameter_sweep( base_config, param_name,param_values,collect_ncu=True)
            for result in sweep_results:
                bench_result = BenchmarkResult(
                    benchmark_name=benchmark.name,benchmark_type=benchmark.benchmark_type.value,
                    contention_type=ContentionType.L2_CACHE.value, config_params=asdict(result.config),
                    execution_time_ms=result.victim_concurrent_ms, baseline_time_ms=baseline_time,
                    slowdown_factor=result.slowdown_factor,
                    metrics={
                        'victim_alone_ms': result.victim_alone_ms,
                        'enemy_ms': result.enemy_ms,
                        'lts_sectors_avg_alone': result.lts_sectors_avg_alone,
                        'lts_miss_avg_alone': result.lts_miss_avg_alone,
                        'lts_sectors_avg_concurrent': result.lts_sectors_avg_concurrent,
                        'lts_miss_avg_concurrent': result.lts_miss_avg_concurrent,
                        'miss_increase': result.miss_increase,
                    },# timestamp=datetime.now().isoformat(),
                    #notes=f"L2 sweep: {param_name}"
               )
                results.append(bench_result)
        logger.info(f"Collected {len(results)} results for {benchmark.name} under L2 contention")
        return results
    
    def _run_under_l1_contention(self, benchmark: BenchmarkInfo) -> List[BenchmarkResult]:
        """Run benchmark under L1 cache contention"""
        logger.info(f"L1 contention for {benchmark.name} - NOT IMPLEMENTED YET")
        # TODO: implement when L1 interference runner is ready
        return []
    def _run_under_compute_contention(self, benchmark: BenchmarkInfo) -> List[BenchmarkResult]:
        """Run benchmark under compute contention"""
        logger.info(f"Compute contention for {benchmark.name} - NOT IMPLEMENTED YET")
        # TODO: implement it when compute interference runner is ready
        return []
    def _run_under_combined_contention(self, benchmark: BenchmarkInfo, 
                                      contention_type: ContentionType) -> List[BenchmarkResult]:
        """Run benchmark under multiple simultaneous contention types"""
        logger.info(f"Combined contention ({contention_type.value}) for {benchmark.name} - NOT IMPLEMENTED YET")
        # TODO: idem
        return []
    
    def run_benchmark(self, benchmark_name: str, 
                     contention_types: Optional[List[ContentionType]] = None) -> List[BenchmarkResult]:
        """
        Run a single benchmark under specified contention types
        
        Args:
            benchmark_name: Name of registered benchmark
            contention_types: List of contention types to test, or None for all registered types
        
        Returns:
            List of BenchmarkResult for all tested scenarios
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark {benchmark_name} not registered")
        
        benchmark = self.benchmarks[benchmark_name]
        
        # Determine which contention types to test
        if contention_types is None:
            contention_types = list(benchmark.test_contention_types)
        
        all_results = []
        
        for contention in contention_types:
            logger.info(f"\nTesting {benchmark_name} under {contention.value} contention")
            
            if contention == ContentionType.NONE:
                # Just run baseline
                baseline_time = self._run_baseline(benchmark)
                result = BenchmarkResult(
                    benchmark_name=benchmark.name,
                    benchmark_type=benchmark.benchmark_type.value,
                    contention_type=ContentionType.NONE.value,
                    config_params=benchmark.default_params,
                    execution_time_ms=baseline_time,
                    baseline_time_ms=baseline_time,
                    slowdown_factor=1.0,
                    timestamp=datetime.now().isoformat()
                )
                all_results.append(result)
                
            elif contention == ContentionType.L2_CACHE:
                results = self._run_under_l2_contention(benchmark)
                all_results.extend(results)
                
            elif contention == ContentionType.L1_CACHE:
                results = self._run_under_l1_contention(benchmark)
                all_results.extend(results)
                
            elif contention == ContentionType.COMPUTE:
                results = self._run_under_compute_contention(benchmark)
                all_results.extend(results)
                
            else:  # Combined contention
                results = self._run_under_combined_contention(benchmark, contention)
                all_results.extend(results)
        
        self.results.extend(all_results)
        return all_results
    
    def run_all_benchmarks(self):
        """Run all registered benchmarks under their designated contention types"""
        logger.info("\n" + "="*70)
        logger.info(f"Running {len(self.benchmarks)} benchmarks")
        logger.info("="*70 + "\n")
        
        for benchmark_name in self.benchmarks:
            try:
                self.run_benchmark(benchmark_name)
            except Exception as e:
                logger.error(f"Failed to run {benchmark_name}: {e}")
                continue
        logger.info(f"\nAll benchmarks complete! Total results: {len(self.results)}")
    
    def save_results(self, filename: Optional[str] = None):
        """Save all results to CSV file"""
        if filename is None:
            filename = self.output_dir / f"all_benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" 
        if not self.results:
            logger.warning("No results to save")
            return
        rows = []
        for result in self.results:
            row = {
                'benchmark_name': result.benchmark_name,
                'benchmark_type': result.benchmark_type,
                'contention_type': result.contention_type,
                'execution_time_ms': result.execution_time_ms,
                'baseline_time_ms': result.baseline_time_ms,
                'slowdown_factor': result.slowdown_factor,
                'timestamp': result.timestamp,
                'notes': result.notes
            }
            for key, value in result.config_params.items():
                row[f'config_{key}'] = value
            if result.metrics:
                for key, value in result.metrics.items():
                    row[f'metric_{key}'] = value
            
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        logger.info(f"Saved {len(rows)} results to {filename}")
        logger.info("\n" + "="*70)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*70)
        
        summary = df.groupby(['benchmark_name', 'contention_type']).agg({
            'execution_time_ms': ['count', 'mean', 'std', 'min', 'max'],
            'slowdown_factor': ['mean', 'max']
        }).round(2)
        print(summary)


def main():
    """Example: Register and run benchmarks"""
    orchestrator = BenchmarkOrchestrator(output_dir="benchmark_results")
    # register L2 cache contention benchmark 
    l2_benchmark = BenchmarkInfo(name="l2_victim_v1",executable_path="enemy.exe",
        benchmark_type=BenchmarkType.MEMORY_BOUND,description="Memory-intensive victim kernel under L2 cache contention",
        test_contention_types={ContentionType.NONE,ContentionType.L2_CACHE},
        default_params={'runtime_seconds': 10,'num_enemy_sms': 8,
            'victim_ws_kb': 512,'enemy_array_mb': 8, 'victim_grid_x': 1,
            'victim_block_x': 32,'enemy_block_x': 128})
    orchestrator.register_benchmark(l2_benchmark)
    # TODO: benchmarks registration shouldnt be like this ; ill provide more details to automate this later // not a prio now
    # Example: Compute-bound benchmark
    # compute_benchmark = BenchmarkInfo(
    #     name="matmul_benchmark",
    #     executable_path="matmul.exe",
    #     benchmark_type=BenchmarkType.COMPUTE_BOUND,
    #     description="Dense matrix multiplication",
    #     test_contention_types={
    #         ContentionType.NONE,
    #         ContentionType.COMPUTE,
    #         ContentionType.L2_COMPUTE
    #     },
    #     default_params={...}
    # )
    # orchestrator.register_benchmark(compute_benchmark)
    orchestrator.run_all_benchmarks()
    orchestrator.save_results()
    logger.info("\nBenchmark suite complete!")


if __name__ == "__main__":
    main()