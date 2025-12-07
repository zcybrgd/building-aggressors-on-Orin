# in this file we'll launch 2 kernels : the enemy and the victim
#they should be launched concurrently through the same context by the use of NVIDIA MPS


# for now we dont have MPS here because i dont have linux yet i have windows so this is just a placeholder for now
# for now the exe that ill input just launch the victim alone AND the victim+enemy (they are in the same .cuda
# just in 2 different streams) and collect the metrics
#we'll collect the timing metrics using nsys because they are more accurate and less overhead
# and the lts_sector_lookup_miss using ncu
# the program contain a lot of params (n) so here for each param  ; we'll fix a combination (n-1) and loop over 1 to generate the data
#and we'll do it over all the params one by one to generate the data and append them into the csv
# the params are independent from the victim as much as we can

"""
L2 Cache Interference Experiment Runner
========================================
This script runs L2 cache contention experiments by launching victim and enemy kernels,
collecting performance metrics (execution time via nsys, cache misses via ncu).

For now, runs on Windows with separate streams. Will be updated to use NVIDIA MPS on Linux.

The script systematically varies one parameter at a time while keeping others fixed,
generating comprehensive training data for transformer models.
"""

import subprocess, re, csv, logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

#configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    runtime_seconds: int  # -t, ofc this one will disappear most likely when we use NVIDIA MPS since im letting it run infinetely against the victim
    num_enemy_sms: int    # -m , this will be just a param for the enemy to decide how many sms to use
    matrix_size: int      # -s (matrix dimension, e.g., 512 creates 512x512 matrices)
    enemy_array_mb: int   # -e
    victim_block_dim: int # -vb (block dimension for GEMM, e.g., 16 creates 16x16 blocks)
    enemy_block_x: int    # -eb
    

@dataclass
class MetricsResult:
    """Results from a single experiment"""
    #configuration
    config: ExperimentConfig
    # timing metrics (from nsys or direct execution)
    victim_alone_ms: float
    victim_concurrent_ms: float
    enemy_ms: float
    
    #cache metrics (from ncu)
    lts_sectors_avg_alone: Optional[float] = None
    lts_miss_avg_alone: Optional[float] = None
    lts_sectors_avg_concurrent: Optional[float] = None
    lts_miss_avg_concurrent: Optional[float] = None


class L2InterferenceRunner:
    """Manages L2 cache contention experiments"""
    
    def __init__(self, exe_path: str, output_csv: str = "l2_contention_results.csv"):
        #self.contention_exe = Path(contention_exe) + add the error checking
        self.exe_path = Path(exe_path)
        if not self.exe_path.exists():
            raise FileNotFoundError(f" Victim Executable not found: {exe_path}")
        self.output_csv = Path(output_csv)
        self.results: List[MetricsResult] = []
        
        # Query GPU properties
        self.gpu_props = self._get_gpu_properties()
        logger.info(f"Initialized L2InterferenceRunner with exe: {self.exe_path}")
        logger.info(f"GPU: {self.gpu_props['name']}, SMs: {self.gpu_props['multiProcessorCount']}, "
                   f"Max threads/block: {self.gpu_props['maxThreadsPerBlock']}, "
                   f"Max blocks/SM: {self.gpu_props['maxBlocksPerMultiProcessor']}")
    
    def _get_gpu_properties(self) -> Dict[str, Any]:
        """Query GPU device properties using CUDA"""
        try:
            # Run a small CUDA program to get device properties
            query_cmd = [
                "python", "-c",
                "import cupy as cp; "
                "d = cp.cuda.Device(0); "
                "print(f'{d.attributes[\"MultiProcessorCount\"]};{d.attributes[\"MaxThreadsPerBlock\"]};{d.attributes[\"MaxBlocksPerMultiProcessor\"]};{d.name.decode()}')"
            ]
            result = subprocess.run(query_cmd, capture_output=True, text=True, shell=True, timeout=10)
            if result.returncode == 0:
                parts = result.stdout.strip().split(';')
                return {
                    'multiProcessorCount': int(parts[0]),
                    'maxThreadsPerBlock': int(parts[1]),
                    'maxBlocksPerMultiProcessor': int(parts[2]),
                    'name': parts[3]
                }
        except Exception as e:
            logger.warning(f"Could not query GPU properties via CuPy: {e}")
        
        # Fallback: use default conservative values
        logger.warning("Using default GPU properties (conservative estimates)")
        return {
            'multiProcessorCount': 16,
            'maxThreadsPerBlock': 1024,
            'maxBlocksPerMultiProcessor': 16,
            'name': 'NVIDIA GeForce GTX 1650 Ti'
        }
    
    def _is_config_valid(self, config: ExperimentConfig) -> tuple[bool, str]:
        """
        Validate if configuration minimizes compute interference (we want L2 interference, not compute interference)
        
        Returns:
            (is_valid, reason) - True if valid, False with reason if invalid
        """
        # Check 1: Calculate victim SM occupancy
        victim_threads_per_block = config.victim_block_dim * config.victim_block_dim
        victim_blocks_needed = ((config.matrix_size + config.victim_block_dim - 1) // config.victim_block_dim) ** 2
        
        # Estimate active victim blocks per SM (simplified, assumes no resource limits except thread count)
        max_blocks_per_sm = self.gpu_props['maxThreadsPerBlock'] // victim_threads_per_block
        max_blocks_per_sm = min(max_blocks_per_sm, self.gpu_props['maxBlocksPerMultiProcessor'])
        
        # Estimate how many SMs the victim will occupy
        victim_sms_needed = (victim_blocks_needed + max_blocks_per_sm - 1) // max_blocks_per_sm
        victim_sms_needed = min(victim_sms_needed, self.gpu_props['multiProcessorCount'])
        
        # Check 2: Ensure enemy and victim don't both saturate the GPU (compute interference)
        total_sms = self.gpu_props['multiProcessorCount']
        
        # We want some compute overlap but not full saturation to isolate L2 interference
        # Allow enemy to use at most 75% of total SMs to leave room for victim
        if config.num_enemy_sms > int(total_sms * 0.75):
            return False, f"Enemy uses too many SMs ({config.num_enemy_sms}/{total_sms}), would cause compute interference"
        
        # Check 3: Ensure victim + enemy don't oversaturate (want L2 interference, not compute starvation)
        # If both kernels together would use >90% of SMs, they interfere at compute level
        estimated_sm_usage = victim_sms_needed + config.num_enemy_sms
        if estimated_sm_usage > int(total_sms * 0.9):
            return False, f"Combined SM usage too high ({estimated_sm_usage}/{total_sms}): victim≈{victim_sms_needed} + enemy={config.num_enemy_sms} > 90%, causes compute interference"
        
        # Check 4: Victim block dimensions are valid
        if victim_threads_per_block > self.gpu_props['maxThreadsPerBlock']:
            return False, f"Victim threads/block ({victim_threads_per_block}) > max ({self.gpu_props['maxThreadsPerBlock']})"
        
        # Check 5: Enemy block dimensions are valid
        if config.enemy_block_x > self.gpu_props['maxThreadsPerBlock']:
            return False, f"Enemy threads/block ({config.enemy_block_x}) > max ({self.gpu_props['maxThreadsPerBlock']})"
        
        # Check 6: Matrix size is reasonable (prevent out-of-memory)
        matrix_memory_mb = (3 * config.matrix_size * config.matrix_size * 4) / (1024 * 1024)
        if matrix_memory_mb > 4096:
            return False, f"Matrix memory ({matrix_memory_mb:.1f} MB) too large (>4GB)"
        
        return True, "Valid configuration"
    
    
    
    def _build_command(self, config: ExperimentConfig, use_profiler: str = None) -> List[str]:
        """Build command line for experiment"""
        base_cmd = []
        if use_profiler == "nsys":
            base_cmd = [
                "nsys", "profile",
                "--stats=true",
                f"--output={self.output_csv.stem}_nsys",
                "--force-overwrite=true"
            ]
        elif use_profiler == "ncu":
            base_cmd = [
                "ncu",
                "--replay-mode", "application",
                "--kernel-name", "victimKernel",
                "--metrics", "lts__t_sectors_lookup_miss.avg,lts__t_sectors.avg"
            ]
        
        base_cmd.append(str(self.exe_path))
        base_cmd.extend([
            "-t", str(config.runtime_seconds),"-m", str(config.num_enemy_sms),
            "-s", str(config.matrix_size), "-e", str(config.enemy_array_mb),
            "-vb", str(config.victim_block_dim), "-eb", str(config.enemy_block_x)])  
        return base_cmd
    
    #shouldve did it from nsys
    def _parse_timing_output(self, output: str) -> Dict[str, float]:
        """Parse timing information from executable output // the console output"""
        timings = {}
        #parse victim alone time (updated for GEMM output)
        match = re.search(r'Victim.*?completed in ([\d.]+) ms', output)
        if match:
            timings['victim_alone_ms'] = float(match.group(1))
        #parse concurrent victim time
        match = re.search(r'Victim kernel finished in ([\d.]+) ms \(concurrent scenario\)', output)
        if match:
            timings['victim_concurrent_ms'] = float(match.group(1))
        #parse enemy time
        match = re.search(r'Enemy kernel finished in ([\d.]+) ms', output)
        if match:
            timings['enemy_ms'] = float(match.group(1))
        
        print("\ntimings information parsed:", timings)
        
        return timings
    
    def _parse_ncu_output(self, output: str) -> Dict[str, Any]:
        """Parse cache metrics from ncu output"""
        metrics = {}
        # find all victimKernel sections
        sections = re.findall(r'victimKernel.*?lts__t_sectors\.avg\s+sector\s+([\d,]+\.?\d*)\s+' +
            r'lts__t_sectors_lookup_miss\.avg\s+sector\s+([\d,]+\.?\d*)',
            output,re.DOTALL)
        
        if len(sections) >= 2:
            # 1st section is "alone", second is "concurrent"
            sectors_alone = float(sections[0][0].replace(',', ''))
            miss_alone = float(sections[0][1].replace(',', ''))
            sectors_concurrent = float(sections[1][0].replace(',', ''))
            miss_concurrent = float(sections[1][1].replace(',', ''))
            metrics['lts_sectors_avg_alone'] = sectors_alone
            metrics['lts_miss_avg_alone'] = miss_alone
            metrics['lts_sectors_avg_concurrent'] = sectors_concurrent
            metrics['lts_miss_avg_concurrent'] = miss_concurrent
        return metrics
    
    def run_experiment(self, config: ExperimentConfig, collect_ncu: bool = True) -> MetricsResult:
        """
        Run a single experiment with given configuration
        
        Args:
            config: Experiment configuration
            collect_ncu: Whether to collect NCU cache metrics (slower)
        
        Returns:
            MetricsResult containing all collected metrics
        """
        # Validate configuration before running
        is_valid, reason = self._is_config_valid(config)
        if not is_valid:
            logger.warning(f"Skipping invalid configuration: {reason}")
            raise ValueError(f"Invalid configuration: {reason}")
        
        logger.info(f"Running experiment: t={config.runtime_seconds}s, "f"m={config.num_enemy_sms}, s={config.matrix_size}x{config.matrix_size}, "f"e={config.enemy_array_mb}MB")
        # Step 1: run without profiler for accurate timing
        cmd = self._build_command(config, use_profiler=None) #we can replace it with nsys later
        logger.debug(f"Command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd,capture_output=True,text=True,
                timeout=config.runtime_seconds * 3 + 60 , # Safety timeout
                shell=True)
            timings = self._parse_timing_output(result.stdout + result.stderr)
            if not timings:
                logger.error("Failed to parse timing output")
                logger.debug(f"Output: {result.stdout}")
                raise ValueError("Could not extract timing metrics")   
        except subprocess.TimeoutExpired:
            logger.error("Experiment timed out")
            raise
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        
        # Step 2: Run with NCU for cache metrics
        cache_metrics = {}
        if collect_ncu:
            logger.info("Collecting NCU cache metrics...")
            cmd_ncu = self._build_command(config, use_profiler="ncu")
            try:
                result_ncu = subprocess.run(cmd_ncu, capture_output=True,text=True,
                    timeout=config.runtime_seconds * 5 + 120  # NCU is slower
                    ,shell=True)
                cache_metrics = self._parse_ncu_output(result_ncu.stdout + result_ncu.stderr)     
            except Exception as e:
                logger.warning(f"NCU collection failed: {e}")
        
        #combine results
        metrics_result = MetricsResult(config=config, victim_alone_ms=timings.get('victim_alone_ms', 0.0),
            victim_concurrent_ms=timings.get('victim_concurrent_ms', 0.0),
            enemy_ms=timings.get('enemy_ms', 0.0),
            **cache_metrics)
        self.results.append(metrics_result)
        return metrics_result
    
    def run_parameter_sweep(self, base_config: ExperimentConfig, param_name: str, param_values: List[Any],collect_ncu: bool = True) -> List[MetricsResult]:
        """
        Sweep a single parameter while keeping others fixed
        
        Args:
            base_config: Base configuration with fixed values
            param_name: Name of parameter to sweep (e.g., 'matrix_size')
            param_values: List of values to test
            collect_ncu: Whether to collect NCU metrics
        
        Returns:
            List of MetricsResult for each parameter value
        """
        logger.info(f"\nParameter sweep: {param_name}")
        logger.info(f"Values: {param_values}")
        sweep_results = []
        for value in param_values:
            #create new config with one modified parameter
            config_dict = asdict(base_config)
            config_dict[param_name] = value
            config = ExperimentConfig(**config_dict)
            try:
                result = self.run_experiment(config, collect_ncu=collect_ncu)
                sweep_results.append(result)
            except Exception as e:
                logger.error(f"Failed for {param_name}={value}: {e}")
                continue
        return sweep_results
    
    def save_results_to_csv(self, filename: Optional[str] = None):
        """Save all collected results to CSV file"""
        if filename is None:
            filename = self.output_csv
        
        if not self.results:
            logger.warning("No results to save")
            return
        
        fieldnames = [
            #i should add here the name of the benchmark later bc ill have multiple benchmarks, or each one in its file? idk for now
            # configuration parameters
            'runtime_seconds', 'num_enemy_sms', 'matrix_size', 'enemy_array_mb',
            'victim_block_dim', 'enemy_block_x',
            # timing metrics
            'victim_alone_ms', 'victim_concurrent_ms', 'enemy_ms',
            # cache metrics
            'lts_sectors_avg_alone', 'lts_miss_avg_alone',
            'lts_sectors_avg_concurrent', 'lts_miss_avg_concurrent']
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = asdict(result.config)
                row.update({
                    'victim_alone_ms': result.victim_alone_ms,
                    'victim_concurrent_ms': result.victim_concurrent_ms,
                    'enemy_ms': result.enemy_ms,
                    'lts_sectors_avg_alone': result.lts_sectors_avg_alone,
                    'lts_miss_avg_alone': result.lts_miss_avg_alone,
                    'lts_sectors_avg_concurrent': result.lts_sectors_avg_concurrent,
                    'lts_miss_avg_concurrent': result.lts_miss_avg_concurrent,
                })
                writer.writerow(row)
        logger.info(f"Saved {len(self.results)} results to {filename}")


def main():
    # init runner // should be an argument so the first script can inject the executables files names here
    runner = L2InterferenceRunner(exe_path="enemy.exe",output_csv="l2_contention_dataset.csv")
    # Define FIXED baseline values for all parameters
    FIXED_VALUES = {
        'runtime_seconds': 10,
        'num_enemy_sms': 8,
        'matrix_size': 512,         # Matrix dimension (512x512 matrices)
        'enemy_array_mb': 8,
        'victim_block_dim': 16,     # 16x16 thread blocks for GEMM
        'enemy_block_x': 128
    }
    
    #define sweep ranges for each parameter (values to loop over)
    #i should add an intelligent function that depending on the gpu target platform decides the ranges and values to test 
    #of the victim and enemy grid dimensions and working set sizes
    SWEEP_RANGES = {
        'num_enemy_sms': [4, 6,8,10,12],
        #'num_enemy_sms': [1, 2, 4, 6, 8, 10, 12, 16], #depending on the target platform, my gpu now has 16 sms
        'matrix_size': [128, 256, 384, 512, 768, 1024, 1536, 2048],  # Matrix dimensions
        'enemy_array_mb': [2, 4, 6, 12, 16, 24, 32, 48, 64],
        'victim_block_dim': [8, 16, 32],  # Block dimensions for GEMM (NxN blocks)
        'enemy_block_x': [32, 64, 256, 512, 1024]
    }
    
    #for each parameter, fix all others and sweep over this one
    for param_to_sweep, values_to_test in SWEEP_RANGES.items():
        logger.info(f"SWEEPING: {param_to_sweep}")
        logger.info(f"Fixed values: {FIXED_VALUES}")
        logger.info(f"Testing values: {values_to_test}")
        
        valid_count = 0
        skipped_count = 0
        
        # loop over values for this single parameter
        for value in values_to_test:
            #create config with this parameter modified
            config_dict = FIXED_VALUES.copy()
            config_dict[param_to_sweep] = value
            config = ExperimentConfig(**config_dict)
            
            # Validate configuration before running
            is_valid, reason = runner._is_config_valid(config)
            if not is_valid:
                logger.warning(f"SKIPPED {param_to_sweep}={value}: {reason}")
                skipped_count += 1
                continue
            
            try:
                # Run single experiment
                result = runner.run_experiment(config, collect_ncu=True)
                logger.info(f"\n{param_to_sweep}={value}")
                valid_count += 1
            except Exception as e:
                logger.error(f"\n{param_to_sweep}={value}: FAILED - {e}")
                continue
        
        logger.info(f"Completed {param_to_sweep} sweep: {valid_count} valid, {skipped_count} skipped")
    
    # save all results
    runner.save_results_to_csv()
    logger.info(f"Experiment complete! Total runs: {len(runner.results)}")
    logger.info(f"Results saved to: {runner.output_csv}")

if __name__ == "__main__":
    main()