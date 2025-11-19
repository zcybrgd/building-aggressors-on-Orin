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
    victim_ws_kb: int     # -v (working set in KB)
    enemy_array_mb: int   # -e
    victim_grid_x: int    # -vg
    victim_block_x: int   # -vb
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
        logger.info(f"Initialized L2InterferenceRunner with exe: {self.exe_path}")
    
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
            "-v", str(config.victim_ws_kb), "-e", str(config.enemy_array_mb), "-vg", str(config.victim_grid_x),
            "-vb", str(config.victim_block_x), "-eb", str(config.enemy_block_x)])  
        return base_cmd
    
    #shouldve did it from nsys
    def _parse_timing_output(self, output: str) -> Dict[str, float]:
        """Parse timing information from executable output // the console output"""
        timings = {}
        #parse victim alone time
        match = re.search(r'Victim alone completed in ([\d.]+) ms', output)
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
        logger.info(f"Running experiment: t={config.runtime_seconds}s, "f"m={config.num_enemy_sms}, v={config.victim_ws_kb}KB, "f"e={config.enemy_array_mb}MB")
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
            param_name: Name of parameter to sweep (e.g., 'victim_ws_kb')
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
            'runtime_seconds', 'num_enemy_sms', 'victim_ws_kb', 'enemy_array_mb',
            'victim_grid_x', 'victim_block_x', 'enemy_block_x',
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
        'victim_ws_kb': 512,
        'enemy_array_mb': 8,
        'victim_grid_x': 1,
        'victim_block_x': 32,
        'enemy_block_x': 128
    }
    
    #define sweep ranges for each parameter (values to loop over)
    #i should add an intelligent function that depending on the gpu target platform decides the ranges and values to test 
    #of the victim and enemy grid dimensions and working set sizes
    SWEEP_RANGES = {
        'runtime_seconds': [10],
        'num_enemy_sms': [4, 6, 8], 
        #'num_enemy_sms': [1, 2, 4, 6, 8, 10, 12, 16], #depending on the target platform, my gpu now has 16 sms
        #'victim_ws_kb': [64, 128, 256, 384, 512, 768, 1024, 2048, 4096],
        'enemy_array_mb': [2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
        'victim_grid_x': [1, 2, 4, 8, 16],
        'victim_block_x': [32, 64, 128, 256, 512],
        'enemy_block_x': [32, 64, 128, 256, 512, 1024]
    }
    
    #for each parameter, fix all others and sweep over this one
    for param_to_sweep, values_to_test in SWEEP_RANGES.items():
        logger.info(f"SWEEPING: {param_to_sweep}")
        logger.info(f"Fixed values: {FIXED_VALUES}")
        logger.info(f"Testing values: {values_to_test}")
        # loop over values for this single parameter
        for value in values_to_test:
            #create config with this parameter modified
            config_dict = FIXED_VALUES.copy()
            config_dict[param_to_sweep] = value
            config = ExperimentConfig(**config_dict)
            try:
                # Run single experiment
                result = runner.run_experiment(config, collect_ncu=True)
                logger.info(f"\n{param_to_sweep}={value}")
            except Exception as e:
                logger.error(f"\n{param_to_sweep}={value}: FAILED - {e}\n")
                continue
    # save all results
    runner.save_results_to_csv()
    logger.info(f"Experiment complete! Total runs: {len(runner.results)}")
    logger.info(f"Results saved to: {runner.output_csv}")

if __name__ == "__main__":
    main()