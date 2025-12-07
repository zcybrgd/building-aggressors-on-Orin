#here we pull the csv and generate plots from it like for example for victims that faced L2 contention
#we'll show the variation of execution time as a function of a certain working set size for different grid sizes or idk 

"""
L2 Cache Contention Results Visualization
==========================================
This script generates comprehensive plots from the L2 interference experiment results.

For each varying parameter (x-axis), it plots:
1. L2 cache miss metrics (alone vs concurrent)
2. Execution time metrics (alone vs concurrent)
3. Derived metrics (slowdown factor, miss increase)

Each plot shows the fixed parameters in the title/annotation.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import logging, argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class L2ResultsPlotter:
    """Generate comprehensive visualizations from L2 interference results"""
    def __init__(self, csv_path: str, output_dir: str = "plots"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} rows from {csv_path}")
        self.config_params = [
            #'runtime_seconds', 
            'num_enemy_sms', 'victim_ws_kb', 'enemy_array_mb', 'victim_grid_x', 'victim_block_x', 'enemy_block_x']
        # derived metrics
        if 'slowdown_factor' not in self.df.columns:
            self.df['slowdown_factor'] = (
                self.df['victim_concurrent_ms'] / self.df['victim_alone_ms']
            )
        #mhmmm...
        if 'miss_increase' not in self.df.columns:
            self.df['miss_increase'] = (self.df['lts_miss_avg_concurrent'] - self.df['lts_miss_avg_alone'])
    
    def _get_varying_param(self, subset_df: pd.DataFrame) -> str:
        """Identify which param is varying in this subset"""
        for param in self.config_params:
            if subset_df[param].nunique() > 1:
                return param
        return None
    def _get_fixed_params(self, subset_df: pd.DataFrame, varying_param: str) -> Dict[str, any]:
        """Get dictionary of fixed parameters and their values"""
        fixed_params = {}
        for param in self.config_params:
            if param != varying_param:
                unique_vals = subset_df[param].unique()
                if len(unique_vals) == 1:
                    fixed_params[param] = unique_vals[0]
        return fixed_params
    
    def _format_fixed_params_text(self, fixed_params: Dict[str, any]) -> str:
        """Format fixed parameters for plot annotation"""
        lines = []
        for param, value in fixed_params.items():
            abbrev = {
                'num_enemy_sms': 'm',
                'victim_ws_kb': 'v_ws',
                'enemy_array_mb': 'e_arr',
                'victim_grid_x': 'v_grid',
                'victim_block_x': 'v_blk',
                'enemy_block_x': 'e_blk'
            }
            param_name = abbrev.get(param, param)
            lines.append(f"{param_name}={value}")
        return '\n'.join(lines)
    
    def _format_param_label(self, param: str) -> str:
        """Format parameter name for axis labels"""
        labels = {
            'num_enemy_sms': 'Number of Enemy SMs',
            'victim_ws_kb': 'Victim Working Set (KB)',
            'enemy_array_mb': 'Enemy Array Size (MB)',
            'victim_grid_x': 'Victim Grid Size',
            'victim_block_x': 'Victim Block Size',
            'enemy_block_x': 'Enemy Block Size'
        }
        return labels.get(param, param)
    
    def plot_cache_metrics(self, varying_param: str, subset_df: pd.DataFrame, 
                          fixed_params: Dict[str, any]):
        """Plot L2 cache miss metrics (alone vs concurrent)"""
        # sort by varying parameter
        plot_df = subset_df.sort_values(varying_param)
        fig, (ax2) = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f'L2 Cache Metrics vs {self._format_param_label(varying_param)}', fontsize=14, fontweight='bold')
        x = plot_df[varying_param]
        
        # Plot 1: L2 sectors accessed i think ill remove this
        '''
        ax1.plot(x, plot_df['lts_sectors_avg_alone'], 
                marker='o', linewidth=2, label='Victim Alone', color='#2ecc71')
        ax1.plot(x, plot_df['lts_sectors_avg_concurrent'], 
                marker='s', linewidth=2, label='Victim Concurrent', color='#e74c3c')
        ax1.set_xlabel(self._format_param_label(varying_param), fontsize=11)
        ax1.set_ylabel('L2 Sectors Accessed (avg)', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('L2 Cache Sector Access', fontsize=12)
        '''
        # Plot 2: L2 cache misses
        ax2.plot(x, plot_df['lts_miss_avg_alone'], 
                marker='o', linewidth=2, label='Victim Alone', color='#2ecc71')
        ax2.plot(x, plot_df['lts_miss_avg_concurrent'], 
                marker='s', linewidth=2, label='Victim Concurrent', color='#e74c3c')
        
        ax2.set_xlabel(self._format_param_label(varying_param), fontsize=11)
        ax2.set_ylabel('L2 Sector Lookup Miss (avg)', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('L2 Cache Misses', fontsize=12)
        # Add fixed parameters text
        fixed_text = self._format_fixed_params_text(fixed_params)
        fig.text(0.02, 0.98, f"Fixed:\n{fixed_text}", fontsize=9, verticalalignment='top',bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        plt.tight_layout(rect=[0.12, 0, 1, 0.96])
        filename = f"cache_metrics_vs_{varying_param}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_timing_metrics(self, varying_param: str, subset_df: pd.DataFrame, 
                           fixed_params: Dict[str, any]):
        """Plot execution time metrics (alone vs concurrent)"""
        plot_df = subset_df.sort_values(varying_param)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Execution Time vs {self._format_param_label(varying_param)}', 
                     fontsize=14, fontweight='bold')
        
        x = plot_df[varying_param]
        
        # Plot 1: Victim execution times
        ax1.plot(x, plot_df['victim_alone_ms'], 
                marker='o', linewidth=2, label='Victim Alone', color='#3498db')
        ax1.plot(x, plot_df['victim_concurrent_ms'], 
                marker='s', linewidth=2, label='Victim Concurrent', color='#e67e22')
        #ax1.plot(x, plot_df['enemy_ms'],  marker='^', linewidth=2, label='Enemy', color='#9b59b6', alpha=0.7)
        ax1.set_xlabel(self._format_param_label(varying_param), fontsize=11)
        ax1.set_ylabel('Execution Time (ms)', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Kernel Execution Times', fontsize=12)
        
        # Plot 2: Slowdown factor
        slowdown = plot_df['victim_concurrent_ms'] / plot_df['victim_alone_ms']
        ax2.plot(x, slowdown, 
                marker='D', linewidth=2.5, color='#c0392b', label='Slowdown Factor')
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
        ax2.set_xlabel(self._format_param_label(varying_param), fontsize=11)
        ax2.set_ylabel('Slowdown Factor (×)', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Performance Degradation', fontsize=12)
    
        fixed_text = self._format_fixed_params_text(fixed_params)
        fig.text(0.02, 0.98, f"Fixed:\n{fixed_text}",  fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        plt.tight_layout(rect=[0.12, 0, 1, 0.96])
        filename = f"timing_vs_{varying_param}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {filepath}")
    
    def plot_derived_metrics(self, varying_param: str, subset_df: pd.DataFrame, 
                            fixed_params: Dict[str, any]):
        """Plot derived metrics (slowdown, miss increase, efficiency)"""
        plot_df = subset_df.sort_values(varying_param)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Interference Impact vs {self._format_param_label(varying_param)}', 
                     fontsize=14, fontweight='bold')
        x = plot_df[varying_param]
        
        # Plot 1: Miss increase
        miss_increase = plot_df['lts_miss_avg_concurrent'] - plot_df['lts_miss_avg_alone']
        ax1.plot(x, miss_increase, 
                marker='o', linewidth=2.5, color='#e74c3c', label='Miss Increase')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel(self._format_param_label(varying_param), fontsize=11)
        ax1.set_ylabel('L2 Miss Increase (sectors)', fontsize=11)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Cache Miss Increase (Concurrent - Alone)', fontsize=12)
        
        # Plot 2: Slowdown vs Miss correlation
        slowdown = plot_df['victim_concurrent_ms'] / plot_df['victim_alone_ms']
        miss_ratio = plot_df['lts_miss_avg_concurrent'] / plot_df['lts_miss_avg_alone'].replace(0, 1)
    
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(x, slowdown, 
                marker='D', linewidth=2, color='#c0392b', label='Slowdown Factor')
        line2 = ax2_twin.plot(x, miss_ratio, 
                marker='s', linewidth=2, color='#8e44ad', label='Miss Ratio')
        
        ax2.set_xlabel(self._format_param_label(varying_param), fontsize=11)
        ax2.set_ylabel('Slowdown Factor (×)', fontsize=11, color='#c0392b')
        ax2_twin.set_ylabel('Miss Ratio (×)', fontsize=11, color='#8e44ad')
        ax2.tick_params(axis='y', labelcolor='#c0392b')
        ax2_twin.tick_params(axis='y', labelcolor='#8e44ad')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best', fontsize=10)
        
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Performance Degradation vs Cache Impact', fontsize=12)
        
        fixed_text = self._format_fixed_params_text(fixed_params)
        fig.text(0.02, 0.98, f"Fixed:\n{fixed_text}", 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
        
        plt.tight_layout(rect=[0.12, 0, 1, 0.96])
        filename = f"derived_metrics_vs_{varying_param}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {filepath}")
    
    def plot_all_sweeps(self):
        """
        Automatically detect parameter sweeps and generate all plots
        
        A sweep is detected when one parameter varies while others are fixed.
        """
        logger.info("Detecting parameter sweeps and generating plots...")
        sweeps_found = 0
        
        # group by all config params to find unique configurations
        # then identify sweeps where only one param varies
        for varying_param in self.config_params:
            other_params = [p for p in self.config_params if p != varying_param]
            grouped = self.df.groupby(other_params)
            for fixed_values, group_df in grouped:
                # check if this group has multiple values for varying_param
                if group_df[varying_param].nunique() > 1:
                    sweeps_found += 1
                    fixed_params = dict(zip(other_params, fixed_values))
                    logger.info(f"\nSweep #{sweeps_found}: Varying {varying_param}")
                    logger.info(f"  Values: {sorted(group_df[varying_param].unique())}")
                    logger.info(f"  Fixed: {fixed_params}")
                    try:
                        self.plot_cache_metrics(varying_param, group_df, fixed_params)
                        self.plot_timing_metrics(varying_param, group_df, fixed_params)
                        self.plot_derived_metrics(varying_param, group_df, fixed_params)
                        logger.info(f"  Generated 3 plots")
                    except Exception as e:
                        logger.error(f"  Failed: {e}")
        logger.info("\n" + "="*60)
        logger.info(f"Complete! Generated plots for {sweeps_found} parameter sweeps")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("="*60)
    
    def generate_summary_report(self):
        """Generate a summary statistics report"""
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        print("\nDataset Overview:")
        print(f"  Total experiments: {len(self.df)}")
        print(f"  Parameters tested: {', '.join(self.config_params)}")
        print("\nSlowdown Statistics:")
        print(f"  Mean slowdown: {self.df['slowdown_factor'].mean():.2f}×")
        print(f"  Max slowdown: {self.df['slowdown_factor'].max():.2f}×")
        print(f"  Min slowdown: {self.df['slowdown_factor'].min():.2f}×")
        print(f"  Std deviation: {self.df['slowdown_factor'].std():.2f}")
        print("\nCache Miss Increase:")
        print(f"  Mean increase: {self.df['miss_increase'].mean():.2f} sectors")
        print(f"  Max increase: {self.df['miss_increase'].max():.2f} sectors")
        print(f"  Min increase: {self.df['miss_increase'].min():.2f} sectors")
        print("\nExecution Time Ranges:")
        print(f"  Victim alone: {self.df['victim_alone_ms'].min():.2f} - {self.df['victim_alone_ms'].max():.2f} ms")
        print(f"  Victim concurrent: {self.df['victim_concurrent_ms'].min():.2f} - {self.df['victim_concurrent_ms'].max():.2f} ms")


def main():
    """Generate all plots from L2 interference results"""
    parser = argparse.ArgumentParser(description='Plot L2 cache interference results')
    parser.add_argument('csv_file', type=str, 
                       help='Path to CSV file with results (e.g., l2_contention_dataset.csv)')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots/)')
    args = parser.parse_args()
    plotter = L2ResultsPlotter(args.csv_file, args.output_dir)
    plotter.generate_summary_report()
    plotter.plot_all_sweeps()
    logger.info(f"\nAll plots saved to: {plotter.output_dir.absolute()}")


if __name__ == "__main__":
    main()