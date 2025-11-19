"""
L2 Cache Contention - CLEAN Individual Line Plots
=================================================

CORRECT APPROACH: For each fixed parameter combination, create ONE separate plot.
This way:
- ONE line per plot (no overlapping mess)
- X-axis: The varying parameter
- Y-axis: The metric
- Title: Shows which parameters are fixed

Result: CLEAN, PROFESSIONAL, PUBLICATION-READY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['font.size'] = 10


class L2ContentionVisualizer:
    """Create CLEAN individual plots - ONE per fixed combination"""
    
    ALL_PARAMS = [
        'runtime_seconds', 'num_enemy_sms', 'victim_ws_kb', 'enemy_array_mb',
        'victim_grid_x', 'victim_block_x', 'enemy_block_x'
    ]
    
    def __init__(self, csv_path: str, output_dir: str = "l2_plots"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} experiments\n")
        
        self.df['slowdown_ratio'] = self.df['victim_concurrent_ms'] / self.df['victim_alone_ms']
        self.df['l2_miss_increase'] = self.df['lts_miss_avg_concurrent'] - self.df['lts_miss_avg_alone']
    
    def create_individual_plots(self):
        """Create ONE separate plot for EACH row (experiment)"""
        logger.info("="*70)
        logger.info("Creating Individual CLEAN Plots (ONE per experiment)...")
        logger.info("="*70)
        
        varying_params = [p for p in self.ALL_PARAMS if self.df[p].nunique() > 1]
        
        total_plots = 0
        
        for varying_param in varying_params:
            logger.info(f"\n  Processing: {varying_param}")
            
            # Get all rows and group by this parameter
            param_values = sorted(self.df[varying_param].unique())
            
            logger.info(f"    Found {len(param_values)} unique values: {param_values}")
            
            # For each value of the parameter, create a plot showing all metrics
            for idx, param_value in enumerate(param_values, 1):
                # Get ALL rows with this parameter value
                rows_with_value = self.df[self.df[varying_param] == param_value]
                
                if len(rows_with_value) == 0:
                    continue
                
                total_plots += 1
                
                # Extract all metrics for this parameter value
                exec_alone = rows_with_value['victim_alone_ms'].values
                exec_conc = rows_with_value['victim_concurrent_ms'].values
                l2_alone = rows_with_value['lts_miss_avg_alone'].values
                l2_conc = rows_with_value['lts_miss_avg_concurrent'].values
                slowdown = rows_with_value['slowdown_ratio'].values
                
                # Create x-positions for bars (one bar per experiment with this param value)
                x_pos = np.arange(len(exec_alone))
                
                # ===== PLOT 1: Execution Time =====
                fig, ax = plt.subplots(figsize=(12, 6))
                
                width = 0.35
                ax.bar(x_pos - width/2, exec_alone, width, label='Alone', 
                       color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
                ax.bar(x_pos + width/2, exec_conc, width, label='Concurrent', 
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
                
                ax.set_xlabel(f'Experiment #{x_pos + 1}', fontweight='bold', fontsize=11)
                ax.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=12)
                ax.set_title(f'Execution Time | {varying_param}={int(param_value)}', 
                           fontsize=13, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'#{i+1}' for i in x_pos], fontsize=9)
                ax.legend(fontsize=11, loc='best')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax.set_facecolor('#fafafa')
                
                # Add value labels
                for i, (a, c) in enumerate(zip(exec_alone, exec_conc)):
                    ax.text(i - width/2, a, f'{int(a)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                    ax.text(i + width/2, c, f'{int(c)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                filename = f"01_exec_time_{varying_param}_{idx:02d}_{int(param_value)}.png"
                filepath = self.output_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                # ===== PLOT 2: L2 Misses =====
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.bar(x_pos - width/2, l2_alone, width, label='Alone', 
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
                ax.bar(x_pos + width/2, l2_conc, width, label='Concurrent', 
                       color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)
                
                ax.set_xlabel(f'Experiment #{x_pos + 1}', fontweight='bold', fontsize=11)
                ax.set_ylabel('L2 Sector Lookups Misses', fontweight='bold', fontsize=12)
                ax.set_title(f'L2 Cache Misses | {varying_param}={int(param_value)}', 
                           fontsize=13, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'#{i+1}' for i in x_pos], fontsize=9)
                ax.legend(fontsize=11, loc='best')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax.set_facecolor('#fafafa')
                
                # Add value labels
                for i, (a, c) in enumerate(zip(l2_alone, l2_conc)):
                    ax.text(i - width/2, a, f'{int(a)}', ha='center', va='bottom', fontsize=7, fontweight='bold')
                    ax.text(i + width/2, c, f'{int(c)}', ha='center', va='bottom', fontsize=7, fontweight='bold')
                
                plt.tight_layout()
                filename = f"02_l2_misses_{varying_param}_{idx:02d}_{int(param_value)}.png"
                filepath = self.output_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                # ===== PLOT 3: Slowdown Factor =====
                fig, ax = plt.subplots(figsize=(12, 6))
                
                bars = ax.bar(x_pos, slowdown, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
                
                # Color bars by value (gradient)
                for bar in bars:
                    height = bar.get_height()
                    if height > slowdown.mean() + slowdown.std():
                        bar.set_color('#c0143c')  # Dark red for high slowdown
                    elif height < slowdown.mean() - slowdown.std():
                        bar.set_color('#90ee90')  # Light green for low slowdown
                
                ax.axhline(y=slowdown.mean(), color='blue', linestyle='--', linewidth=2, 
                          label=f'Mean: {slowdown.mean():.2f}x', alpha=0.7)
                
                ax.set_xlabel(f'Experiment #{x_pos + 1}', fontweight='bold', fontsize=11)
                ax.set_ylabel('Slowdown Factor (×)', fontweight='bold', fontsize=12)
                ax.set_title(f'Slowdown Factor | {varying_param}={int(param_value)}', 
                           fontsize=13, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'#{i+1}' for i in x_pos], fontsize=9)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax.set_facecolor('#fafafa')
                
                # Add value labels
                for i, s in enumerate(slowdown):
                    ax.text(i, s, f'{s:.2f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                filename = f"03_slowdown_{varying_param}_{idx:02d}_{int(param_value)}.png"
                filepath = self.output_dir / filename
                fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
            
            logger.info(f"    Created {len(param_values)} plot sets")
        
        logger.info(f"\n  Total: {total_plots} individual plots\n")
    
    def generate_summary(self):
        """Generate summary"""
        logger.info("="*70)
        logger.info("Generating Summary...")
        logger.info("="*70)
        
        report_path = self.output_dir / "00_README.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("L2 CACHE CONTENTION - CLEAN INDIVIDUAL PLOTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("VISUALIZATION APPROACH:\n")
            f.write("-"*80 + "\n")
            f.write("Each plot shows all experiments with a SPECIFIC parameter value:\n")
            f.write("  - X-axis: Each experiment with this parameter value\n")
            f.write("  - Y-axis: The metric (execution time, L2 misses, or slowdown)\n")
            f.write("  - TWO bars per experiment: Alone (green/blue) vs Concurrent (red/orange)\n")
            f.write("  - Clear, readable, publication-ready\n\n")
            
            f.write("FILE NAMING:\n")
            f.write("-"*80 + "\n")
            f.write("01_exec_time_PARAM_NN_VALUE.png  - Execution time plots\n")
            f.write("02_l2_misses_PARAM_NN_VALUE.png  - L2 misses plots\n")
            f.write("03_slowdown_PARAM_NN_VALUE.png   - Slowdown factor plots\n")
            f.write("(PARAM = parameter name, NN = plot number, VALUE = parameter value)\n\n")
            
            f.write("DATA SUMMARY:\n")
            f.write("-"*80 + "\n")
            f.write(f"Total experiments: {len(self.df)}\n")
            f.write(f"Slowdown: {self.df['slowdown_ratio'].min():.2f}x to {self.df['slowdown_ratio'].max():.2f}x\n")
            f.write(f"Average slowdown: {self.df['slowdown_ratio'].mean():.2f}x\n")
            f.write(f"L2 miss increase: {self.df['l2_miss_increase'].min():.0f} to {self.df['l2_miss_increase'].max():.0f}\n")
        
        logger.info(f"  Saved: {report_path.name}\n")
    
    def generate_all(self):
        """Generate all plots"""
        logger.info("\n" + "="*70)
        logger.info("L2 CACHE CONTENTION - INDIVIDUAL CLEAN PLOTS")
        logger.info("="*70 + "\n")
        
        self.create_individual_plots()
        self.generate_summary()
        
        png_count = len(list(self.output_dir.glob('*.png')))
        logger.info("="*70)
        logger.info(f"SUCCESS: {png_count} CLEAN individual plots saved!")
        logger.info("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="L2 Cache Contention - Individual Clean Plots")
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--output', default='l2_plots', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        visualizer = L2ContentionVisualizer(args.csv, args.output)
        visualizer.generate_all()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
