import subprocess
import re
import csv
import os
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
exe_path = os.path.join(BASE, "enemy.exe")
ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.0\ncu.bat"

metrics = [
    "lts__t_sectors_lookup_miss",
    "l1tex__t_sectors_lookup_miss",
    "dram__bytes_read",
    "dram__bytes_write"
]

fixed_params = {"-t": 10, "-m": 8, "-v": 512, "-e": 8}
vary_param = "-v"
sweep_values = [256, 512, 1024]

output_folder = os.path.join(BASE, "results")
os.makedirs(output_folder, exist_ok=True)

def run_and_profile(cmd_list):
    cmd_parts = []
    for arg in cmd_list:
        arg_str = str(arg)
        if ' ' in arg_str:
            cmd_parts.append(f'"{arg_str}"')
        else:
            cmd_parts.append(arg_str)
    
    cmd_string = ' '.join(cmd_parts)
    print(f"Executing: {cmd_string}\n")
    
    result = subprocess.run(cmd_string, capture_output=True, text=True, shell=True)
    
    # Print output for debugging
    print("=== NCU OUTPUT ===")
    print(result.stdout)
    print("=== NCU STDERR ===")
    print(result.stderr)
    print("==================\n")
    
    return result.stdout

def parse_ncu_metrics(output):
    metrics_data = {}
    for m in metrics:
        # Try multiple patterns to capture the metric value
        # Pattern 1: metric_name followed by value with optional units
        pattern1 = rf'{re.escape(m)}\s+([0-9,]+(?:\.[0-9]+)?)'
        # Pattern 2: metric_name with any characters then number
        pattern2 = rf'{re.escape(m)}.*?([0-9,]+(?:\.[0-9]+)?)\s'
        
        match = re.search(pattern1, output) or re.search(pattern2, output)
        if match:
            value_str = match.group(1).replace(',', '')
            try:
                metrics_data[m] = float(value_str)
            except ValueError:
                print(f"Warning: Could not parse value '{value_str}' for metric {m}")
                metrics_data[m] = None
        else:
            print(f"Warning: Metric {m} not found in output")
            metrics_data[m] = None
    return metrics_data

def parse_victim_time(output):
    m1 = re.search(r'Victim alone completed in ([\d\.]+) ms', output)
    if m1:
        return float(m1.group(1))
    m2 = re.search(r'Victim kernel finished after ([\d\.]+) ms', output)
    if m2:
        return float(m2.group(1))
    return None

results = []

for val in sweep_values:
    params = fixed_params.copy()
    params[vary_param] = val
    
    cmd = [
        ncu_path,
        "--kernel-name", "victimKernel",
        "--metrics", ",".join(metrics),
        exe_path
    ]
    
    for flag, param_val in params.items():
        cmd.extend([flag, str(param_val)])
    
    print(f"Running for {vary_param}={val} ...")
    
    output = run_and_profile(cmd)
    
    metric_values = parse_ncu_metrics(output)
    victim_time = parse_victim_time(output)
    
    results.append({vary_param: val, "victim_time_ms": victim_time, **metric_values})
    print(f"Results: {results[-1]}\n")

csv_file = os.path.join(output_folder, f"sweep_{vary_param.replace('-', '')}.csv")
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Saved results to {csv_file}")

x = [r[vary_param] for r in results]

for m in ["victim_time_ms"] + metrics:
    y = [r[m] for r in results]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linewidth=2, markersize=8)
    plt.xlabel(f'{vary_param} (KB)', fontsize=12)
    plt.ylabel(m, fontsize=12)
    plt.title(f"{m} vs {vary_param}\n(fixed params: {fixed_params})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname = f"{m}_vs_{vary_param.replace('-', '')}.png"
    png_file = os.path.join(output_folder, fname)
    plt.savefig(png_file, dpi=150)
    plt.close()
    print(f"Saved plot {png_file}")

print("\nSweep completed!")