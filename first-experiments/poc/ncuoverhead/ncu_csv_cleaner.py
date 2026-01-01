#!/usr/bin/env python3
"""
NCU Metrics CSV Cleaner with Statistics
Cleans NVIDIA NCU profiler CSV output and generates statistics
"""

import csv
import sys
import os
from collections import defaultdict
import math

def calculate_statistics(values):
    """Calculate min, max, avg, std for a list of numeric values"""
    if not values:
        return None, None, None, None

    # Convert to floats
    numeric_values = []
    for v in values:
        try:
            numeric_values.append(float(v))
        except ValueError:
            continue

    if not numeric_values:
        return None, None, None, None

    n = len(numeric_values)
    min_val = min(numeric_values)
    max_val = max(numeric_values)
    avg_val = sum(numeric_values) / n

    # Calculate standard deviation
    if n > 1:
        variance = sum((x - avg_val) ** 2 for x in numeric_values) / n
        std_val = math.sqrt(variance)
    else:
        std_val = 0.0

    return min_val, max_val, avg_val, std_val

def clean_ncu_csv(input_file, output_folder):
    """
    Clean NCU CSV and create outputs in organized folder
    """

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Output files
    cleaned_file = os.path.join(output_folder, "metrics_cleaned.csv")
    pivot_file = os.path.join(output_folder, "metrics_pivot.csv")
    stats_file = os.path.join(output_folder, "metrics_statistics.txt")
    stats_csv_file = os.path.join(output_folder, "metrics_statistics.csv")

    # Storage for cleaned data
    all_metrics = []
    metrics_by_iteration = defaultdict(dict)
    metric_names = set()
    metrics_by_name = defaultdict(list)  # For statistics

    print(f"Processing: {input_file}")
    print("=" * 60)

    # Open with UTF-8-sig to handle BOM
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)

        for row in reader:
            # Skip empty rows
            if not row or len(row) < 2:
                continue

            # Skip rows containing profiler messages
            if any('==PROF==' in str(cell) for cell in row):
                continue

            # Skip header rows
            if 'Metric Name' in str(row) or 'Metric Unit' in str(row):
                continue

            # Process data rows (need at least 16 columns for value)
            if len(row) >= 16:
                try:
                    iteration = row[0]
                    # Verify iteration is a number
                    int(iteration)

                    # Extract metric data
                    metric_name = row[13]      # Metric Name
                    metric_unit = row[14]      # Metric Unit  
                    metric_value = row[15]     # Metric Value

                    # Clean value (remove commas from numbers)
                    metric_value_clean = metric_value.replace(',', '').strip()

                    # Skip empty values
                    if not metric_value_clean:
                        continue

                    # Skip if value doesn't contain any digits
                    if not any(c.isdigit() for c in metric_value_clean):
                        continue

                    # Store for long format
                    all_metrics.append({
                        'Iteration': iteration,
                        'Metric_Name': metric_name,
                        'Metric_Value': metric_value_clean,
                        'Metric_Unit': metric_unit
                    })

                    # Store for pivot format
                    metrics_by_iteration[iteration][metric_name] = metric_value_clean
                    metric_names.add(metric_name)

                    # Store for statistics
                    metrics_by_name[metric_name].append(metric_value_clean)

                except (ValueError, IndexError) as e:
                    continue

    if not all_metrics:
        print("ERROR: No valid data found!")
        return False

    # Write long format (cleaned CSV)
    with open(cleaned_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Iteration', 'Metric_Name', 'Metric_Value', 'Metric_Unit']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"✓ Long format: {len(all_metrics)} metric entries")
    print(f"  Saved to: {cleaned_file}")

    # Write pivot format (wide table)
    metric_list = sorted(list(metric_names))

    with open(pivot_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Iteration'] + metric_list
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for iteration in sorted(metrics_by_iteration.keys(), key=int):
            row_data = {'Iteration': iteration}
            row_data.update(metrics_by_iteration[iteration])
            writer.writerow(row_data)

    print(f"✓ Pivot format: {len(metrics_by_iteration)} iterations × {len(metric_names)} metrics")
    print(f"  Saved to: {pivot_file}")

    # Calculate and write statistics
    stats_data = []

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NCU METRICS STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Iterations: {len(metrics_by_iteration)}\n")
        f.write(f"Total Metrics: {len(metric_names)}\n\n")

        for metric_name in sorted(metric_names):
            values = metrics_by_name[metric_name]
            min_val, max_val, avg_val, std_val = calculate_statistics(values)

            if min_val is not None:
                f.write("-" * 80 + "\n")
                f.write(f"Metric: {metric_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Minimum:    {min_val:.6f}\n")
                f.write(f"  Maximum:    {max_val:.6f}\n")
                f.write(f"  Average:    {avg_val:.6f}\n")
                f.write(f"  Std Dev:    {std_val:.6f}\n")
                f.write(f"  Range:      {max_val - min_val:.6f}\n")
                f.write(f"  CV (%):     {(std_val / avg_val * 100) if avg_val != 0 else 0:.2f}\n")
                f.write(f"  Samples:    {len(values)}\n\n")

                # Store for CSV
                stats_data.append({
                    'Metric_Name': metric_name,
                    'Min': f"{min_val:.6f}",
                    'Max': f"{max_val:.6f}",
                    'Average': f"{avg_val:.6f}",
                    'Std_Dev': f"{std_val:.6f}",
                    'Range': f"{max_val - min_val:.6f}",
                    'CV_Percent': f"{(std_val / avg_val * 100) if avg_val != 0 else 0:.2f}",
                    'Samples': len(values)
                })

    print(f"✓ Statistics (text): {len(metric_names)} metrics analyzed")
    print(f"  Saved to: {stats_file}")

    # Write statistics CSV
    if stats_data:
        with open(stats_csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Metric_Name', 'Min', 'Max', 'Average', 'Std_Dev', 'Range', 'CV_Percent', 'Samples']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_data)

        print(f"✓ Statistics (CSV): Saved to: {stats_csv_file}")

    print("=" * 60)
    print(f"✓ All outputs saved to folder: {output_folder}/")
    print("\nContents:")
    print(f"  - metrics_cleaned.csv      (long format)")
    print(f"  - metrics_pivot.csv        (wide format)")
    print(f"  - metrics_statistics.txt   (readable stats)")
    print(f"  - metrics_statistics.csv   (stats for analysis)")

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 clean_ncu_metrics.py <input_csv>")
        print("Example: python3 clean_ncu_metrics.py ncu_metrics_30iter.csv")
        sys.exit(1)

    input_csv = sys.argv[1]

    # Create output folder name from input file
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    output_folder = f"{base_name}_analysis"

    # Process the file
    success = clean_ncu_csv(input_csv, output_folder)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()