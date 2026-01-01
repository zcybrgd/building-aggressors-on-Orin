#!/usr/bin/env python3
"""
NCU Metrics CSV Cleaner
Cleans NVIDIA NCU profiler CSV output by removing profiler messages
and creating both long-format and pivot tables.
"""

import csv
import sys
from collections import defaultdict

def clean_ncu_csv(input_file, cleaned_file, pivot_file):
    """
    Clean NCU CSV and create both cleaned and pivot versions
    """
    
    # Storage for cleaned data
    all_metrics = []
    metrics_by_iteration = defaultdict(dict)
    metric_names = set()
    
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
                    
                    # Extract metric data (corrected indices)
                    metric_name = row[13]      # Metric Name
                    metric_unit = row[14]      # Metric Unit  
                    metric_value = row[15]     # Metric Value (THIS WAS THE BUG!)
                    
                    # Clean value (remove commas from numbers)
                    metric_value_clean = metric_value.replace(',', '').strip()
                    
                    # Skip empty values
                    if not metric_value_clean:
                        continue
                    
                    # Skip if value doesn't contain any digits (it's just units)
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
                    
                except (ValueError, IndexError) as e:
                    continue
    
    # Write long format (cleaned CSV)
    if all_metrics:
        with open(cleaned_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Iteration', 'Metric_Name', 'Metric_Value', 'Metric_Unit']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        print(f"✓ Long format: {len(all_metrics)} metric entries")
        print(f"  Saved to: {cleaned_file}")
    else:
        print("ERROR: No valid data found!")
        return False
    
    # Write pivot format (wide table)
    if metrics_by_iteration:
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
    
    print("=" * 60)
    print("✓ Processing complete!")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 clean_ncu_metrics.py <input_csv>")
        print("Example: python3 clean_ncu_metrics.py ncu_metrics_30iter.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    
    # Generate output filenames
    base_name = input_csv.replace('.csv', '')
    cleaned_csv = f"{base_name}_cleaned.csv"
    pivot_csv = f"{base_name}_pivot.csv"
    
    # Process the file
    success = clean_ncu_csv(input_csv, cleaned_csv, pivot_csv)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
