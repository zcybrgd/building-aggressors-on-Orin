#!/usr/bin/env python3
from pathlib import Path
import csv
import statistics as stats
import sys

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: summarize_timings.py <timings_csv> <summary_csv>")

    timings_path = Path(sys.argv[1])
    summary_path = Path(sys.argv[2])

    values = []
    with timings_path.open() as f:
        next(f, None)  
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, elapsed = line.split(",")
            values.append(float(elapsed))

    if not values:
        raise SystemExit(f"No timing data found in {timings_path}")

    summary = [
        ("runs", len(values)),
        ("mean_ms", f"{stats.mean(values):.3f}"),
        ("std_ms", f"{stats.pstdev(values):.3f}"),
        ("min_ms", f"{min(values):.3f}"),
        ("max_ms", f"{max(values):.3f}"),
    ]

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary)

if __name__ == "__main__":
    main()

