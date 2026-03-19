import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ── Load CSV ──────────────────────────────────────────────────────────────────
# Expected columns (header required):
#   scenario, sm_count, run, gpu_time_ns, sm_cycles_avg, warps_launched,
#   warps_active_avg, inst_executed, l2_read_hit_pct, l2_read_misses

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
out_dir  = os.path.dirname(os.path.abspath(csv_path))   # same folder as CSV

raw = []
skipped = 0
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=2):   # line 1 = header
        # Skip rows with any empty or whitespace-only field
        if any(v.strip() == "" for v in row.values()):
            print(f"  WARNING: skipping line {i} — empty field(s)")
            skipped += 1
            continue
        try:
            raw.append((
                int(row["scenario"]),
                int(row["sm_count"]),
                int(row["run"]),
                float(row["gpu_time_ns"]),
                float(row["sm_cycles_avg"]),
                int(row["warps_launched"]),
                float(row["warps_active_avg"]),
                float(row["inst_executed"]),
                float(row["l2_read_hit_pct"]),
                float(row["l2_read_misses"]),
            ))
        except ValueError as e:
            print(f"  WARNING: skipping line {i} — parse error: {e}")
            skipped += 1

print(f"Loaded {len(raw)} rows ({skipped} skipped) from '{csv_path}'")
print(f"Charts will be saved to: {out_dir}/\n")

# ── Per-scenario arrays ───────────────────────────────────────────────────────
scenarios    = sorted(set(r[0] for r in raw))
sm_counts    = {r[0]: r[1] for r in raw}
short_labels = {s: f"{sm_counts[s]} SMs" for s in scenarios}

COLS = {
    "gpu_time_ns":      3,
    "sm_cycles_avg":    4,
    "warps_active_avg": 6,
    "l2_read_hit_pct":  8,
    "l2_read_misses":   9,
}

def get(scenario, col_idx):
    return np.array([r[col_idx] for r in raw if r[0] == scenario], dtype=float)

data = {s: {k: get(s, v) for k, v in COLS.items()} for s in scenarios}
for s in scenarios:
    data[s]["gpu_time_ms"] = data[s]["gpu_time_ns"] / 1e6

max_run = max(len(data[s]["gpu_time_ms"]) for s in scenarios)

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}
BG    = "#0F1117"
PANEL = "#1A1D27"
GRID  = "#2A2D3A"
TEXT  = "#E8EAF0"
MUTED = "#7B7F93"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

SUPTITLE = "Green Context SM Isolation — Jetson Orin Nano  |  victimKernel  |  16 blocks × 256 threads"
SAVE_KW  = dict(dpi=150, bbox_inches="tight", facecolor=BG)

def save(filename):
    path = os.path.join(out_dir, filename)
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"  Saved → {path}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — GPU Execution Time per Run
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

for s in scenarios:
    t    = data[s]["gpu_time_ms"]
    runs = np.arange(1, len(t) + 1)
    ax.plot(runs, t, "o-", color=COLORS[s], label=short_labels[s],
            linewidth=2, markersize=5, markeredgewidth=0)
    ax.axhline(t.mean(), color=COLORS[s], linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_title("GPU Execution Time per Run", fontweight="bold", pad=8)
ax.set_xlabel("Run #")
ax.set_ylabel("Time (ms)")
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, axis="y")
ax.set_xticks(range(1, max_run + 1))
save("chart1_gpu_time_per_run.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Mean GPU Time ± σ
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

means = [data[s]["gpu_time_ms"].mean() for s in scenarios]
stds  = [data[s]["gpu_time_ms"].std()  for s in scenarios]
bars  = ax.bar(
    [short_labels[s] for s in scenarios], means,
    color=[COLORS[s] for s in scenarios],
    yerr=stds, capsize=5, error_kw={"ecolor": TEXT, "linewidth": 1.5},
    width=0.5, zorder=3
)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{m:.1f} ms", ha="center", va="bottom", fontsize=9,
            color=TEXT, fontweight="bold")

base = means[0]
for i, (s, m) in enumerate(zip(scenarios[1:], means[1:]), 1):
    pct = (m - base) / base * 100
    ax.text(i, 1.5, f"+{pct:.1f}%", ha="center", va="bottom",
            fontsize=8, color=COLORS[s], fontweight="bold")

ax.set_ylim(0, max(means) * 1.15)
ax.set_title("Mean GPU Time ± σ", fontweight="bold", pad=8)
ax.set_ylabel("Time (ms)")
ax.grid(True, axis="y", zorder=0)
save("chart2_mean_gpu_time.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — SM Cycles avg (boxplot)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

bp = ax.boxplot(
    [data[s]["sm_cycles_avg"] / 1e6 for s in scenarios],
    patch_artist=True, widths=0.5,
    medianprops={"color": TEXT, "linewidth": 2},
    whiskerprops={"color": MUTED}, capprops={"color": MUTED},
    flierprops={"markerfacecolor": MUTED, "marker": "o", "markersize": 4}
)
for patch, s in zip(bp["boxes"], scenarios):
    patch.set_facecolor(COLORS[s])
    patch.set_alpha(0.85)

ax.set_xticklabels([short_labels[s] for s in scenarios])
ax.set_title("SM Cycles avg", fontweight="bold", pad=8)
ax.set_ylabel("Cycles (×10⁶)")
ax.grid(True, axis="y")
save("chart3_sm_cycles_boxplot.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Active Warps avg per Cycle
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

for s in scenarios:
    w    = data[s]["warps_active_avg"] / 1e6
    runs = np.arange(1, len(w) + 1)
    ax.plot(runs, w, "s-", color=COLORS[s], label=short_labels[s],
            linewidth=2, markersize=5, markeredgewidth=0)
    ax.axhline(w.mean(), color=COLORS[s], linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_title("Active Warps avg per Cycle", fontweight="bold", pad=8)
ax.set_xlabel("Run #")
ax.set_ylabel("Warps (×10⁶)")
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, axis="y")
ax.set_xticks(range(1, max_run + 1))
save("chart4_warps_active.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 5 — L2 Read Hit Rate
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

means_hit = [data[s]["l2_read_hit_pct"].mean() for s in scenarios]
stds_hit  = [data[s]["l2_read_hit_pct"].std()  for s in scenarios]
bars5 = ax.bar(
    [short_labels[s] for s in scenarios], means_hit,
    color=[COLORS[s] for s in scenarios],
    yerr=stds_hit, capsize=5, error_kw={"ecolor": TEXT, "linewidth": 1.5},
    width=0.5, zorder=3
)
for bar, m in zip(bars5, means_hit):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{m:.1f}%", ha="center", va="bottom", fontsize=9,
            color=TEXT, fontweight="bold", clip_on=True)

ax.set_title("L2 Read Hit Rate", fontweight="bold", pad=8)
ax.set_ylabel("Hit Rate (%)")
ax.set_ylim(0, 100)
ax.axhline(50, color=MUTED, linewidth=0.7, linestyle=":")
ax.grid(True, axis="y", zorder=0)
save("chart5_l2_hit_rate.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 6 — L2 Read Misses
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

means_miss = [data[s]["l2_read_misses"].mean() / 1e6 for s in scenarios]
bars6 = ax.bar(
    [short_labels[s] for s in scenarios], means_miss,
    color=[COLORS[s] for s in scenarios],
    width=0.5, zorder=3
)
# Use relative offset (2% of y-range) so annotation never escapes the axes
y_range = max(means_miss) - min(means_miss) if max(means_miss) != min(means_miss) else max(means_miss)
y_offset = max(means_miss) * 0.02 + 0.01
for bar, m in zip(bars6, means_miss):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
            f"{m:.1f}M", ha="center", va="bottom", fontsize=9,
            color=TEXT, fontweight="bold", clip_on=True)

ax.set_ylim(0, max(means_miss) * 1.15)
ax.set_title("L2 Read Misses (mean)", fontweight="bold", pad=8)
ax.set_ylabel("Misses (×10⁶)")
ax.grid(True, axis="y", zorder=0)
save("chart6_l2_misses.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 7 — SM Cycles vs GPU Time (scatter)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)

for s in scenarios:
    ax.scatter(
        data[s]["sm_cycles_avg"] / 1e6,
        data[s]["gpu_time_ms"],
        color=COLORS[s], label=short_labels[s],
        s=70, edgecolors=BG, linewidths=0.5, zorder=3
    )

ax.set_title("SM Cycles vs GPU Time", fontweight="bold", pad=8)
ax.set_xlabel("SM Cycles avg (×10⁶)")
ax.set_ylabel("GPU Time (ms)")
ax.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
ax.grid(True, zorder=0)
save("chart7_cycles_vs_time_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Summary Table
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3), facecolor=BG)
fig.suptitle(SUPTITLE, fontsize=9, color=MUTED)
ax.axis("off")

col_labels = ["Scenario", "SMs", "Time (ms)  μ ± σ",
              "SM Cycles (μ, ×10⁶)", "L2 Hit (%)", "L2 Misses (×10⁶)"]
table_data = []
for s in scenarios:
    t = data[s]["gpu_time_ms"]
    c = data[s]["sm_cycles_avg"] / 1e6
    h = data[s]["l2_read_hit_pct"]
    m = data[s]["l2_read_misses"] / 1e6
    table_data.append([
        f"S{s}  ({'no green ctx' if s == 0 else 'green ctx'})",
        str(sm_counts[s]),
        f"{t.mean():.1f} ± {t.std():.1f}",
        f"{c.mean():.1f}",
        f"{h.mean():.1f}",
        f"{m.mean():.1f}",
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(GRID)
    if row == 0:
        cell.set_facecolor("#12162A")
        cell.set_text_props(color="#AAB4FF", fontweight="bold")
    else:
        s = scenarios[row - 1]
        cell.set_facecolor(COLORS[s] + "22")
        cell.set_text_props(color=TEXT)

ax.set_title("Summary Table", fontweight="bold", pad=12, color=TEXT)
save("chart8_summary_table.png")

print(f"\nDone — 8 charts saved in: {out_dir}/")