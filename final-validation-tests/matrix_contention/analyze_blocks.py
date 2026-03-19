#!/usr/bin/env python3
"""
SM Block Distribution — Green Context Isolation Proof
Parses [VICTIM] block X -> SM Y lines from stdout files and shows
which SMs were used in alone vs concurrent scenarios.

Usage:
    python plot_sm_distribution.py <results_dir>
"""

import sys
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python plot_sm_distribution.py <results_dir>")
    sys.exit(1)

results_dir = os.path.abspath(sys.argv[1])

if not os.path.isdir(results_dir):
    print(f"ERROR: directory not found: {results_dir}")
    sys.exit(1)

# ── Style ─────────────────────────────────────────────────────────────────────
BG    = "#0F1117"
PANEL = "#1A1D27"
GRID  = "#2A2D3A"
TEXT  = "#E8EAF0"
MUTED = "#7B7F93"

COLOR_ALONE = "#2196F3"
COLOR_CONC  = "#FF9800"
COLOR_FORB  = "#F44336"
COLOR_BOUND = "#FF5252"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID, "axes.labelcolor":  TEXT,
    "axes.titlecolor":  TEXT, "xtick.color":      MUTED,
    "ytick.color":      MUTED, "text.color":       TEXT,
    "grid.color":       GRID, "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "axes.spines.top":  False, "axes.spines.right": False,
})

PATTERN = re.compile(r'\[VICTIM\]\s+block\s+(\d+)\s+->\s+SM\s+(\d+)')

# ── Load & parse ──────────────────────────────────────────────────────────────
def load_scenario(results_dir, scenario_tag):
    """
    Read every file in results_dir whose name contains scenario_tag.
    Parse all [VICTIM] block X -> SM Y lines and return (blocks, sms) lists.
    Exits with error if no files found or no lines parsed.
    """
    blocks, sms = [], []
    found_files = 0

    for fname in sorted(os.listdir(results_dir)):
        if scenario_tag not in fname:
            continue
        fpath = os.path.join(results_dir, fname)
        if not os.path.isfile(fpath):
            continue
        found_files += 1
        with open(fpath, errors="replace") as fh:
            raw = fh.read()
        before = len(sms)
        for m in PATTERN.finditer(raw):
            blocks.append(int(m.group(1)))
            sms.append(int(m.group(2)))
        print(f"  FILE: {fname} — {len(sms) - before} [VICTIM] lines")

    if found_files == 0:
        print(f"ERROR: no files containing '{scenario_tag}' in {results_dir}")
        sys.exit(1)
    if not sms:
        print(f"ERROR: found {found_files} file(s) for '{scenario_tag}' but zero [VICTIM] lines.")
        print(f"       Make sure the printf in GPUMultiplyMatrix is uncommented.")
        sys.exit(1)

    return blocks, sms

print(f"\nLoading ALONE...")
alone_blocks, alone_sms = load_scenario(results_dir, "alone")

print(f"\nLoading CONCURRENT...")
conc_blocks, conc_sms = load_scenario(results_dir, "concurrent")

print(f"\n  Alone     : {len(alone_sms)} dispatches | SMs used: {sorted(set(alone_sms))}")
print(f"  Concurrent: {len(conc_sms)} dispatches | SMs used: {sorted(set(conc_sms))}")

# ── Derive everything from parsed data ────────────────────────────────────────
total_sms     = max(max(alone_sms), max(conc_sms)) + 1
sm_ids        = list(range(total_sms))

victim_sms    = sorted(set(conc_sms))
enemy_sms     = sorted(set(sm_ids) - set(victim_sms))
max_victim_sm = max(victim_sms)
boundary      = max_victim_sm + 0.5

alone_count   = Counter(alone_sms)
conc_count    = Counter(conc_sms)

# ── Isolation validation ──────────────────────────────────────────────────────
violations = [sm for sm in conc_sms if sm in set(enemy_sms)]

print()
print("=" * 55)
print("  GREEN CONTEXT ISOLATION — VALIDATION REPORT")
print("=" * 55)
print(f"  Victim SMs (from concurrent stdout) : {victim_sms}")
print(f"  Enemy  SMs (not seen in concurrent) : {enemy_sms}")
print(f"  Total dispatches checked            : {len(conc_sms)}")
print()
if violations:
    vcount = Counter(violations)
    print(f"  ✗  FAIL — {len(violations)} block(s) landed on enemy SMs!")
    for sm, n in sorted(vcount.items()):
        print(f"       SM {sm} : {n} block(s)  ← VIOLATION")
    print("  Green context isolation is BROKEN.")
else:
    print(f"  ✓  PASS — all {len(conc_sms)} blocks stayed on SM {victim_sms[0]}–{victim_sms[-1]}")
    unused = set(victim_sms) - set(conc_sms)
    if unused:
        print(f"  ⚠  Victim SMs with no blocks: {sorted(unused)} (normal for small grids)")
    else:
        print(f"  ✓  All {len(victim_sms)} victim SMs received at least one block")
print("=" * 55)

# ── Plot ──────────────────────────────────────────────────────────────────────
x = np.arange(total_sms)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle(
    f"Green Context SM Isolation — Victim Block Distribution\n"
    f"Alone: {len(set(alone_sms))}/{total_sms} SMs used   |   "
    f"Concurrent: {len(victim_sms)} victim SMs  +  {len(enemy_sms)} enemy SMs",
    fontsize=11, fontweight="bold", color=TEXT, y=1.01
)

# ── Chart 1 — ALONE ───────────────────────────────────────────────────────────
counts_alone = [alone_count.get(s, 0) for s in sm_ids]
ax1.bar(x, counts_alone, color=COLOR_ALONE, width=0.6, zorder=3,
        edgecolor=BG, linewidth=0.5)

y_off1 = max(counts_alone) * 0.02 + 0.5
for xi, c in zip(x, counts_alone):
    if c > 0:
        ax1.text(xi, c + y_off1, str(c), ha="center", va="bottom",
                 fontsize=9, color=TEXT, fontweight="bold")

ax1.set_title(f"ALONE — all {total_sms} SMs available",
              fontweight="bold", pad=10, color=COLOR_ALONE)
ax1.set_xlabel("Physical SM ID")
ax1.set_ylabel("Blocks dispatched")
ax1.set_xticks(x)
ax1.set_xticklabels([f"SM {i}" for i in sm_ids])
ax1.set_ylim(0, max(counts_alone) * 1.2 + 1)
ax1.grid(True, axis="y", zorder=0)
ax1.text(0.5, 0.92,
         f"✓  {len(set(alone_sms))} / {total_sms} SMs used",
         transform=ax1.transAxes, ha="center", fontsize=10,
         color=COLOR_ALONE, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=COLOR_ALONE))

# ── Chart 2 — CONCURRENT ─────────────────────────────────────────────────────
counts_conc = [conc_count.get(s, 0) for s in sm_ids]
bar_colors  = [COLOR_CONC if s in set(victim_sms) else COLOR_FORB for s in sm_ids]
max_conc    = max(counts_conc) if max(counts_conc) > 0 else 1

ax2.bar(x, counts_conc, color=bar_colors, width=0.6, zorder=3,
        edgecolor=BG, linewidth=0.5)

y_off2 = max_conc * 0.02 + 0.5
for xi, c, s in zip(x, counts_conc, sm_ids):
    label = str(c) if c > 0 else "0\n(blocked)"
    color = TEXT if c > 0 else COLOR_FORB
    ax2.text(xi, c + y_off2, label, ha="center", va="bottom",
             fontsize=8.5, color=color, fontweight="bold")

ax2.axvline(boundary, color=COLOR_BOUND, linewidth=2, linestyle="--", zorder=5)
ax2.text(boundary + 0.08, max_conc * 1.05,
         f"← victim ({len(victim_sms)} SMs)  |  enemy ({len(enemy_sms)} SMs) →",
         color=COLOR_BOUND, fontsize=8.5, va="top")

if enemy_sms:
    ax2.axvspan(min(enemy_sms) - 0.5, max(enemy_sms) + 0.5,
                alpha=0.08, color=COLOR_FORB, zorder=1)
    enemy_center = (min(enemy_sms) + max(enemy_sms)) / 2
    ax2.text(enemy_center, max_conc * 0.55, "ENEMY\nSMs",
             ha="center", color=COLOR_FORB, fontsize=9,
             fontweight="bold", alpha=0.8)

ax2.set_title(f"CONCURRENT — Green Context (victim = {len(victim_sms)} SMs)",
              fontweight="bold", pad=10, color=COLOR_CONC)
ax2.set_xlabel("Physical SM ID")
ax2.set_ylabel("Blocks dispatched")
ax2.set_xticks(x)
ax2.set_xticklabels([f"SM {i}" for i in sm_ids])
ax2.set_ylim(0, max_conc * 1.25 + 1)
ax2.grid(True, axis="y", zorder=0)

status       = "✓  PASS" if not violations else "✗  FAIL"
color_status = COLOR_CONC if not violations else COLOR_FORB
ax2.text(0.5, 0.92,
         f"{status} — SM {victim_sms[0]}–{victim_sms[-1]} only  "
         f"({len(victim_sms)}/{total_sms} SMs)",
         transform=ax2.transAxes, ha="center", fontsize=10,
         color=color_status, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=BG, edgecolor=color_status))

fig.legend(handles=[
    mpatches.Patch(color=COLOR_ALONE, label="Alone — SM active"),
    mpatches.Patch(color=COLOR_CONC,  label="Concurrent — victim SM"),
    mpatches.Patch(color=COLOR_FORB,  label="Concurrent — enemy SM (blocked)"),
], loc="lower center", ncol=3, fontsize=9,
   facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
   bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out_path = os.path.join(results_dir, "sm_block_distribution.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n  Saved → {out_path}")