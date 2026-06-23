#!/usr/bin/env python3
"""
pop_size_comparison.py
======================
Prints, for each pop-size alternative vs the reference:
  - average difference in total ΔV across all launch windows
  - difference in the single best (minimum) total ΔV

Usage:  python pop_size_comparison.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent
DEFAULT_FILE = "VEEv2.csv"

ALT_FILES = {
    "pop = 5":    "VEEpop_size5v2.csv",
    "pop = 10":   "VEEpop_size10v2.csv",
    "pop = 30":   "VEEpop_size30v2.csv",
    "pop = 300":  "VEEpop_size300v2.csv",
    "pop = 1000": "VEEpop_size1000v3.csv",
}
COL = "delta_v_ms"

def load_csv(path):
    return pd.read_csv(path, parse_dates=["launch_date"])

def extract_pop(label):
    m = re.search(r"\d+", label)
    return int(m.group()) if m else 0

# ── Load reference ─────────────────────────────────────────────────────────────
ref_path = DATA_DIR / DEFAULT_FILE
if not ref_path.exists():
    sys.exit(f"[ERROR] Reference file not found: {ref_path}")

df_ref   = load_csv(ref_path)
ref_best = df_ref[COL].min()
ref_mean = df_ref[COL].mean()

print(f"Reference ({DEFAULT_FILE})")
print(f"  best  : {ref_best / 1000:.3f} km/s")
print(f"  mean  : {ref_mean / 1000:.3f} km/s")
print()

# ── Header ─────────────────────────────────────────────────────────────────────
w = 14
print(f"{'':>{w}}  {'avg diff (m/s)':>16}  {'best diff (m/s)':>16}")
print("  " + "─" * (w + 38))

# ── Compare each alternative ───────────────────────────────────────────────────
for label in sorted(ALT_FILES, key=extract_pop):
    fpath = DATA_DIR / ALT_FILES[label]
    if not fpath.exists():
        print(f"  {label:>{w}}  {'[file not found]':>16}")
        continue

    df_alt   = load_csv(fpath)
    n        = min(len(df_ref), len(df_alt))
    avg_diff = (df_alt[COL].iloc[:n].values - df_ref[COL].iloc[:n].values).mean()
    best_diff = df_alt[COL].min() - ref_best

    print(f"  {label:>{w}}  {avg_diff:>+16.1f}  {best_diff:>+16.1f}")