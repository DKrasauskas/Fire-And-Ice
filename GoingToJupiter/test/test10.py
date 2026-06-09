"""
MGA Multi-Sequence Analysis
============================
Reads all CSV files from DATA_FOLDER. Each file = one gravity-assist sequence.

Expected CSV columns (from MGA Configurator output):
  launch_date, delta_v_ms, total_tof_days,
  dv_node_0_{Body}_ms, dv_node_1_{Body}_ms, ...

Produces four plots:
  1.  Total ΔV vs launch date       (shape = sequence, color = ToF)
  2.  Mass at Jupiter, single Isp   (shape = sequence, color = ToF)
  3.  Mass at Jupiter, dual Isp     (shape = sequence, color = ToF)
  4.  Stacked bar: ΔV per mission phase (departure / flyby / arrival)
"""

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ==============================================================================
# CONFIGURATION  — only edit this section
# ==============================================================================

# Folder that contains the CSV files (use "." for current directory)
DATA_FOLDER = "."

# Filter out failed trajectories above this threshold [m/s]
MAX_VALID_DV_MS = 6_500

# ── Mass calculation ──────────────────────────────────────────────────────────
START_MASS  = 15_000.0    # [kg]  total wet mass at Earth departure (spacecraft + transfer stage)
G0          = 9.80665    # [m/s²]

# Transfer stage: used for Earth departure only, then jettisoned.
# STAGE_MF = propellant mass fraction of the transfer stage.
#   0.9 means 90 % of the stage mass is propellant, 10 % is dry structure.
# The stage total mass and dry mass are derived from the ΔV of node 0 alone.
STAGE_MF    = 0.9

# Plot 2: single Isp used for every node
ISP_ALL     = 340        # [s]

# Plot 3: separate Isp per mission phase
ISP_DEP     = 430        # [s]  transfer-stage engine  (node 0 only)
ISP_CRUISE  = 340        # [s]  spacecraft engine      (all subsequent nodes)

# ── Appearance ────────────────────────────────────────────────────────────────
TOF_CMAP    = "plasma"   # colormap for Time of Flight
MARKER_SIZE = 45         # scatter marker size
ALPHA       = 0.85       # marker transparency

# ── Show / hide entire figures ────────────────────────────────────────────────
SHOW_DV_PLOT      = True   # Plot 1: Total ΔV
SHOW_MASS_SINGLE  = True   # Plot 2: mass at Jupiter, single Isp
SHOW_MASS_DUAL    = True   # Plot 3: mass at Jupiter, dual Isp
SHOW_DV_BREAKDOWN = True   # Plot 4: stacked ΔV per mission phase


# ==============================================================================
# HELPERS
# ==============================================================================

MARKERS  = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '<', '>']
_NODE_RE = re.compile(r'dv_node_(\d+)_(\w+)_ms')


def _get_cmap(name: str):
    try:                            # matplotlib >= 3.7
        return matplotlib.colormaps[name]
    except AttributeError:
        return plt.cm.get_cmap(name)


def parse_node_columns(df: pd.DataFrame) -> list:
    """Return [(node_index, body_name, col_name), …] sorted by node index."""
    nodes = []
    for col in df.columns:
        m = _NODE_RE.match(col)
        if m:
            nodes.append((int(m.group(1)), m.group(2), col))
    return sorted(nodes, key=lambda x: x[0])


def mass_at_arrival(row, node_cols: list, isp_dep: float, isp_cruise: float) -> float:
    """
    Compute spacecraft mass at the final node (Jupiter) accounting for:

      Node 0  — Earth departure burn using isp_dep.
                After the burn the transfer stage dry mass is jettisoned.
                  stage_total   = m_propellant_used / STAGE_MF
                  stage_dry     = stage_total × (1 − STAGE_MF)
                                = m_propellant_used × (1 − STAGE_MF) / STAGE_MF

      Node 1+ — All remaining burns use isp_cruise; no further staging.

    Falls back to total delta_v_ms with isp_cruise if no per-node columns exist.
    """
    if not node_cols:
        return START_MASS * np.exp(-row["delta_v_ms"] / (isp_cruise * G0))

    m = START_MASS
    for node_idx, _body, col in node_cols:
        dv  = float(row[col])
        isp = isp_dep if node_idx == 0 else isp_cruise

        if isp <= 0 or dv < 0:
            continue

        m_before = m
        m *= np.exp(-dv / (isp * G0))          # Tsiolkovsky — propellant consumed

        if node_idx == 0:
            # Jettison transfer stage dry structure after departure burn
            m_prop      = m_before - m                              # propellant used [kg]
            stage_total = m_prop / STAGE_MF                        # total stage mass [kg]
            print(stage_total)
            stage_dry   = stage_total * (1.0 - STAGE_MF)           # dry / structural [kg]
            m          -= stage_dry
            if m <= 0.0:
                return 0.0

    return m


def load_datasets(folder: str) -> dict:
    """Load every CSV in folder. Returns {sequence_label: DataFrame}."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(folder)}")
    datasets = {}
    for f in files:
        label = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f, parse_dates=["launch_date"])
        df = df[df["delta_v_ms"] <= MAX_VALID_DV_MS].copy()
        if df.empty:
            print(f"  [warning] {label}: no valid rows after filtering — skipped")
            continue
        datasets[label] = df
        print(f"  Loaded {label}: {len(df)} launch opportunities")
    return datasets


def build_tof_norm(datasets: dict) -> mcolors.Normalize:
    """Shared ToF colormap normalizer across all datasets."""
    all_tof = pd.concat([df["total_tof_days"] for df in datasets.values()])
    return mcolors.Normalize(vmin=all_tof.min(), vmax=all_tof.max())


# ==============================================================================
# PLOTS 1–3: scatter
# ==============================================================================

def make_scatter_plot(title: str, ylabel: str, y_fn,
                      datasets: dict, tof_norm: mcolors.Normalize):
    """
    Scatter: x = launch_date, y = y_fn(df, node_cols),
    shape = sequence (marker), color = ToF (colormap).
    """
    cmap = _get_cmap(TOF_CMAP)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.08, bottom=0.14, top=0.88, right=0.88)

    for i, (label, df) in enumerate(datasets.items()):
        node_cols = parse_node_columns(df)
        x      = df["launch_date"]
        y      = y_fn(df, node_cols)
        colors = cmap(tof_norm(df["total_tof_days"]))
        ax.scatter(x, y, c=colors, marker=MARKERS[i % len(MARKERS)],
                   s=MARKER_SIZE, alpha=ALPHA, zorder=3,
                   edgecolors="none", label=label)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=tof_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Total ToF  [days]", fontsize=9)

    ax.set_xlabel("Launch date", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=30)
    return fig


# ==============================================================================
# PLOT 4: ΔV breakdown stacked bar
# ==============================================================================

DEP_COLOR    = "#2563eb"
FLYBY_COLORS = ["#f59e0b", "#10b981", "#8b5cf6", "#06b6d4"]
ARR_COLOR    = "#dc2626"


def make_dv_breakdown_plot(datasets: dict):
    """
    One subplot per sequence. Stacked bars:
      bottom  = Earth departure ΔV
      middle  = flyby ΔV(s) — one colour per flyby node
      top     = Jupiter arrival ΔV
    """
    n_seq = len(datasets)
    if n_seq == 0:
        return None

    fig, axes = plt.subplots(n_seq, 1,
                              figsize=(13, 4 * n_seq),
                              squeeze=False)
    fig.suptitle("ΔV Breakdown by Mission Phase", fontsize=13, y=1.01)

    for ax, (label, df) in zip(axes[:, 0], datasets.items()):
        node_cols = parse_node_columns(df)

        if not node_cols:
            ax.text(0.5, 0.5, "No per-node ΔV columns in this CSV",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_title(f"Sequence: {label}")
            continue

        n_nodes = len(node_cols)
        dates   = df["launch_date"]
        x       = np.arange(len(dates))
        width   = 0.75

        dv_dep   = df[node_cols[0][2]].values / 1000
        dep_body = node_cols[0][1]
        ax.bar(x, dv_dep, width, label=f"Departure  ({dep_body})",
               color=DEP_COLOR, alpha=ALPHA)
        bottom = dv_dep.copy()

        flyby_nodes = [(idx, body, col)
                       for idx, body, col in node_cols
                       if 0 < idx < n_nodes - 1]
        for fi, (idx, body, col) in enumerate(flyby_nodes):
            dv_fly = df[col].values / 1000
            ax.bar(x, dv_fly, width, bottom=bottom,
                   label=f"Flyby  ({body})",
                   color=FLYBY_COLORS[fi % len(FLYBY_COLORS)], alpha=ALPHA)
            bottom += dv_fly

        dv_arr   = df[node_cols[-1][2]].values / 1000
        arr_body = node_cols[-1][1]
        ax.bar(x, dv_arr, width, bottom=bottom,
               label=f"Arrival  ({arr_body})",
               color=ARR_COLOR, alpha=ALPHA)

        step = max(1, len(dates) // 14)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [d.strftime("%b %Y") for d in dates.iloc[::step]],
            rotation=35, ha="right", fontsize=8)

        ax.set_ylabel("ΔV  [km/s]", fontsize=9)
        ax.set_title(f"Sequence: {label}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    return fig


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\nLoading CSV files from: {os.path.abspath(DATA_FOLDER)}")
    datasets = load_datasets(DATA_FOLDER)
    if not datasets:
        print("No valid data loaded — exiting.")
        return

    tof_norm = build_tof_norm(datasets)

    if SHOW_DV_PLOT:
        make_scatter_plot(
            title    = "Total ΔV  ·  shape: sequence  |  color: ToF",
            ylabel   = "Total ΔV  [km/s]",
            y_fn     = lambda df, _nc: df["delta_v_ms"] / 1000,
            datasets = datasets,
            tof_norm = tof_norm,
        )

    if SHOW_MASS_SINGLE:
        make_scatter_plot(
            title    = (f"Mass at Jupiter  ·  single Isp = {ISP_ALL} s  "
                        f"[start = {START_MASS:.0f} kg,  stage MF = {STAGE_MF}]  "
                        f"|  shape: sequence  |  color: ToF"),
            ylabel   = "Mass at Jupiter  [kg]",
            y_fn     = lambda df, nc: df.apply(
                            lambda row: mass_at_arrival(row, nc, ISP_ALL, ISP_ALL), axis=1),
            datasets = datasets,
            tof_norm = tof_norm,
        )

    if SHOW_MASS_DUAL:
        make_scatter_plot(
            title    = (f"Mass at Jupiter  ·  Isp dep = {ISP_DEP} s  |  "
                        f"Isp cruise = {ISP_CRUISE} s  "
                        f"[start = {START_MASS:.0f} kg,  stage MF = {STAGE_MF}]  "
                        f"|  shape: sequence  |  color: ToF"),
            ylabel   = "Mass at Jupiter  [kg]",
            y_fn     = lambda df, nc: df.apply(
                            lambda row: mass_at_arrival(row, nc, ISP_DEP, ISP_CRUISE), axis=1),
            datasets = datasets,
            tof_norm = tof_norm,
        )

    if SHOW_DV_BREAKDOWN:
        make_dv_breakdown_plot(datasets)

    plt.show()


if __name__ == "__main__":
    main()