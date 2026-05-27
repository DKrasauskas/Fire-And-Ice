"""
Monthly Launch Window Sweep
===========================
Runs a full DE optimisation for each calendar month in the sweep range.
Departure window = first day to last day of each month.

Paste your trajectory configuration in the CONFIG section below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import calendar

import pygmo as pg
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.dynamics import environment_setup
from tudatpy.astro.time_representation import DateTime


# ── Helpers ───────────────────────────────────────────────────────────────────

J2000 = datetime(2000, 1, 1, 12, 0, 0)

def epoch_to_date(epoch_seconds):
    return J2000 + timedelta(seconds=float(epoch_seconds))

def convert_trajectory_parameters(transfer_trajectory_object, trajectory_parameters):
    node_times, leg_free_parameters, node_free_parameters = [], [], []
    departure_time = trajectory_parameters[0]
    node_times.append(departure_time)
    accumulated_time = departure_time
    for i in range(transfer_trajectory_object.number_of_nodes - 1):
        accumulated_time += trajectory_parameters[i + 1]
        node_times.append(accumulated_time)
    for _ in range(transfer_trajectory_object.number_of_legs):
        leg_free_parameters.append([])
    for _ in range(transfer_trajectory_object.number_of_nodes):
        node_free_parameters.append([])
    return node_times, leg_free_parameters, node_free_parameters


class TransferTrajectoryProblem:
    def __init__(self, transfer_trajectory_object,
                 departure_date_lb, departure_date_ub,
                 legs_tof_lb, legs_tof_ub):
        self.departure_date_lb = departure_date_lb
        self.departure_date_ub = departure_date_ub
        self.legs_tof_lb = legs_tof_lb
        self.legs_tof_ub = legs_tof_ub
        self.transfer_trajectory_function = lambda: transfer_trajectory_object

    def get_bounds(self):
        obj = self.transfer_trajectory_function()
        n = obj.number_of_nodes
        lb = [0.0] * n
        ub = [0.0] * n
        lb[0] = self.departure_date_lb
        ub[0] = self.departure_date_ub
        for i in range(obj.number_of_legs):
            lb[i + 1] = self.legs_tof_lb[i]
            ub[i + 1] = self.legs_tof_ub[i]
        return lb, ub

    def get_number_of_parameters(self):
        return self.transfer_trajectory_function().number_of_nodes

    def fitness(self, trajectory_parameters):
        traj = self.transfer_trajectory_function()
        node_times, leg_fp, node_fp = convert_trajectory_parameters(traj, trajectory_parameters)
        try:
            traj.evaluate(node_times, leg_fp, node_fp)
            return [traj.delta_v]
        except Exception:
            return [1e10]


# ── CONFIG — edit this section ────────────────────────────────────────────────

# Sweep range: every month from SWEEP_START to SWEEP_END (inclusive)
SWEEP_START = datetime(2032, 3, 1)
SWEEP_END   = datetime(2038, 4, 1)

# Planet sequence
transfer_body_order = ["Earth","Mars", "Earth", "Earth", "Jupiter"]

# Departure orbit (np.inf = hyperbolic / unbound)
departure_semi_major_axis = 6700000
departure_eccentricity    = 0.0

# Arrival orbit
arrival_semi_major_axis = 9.4237e9
arrival_eccentricity    = 0.89

# Time-of-flight bounds per leg [days] — one row per leg
#   [ [min_days, max_days], ... ]
TOF_BOUNDS_DAYS = [
    [100,  400],   # Earth  → Mars
    [50,  400],   # Mars  → Earth
    [300,  800],   # Earth  → Earth
    [400, 1200],  # Earth  → Jupiter
]

# DE optimiser settings
POPULATION_SIZE      = 20
GENERATIONS_PER_EVOL = 10
NUM_EVOLUTIONS       = 800
SEED                 = 4444
F                    = 0.5

# ── Build trajectory object ────────────────────────────────────────────────────

bodies = environment_setup.create_simplified_system_of_bodies()

transfer_leg_settings, transfer_node_settings = (
    transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
)

transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    "Sun",
)

n_legs = transfer_trajectory_object.number_of_legs
legs_tof_lb = np.array([b[0] for b in TOF_BOUNDS_DAYS]) * constants.JULIAN_DAY
legs_tof_ub = np.array([b[1] for b in TOF_BOUNDS_DAYS]) * constants.JULIAN_DAY


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_sweep():
    # Build list of (first_day, last_day) tuples for each month in range
    months = []
    cur = SWEEP_START.replace(day=1)
    while cur <= SWEEP_END:
        last_day = calendar.monthrange(cur.year, cur.month)[1]
        months.append((cur, cur.replace(day=last_day)))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    records = []
    n = len(months)

    for idx, (first, last) in enumerate(months):
        dep_lb = (first - J2000).total_seconds()
        dep_ub = (last  - J2000).total_seconds()

        optimizer = TransferTrajectoryProblem(
            transfer_trajectory_object,
            dep_lb, dep_ub,
            legs_tof_lb, legs_tof_ub,
        )

        prob = pg.problem(optimizer)
        algo = pg.algorithm(pg.de(gen=GENERATIONS_PER_EVOL, seed=SEED, F=F))
        pop  = pg.population(prob, size=POPULATION_SIZE, seed=SEED)

        for _ in range(NUM_EVOLUTIONS):
            pop = algo.evolve(pop)

        best_x        = pop.champion_x
        dv            = pop.champion_f[0]
        departure     = epoch_to_date(best_x[0])
        total_tof_days = float(np.sum(best_x[1:])) / constants.JULIAN_DAY

        print(
            f"[{idx+1:>3}/{n}]  {first.strftime('%b %Y')}  "
            f"departure {departure.strftime('%d %b %Y')}  |  "
            f"ΔV {dv/1000:6.2f} km/s  |  "
            f"ToF {total_tof_days:.0f} d"
        )

        records.append({
            "launch_date"    : departure,
            "delta_v_ms"     : dv,
            "total_tof_days" : total_tof_days,
        })

    return pd.DataFrame(records)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_sweep(df):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    c_dv  = "#2563eb"
    c_tof = "#ea580c"

    ax1.plot(df["launch_date"], df["delta_v_ms"] / 1000,
             color=c_dv, marker="o", ms=4, lw=1.5, label="ΔV [km/s]")
    ax1.set_ylabel("Total ΔV  [km/s]", color=c_dv)
    ax1.tick_params(axis="y", labelcolor=c_dv)
    ax1.set_xlabel("Launch date")

    ax2 = ax1.twinx()
    ax2.plot(df["launch_date"], df["total_tof_days"],
             color=c_tof, marker="s", ms=4, lw=1.5, ls="--", label="ToF [days]")
    ax2.set_ylabel("Total transfer time  [days]", color=c_tof)
    ax2.tick_params(axis="y", labelcolor=c_tof)

    best = df.loc[df["delta_v_ms"].idxmin()]
    ax1.scatter([best["launch_date"]], [best["delta_v_ms"] / 1000],
                color="red", zorder=5, s=80,
                label=f"Best  {best['delta_v_ms']/1000:.2f} km/s  ({best['launch_date'].strftime('%b %Y')})")

    lines  = ax1.get_lines() + ax2.get_lines() + [ax1.collections[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title(f"Monthly launch opportunities  ·  {' → '.join(transfer_body_order)}")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = run_sweep()

    print("\n", df.to_string(index=False))

    # Optional: save to CSV
    df.to_csv("launch_window_sweep.csv", index=False)

    plot_sweep(df)
