import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


# -----------------------------------------------------
# Load data
# -----------------------------------------------------
script_dir = Path(__file__).parent


data_EVE = pd.read_csv(
    script_dir / "EVE.csv",
    usecols=[0, 1, 2]
)

data_EVEE = pd.read_csv(
    script_dir / "EVEE.csv",
    usecols=[0, 1, 2]
)

data_VE = pd.read_csv(
    script_dir / "VE.csv",
    usecols=[0, 1, 2]
)

data_VEE = pd.read_csv(
    script_dir / "VEE.csv",
    usecols=[0, 1, 2]
)

data_VVE = pd.read_csv(
    script_dir / "VVE.csv",
    usecols=[0, 1, 2]
)

data_VVEE = pd.read_csv(
    script_dir / "VVEE.csv",
    usecols=[0, 1, 2]
)



# -----------------------------------------------------
# Format columns
# -----------------------------------------------------

data_EVE.columns = ["Launch date", "Delta V [m/s]", "Transfer time [days]"]
data_EVEE.columns = ["Launch date", "Delta V [m/s]", "Transfer time [days]"]
data_VE.columns = ["Launch date", "Delta V [m/s]", "Transfer time [days]"]
data_VEE.columns = ["Launch date", "Delta V [m/s]", "Transfer time [days]"]
data_VVE.columns = ["Launch date", "Delta V [m/s]", "Transfer time [days]"]
data_VVEE.columns = ["Launch date", "Delta V [m/s]", "Transfer time [days]"]

data_EVE["Trajectory"] = "EVE"
data_EVEE["Trajectory"] = "EVEE"
data_VE["Trajectory"] = "VE"
data_VEE["Trajectory"] = "VEE"
data_VVE["Trajectory"] = "VVE"
data_VVEE["Trajectory"] = "VVEE"


# -----------------------------------------------------
# Combine all trajectories into one DataFrame
# -----------------------------------------------------

all_data = pd.concat(
    [data_VE, data_VEE, data_VVE, data_VVEE],
    ignore_index=True
)


# Convert dates and Delta V values
all_data["Launch date"] = pd.to_datetime(all_data["Launch date"])
all_data["Delta V [m/s]"] = pd.to_numeric(all_data["Delta V [m/s]"]) - 3000 - 1000


# Keep only the calendar date, ignoring hours/minutes/seconds
all_data["Launch day"] = all_data["Launch date"].dt.date


# -----------------------------------------------------
# Find minimum Delta V trajectory for each launch day
# -----------------------------------------------------

idx_min = all_data.groupby("Launch day")["Delta V [m/s]"].idxmin()

best_trajectory_each_day = all_data.loc[idx_min].copy()

best_trajectory_each_day = best_trajectory_each_day.sort_values("Launch day")


# -----------------------------------------------------
# Save results
# -----------------------------------------------------

output_file = script_dir / "minimum_delta_v_trajectory_by_date.xlsx"

best_trajectory_each_day.to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")


print(best_trajectory_each_day[[
    "Launch day",
    "Delta V [m/s]",
    "Transfer time [days]",
    "Trajectory"
]].head(30))

# -----------------------------------------------------
# Plot minimum Delta V values <= 8000 m/s
# Color = transfer time bin
# Marker = trajectory
# -----------------------------------------------------

best_below_4000 = best_trajectory_each_day[
    best_trajectory_each_day["Delta V [m/s]"] <= 4000
].copy()

# Convert transfer time from days to years
best_below_4000["Transfer time [years]"] = (
    best_below_4000["Transfer time [days]"] / 365.25
)

# Define transfer-time bins
bins = [0, 1, 2, 3, 4, 5, 6, np.inf]

labels = [
    "0-1 years",
    "1-2 years",
    "2-3 years",
    "3-4 years",
    "4-5 years",
    "5-6 years",
    ">6 years"
]

best_below_4000["Transfer time bin"] = pd.cut(
    best_below_4000["Transfer time [years]"],
    bins=bins,
    labels=labels,
    right=False
)

# Colours for transfer-time bins
time_bin_colours = {
    "0-1 years": "tab:blue",
    "1-2 years": "tab:orange",
    "2-3 years": "tab:green",
    "3-4 years": "tab:red",
    "4-5 years": "tab:purple",
    "5-6 years": "tab:brown",
    ">6 years": "tab:pink"
}

# Markers for trajectories
trajectory_markers = {
    "VE": "o",
    "VEE": "x",
    "VVE": "^",
    "VVEE": "s",
    "EVE": "D",
    "EVEE": "*"
}

# Only include transfer-time bins and trajectories that actually appear
used_time_bins = best_below_4000["Transfer time bin"].dropna().unique()
used_trajectories = best_below_4000["Trajectory"].dropna().unique()


plt.figure()

for trajectory, marker in trajectory_markers.items():
    for time_bin, colour in time_bin_colours.items():

        subset = best_below_4000[
            (best_below_4000["Trajectory"] == trajectory) &
            (best_below_4000["Transfer time bin"] == time_bin)
        ]

        if subset.empty:
            continue

        plt.scatter(
            subset["Launch day"],
            subset["Delta V [m/s]"],
            marker=marker,
            color=colour
        )


# -----------------------------------------------------
# Legend 1: colours = transfer time
# only includes transfer-time bins present in the data
# -----------------------------------------------------

colour_legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=time_bin,
        markerfacecolor=time_bin_colours[time_bin],
        markersize=8
    )
    for time_bin in labels
    if time_bin in used_time_bins
]

colour_legend = plt.legend(
    handles=colour_legend_handles,
    title="Transfer time",
    loc="upper right"
)

plt.gca().add_artist(colour_legend)


# -----------------------------------------------------
# Legend 2: markers = trajectory
# only includes trajectories present in the data
# -----------------------------------------------------

marker_legend_handles = [
    Line2D(
        [0],
        [0],
        marker=trajectory_markers[trajectory],
        color="black",
        label=trajectory,
        linestyle="None",
        markersize=8
    )
    for trajectory in trajectory_markers
    if trajectory in used_trajectories
]

plt.legend(
    handles=marker_legend_handles,
    title="Trajectory",
    loc="upper left"
)


plt.xlabel("Launch date")
plt.ylabel("Minimum Delta V [m/s]")
plt.title("Optimal launch windows by transfer time and trajectory")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# Spacecraft propulsion sizing for optimal launch dates
# -----------------------------------------------------

def compute_spacecraft_propulsion_mass_fraction(
    delta_v,
    Isp,
    g0,
    spacecraft_dry_mass
):
    """
    Computes spacecraft propellant mass and mass fractions when the transfer
    is performed by the spacecraft propulsion system directly.
    """

    delta_v = np.asarray(delta_v)

    mass_ratio = np.exp(delta_v / (Isp * g0))

    spacecraft_prop_mass = spacecraft_dry_mass * (mass_ratio - 1)

    initial_spacecraft_mass = spacecraft_dry_mass + spacecraft_prop_mass

    final_spacecraft_mass = spacecraft_dry_mass

    spacecraft_propellant_fraction = (
        spacecraft_prop_mass / initial_spacecraft_mass
    )

    spacecraft_final_mass_fraction = (
        final_spacecraft_mass / initial_spacecraft_mass
    )

    return {
        "mass_ratio": mass_ratio,
        "spacecraft_prop_mass": spacecraft_prop_mass,
        "initial_spacecraft_mass": initial_spacecraft_mass,
        "final_spacecraft_mass": final_spacecraft_mass,
        "spacecraft_propellant_fraction": spacecraft_propellant_fraction,
        "spacecraft_final_mass_fraction": spacecraft_final_mass_fraction,
    }


# -----------------------------------------------------
# Known / assumed parameters
# -----------------------------------------------------

Isp = 345                 # s
g0 = 9.80665              # m/s^2
spacecraft_dry_mass = 3715   # kg, excluding transfer propellant


# Use only the optimal launch opportunities below 8000 m/s
optimal_launches = best_below_4000.copy()

delta_v_optimal = optimal_launches["Delta V [m/s]"].to_numpy()


# -----------------------------------------------------
# Compute spacecraft propellant sizing
# -----------------------------------------------------

results = compute_spacecraft_propulsion_mass_fraction(
    delta_v=delta_v_optimal,
    Isp=Isp,
    g0=g0,
    spacecraft_dry_mass=spacecraft_dry_mass
)


optimal_launches["Mass ratio [-]"] = results["mass_ratio"]
optimal_launches["Spacecraft dry mass [kg]"] = spacecraft_dry_mass
optimal_launches["Spacecraft propellant mass [kg]"] = results["spacecraft_prop_mass"]
optimal_launches["Initial spacecraft mass [kg]"] = results["initial_spacecraft_mass"]
optimal_launches["Final spacecraft mass [kg]"] = results["final_spacecraft_mass"]
optimal_launches["Spacecraft propellant fraction [-]"] = results["spacecraft_propellant_fraction"]
optimal_launches["Spacecraft final mass fraction [-]"] = results["spacecraft_final_mass_fraction"]


# -----------------------------------------------------
# Save and print results
# -----------------------------------------------------

output_file = r"C:\Users\gonza\DSE\optimal_launches_spacecraft_propulsion_results.xlsx"
optimal_launches.to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")

print(optimal_launches[[
    "Launch day",
    "Trajectory",
    "Delta V [m/s]",
    "Mass ratio [-]",
    "Spacecraft propellant mass [kg]",
    "Initial spacecraft mass [kg]",
    "Spacecraft propellant fraction [-]"
]].head(30))

# -----------------------------------------------------
# Plot propellant mass for optimal launch opportunities
# -----------------------------------------------------

plt.figure()

for trajectory in optimal_launches["Trajectory"].unique():
    subset = optimal_launches[
        optimal_launches["Trajectory"] == trajectory
    ]

    plt.scatter(
        subset["Launch day"],
        subset["Spacecraft propellant mass [kg]"],
        marker="o",
        label=trajectory
    )

plt.xlabel("Launch date")
plt.ylabel("Required spacecraft propellant mass [kg]")
plt.title("Required propellant mass for optimal launch opportunities")
plt.grid(True)
plt.legend(title="Best trajectory")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()