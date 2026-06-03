import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------------------------------
# Load data
# -----------------------------------------------------
script_dir = Path(__file__).parent


data_EVE = pd.read_csv(
    script_dir / "EVE.csv",
    usecols=[0, 1]
)

data_EVEE = pd.read_csv(
    script_dir / "EVEE.csv",
    usecols=[0, 1]
)

data_VE = pd.read_csv(
    script_dir / "VE.csv",
    usecols=[0, 1]
)

data_VEE = pd.read_csv(
    script_dir / "VEE.csv",
    usecols=[0, 1]
)

data_VVE = pd.read_csv(
    script_dir / "VVE.csv",
    usecols=[0, 1]
)

data_VVEE = pd.read_csv(
    script_dir / "VVEE.csv",
    usecols=[0, 1]
)



# -----------------------------------------------------
# Format columns
# -----------------------------------------------------

data_EVE.columns = ["Launch date", "Delta V [m/s]"]
data_EVEE.columns = ["Launch date", "Delta V [m/s]"]
data_VE.columns = ["Launch date", "Delta V [m/s]"]
data_VEE.columns = ["Launch date", "Delta V [m/s]"]
data_VVE.columns = ["Launch date", "Delta V [m/s]"]
data_VVEE.columns = ["Launch date", "Delta V [m/s]"]

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

output_file = r"C:\Users\gonza\DSE\minimum_delta_v_trajectory_by_date.xlsx"

best_trajectory_each_day.to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")


print(best_trajectory_each_day[[
    "Launch day",
    "Delta V [m/s]",
    "Trajectory"
]].head(30))


best_below_8000 = best_trajectory_each_day[
    best_trajectory_each_day["Delta V [m/s]"] <= 8000
].copy()

plt.figure()

for trajectory in best_below_8000["Trajectory"].unique():
    subset = best_below_8000[
        best_below_8000["Trajectory"] == trajectory
    ]

    plt.scatter(
        subset["Launch day"],
        subset["Delta V [m/s]"],
        marker="o",
        label=trajectory
    )

plt.xlabel("Launch date")
plt.ylabel("Minimum Delta V [m/s]")
plt.grid(True)
plt.legend(title="Best trajectory")
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
optimal_launches = best_below_8000.copy()

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