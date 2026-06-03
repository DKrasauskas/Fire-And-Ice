import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------------------------------
# Function: maximum dry mass for propellant limit
# -----------------------------------------------------

script_dir = Path(__file__).parent

def compute_max_dry_mass_for_propellant_limit(
    delta_v,
    Isp,
    g0,
    max_propellant_mass
):
    """
    Computes the maximum spacecraft dry mass such that the required
    propellant mass stays below a specified limit.

    From:
        m_prop = m_dry * (mass_ratio - 1)

    Therefore:
        m_dry_max = m_prop_max / (mass_ratio - 1)
    """

    delta_v = np.asarray(delta_v)

    mass_ratio = np.exp(delta_v / (Isp * g0))

    max_dry_mass = max_propellant_mass / (mass_ratio - 1)

    return {
        "mass_ratio": mass_ratio,
        "max_dry_mass": max_dry_mass
    }


# -----------------------------------------------------
# Load trajectory data
# -----------------------------------------------------

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
# Rename columns and add trajectory labels
# -----------------------------------------------------

data_EVE.columns = ["Launch date", "Delta V [m/s]"]
data_EVEE.columns = ["Launch date", "Delta V [m/s]"]
data_VE.columns = ["Launch date", "Delta V [m/s]"]
data_VEE.columns = ["Launch date", "Delta V [m/s]"]
# data_VEM.columns = ["Launch date", "Delta V [m/s]"]
# data_VME.columns = ["Launch date", "Delta V [m/s]"]
# data_VEVE.columns = ["Launch date", "Delta V [m/s]"]
data_VVE.columns = ["Launch date", "Delta V [m/s]"]
data_VVEE.columns = ["Launch date", "Delta V [m/s]"]

data_EVE["Trajectory"] = "EVE"
data_EVEE["Trajectory"] = "EVEE"
data_VE["Trajectory"] = "VE"
data_VEE["Trajectory"] = "VEE"
# data_VEM["Trajectory"] = "VEM"
# data_VME["Trajectory"] = "VME"
# data_VEVE["Trajectory"] = "VEVE"
data_VVE["Trajectory"] = "VVE"
data_VVEE["Trajectory"] = "VVEE"

# -----------------------------------------------------
# Combine all data
# -----------------------------------------------------

all_data = pd.concat(
    [data_EVE, data_EVEE, data_VE, data_VEE, data_VVE, data_VVEE],
    ignore_index=True
)

all_data["Launch date"] = pd.to_datetime(all_data["Launch date"])
all_data["Delta V [m/s]"] = pd.to_numeric(all_data["Delta V [m/s]"]) - 3000

# Compare by calendar day only
all_data["Launch day"] = all_data["Launch date"].dt.date


# -----------------------------------------------------
# Find best trajectory for each launch day
# -----------------------------------------------------

idx_min = all_data.groupby("Launch day")["Delta V [m/s]"].idxmin()

optimal_launches = all_data.loc[idx_min].copy()
optimal_launches = optimal_launches.sort_values("Launch day")


# -----------------------------------------------------
# Optional: only keep optimal launch dates with Delta V <= 8000 m/s
# -----------------------------------------------------

optimal_launches = optimal_launches[
    optimal_launches["Delta V [m/s]"] <= 8000
].copy()


# -----------------------------------------------------
# Inputs
# -----------------------------------------------------

Isp = 345                       # s
g0 = 9.80665                    # m/s^2
max_propellant_mass = 7500      # kg
current_spacecraft_dry_mass = 3715  # kg


# -----------------------------------------------------
# Compute maximum allowed dry mass
# -----------------------------------------------------

delta_v_optimal = optimal_launches["Delta V [m/s]"].to_numpy()

results = compute_max_dry_mass_for_propellant_limit(
    delta_v=delta_v_optimal,
    Isp=Isp,
    g0=g0,
    max_propellant_mass=max_propellant_mass
)

optimal_launches["Mass ratio [-]"] = results["mass_ratio"]
optimal_launches["Maximum allowed dry mass [kg]"] = results["max_dry_mass"]
optimal_launches["Maximum propellant mass [kg]"] = max_propellant_mass

optimal_launches["Current dry mass feasible?"] = (
    optimal_launches["Maximum allowed dry mass [kg]"] >= current_spacecraft_dry_mass
)


# -----------------------------------------------------
# Save results
# -----------------------------------------------------

output_file = script_dir / "optimal_launches_max_dry_mass_results.xlsx"

optimal_launches.to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")

print(optimal_launches[[
    "Launch day",
    "Trajectory",
    "Delta V [m/s]",
    "Mass ratio [-]",
    "Maximum allowed dry mass [kg]",
    "Current dry mass feasible?"
]].head(30))


# -----------------------------------------------------
# Plot maximum allowed dry mass
# -----------------------------------------------------

plt.figure()

for trajectory in optimal_launches["Trajectory"].unique():
    subset = optimal_launches[
        optimal_launches["Trajectory"] == trajectory
    ]

    plt.scatter(
        subset["Launch day"],
        subset["Maximum allowed dry mass [kg]"],
        marker="o",
        label=trajectory
    )

plt.axhline(
    y=current_spacecraft_dry_mass,
    linestyle="--",
    label=f"Current dry mass = {current_spacecraft_dry_mass} kg"
)

plt.xlabel("Launch date")
plt.ylabel("Maximum allowed spacecraft dry mass [kg]")
plt.title("Maximum dry mass for propellant mass ≤ 20 tonnes")
plt.grid(True)
plt.legend(title="Best trajectory")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()