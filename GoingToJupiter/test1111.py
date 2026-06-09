import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent


# -----------------------------------------------------
# Function: two-stage dry mass calculator
# -----------------------------------------------------

def calc_dry_mass(m_total, isp_transfer, isp_sc, dv_transfer, dv_sc, epsilon_transfer):
    """
    Calculate spacecraft dry mass after two-stage propulsion sequence.

    Parameters:
        m_total           : Total (wet) mass at launch [kg]
        isp_transfer      : Specific impulse of transfer stage [s]
        isp_sc            : Specific impulse of spacecraft propulsion [s]
        dv_transfer       : Delta-V of transfer stage [m/s]
        dv_sc             : Delta-V of spacecraft [m/s] (array or scalar)
        epsilon_transfer  : Structural coefficient of transfer stage [-]

    Returns:
        m_dry_sc : Dry mass of spacecraft [kg] (array or scalar)
    """
    g0 = 9.80665

    dv_sc = np.asarray(dv_sc)

    mr_transfer = np.exp(dv_transfer / (isp_transfer * g0))
    m_after_transfer = m_total / mr_transfer

    prop_transfer = m_total - m_after_transfer
    m_struct_transfer = epsilon_transfer * prop_transfer

    m_sc_initial = m_after_transfer - m_struct_transfer

    mr_sc = np.exp(dv_sc / (isp_sc * g0))
    m_dry_sc = m_sc_initial / mr_sc

    return m_dry_sc


# -----------------------------------------------------
# Load trajectory data
# -----------------------------------------------------

data_VE   = pd.read_csv(script_dir / "VE.csv",   usecols=[0, 1, 2])
data_VEE  = pd.read_csv(script_dir / "VEE.csv",  usecols=[0, 1, 2])
data_VVE  = pd.read_csv(script_dir / "VVE.csv",  usecols=[0, 1, 2])
data_VVEE = pd.read_csv(script_dir / "VVEE.csv", usecols=[0, 1, 2])
data_VEVE = pd.read_csv(script_dir / "VEVE.csv", usecols=[0, 1, 2])
data_VEM  = pd.read_csv(script_dir / "VEM.csv",  usecols=[0, 1, 2])
data_ME   = pd.read_csv(script_dir / "ME.csv",   usecols=[0, 1, 2])
data_MEE  = pd.read_csv(script_dir / "MEE.csv",  usecols=[0, 1, 2])


# -----------------------------------------------------
# Rename columns and add trajectory labels
# -----------------------------------------------------

for df, label in [
    (data_VE,   "VE"),
    (data_VEE,  "VEE"),
    (data_VVE,  "VVE"),
    (data_VVEE, "VVEE"),
    (data_VEVE, "VEVE"),
    (data_VEM,  "VEM"),
    (data_ME,   "ME"),
    (data_MEE,  "MEE"),
]:
    df.columns = ["Launch date", "Delta V [m/s]", "ToF [days]"]
    df["Trajectory"] = label


# -----------------------------------------------------
# Combine, clean, find optimal launch per day
# -----------------------------------------------------

all_data = pd.concat(
    [data_VE, data_VEE, data_VVE, data_VVEE, data_VEVE, data_VEM, data_ME, data_MEE],
    ignore_index=True
)

all_data["Launch date"] = pd.to_datetime(all_data["Launch date"])
all_data["Delta V [m/s]"] = pd.to_numeric(all_data["Delta V [m/s]"]) - 3000
all_data["Launch day"] = all_data["Launch date"].dt.date

idx_min = all_data.groupby("Launch day")["Delta V [m/s]"].idxmin()
optimal_launches = all_data.loc[idx_min].copy()
optimal_launches = optimal_launches.sort_values("Launch day")
optimal_launches = optimal_launches[
    optimal_launches["Delta V [m/s]"] <= 8000
].copy()


# -----------------------------------------------------
# Inputs
# -----------------------------------------------------

m_total          = 11000   # kg  — total wet mass at launch
isp_transfer     = 440     # s   — transfer stage (e.g. LH2/LOX)
isp_sc           = 320     # s   — spacecraft propulsion (e.g. hypergolic)
dv_transfer      = 4200    # m/s — fixed transfer stage burn
epsilon_transfer = 0.10    # [-] — transfer stage structural coefficient

current_spacecraft_dry_mass = 3715  # kg — feasibility reference line


# -----------------------------------------------------
# Compute dry mass for each optimal launch date
# -----------------------------------------------------

dv_sc_optimal = optimal_launches["Delta V [m/s]"].to_numpy() - dv_transfer

optimal_launches["S/C dry mass [kg]"] = calc_dry_mass(
    m_total          = m_total,
    isp_transfer     = isp_transfer,
    isp_sc           = isp_sc,
    dv_transfer      = dv_transfer,
    dv_sc            = dv_sc_optimal,
    epsilon_transfer = epsilon_transfer,
)

optimal_launches["Current dry mass feasible?"] = (
    optimal_launches["S/C dry mass [kg]"] >= current_spacecraft_dry_mass
)


# -----------------------------------------------------
# Save results
# -----------------------------------------------------

output_file = script_dir / "optimal_launches_dry_mass_results.xlsx"
optimal_launches.to_excel(output_file, index=False)
print(f"Results saved to: {output_file}")

print(optimal_launches[[
    "Launch day",
    "Trajectory",
    "Delta V [m/s]",
    "ToF [days]",
    "S/C dry mass [kg]",
    "Current dry mass feasible?"
]].head(30))


# -----------------------------------------------------
# Plot
# -----------------------------------------------------

plt.figure()

for trajectory in optimal_launches["Trajectory"].unique():
    subset = optimal_launches[optimal_launches["Trajectory"] == trajectory]
    plt.scatter(
        subset["Launch day"],
        subset["S/C dry mass [kg]"],
        marker="o",
        label=trajectory
    )

plt.axhline(
    y=current_spacecraft_dry_mass,
    linestyle="--",
    label=f"Current dry mass = {current_spacecraft_dry_mass} kg"
)

plt.xlabel("Launch date")
plt.ylabel("Spacecraft dry mass [kg]")
plt.title("S/C dry mass by launch date (two-stage rocket equation)")
plt.grid(True)
plt.legend(title="Best trajectory")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()