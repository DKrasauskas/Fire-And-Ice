import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).parent


def compute_transfer_stage_from_dry_fraction(
    delta_v,
    Isp,
    g0,
    spacecraft_mass,
    transfer_stage_dry_fraction
):
    """
    Computes transfer-stage dry mass, propellant mass, total mass, and mass fractions
    when the transfer-stage dry mass fraction is given instead of the dry mass.

    Parameters
    ----------
    delta_v : float or array-like
        Required Delta V [m/s].
    Isp : float
        Specific impulse [s].
    g0 : float
        Standard gravity [m/s^2].
    spacecraft_mass : float
        Total spacecraft mass, including spacecraft dry mass and spacecraft propellant [kg].
    transfer_stage_dry_fraction : float
        Dry mass fraction of the transfer stage [-].
        Defined as:
            transfer_stage_dry_fraction = m_TS_dry / m_TS_total

    Returns
    -------
    dict
        Dictionary containing mass ratio, transfer-stage dry mass,
        propellant mass, total mass, initial total mass, and mass fractions.
    """

    delta_v = np.asarray(delta_v)

    mass_ratio = np.exp(delta_v / (Isp * g0))

    epsilon = transfer_stage_dry_fraction

    # Check feasibility
    denominator = 1 - mass_ratio * epsilon

    transfer_stage_total_mass = np.full_like(mass_ratio, np.nan, dtype=float)

    feasible = denominator > 0

    transfer_stage_total_mass[feasible] = (
        spacecraft_mass * (mass_ratio[feasible] - 1)
        / denominator[feasible]
    )

    transfer_stage_dry_mass = epsilon * transfer_stage_total_mass

    transfer_stage_prop_mass = (
        transfer_stage_total_mass - transfer_stage_dry_mass
    )

    initial_total_mass = spacecraft_mass + transfer_stage_total_mass

    transfer_stage_initial_mass_fraction = (
        transfer_stage_total_mass / initial_total_mass
    )

    transfer_stage_propellant_fraction = (
        transfer_stage_prop_mass / transfer_stage_total_mass
    )

    return {
        "mass_ratio": mass_ratio,
        "transfer_stage_dry_mass": transfer_stage_dry_mass,
        "transfer_stage_prop_mass": transfer_stage_prop_mass,
        "transfer_stage_total_mass": transfer_stage_total_mass,
        "initial_total_mass": initial_total_mass,
        "transfer_stage_initial_mass_fraction": transfer_stage_initial_mass_fraction,
        "transfer_stage_propellant_fraction": transfer_stage_propellant_fraction,
        "feasible": feasible,
    }


file_name = script_dir / "minimum_delta_v_trajectory_by_date.xlsx"

raw_data = pd.read_excel(file_name)

launch_dates = pd.to_datetime(raw_data["Launch day"])
delta_v_values = raw_data["Delta V [m/s]"].to_numpy()
trajectories = raw_data["Trajectory"]


# -----------------------------------------------------
# Known / assumed parameters
# -----------------------------------------------------

Isp = 320                 # s
g0 = 9.80665              # m/s^2
spacecraft_mass = 3715    # kg

# Transfer-stage dry mass fraction
transfer_stage_dry_fraction = 0.10


# -----------------------------------------------------
# Compute results
# -----------------------------------------------------

results = compute_transfer_stage_from_dry_fraction(
    delta_v=delta_v_values,
    Isp=Isp,
    g0=g0,
    spacecraft_mass=spacecraft_mass,
    transfer_stage_dry_fraction=transfer_stage_dry_fraction
)


df = pd.DataFrame({
    "Launch date": launch_dates,
    "Delta V [m/s]": delta_v_values,
    "Mass ratio [-]": results["mass_ratio"],
    "Transfer stage dry fraction [-]": transfer_stage_dry_fraction,
    "Transfer stage dry mass [kg]": results["transfer_stage_dry_mass"],
    "Transfer stage propellant mass [kg]": results["transfer_stage_prop_mass"],
    "Transfer stage total mass [kg]": results["transfer_stage_total_mass"],
    "Initial total mass [kg]": results["initial_total_mass"],
    "Transfer stage initial mass fraction [-]": results["transfer_stage_initial_mass_fraction"],
    "Transfer stage propellant fraction [-]": results["transfer_stage_propellant_fraction"],
    "Feasible?": results["feasible"],
})


# -----------------------------------------------------
# Save and print results
# -----------------------------------------------------

output_file = script_dir / "transfer_stage_mass_fraction_results.xlsx"
df.to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")

print(df[[
    "Launch date",
    "Delta V [m/s]",
    "Mass ratio [-]",
    "Transfer stage dry fraction [-]",
    "Transfer stage dry mass [kg]",
    "Transfer stage propellant mass [kg]",
    "Transfer stage total mass [kg]",
    "Feasible?"
]].head(22))