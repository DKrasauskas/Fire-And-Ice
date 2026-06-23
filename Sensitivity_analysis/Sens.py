import numpy as np
import matplotlib.pyplot as plt
from Prop import size_prop
from EPS import size_eps
from mech import size_structure
from ADCS import size_adcs

g0     = 9.80665
margin = 1.2

DV_EUROPA = 1859.0
DV_IO     = 1358.0

#                         Europa   Io
M_ADCS_EU   =  34.02  ;  M_ADCS_IO   =  30.11   # ADCS hardware (fixed — not resized)
M_CDH_EU    =  10.50  ;  M_CDH_IO    =  10.50
M_TTC_EU    =  38.42  ;  M_TTC_IO    =  30.19
M_GNC_EU    =   1.80  ;  M_GNC_IO    =   1.80
M_EPS_EU    = 258.76  ;  M_EPS_IO    = 249.32
M_PROP_HW_EU=  85.30  ;  M_PROP_HW_IO=  68.65   # propulsion hardware (tanks, thrusters)
M_MECH_EU   = 102.10  ;  M_MECH_IO   = 105.30   # structure/mechanical (fixed from table)

# Fixed dry mass: all hardware subsystems including ADCS, structure/mech.
# size_structure() is NOT called — it returns the same structural mass already
# captured here as M_MECH, so using it separately would double-count.
FIXED_EU = (
    M_ADCS_EU +
    M_CDH_EU +
    M_TTC_EU +
    M_GNC_EU +
    M_EPS_EU +
    M_PROP_HW_EU
)

FIXED_IO = (
    M_ADCS_IO +
    M_CDH_IO +
    M_TTC_IO +
    M_GNC_IO +
    M_EPS_IO +
    M_PROP_HW_IO
)

# Thermal fraction (back-calculated from table totals)
M_THERMAL_FRAC_EU = 33.88738 / 530.9023   # ≈ 6.38 %
M_THERMAL_FRAC_IO = 31.65132 / 495.8706   # ≈ 6.38 %

M_RADIATION_EU = 164.0   # fixed shield mass [kg]
M_RADIATION_IO = 245.8

# Baseline power numbers (used by size_eps)
P_NOM_IO_BASE  = 484.05
P_NOM_EU_BASE  = 492.35
P_SAFE_IO_BASE = 351.40
P_SAFE_EU_BASE = 366.40


def converge(fixed_eu, fixed_io, dv_eu, dv_io,
             P_nom_io, P_nom_eu, P_safe_io, P_safe_eu,
             n_iter=10):

    wet_eu = fixed_eu + M_RADIATION_EU
    wet_io = fixed_io + M_RADIATION_IO

    for _ in range(n_iter):

        eps = size_eps(
            P_nom_io, P_nom_eu,
            P_safe_io, P_safe_eu
        )

# Allows negative deltas (mass savings)
        m_eps_eu_add = (eps["m_batt_Eu"] + eps["m_sa_Eu"]) - M_EPS_EU
        m_eps_io_add = (eps["m_batt_Io"] + eps["m_sa_Io"]) - M_EPS_IO

        # Dry mass before structure
        dry_eu = fixed_eu + M_RADIATION_EU + m_eps_eu_add
        dry_io = fixed_io + M_RADIATION_IO + m_eps_io_add

        # First prop estimate
        prop_eu = size_prop(dry_eu, dv_eu)
        prop_io = size_prop(dry_io, dv_io)

        # Structure sized from wet mass
        struct = size_structure(
            prop_eu["wet"],
            prop_io["wet"]
        )

        m_struct_eu = struct["m_struct_europa"]
        m_struct_io = struct["m_struct_io"]

        dry_eu += m_struct_eu
        dry_io += m_struct_io

        # Thermal mass
        subtotal_eu = fixed_eu + m_eps_eu_add + m_struct_eu
        subtotal_io = fixed_io + m_eps_io_add + m_struct_io

        dry_eu += subtotal_eu * M_THERMAL_FRAC_EU
        dry_io += subtotal_io * M_THERMAL_FRAC_IO

        # Final propellant
        prop_eu = size_prop(dry_eu, dv_eu)
        prop_io = size_prop(dry_io, dv_io)

        wet_eu = prop_eu["wet"]
        wet_io = prop_io["wet"]

    return {
        "wet_eu": wet_eu,
        "wet_io": wet_io,
        "total": wet_eu + wet_io,

        "m_eps_eu_add": m_eps_eu_add,
        "m_eps_io_add": m_eps_io_add,

        "m_struct_eu": m_struct_eu,
        "m_struct_io": m_struct_io,

        "thermal_eu": subtotal_eu * M_THERMAL_FRAC_EU,
        "thermal_io": subtotal_io * M_THERMAL_FRAC_IO,

        "dry_eu": dry_eu,
        "dry_io": dry_io,

        "prop_mass_eu": wet_eu - dry_eu,
        "prop_mass_io": wet_io - dry_io
    }

# ── Baseline ──────────────────────────────────────────────────────────────────
baseline = converge(
    FIXED_EU, FIXED_IO,
    DV_EUROPA, DV_IO,
    P_NOM_IO_BASE, P_NOM_EU_BASE,
    P_SAFE_IO_BASE, P_SAFE_EU_BASE,
)
print("\n" + "="*70)
print("BASELINE MASS BREAKDOWN (0% DEVIATION)")
print("="*70)

print("\nEUROPA ORBITER")
print(f"ADCS                  : {M_ADCS_EU:8.2f} kg")
print(f"CDH                   : {M_CDH_EU:8.2f} kg")
print(f"TTC                   : {M_TTC_EU:8.2f} kg")
print(f"GNC                   : {M_GNC_EU:8.2f} kg")
print(f"EPS Hardware          : {M_EPS_EU + baseline['m_eps_eu_add']:8.2f} kg")
print(f"Propulsion Hardware   : {M_PROP_HW_EU:8.2f} kg")
print(f"Structure             : {baseline['m_struct_eu']:8.2f} kg")
print(f"Thermal               : {baseline['thermal_eu']:8.2f} kg")
print(f"Radiation Shielding   : {M_RADIATION_EU:8.2f} kg")
print(f"Propellant            : {baseline['prop_mass_eu']:8.2f} kg")
print("-"*50)
print(f"Dry Mass              : {baseline['dry_eu']:8.2f} kg")
print(f"Wet Mass              : {baseline['wet_eu']:8.2f} kg")

print("\nIO ORBITER")
print(f"ADCS                  : {M_ADCS_IO:8.2f} kg")
print(f"CDH                   : {M_CDH_IO:8.2f} kg")
print(f"TTC                   : {M_TTC_IO:8.2f} kg")
print(f"GNC                   : {M_GNC_IO:8.2f} kg")
print(f"EPS Hardware          : {M_EPS_IO + baseline['m_eps_io_add']:8.2f} kg")
print(f"Propulsion Hardware   : {M_PROP_HW_IO:8.2f} kg")
print(f"Structure             : {baseline['m_struct_io']:8.2f} kg")
print(f"Thermal               : {baseline['thermal_io']:8.2f} kg")
print(f"Radiation Shielding   : {M_RADIATION_IO:8.2f} kg")
print(f"Propellant            : {baseline['prop_mass_io']:8.2f} kg")
print("-"*50)
print(f"Dry Mass              : {baseline['dry_io']:8.2f} kg")
print(f"Wet Mass              : {baseline['wet_io']:8.2f} kg")

print("\nMISSION TOTAL")
print("-"*50)
print(f"Total Wet Mass        : {baseline['total']:8.2f} kg")
print("="*70)
M_base = baseline["total"]
print(f"Baseline total wet mass : {M_base:.1f} kg")
print(f"  Europa wet            : {baseline['wet_eu']:.1f} kg  (table: 1334.5 kg)")
print(f"  Io     wet            : {baseline['wet_io']:.1f} kg  (table: 1200.6 kg)")

# ── Parameter sweep ───────────────────────────────────────────────────────────
factors = np.linspace(-0.20, 0.20, 41)

results_dry = []
results_pow = []
results_dv  = []

for f in factors:
    # Dry-mass sweep: scale all fixed hardware masses together
    r = converge(
        FIXED_EU * (1 + f), FIXED_IO * (1 + f),
        DV_EUROPA, DV_IO,
        P_NOM_IO_BASE, P_NOM_EU_BASE, P_SAFE_IO_BASE, P_SAFE_EU_BASE,
    )
    results_dry.append(r["total"])

    # Power sweep: scale power requirements (drives EPS sizing)
    r = converge(
        FIXED_EU, FIXED_IO,
        DV_EUROPA, DV_IO,
        P_NOM_IO_BASE  * (1 + f), P_NOM_EU_BASE  * (1 + f),
        P_SAFE_IO_BASE * (1 + f), P_SAFE_EU_BASE * (1 + f),
    )
    results_pow.append(r["total"])

    # Delta-V sweep
    r = converge(
        FIXED_EU, FIXED_IO,
        DV_EUROPA * (1 + f), DV_IO * (1 + f),
        P_NOM_IO_BASE, P_NOM_EU_BASE, P_SAFE_IO_BASE, P_SAFE_EU_BASE,
    )
    results_dv.append(r["total"])

results_dry = np.array(results_dry)
results_pow = np.array(results_pow)
results_dv  = np.array(results_dv)

pct = factors * 100

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Factor':>8}  {'Dry mass':>12}  {'Power':>12}  {'Delta-V':>12}")
print("-" * 52)
for i, f in enumerate(factors):
    if abs(round(f * 100) % 5) < 1e-6:
        print(f"{f*100:+7.0f}%  "
              f"{results_dry[i]:>10.1f} kg  "
              f"{results_pow[i]:>10.1f} kg  "
              f"{results_dv[i]:>10.1f} kg")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(pct, results_dry, label="Dry mass ±20%", color="#E15759", lw=2)
ax.plot(pct, results_pow, label="Power ±20%",    color="#4E79A7", lw=2)
ax.plot(pct, results_dv,  label="ΔV ±20%",       color="#59A14F", lw=2)
ax.axhline(M_base, color="black", lw=1, ls="--", label=f"Baseline {M_base:.0f} kg")
ax.axvline(0, color="grey", lw=0.7, ls=":")

ax.set_xlabel("Parameter variation [%]")
ax.set_ylabel("Total wet mass [kg]")
ax.set_title("SoIaF — mass sensitivity analysis")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved sensitivity.png")
