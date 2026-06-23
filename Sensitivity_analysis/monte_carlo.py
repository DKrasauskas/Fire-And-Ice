"""
monte_carlo_mass.py
────────────────────────────────────────────────────────────────────────────────
Monte Carlo sensitivity analysis for the SoIaF dual-spacecraft wet-mass model.

Mirrors the structure of the trade-study Monte Carlo (weights + scores sampled
from triangular / normal distributions) but applied to the mass-budget inputs:

  • Fixed hardware masses  → normal, σ = var %
  • Power requirements     → normal, σ = var %
  • Delta-V budgets        → normal, σ = var %
  • Radiation shield mass  → normal, σ = var %
  • Thermal fraction       → triangular (low/mode/high)

N = 100 000 runs.  Output: three plots  +  console summary.
────────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Prop import size_prop
from mech import size_structure
from EPS  import size_eps

# ── Constants ─────────────────────────────────────────────────────────────────
DV_EUROPA = 1859.0
DV_IO     = 1358.0

M_ADCS_EU    =  34.02 ;  M_ADCS_IO    =  30.11
M_CDH_EU     =  10.50 ;  M_CDH_IO     =  10.50
M_TTC_EU     =  38.42 ;  M_TTC_IO     =  30.19
M_GNC_EU     =   1.80 ;  M_GNC_IO     =   1.80
M_EPS_EU     = 258.76 ;  M_EPS_IO     = 249.32
M_PROP_HW_EU =  85.30 ;  M_PROP_HW_IO =  68.65
M_MECH_EU    = 102.10 ;  M_MECH_IO    = 105.30

# FIXED masses now INCLUDE ADCS constants so they scale properly during sweeps.
FIXED_EU = M_CDH_EU + M_TTC_EU + M_GNC_EU + M_EPS_EU + M_PROP_HW_EU + M_MECH_EU + M_ADCS_EU
FIXED_IO = M_CDH_IO + M_TTC_IO + M_GNC_IO + M_EPS_IO + M_PROP_HW_IO + M_MECH_IO + M_ADCS_IO

M_THERMAL_FRAC_EU = 33.88738 / 530.9023
M_THERMAL_FRAC_IO = 31.65132 / 495.8706
M_RADIATION_EU    = 164.0
M_RADIATION_IO    = 245.8

P_NOM_IO_BASE  = 484.05 ;  P_NOM_EU_BASE  = 492.35
P_SAFE_IO_BASE = 351.40 ;  P_SAFE_EU_BASE = 366.40

# ── Deterministic convergence loop ────────────────────────────────────────────
def converge(fixed_eu, fixed_io, dv_eu, dv_io,
             P_nom_io, P_nom_eu, P_safe_io, P_safe_eu,
             rad_eu, rad_io, th_frac_eu, th_frac_io,
             n_iter=10):
    wet_eu = fixed_eu + rad_eu
    wet_io = fixed_io + rad_io

    for _ in range(n_iter):
        eps          = size_eps(P_nom_io, P_nom_eu, P_safe_io, P_safe_eu)
        # Allowed negative mass deltas so power sweeps slope downward correctly
        m_eps_eu_add = (eps["m_batt_Eu"] + eps["m_sa_Eu"]) - M_EPS_EU
        m_eps_io_add = (eps["m_batt_Io"] + eps["m_sa_Io"]) - M_EPS_IO

        # ADCS is now baked into fixed_eu / fixed_io
        dry_eu = fixed_eu + rad_eu + m_eps_eu_add
        dry_io = fixed_io + rad_io + m_eps_io_add

        prop_eu    = size_prop(dry_eu, dv_eu)
        prop_io    = size_prop(dry_io, dv_io)
        wet_eu_tmp = prop_eu["wet"]
        wet_io_tmp = prop_io["wet"]

        struct  = size_structure(wet_eu_tmp, wet_io_tmp)
        dry_eu += struct["m_struct_europa"]
        dry_io += struct["m_struct_io"]

        subtotal_eu = fixed_eu + m_eps_eu_add + struct["m_struct_europa"]
        subtotal_io = fixed_io + m_eps_io_add + struct["m_struct_io"]
        dry_eu += subtotal_eu * th_frac_eu
        dry_io += subtotal_io * th_frac_io

        prop_eu = size_prop(dry_eu, dv_eu)
        prop_io = size_prop(dry_io, dv_io)
        wet_eu  = prop_eu["wet"]
        wet_io  = prop_io["wet"]

    return wet_eu, wet_io


# ── Baseline (deterministic) ──────────────────────────────────────────────────
wet_eu_base, wet_io_base = converge(
    FIXED_EU, FIXED_IO, DV_EUROPA, DV_IO,
    P_NOM_IO_BASE, P_NOM_EU_BASE, P_SAFE_IO_BASE, P_SAFE_EU_BASE,
    M_RADIATION_EU, M_RADIATION_IO,
    M_THERMAL_FRAC_EU, M_THERMAL_FRAC_IO,
)
M_base = wet_eu_base + wet_io_base
print(f"Baseline total wet mass : {M_base:.1f} kg")
print(f"  Europa wet            : {wet_eu_base:.1f} kg")
print(f"  Io     wet            : {wet_io_base:.1f} kg\n")


# ── Uncertain parameters (mirrors trade-study criteria dict) ──────────────────
#
#  Each entry defines how one group of inputs is perturbed:
#    var     : 1-σ fractional uncertainty of the *mean* seed values [%]
#    th_lo/hi: fractional shifts for triangular-distributed params
#
params = [
    {"name": "Fixed hardware masses", "var": 10},   # CDH+TTC+GNC+EPS+PropHW+Mech+ADCS
    {"name": "Power requirements",    "var": 15},
    {"name": "Delta-V budgets",       "var":  5},
    {"name": "Radiation shield mass", "var": 20},
]
# Thermal fraction: triangular  (−20 % / mode / +30 % of baseline value)
TH_FRAC_LO   = 0.80
TH_FRAC_MODE = 1.00
TH_FRAC_HI   = 1.30

# ── Monte Carlo ───────────────────────────────────────────────────────────────
N = 100_000
np.random.seed(42)

def gauss_sample(mu, sigma_frac, size):
    """Normal sample clipped to ±3 σ, always positive."""
    s = sigma_frac / 100.0
    return np.clip(np.random.normal(mu, mu * s, size=size), mu * 0.01, None)

# Sample all stochastic inputs up-front
s_fixed_eu  = gauss_sample(FIXED_EU,         params[0]["var"], N)
s_fixed_io  = gauss_sample(FIXED_IO,         params[0]["var"], N)

s_pnom_io   = gauss_sample(P_NOM_IO_BASE,    params[1]["var"], N)
s_pnom_eu   = gauss_sample(P_NOM_EU_BASE,    params[1]["var"], N)
s_psafe_io  = gauss_sample(P_SAFE_IO_BASE,   params[1]["var"], N)
s_psafe_eu  = gauss_sample(P_SAFE_EU_BASE,   params[1]["var"], N)

s_dv_eu     = gauss_sample(DV_EUROPA,        params[2]["var"], N)
s_dv_io     = gauss_sample(DV_IO,            params[2]["var"], N)

s_rad_eu    = gauss_sample(M_RADIATION_EU,   params[3]["var"], N)
s_rad_io    = gauss_sample(M_RADIATION_IO,   params[3]["var"], N)

s_th_eu     = np.random.triangular(
    TH_FRAC_LO * M_THERMAL_FRAC_EU,
    TH_FRAC_MODE * M_THERMAL_FRAC_EU,
    TH_FRAC_HI * M_THERMAL_FRAC_EU, size=N)
s_th_io     = np.random.triangular(
    TH_FRAC_LO * M_THERMAL_FRAC_IO,
    TH_FRAC_MODE * M_THERMAL_FRAC_IO,
    TH_FRAC_HI * M_THERMAL_FRAC_IO, size=N)

# Run all N cases
wet_eu_mc = np.empty(N)
wet_io_mc = np.empty(N)

print(f"Running {N:,} Monte Carlo iterations …", flush=True)
for i in range(N):
    eu, io = converge(
        s_fixed_eu[i], s_fixed_io[i],
        s_dv_eu[i],    s_dv_io[i],
        s_pnom_io[i],  s_pnom_eu[i],
        s_psafe_io[i], s_psafe_eu[i],
        s_rad_eu[i],   s_rad_io[i],
        s_th_eu[i],    s_th_io[i],
    )
    wet_eu_mc[i] = eu
    wet_io_mc[i] = io
    if (i + 1) % 10_000 == 0:
        print(f"  {i+1:>7,} / {N:,}", flush=True)

total_mc = wet_eu_mc + wet_io_mc

# ── Console summary (mirrors trade-study output) ──────────────────────────────
pcts = [5, 25, 50, 75, 95]
print("\n" + "=" * 60)
print(f"  Monte Carlo  N={N:,}  |  σ from param table  |  thermal: triangular\n")
print(f"  {'Quantity':<28}  {'Mean':>8}  {'Std':>7}  {'P05':>8}  {'P95':>8}")
print("-" * 60)
for label, arr in [("Europa wet mass [kg]", wet_eu_mc),
                   ("Io     wet mass [kg]", wet_io_mc),
                   ("Total  wet mass [kg]", total_mc)]:
    print(f"  {label:<28}  {arr.mean():>8.1f}  {arr.std():>7.1f}"
          f"  {np.percentile(arr,5):>8.1f}  {np.percentile(arr,95):>8.1f}")
print("=" * 60)

margin_mass = np.percentile(total_mc, 95) - M_base
print(f"\n  Baseline total          : {M_base:.1f} kg")
print(f"  Monte Carlo mean        : {total_mc.mean():.1f} kg")
print(f"  95th-percentile mass    : {np.percentile(total_mc,95):.1f} kg")
print(f"  95th-pct margin over BL : +{margin_mass:.1f} kg  "
      f"({margin_mass/M_base*100:.1f} %)\n")

# ── Pearson correlation — which input drives variance most? ───────────────────
inputs = {
    "Fixed hw (EU)":  s_fixed_eu,
    "Fixed hw (Io)":  s_fixed_io,
    "Power (EU)":     s_pnom_eu,
    "Power (Io)":     s_pnom_io,
    "ΔV (EU)":        s_dv_eu,
    "ΔV (Io)":        s_dv_io,
    "Radiation (EU)": s_rad_eu,
    "Radiation (Io)": s_rad_io,
    "Thermal frac EU":s_th_eu,
    "Thermal frac Io":s_th_io,
}
corr = {k: np.corrcoef(v, total_mc)[0, 1] for k, v in inputs.items()}
corr_sorted = dict(sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True))

print("  Pearson r  (input → total wet mass):")
for k, r in corr_sorted.items():
    bar = "█" * int(abs(r) * 30)
    sign = "+" if r >= 0 else "-"
    print(f"    {k:<20}  {sign}{abs(r):.3f}  {bar}")
print()

# ── Plots ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
fig.suptitle(f"SoIaF — Mass Budget Monte Carlo  (N={N:,})", fontsize=13, fontweight='bold')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── 1. Total wet mass distribution ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])

p5, p50, p95 = np.percentile(total_mc, [5, 50, 95])
bins = np.linspace(total_mc.min(), total_mc.max(), 120)
ax1.hist(total_mc, bins=bins, color="#4E79A7", alpha=0.75, edgecolor="white", lw=0.3)
ax1.axvline(M_base,  color="black",     lw=1.8, ls="--",  label=f"Baseline  {M_base:.0f} kg")
ax1.axvline(p50,     color="#E15759",   lw=1.5, ls="-",   label=f"Median    {p50:.0f} kg")
ax1.axvline(p5,      color="#59A14F",   lw=1.2, ls=":",   label=f"P5        {p5:.0f} kg")
ax1.axvline(p95,     color="#F28E2B",   lw=1.2, ls=":",   label=f"P95       {p95:.0f} kg")
ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1],
                  p5, p95, color="#4E79A7", alpha=0.12, label="P5–P95 band")
ax1.set_xlabel("Total wet mass  [kg]", fontsize=10)
ax1.set_ylabel("Count", fontsize=10)
ax1.set_title("Total wet mass distribution (both spacecraft)", fontsize=10)
ax1.legend(fontsize=8.5, framealpha=0.9)
ax1.grid(alpha=0.2)

# ── 2. Europa vs Io scatter ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(wet_eu_mc, wet_io_mc, s=1, alpha=0.15, color="#4E79A7", rasterized=True)
ax2.scatter([wet_eu_base], [wet_io_base], color="red", s=60, zorder=5,
            label=f"Baseline ({wet_eu_base:.0f}, {wet_io_base:.0f})")
ax2.set_xlabel("Europa wet mass  [kg]", fontsize=9)
ax2.set_ylabel("Io wet mass  [kg]", fontsize=9)
ax2.set_title("Europa vs Io wet mass  (joint scatter)", fontsize=9)
ax2.legend(fontsize=8, markerscale=1.5)
ax2.grid(alpha=0.2)

# ── 3. Tornado (Pearson r bar chart) ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
names = list(corr_sorted.keys())
vals  = list(corr_sorted.values())
colors_bar = ["#E15759" if v >= 0 else "#4E79A7" for v in vals]

bars = ax3.barh(names[::-1], vals[::-1], color=colors_bar[::-1], alpha=0.8, height=0.6)
ax3.axvline(0, color="black", lw=0.8)
ax3.set_xlabel("Pearson r  (vs total wet mass)", fontsize=9)
ax3.set_title("Sensitivity tornado  (Pearson r)", fontsize=9)
ax3.set_xlim(-1.05, 1.05)
for bar, v in zip(bars, vals[::-1]):
    ax3.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
             f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
             fontsize=7.5, color="black")
ax3.grid(axis="x", alpha=0.2)

# Fix y-axis labels for ax1 fill (re-draw now axes are finalised)
ax1.fill_betweenx(ax1.get_ylim(), p5, p95,
                  color="#4E79A7", alpha=0.12)

plt.savefig("monte_carlo_mass.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved monte_carlo_mass.png")