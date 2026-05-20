"""
Io Hyperbolic Flyby Trajectory Visualizer
==========================================
Plots Jupiter's moon Io alongside a configurable hyperbolic flyby trajectory.

PARAMETERS (edit below):
  V_INF_KMS   – hyperbolic excess speed [km/s] (speed at infinity)
  R_CLOSEST_KM – closest approach distance from Io's *centre* [km]
                 (must be > R_IO = 1821.6 km to avoid surface impact)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# ─── CONFIGURABLE PARAMETERS ────────────────────────────────────────────────

V_INF_KMS    = 40.0       # km/s  – hyperbolic excess speed (∞ speed)
R_CLOSEST_KM = 40.0     # km above Io surface → centre distance = R_IO + this

# ─── PHYSICAL CONSTANTS ─────────────────────────────────────────────────────

R_IO_KM   = 1821.6                # Io mean radius [km]
M_IO_KG   = 8.9319e22             # Io mass [kg]
G         = 6.67430e-11           # gravitational constant [m³ kg⁻¹ s⁻²]
MU_IO     = G * M_IO_KG * 1e-9   # Io gravitational parameter [km³/s²]

# ─── DERIVED ORBITAL ELEMENTS ────────────────────────────────────────────────

r_p   = R_IO_KM + R_CLOSEST_KM   # periapsis (closest approach) [km]
v_inf = V_INF_KMS                 # hyperbolic excess speed [km/s]

# Orbital mechanics of a hyperbolic trajectory:
#   semi-major axis (negative for hyperbola): a = -μ / v_inf²
#   eccentricity:  e = 1 + r_p * v_inf² / μ
#   semi-latus rectum: p = r_p * (1 + e)
#   polar equation: r(θ) = p / (1 + e·cos θ)

a   = -MU_IO / v_inf**2
e   = 1.0 + r_p * v_inf**2 / MU_IO
p   = r_p * (1.0 + e)

# Asymptote half-angle (angle where r → ∞)
theta_inf = np.arccos(-1.0 / e)   # radians

# Periapsis velocity (vis-viva / hyperbolic)
v_peri = np.sqrt(v_inf**2 + 2.0 * MU_IO / r_p)

# ─── BUILD TRAJECTORY ────────────────────────────────────────────────────────

# Sweep θ just inside the asymptote limits (leave a small margin)
margin = 0.03   # radians
theta  = np.linspace(-theta_inf + margin, theta_inf - margin, 3000)
r      = p / (1.0 + e * np.cos(theta))

x_traj = r * np.cos(theta)
y_traj = r * np.sin(theta)

# Speed along trajectory (vis-viva for hyperbola: v² = v_inf² + 2μ/r)
speed  = np.sqrt(v_inf**2 + 2.0 * MU_IO / r)

# Velocity direction (tangent to trajectory)
dx = np.gradient(x_traj)
dy = np.gradient(y_traj)
ds = np.hypot(dx, dy)

# Asymptote lines (drawn lightly for context)
# Direction of asymptotes: angle ±theta_inf from x-axis
asym_len = r_p * 8
asym_angles = [theta_inf, -theta_inf]

# ─── CLOSEST APPROACH MARKER ─────────────────────────────────────────────────
# Periapsis is at θ = 0  →  (r_p, 0)
x_peri, y_peri = r_p, 0.0

# ─── VELOCITY ARROWS ─────────────────────────────────────────────────────────
arrow_indices = [200, 600, 1000, 1500, 2000, 2400, 2800]
arrow_scale   = r_p * 0.45   # scale for display

# ─── PLOTTING ────────────────────────────────────────────────────────────────

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor("#0a0d14")
ax.set_facecolor("#0a0d14")

# --- star field background ---
rng = np.random.default_rng(42)
n_stars = 350
sx = rng.uniform(-r_p * 9, r_p * 9, n_stars)
sy = rng.uniform(-r_p * 9, r_p * 9, n_stars)
sz = rng.power(3, n_stars) * 1.8 + 0.2
ax.scatter(sx, sy, s=sz, c="white", alpha=0.5, linewidths=0, zorder=0)

# --- Io body ---
# Simple two-tone: dark volcanic surface + sulphur-yellow patches
io_body = plt.Circle((0, 0), R_IO_KM, color="#c8a832", zorder=4, linewidth=0)
ax.add_patch(io_body)

# Volcanic / sulphur texture via small random patches
rng2 = np.random.default_rng(7)
for _ in range(120):
    ang  = rng2.uniform(0, 2 * np.pi)
    dist = rng2.uniform(0, 0.88) * R_IO_KM
    cx   = dist * np.cos(ang)
    cy   = dist * np.sin(ang)
    rad  = rng2.uniform(0.03, 0.18) * R_IO_KM
    col  = rng2.choice(["#8b6914", "#e8d060", "#b03010", "#d4a020",
                         "#602808", "#f0e080", "#783018"])
    spot = plt.Circle((cx, cy), rad, color=col, zorder=5, linewidth=0,
                       alpha=rng2.uniform(0.5, 0.95))
    ax.add_patch(spot)

# Io outline glow
for lw, alpha in [(14, 0.04), (8, 0.08), (3, 0.25), (1.2, 0.85)]:
    ring = plt.Circle((0, 0), R_IO_KM, fill=False, edgecolor="#f0d060",
                       linewidth=lw, alpha=alpha, zorder=6)
    ax.add_patch(ring)

# --- asymptote lines ---
for ang in asym_angles:
    ax.plot([0, asym_len * np.cos(ang)], [0, asym_len * np.sin(ang)],
            color="#3a4a6a", linewidth=0.8, linestyle="--",
            alpha=0.5, zorder=2, label="_nolegend_")

# --- trajectory coloured by speed ---
norm_speed = (speed - speed.min()) / (speed.max() - speed.min())
cmap_traj  = LinearSegmentedColormap.from_list(
    "flyby", ["#1e60d8", "#30b0f0", "#a8e8ff", "#ffffff", "#ffe060", "#ff7020"])

points  = np.array([x_traj, y_traj]).T.reshape(-1, 1, 2)
segs    = np.concatenate([points[:-1], points[1:]], axis=1)
lc      = LineCollection(segs, cmap=cmap_traj, linewidth=2.4, zorder=8,
                          alpha=0.92)
lc.set_array(norm_speed[:-1])
ax.add_collection(lc)

# Glow duplicate (thicker, low alpha)
lc_glow = LineCollection(segs, cmap=cmap_traj, linewidth=7, zorder=7, alpha=0.18)
lc_glow.set_array(norm_speed[:-1])
ax.add_collection(lc_glow)

# --- velocity arrows ---
for idx in arrow_indices:
    if idx >= len(x_traj) - 1:
        continue
    norm = ds[idx] if ds[idx] > 0 else 1
    ux, uy = dx[idx] / norm, dy[idx] / norm
    c = cmap_traj(norm_speed[idx])
    ax.annotate(
        "", xy=(x_traj[idx] + ux * arrow_scale, y_traj[idx] + uy * arrow_scale),
        xytext=(x_traj[idx], y_traj[idx]),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4,
                        mutation_scale=10),
        zorder=9)

# --- periapsis marker ---
ax.plot(x_peri, y_peri, "o", color="#ff5540", markersize=7,
        markeredgecolor="white", markeredgewidth=0.8, zorder=10)
ax.annotate(f"Periapsis\n{r_p - R_IO_KM:.0f} km alt\n{v_peri:.2f} km/s",
            xy=(x_peri, y_peri),
            xytext=(x_peri + r_p * 0.15, y_peri + r_p * 0.55),
            color="#ff9988", fontsize=8.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="#ff5540", lw=1.0),
            zorder=10)

# --- Io label ---
ax.text(0, R_IO_KM * 1.12, "Io", color="#f0d060", fontsize=14,
        fontweight="bold", ha="center", va="bottom", zorder=11,
        fontfamily="monospace")

# --- colourbar (speed) ---
sm = plt.cm.ScalarMappable(cmap=cmap_traj,
                            norm=plt.Normalize(speed.min(), speed.max()))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02, aspect=25)
cbar.set_label("Spacecraft speed  [km/s]", color="#aabbcc", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#aabbcc", labelcolor="#aabbcc")

# --- axes formatting ---
lim = r_p * 5.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim * 0.85, lim * 0.85)
ax.set_aspect("equal")
ax.tick_params(colors="#4a6a8a", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#1e2e40")
ax.set_xlabel("km", color="#4a6a8a", fontsize=9)
ax.set_ylabel("km", color="#4a6a8a", fontsize=9)
ax.grid(color="#1a2535", linewidth=0.5, linestyle=":", alpha=0.6)

# --- title + parameter box ---
ax.set_title("Io Hyperbolic Flyby Trajectory", color="#ccddef",
             fontsize=15, fontweight="bold", pad=14, fontfamily="monospace")

param_text = (
    f"$v_\\infty$ = {v_inf:.2f} km/s\n"
    f"$r_{{closest}}$ = {R_CLOSEST_KM:.0f} km (alt)\n"
    f"$r_p$ = {r_p:.1f} km\n"
    f"Eccentricity  $e$ = {e:.4f}\n"
    f"Turn angle  $\\Delta$ = {2*(np.pi/2 - theta_inf)*180/np.pi:.2f}°\n"
    f"$v_{{peri}}$ = {v_peri:.3f} km/s"
)
ax.text(0.014, 0.985, param_text, transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5, color="#99bbdd",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1520",
                  edgecolor="#2a3f55", alpha=0.85))

# Io radius note
ax.set_xlim(-3000, 3000)
ax.set_ylim(-3000, 3000)
ax.text(0.986, 0.014,
        f"Io radius = {R_IO_KM:.0f} km",
        transform=ax.transAxes, va="bottom", ha="right",
        fontsize=7.5, color="#5a7a9a", fontfamily="monospace")

plt.tight_layout()

out = "io_flyby_trajectory.png"
plt.savefig(out, dpi=160, facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()