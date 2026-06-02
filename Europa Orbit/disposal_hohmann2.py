import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup


# ============================================================
# Setup
# ============================================================

spice.load_standard_kernels()

bodies_to_create = ["Jupiter", "Europa", "Ganymede"]

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    "Jupiter",
    "J2000"
)

bodies = environment_setup.create_system_of_bodies(body_settings)

mu_jupiter = bodies.get("Jupiter").gravitational_parameter
mu_europa = bodies.get("Europa").gravitational_parameter

radius_europa = bodies.get("Europa").shape_model.average_radius


# ============================================================
# Geometric Europa-to-Ganymede Hohmann transfer
# ============================================================

# Mean orbital radii around Jupiter [m]
r_europa = 671100e3
r_ganymede = 1070400e3

# Hohmann transfer semi-major axis
a_transfer = 0.5 * (r_europa + r_ganymede)

# Circular velocities around Jupiter
v_europa = np.sqrt(mu_jupiter / r_europa)
v_ganymede = np.sqrt(mu_jupiter / r_ganymede)

# Transfer velocities around Jupiter
v_transfer_periapsis = np.sqrt(
    mu_jupiter * (2.0/r_europa - 1.0/a_transfer)
)

v_transfer_apoapsis = np.sqrt(
    mu_jupiter * (2.0/r_ganymede - 1.0/a_transfer)
)

# Hohmann ΔV values in Jupiter-centred frame
dv_departure_jupiter_frame = v_transfer_periapsis - v_europa
dv_arrival_jupiter_frame = v_ganymede - v_transfer_apoapsis

# Transfer time
transfer_time = np.pi * np.sqrt(a_transfer**3 / mu_jupiter)


# ============================================================
# Convert Jupiter-frame Hohmann departure ΔV to Europa orbit burn
# ============================================================

parking_altitude = 100.0e3
r_parking = radius_europa + parking_altitude

v_circ_parking = np.sqrt(mu_europa / r_parking)
v_esc_parking = np.sqrt(2.0 * mu_europa / r_parking)

# Required hyperbolic excess velocity with respect to Europa
v_inf_europa = abs(dv_departure_jupiter_frame)

# Burn from circular Europa orbit into hyperbolic escape
dv_escape_from_europa = np.sqrt(
    v_esc_parking**2 + v_inf_europa**2
) - v_circ_parking

# Disposal by Ganymede impact: no arrival burn
dv_total_impact = dv_escape_from_europa


# ============================================================
# Print results
# ============================================================

print("Geometric Europa-to-Ganymede Hohmann transfer")
print("---------------------------------------------")
print(f"Europa orbital radius around Jupiter: {r_europa/1000:.0f} km")
print(f"Ganymede orbital radius around Jupiter: {r_ganymede/1000:.0f} km")
print(f"Transfer semi-major axis: {a_transfer/1000:.0f} km")
print()
print(f"Europa circular velocity around Jupiter: {v_europa/1000:.3f} km/s")
print(f"Ganymede circular velocity around Jupiter: {v_ganymede/1000:.3f} km/s")
print()
print(f"Departure ΔV in Jupiter frame: {dv_departure_jupiter_frame/1000:.3f} km/s")
print(f"Arrival relative velocity at Ganymede: {abs(dv_arrival_jupiter_frame)/1000:.3f} km/s")
print(f"Transfer time: {transfer_time/(24*3600):.2f} days")
print()
print("Including escape from 100 km Europa orbit")
print("-----------------------------------------")
print(f"Europa parking orbit altitude: {parking_altitude/1000:.1f} km")
print(f"Europa parking orbit velocity: {v_circ_parking/1000:.3f} km/s")
print(f"Europa escape velocity: {v_esc_parking/1000:.3f} km/s")
print(f"Required Europa v_inf: {v_inf_europa/1000:.3f} km/s")
print(f"Europa departure burn: {dv_escape_from_europa/1000:.3f} km/s")
print()
print(f"Total ΔV for Ganymede impact disposal: {dv_total_impact/1000:.3f} km/s")
print(f"Impact-disposal ΔV with 20% margin: {1.2*dv_total_impact/1000:.3f} km/s")


# ============================================================
# Plot transfer geometry
# ============================================================

theta = np.linspace(0.0, np.pi, 1000)

e_transfer = (r_ganymede - r_europa) / (r_ganymede + r_europa)
p_transfer = a_transfer * (1.0 - e_transfer**2)

r_transfer = p_transfer / (1.0 + e_transfer * np.cos(theta))

x_transfer = r_transfer * np.cos(theta)
y_transfer = r_transfer * np.sin(theta)
z_transfer = np.zeros_like(theta)

theta_orbit = np.linspace(0.0, 2.0*np.pi, 1000)

x_europa = r_europa * np.cos(theta_orbit)
y_europa = r_europa * np.sin(theta_orbit)

x_ganymede = r_ganymede * np.cos(theta_orbit)
y_ganymede = r_ganymede * np.sin(theta_orbit)

fig = plt.figure(figsize=(9, 8), dpi=125)
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Geometric Europa-to-Ganymede Hohmann Transfer")

# Jupiter
ax.scatter(
    0.0,
    0.0,
    0.0,
    color="tab:orange",
    s=300,
    label="Jupiter"
)

# Europa orbit
ax.plot(
    x_europa/1000,
    y_europa/1000,
    np.zeros_like(x_europa),
    linestyle="--",
    color="tab:blue",
    label="Europa orbit"
)

# Ganymede orbit
ax.plot(
    x_ganymede/1000,
    y_ganymede/1000,
    np.zeros_like(x_ganymede),
    linestyle="--",
    color="tab:green",
    label="Ganymede orbit"
)

# Transfer ellipse
ax.plot(
    x_transfer/1000,
    y_transfer/1000,
    z_transfer/1000,
    color="tab:red",
    linewidth=2.0,
    label="Hohmann transfer"
)

# Europa at departure
ax.scatter(
    r_europa/1000,
    0.0,
    0.0,
    color="tab:blue",
    s=100,
    label="Europa at departure"
)

# Ganymede at arrival
ax.scatter(
    -r_ganymede/1000,
    0.0,
    0.0,
    color="tab:green",
    s=100,
    label="Ganymede at arrival"
)

ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")

ax.set_aspect("equal")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
