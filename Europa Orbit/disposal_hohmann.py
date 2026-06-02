import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup
from tudatpy.astro.time_representation import DateTime

# Load SPICE kernels
spice.load_standard_kernels()

# Bodies
bodies_to_create = ["Jupiter", "Europa", "Ganymede"]

global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
)

bodies = environment_setup.create_system_of_bodies(body_settings)

# Gravitational parameters
mu_jupiter = bodies.get("Jupiter").gravitational_parameter
mu_europa = bodies.get("Europa").gravitational_parameter

# Radii
radius_jupiter = bodies.get("Jupiter").shape_model.average_radius
radius_europa = bodies.get("Europa").shape_model.average_radius
radius_ganymede = bodies.get("Ganymede").shape_model.average_radius

# Mean orbital radii around Jupiter [m]
r_europa_orbit = 671100e3
r_ganymede_orbit = 1070400e3

# Europa science orbit before disposal
europa_orbit_altitude = 100e3
r_sc_europa = radius_europa + europa_orbit_altitude

# Europa escape from circular orbit
v_circ_europa = np.sqrt(mu_europa / r_sc_europa)
v_esc_europa = np.sqrt(2 * mu_europa / r_sc_europa)
dv_escape_europa = v_esc_europa - v_circ_europa

# Jupiter-centred Hohmann transfer
r1 = r_europa_orbit
r2 = r_ganymede_orbit

a_transfer = 0.5 * (r1 + r2)

v_europa_jupiter = np.sqrt(mu_jupiter / r1)
v_ganymede_jupiter = np.sqrt(mu_jupiter / r2)

v_transfer_departure = np.sqrt(mu_jupiter * (2/r1 - 1/a_transfer))
v_transfer_arrival = np.sqrt(mu_jupiter * (2/r2 - 1/a_transfer))

dv_hohmann_departure = v_transfer_departure - v_europa_jupiter
v_inf_ganymede = abs(v_ganymede_jupiter - v_transfer_arrival)

transfer_time = np.pi * np.sqrt(a_transfer**3 / mu_jupiter)

dv_total_ideal = dv_escape_europa + dv_hohmann_departure
dv_total_margin = 1.2 * dv_total_ideal

print("Europa disposal transfer estimate")
print("---------------------------------")
print(f"Europa parking orbit altitude: {europa_orbit_altitude/1000:.1f} km")
print(f"Europa circular orbit velocity: {v_circ_europa/1000:.3f} km/s")
print(f"Europa local escape velocity: {v_esc_europa/1000:.3f} km/s")
print(f"Europa escape ΔV: {dv_escape_europa/1000:.3f} km/s")
print()
print(f"Jupiter-centred Hohmann departure ΔV: {dv_hohmann_departure/1000:.3f} km/s")
print(f"Ganymede arrival relative velocity: {v_inf_ganymede/1000:.3f} km/s")
print(f"Transfer time: {transfer_time/(24*3600):.2f} days")
print()
print(f"Ideal disposal ΔV, no Ganymede capture: {dv_total_ideal/1000:.3f} km/s")
print(f"Disposal ΔV with 20% margin: {dv_total_margin/1000:.3f} km/s")

# Build Hohmann transfer trajectory
theta = np.linspace(0, np.pi, 1000)

e_transfer = (r2 - r1) / (r2 + r1)
p_transfer = a_transfer * (1 - e_transfer**2)

r_transfer = p_transfer / (1 + e_transfer * np.cos(theta))

x_transfer = r_transfer * np.cos(theta)
y_transfer = r_transfer * np.sin(theta)
z_transfer = np.zeros_like(theta)

# Circular moon orbits around Jupiter
theta_orbit = np.linspace(0, 2*np.pi, 1000)

x_europa_orbit = r1 * np.cos(theta_orbit)
y_europa_orbit = r1 * np.sin(theta_orbit)
z_europa_orbit = np.zeros_like(theta_orbit)

x_ganymede_orbit = r2 * np.cos(theta_orbit)
y_ganymede_orbit = r2 * np.sin(theta_orbit)
z_ganymede_orbit = np.zeros_like(theta_orbit)

# Moon positions
# Europa is placed at departure point.
europa_departure_position = np.array([r1, 0.0, 0.0])

# Ganymede is placed at arrival point.
ganymede_arrival_position = np.array([-r2, 0.0, 0.0])

# Optional: compute required Ganymede phase angle at departure
n_ganymede = np.sqrt(mu_jupiter / r2**3)
ganymede_phase_departure = np.pi - n_ganymede * transfer_time

ganymede_departure_position = np.array([
    r2 * np.cos(ganymede_phase_departure),
    r2 * np.sin(ganymede_phase_departure),
    0.0
])

print()
print(f"Required Ganymede phase angle at departure: {np.rad2deg(ganymede_phase_departure):.2f} deg")

# Plot
fig = plt.figure(figsize=(9, 8), dpi=125)
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Simplified Europa-to-Ganymede Disposal Hohmann Transfer")

# Orbits
ax.plot(
    x_europa_orbit/1000,
    y_europa_orbit/1000,
    z_europa_orbit/1000,
    linestyle="--",
    color="tab:blue",
    label="Europa orbit around Jupiter"
)

ax.plot(
    x_ganymede_orbit/1000,
    y_ganymede_orbit/1000,
    z_ganymede_orbit/1000,
    linestyle="--",
    color="tab:green",
    label="Ganymede orbit around Jupiter"
)

# Spacecraft transfer
ax.plot(
    x_transfer/1000,
    y_transfer/1000,
    z_transfer/1000,
    color="tab:red",
    linewidth=2.0,
    label="SoIaF Hohmann transfer"
)

# Jupiter
ax.scatter(
    0.0,
    0.0,
    0.0,
    color="tab:orange",
    s=300,
    label="Jupiter"
)

# Europa at departure
ax.scatter(
    europa_departure_position[0]/1000,
    europa_departure_position[1]/1000,
    europa_departure_position[2]/1000,
    color="tab:blue",
    s=80,
    label="Europa at departure"
)

# Ganymede at departure
ax.scatter(
    ganymede_departure_position[0]/1000,
    ganymede_departure_position[1]/1000,
    ganymede_departure_position[2]/1000,
    color="lightgreen",
    s=60,
    label="Ganymede at departure"
)

# Ganymede at arrival
ax.scatter(
    ganymede_arrival_position[0]/1000,
    ganymede_arrival_position[1]/1000,
    ganymede_arrival_position[2]/1000,
    color="tab:green",
    s=100,
    label="Ganymede at arrival"
)

# Start and end points
ax.scatter(
    x_transfer[0]/1000,
    y_transfer[0]/1000,
    z_transfer[0]/1000,
    color="black",
    s=40,
    marker="x",
    label="Transfer start"
)

ax.scatter(
    x_transfer[-1]/1000,
    y_transfer[-1]/1000,
    z_transfer[-1]/1000,
    color="black",
    s=40,
    marker="o",
    label="Transfer end"
)

ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")

ax.set_aspect("equal")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()