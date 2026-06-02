import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup
from tudatpy.astro.time_representation import DateTime


# ============================================================
# Helper functions
# ============================================================

def get_state(body, epoch):
    return spice.get_body_cartesian_state_at_epoch(
        target_body_name=body,
        observer_body_name="Jupiter",
        reference_frame_name="J2000",
        aberration_corrections="NONE",
        ephemeris_time=epoch
    )


def two_body_acceleration(r, mu):
    return -mu * r / np.linalg.norm(r)**3


def rk4_step(state, dt, mu):
    def f(s):
        r = s[:3]
        v = s[3:]
        a = two_body_acceleration(r, mu)
        return np.hstack((v, a))

    k1 = f(state)
    k2 = f(state + 0.5*dt*k1)
    k3 = f(state + 0.5*dt*k2)
    k4 = f(state + dt*k3)

    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0


def propagate_jupiter_two_body(initial_state, tof, mu_jupiter, n_steps=1200):
    state = initial_state.copy()
    dt = tof / n_steps

    for _ in range(n_steps):
        state = rk4_step(state, dt, mu_jupiter)

    return state


def propagate_full_arc(initial_state, tof, mu_jupiter, n_steps=4000):
    states = np.zeros((n_steps + 1, 6))
    states[0, :] = initial_state

    dt = tof / n_steps

    for i in range(n_steps):
        states[i+1, :] = rk4_step(states[i, :], dt, mu_jupiter)

    return states


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
radius_ganymede = bodies.get("Ganymede").shape_model.average_radius


# ============================================================
# Mission dates
# ============================================================

science_start_epoch = DateTime(2026, 6, 1).to_epoch()

science_duration_days = 170.0
science_duration = science_duration_days * 24 * 3600

departure_epoch = science_start_epoch + science_duration

europa_state_dep = get_state("Europa", departure_epoch)

r_europa_dep = europa_state_dep[:3]
v_europa_dep = europa_state_dep[3:]


# ============================================================
# Europa parking orbit
# ============================================================

parking_altitude = 100.0e3
r_parking = radius_europa + parking_altitude

v_circ_europa = np.sqrt(mu_europa / r_parking)
v_esc_europa = np.sqrt(2 * mu_europa / r_parking)


# ============================================================
# Optimization variables
# ============================================================
# x[0] = transfer time [days]
# x[1] = Europa departure v_inf magnitude [m/s]
# x[2] = azimuth angle [rad]
# x[3] = elevation angle [rad]
#
# The optimizer searches for trajectories that arrive close to Ganymede.
# The objective strongly penalizes missing Ganymede, then minimizes ΔV.

tof_min_days = 5.0
tof_max_days = 5.0 * 365.25

v_inf_min = 0.0
v_inf_max = 2500.0  # m/s

bounds = [
    (tof_min_days, tof_max_days),
    (v_inf_min, v_inf_max),
    (0.0, 2*np.pi),
    (-0.5*np.pi, 0.5*np.pi)
]


def objective(x):
    tof_days = x[0]
    v_inf_mag = x[1]
    azimuth = x[2]
    elevation = x[3]

    tof = tof_days * 24 * 3600
    arrival_epoch = departure_epoch + tof

    direction = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ])

    v_inf_vec = v_inf_mag * direction

    spacecraft_initial_state = np.hstack((
        r_europa_dep,
        v_europa_dep + v_inf_vec
    ))

    spacecraft_final_state = propagate_jupiter_two_body(
        spacecraft_initial_state,
        tof,
        mu_jupiter,
        n_steps=1000
    )

    ganymede_state_arr = get_state("Ganymede", arrival_epoch)

    miss_distance = np.linalg.norm(
        spacecraft_final_state[:3] - ganymede_state_arr[:3]
    )

    dv_departure = np.sqrt(v_inf_mag**2 + v_esc_europa**2) - v_circ_europa

    # Strong miss-distance penalty.
    # If miss distance is small, ΔV dominates.
    miss_penalty = 1000.0 * (miss_distance / radius_ganymede)**2

    return dv_departure/1000 + miss_penalty


# ============================================================
# Run global optimization
# ============================================================

result = differential_evolution(
    objective,
    bounds,
    strategy="best1bin",
    maxiter=120,
    popsize=20,
    tol=1e-6,
    polish=True,
    workers=1,
    updating="immediate",
    seed=42
)

best_x = result.x

best_tof_days = best_x[0]
best_v_inf_mag = best_x[1]
best_azimuth = best_x[2]
best_elevation = best_x[3]

best_tof = best_tof_days * 24 * 3600
best_arrival_epoch = departure_epoch + best_tof

best_direction = np.array([
    np.cos(best_elevation) * np.cos(best_azimuth),
    np.cos(best_elevation) * np.sin(best_azimuth),
    np.sin(best_elevation)
])

best_v_inf_vec = best_v_inf_mag * best_direction

best_initial_state = np.hstack((
    r_europa_dep,
    v_europa_dep + best_v_inf_vec
))

best_final_state = propagate_jupiter_two_body(
    best_initial_state,
    best_tof,
    mu_jupiter,
    n_steps=3000
)

ganymede_state_arr = get_state("Ganymede", best_arrival_epoch)

miss_distance = np.linalg.norm(
    best_final_state[:3] - ganymede_state_arr[:3]
)

dv_departure = np.sqrt(best_v_inf_mag**2 + v_esc_europa**2) - v_circ_europa


# ============================================================
# Print results
# ============================================================

print("Optimized long-duration Europa-to-Ganymede disposal")
print("---------------------------------------------------")
print(f"Science start date: 1 June 2026")
print(f"Science duration: {science_duration_days:.1f} days")
print(f"Search range: {tof_min_days:.1f} to {tof_max_days:.1f} days")
print()
print(f"Best transfer time: {best_tof_days:.2f} days")
print(f"Europa departure v_inf: {best_v_inf_mag/1000:.3f} km/s")
print(f"Europa parking orbit velocity: {v_circ_europa/1000:.3f} km/s")
print(f"Europa escape velocity: {v_esc_europa/1000:.3f} km/s")
print(f"Required Europa departure ΔV: {dv_departure/1000:.3f} km/s")
print(f"Required Europa departure ΔV with 20% margin: {1.2*dv_departure/1000:.3f} km/s")
print()
print(f"Miss distance at Ganymede arrival: {miss_distance/1000:.2f} km")
print(f"Ganymede radius: {radius_ganymede/1000:.2f} km")

if miss_distance <= radius_ganymede:
    print("Result: impact trajectory found.")
else:
    print("Result: close approach found, but not direct impact. Increase maxiter/popsize or search range.")


# ============================================================
# Propagate and plot best trajectory
# ============================================================

transfer_states = propagate_full_arc(
    best_initial_state,
    best_tof,
    mu_jupiter,
    n_steps=5000
)

times = np.linspace(departure_epoch, best_arrival_epoch, len(transfer_states))

europa_positions = np.zeros((len(times), 3))
ganymede_positions = np.zeros((len(times), 3))

for i, epoch in enumerate(times):
    europa_positions[i, :] = get_state("Europa", epoch)[:3]
    ganymede_positions[i, :] = get_state("Ganymede", epoch)[:3]


# ============================================================
# 3D plot
# ============================================================

fig = plt.figure(figsize=(10, 9), dpi=125)
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Optimized Long-Duration Europa-to-Ganymede Disposal Trajectory")

# Jupiter
ax.scatter(
    0.0,
    0.0,
    0.0,
    s=300,
    color="tab:orange",
    label="Jupiter"
)

# Europa path
ax.plot(
    europa_positions[:, 0]/1000,
    europa_positions[:, 1]/1000,
    europa_positions[:, 2]/1000,
    linestyle="--",
    color="tab:blue",
    label="Europa trajectory"
)

# Ganymede path
ax.plot(
    ganymede_positions[:, 0]/1000,
    ganymede_positions[:, 1]/1000,
    ganymede_positions[:, 2]/1000,
    linestyle="--",
    color="tab:green",
    label="Ganymede trajectory"
)

# Spacecraft path
ax.plot(
    transfer_states[:, 0]/1000,
    transfer_states[:, 1]/1000,
    transfer_states[:, 2]/1000,
    linewidth=2.0,
    color="tab:red",
    label="SoIaF disposal trajectory"
)

# Departure and arrival markers
ax.scatter(
    r_europa_dep[0]/1000,
    r_europa_dep[1]/1000,
    r_europa_dep[2]/1000,
    color="tab:blue",
    s=100,
    label="Europa at departure"
)

ax.scatter(
    ganymede_state_arr[0]/1000,
    ganymede_state_arr[1]/1000,
    ganymede_state_arr[2]/1000,
    color="tab:green",
    s=100,
    label="Ganymede at arrival"
)

ax.scatter(
    transfer_states[0, 0]/1000,
    transfer_states[0, 1]/1000,
    transfer_states[0, 2]/1000,
    color="black",
    marker="x",
    s=80,
    label="Departure"
)

ax.scatter(
    transfer_states[-1, 0]/1000,
    transfer_states[-1, 1]/1000,
    transfer_states[-1, 2]/1000,
    color="black",
    marker="o",
    s=80,
    label="Arrival"
)

ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")

ax.set_aspect("equal")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()