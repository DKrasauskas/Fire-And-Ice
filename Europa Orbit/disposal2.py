import numpy as np
from matplotlib import pyplot as plt

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
    r_norm = np.linalg.norm(r)
    return -mu * r / r_norm**3


def rk4_step(state, dt, mu):
    def f(s):
        r = s[:3]
        v = s[3:]
        a = two_body_acceleration(r, mu)
        return np.hstack((v, a))

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)

    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def propagate_jupiter_two_body(initial_state, tof, mu_jupiter, n_steps=300):
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


def angle_between_vectors(r1, r2):
    cos_angle = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


# ============================================================
# Tudat / SPICE setup
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
science_duration = science_duration_days * 24.0 * 3600.0

departure_epoch = science_start_epoch + science_duration

europa_state_dep = get_state("Europa", departure_epoch)
ganymede_state_dep = get_state("Ganymede", departure_epoch)

r_europa_dep = europa_state_dep[:3]
v_europa_dep = europa_state_dep[3:]

departure_phase_angle = angle_between_vectors(
    europa_state_dep[:3],
    ganymede_state_dep[:3]
)

departure_distance = np.linalg.norm(
    ganymede_state_dep[:3] - europa_state_dep[:3]
)


# ============================================================
# Europa parking orbit
# ============================================================

parking_altitude = 100.0e3
r_parking = radius_europa + parking_altitude

v_circ_europa = np.sqrt(mu_europa / r_parking)
v_esc_europa = np.sqrt(2.0 * mu_europa / r_parking)


# ============================================================
# Coarse trajectory search
# ============================================================
# This avoids scipy completely.
# It searches:
# - transfer time
# - departure v_inf magnitude
# - departure v_inf direction in Jupiter-centred XY plane
#
# For first run, keep this coarse. Increase values later if needed.



tof_days_array = np.linspace(20.0, 500.0, 40)
v_inf_array = np.linspace(0.2e3, 2.2e3, 20)
azimuth_array = np.linspace(0.0, 2.0*np.pi, 18)

elevation = 0.0

best = None
best_score = np.inf

total_cases = len(tof_days_array) * len(v_inf_array) * len(azimuth_array)
case_counter = 0

print("Mission timing")
print("--------------")
print("Science start: 1 June 2026")
print(f"Science duration: {science_duration_days:.1f} days")
print(f"Disposal departure is {science_duration_days:.1f} days after science start")
print()

print("Moon geometry at disposal departure")
print("-----------------------------------")
print(f"Europa-Ganymede phase angle: {departure_phase_angle:.2f} deg")
print(f"Europa-Ganymede distance: {departure_distance/1000:.0f} km")
print()

print("Starting coarse search")
print("----------------------")
print(f"Transfer time range: {tof_days_array[0]:.1f} to {tof_days_array[-1]:.1f} days")
print(f"v_inf range: {v_inf_array[0]/1000:.2f} to {v_inf_array[-1]/1000:.2f} km/s")
print(f"Total cases: {total_cases}")
print()

for tof_days in tof_days_array:

    tof = tof_days * 24.0 * 3600.0
    arrival_epoch = departure_epoch + tof
    ganymede_state_arr = get_state("Ganymede", arrival_epoch)

    for v_inf_mag in v_inf_array:

        for azimuth in azimuth_array:

            case_counter += 1

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
                n_steps=300
            )

            miss_distance = np.linalg.norm(
                spacecraft_final_state[:3] - ganymede_state_arr[:3]
            )

            dv_departure = np.sqrt(v_inf_mag**2 + v_esc_europa**2) - v_circ_europa

            # Score:
            # first prioritize getting close to Ganymede,
            # then prefer lower ΔV.
            score = miss_distance/radius_ganymede + 0.05*(dv_departure/1000.0)

            if score < best_score:
                best_score = score
                best = {
                    "tof_days": tof_days,
                    "tof": tof,
                    "arrival_epoch": arrival_epoch,
                    "v_inf_mag": v_inf_mag,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "v_inf_vec": v_inf_vec,
                    "initial_state": spacecraft_initial_state,
                    "final_state": spacecraft_final_state,
                    "miss_distance": miss_distance,
                    "dv_departure": dv_departure,
                    "ganymede_state_arr": ganymede_state_arr
                }

    print(
        f"Completed TOF = {tof_days:.1f} days | "
        f"Best miss = {best['miss_distance']/1000:.0f} km | "
        f"Best ΔV = {best['dv_departure']/1000:.3f} km/s"
    )


# ============================================================
# Print best coarse-search result
# ============================================================

arrival_epoch = best["arrival_epoch"]

europa_state_arr = get_state("Europa", arrival_epoch)
ganymede_state_arr = get_state("Ganymede", arrival_epoch)

arrival_phase_angle = angle_between_vectors(
    europa_state_arr[:3],
    ganymede_state_arr[:3]
)

arrival_distance = np.linalg.norm(
    ganymede_state_arr[:3] - europa_state_arr[:3]
)

print()
print("Best coarse-search Europa-to-Ganymede disposal trajectory")
print("---------------------------------------------------------")
print(f"Best transfer time: {best['tof_days']:.2f} days")
print(f"Europa departure v_inf: {best['v_inf_mag']/1000:.3f} km/s")
print(f"Departure azimuth: {np.rad2deg(best['azimuth']):.2f} deg")
print(f"Miss distance at Ganymede arrival: {best['miss_distance']/1000:.1f} km")
print(f"Ganymede radius: {radius_ganymede/1000:.1f} km")
print()
print(f"Europa parking orbit altitude: {parking_altitude/1000:.1f} km")
print(f"Europa parking orbit velocity: {v_circ_europa/1000:.3f} km/s")
print(f"Europa escape velocity: {v_esc_europa/1000:.3f} km/s")
print(f"Required Europa departure ΔV: {best['dv_departure']/1000:.3f} km/s")
print(f"Required Europa departure ΔV with 20% margin: {1.2*best['dv_departure']/1000:.3f} km/s")
print()
print("Moon geometry at arrival")
print("------------------------")
print(f"Europa-Ganymede phase angle at arrival: {arrival_phase_angle:.2f} deg")
print(f"Europa-Ganymede distance at arrival: {arrival_distance/1000:.0f} km")

if best["miss_distance"] <= radius_ganymede:
    print("Result: impact trajectory found.")
else:
    print("Result: close approach only. Increase search resolution or use a local optimizer around this result.")


# ============================================================
# Propagate full best trajectory for plotting
# ============================================================

transfer_states = propagate_full_arc(
    best["initial_state"],
    best["tof"],
    mu_jupiter,
    n_steps=5000
)

times = np.linspace(departure_epoch, arrival_epoch, len(transfer_states))

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

ax.set_title("Coarse-Search Europa-to-Ganymede Disposal Trajectory")

# Jupiter
ax.scatter(
    0.0,
    0.0,
    0.0,
    s=300,
    color="tab:orange",
    label="Jupiter"
)

# Europa trajectory during transfer
ax.plot(
    europa_positions[:, 0]/1000,
    europa_positions[:, 1]/1000,
    europa_positions[:, 2]/1000,
    linestyle="--",
    color="tab:blue",
    label="Europa trajectory"
)

# Ganymede trajectory during transfer
ax.plot(
    ganymede_positions[:, 0]/1000,
    ganymede_positions[:, 1]/1000,
    ganymede_positions[:, 2]/1000,
    linestyle="--",
    color="tab:green",
    label="Ganymede trajectory"
)

# Spacecraft trajectory
ax.plot(
    transfer_states[:, 0]/1000,
    transfer_states[:, 1]/1000,
    transfer_states[:, 2]/1000,
    linewidth=2.0,
    color="tab:red",
    label="SoIaF disposal trajectory"
)

# Europa at departure
ax.scatter(
    europa_state_dep[0]/1000,
    europa_state_dep[1]/1000,
    europa_state_dep[2]/1000,
    color="tab:blue",
    s=100,
    label="Europa at departure"
)

# Ganymede at departure
ax.scatter(
    ganymede_state_dep[0]/1000,
    ganymede_state_dep[1]/1000,
    ganymede_state_dep[2]/1000,
    color="lime",
    s=100,
    label="Ganymede at departure"
)

# Europa at arrival
ax.scatter(
    europa_state_arr[0]/1000,
    europa_state_arr[1]/1000,
    europa_state_arr[2]/1000,
    color="cyan",
    s=100,
    label="Europa at arrival"
)

# Ganymede at arrival
ax.scatter(
    ganymede_state_arr[0]/1000,
    ganymede_state_arr[1]/1000,
    ganymede_state_arr[2]/1000,
    color="tab:green",
    s=100,
    label="Ganymede at arrival"
)

# Transfer start and end
ax.scatter(
    transfer_states[0, 0]/1000,
    transfer_states[0, 1]/1000,
    transfer_states[0, 2]/1000,
    color="black",
    marker="x",
    s=80,
    label="Transfer start"
)

ax.scatter(
    transfer_states[-1, 0]/1000,
    transfer_states[-1, 1]/1000,
    transfer_states[-1, 2]/1000,
    color="black",
    marker="o",
    s=80,
    label="Transfer end"
)

ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_zlabel("z [km]")

ax.set_aspect("equal")
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()