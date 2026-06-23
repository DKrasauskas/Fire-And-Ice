import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup
from tudatpy.astro.time_representation import DateTime


# ============================================================
# Lambert solver
# ============================================================

def stumpff_C(z):
    if z > 0:
        s = np.sqrt(z)
        return (1 - np.cos(s)) / z
    elif z < 0:
        s = np.sqrt(-z)
        return (np.cosh(s) - 1) / (-z)
    else:
        return 0.5


def stumpff_S(z):
    if z > 0:
        s = np.sqrt(z)
        return (s - np.sin(s)) / s**3
    elif z < 0:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / s**3
    else:
        return 1.0 / 6.0


def solve_lambert(r1_vec, r2_vec, tof, mu, prograde=True):
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cos_dtheta = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)

    cross = np.cross(r1_vec, r2_vec)

    if prograde:
        if cross[2] >= 0:
            dtheta = np.arccos(cos_dtheta)
        else:
            dtheta = 2*np.pi - np.arccos(cos_dtheta)
    else:
        if cross[2] < 0:
            dtheta = np.arccos(cos_dtheta)
        else:
            dtheta = 2*np.pi - np.arccos(cos_dtheta)

    if abs(1 - np.cos(dtheta)) < 1e-12:
        raise RuntimeError("Lambert angle too small.")

    A = np.sin(dtheta) * np.sqrt(r1*r2 / (1 - np.cos(dtheta)))

    if abs(A) < 1e-12:
        raise RuntimeError("Lambert A too small.")

    def tof_from_z(z):
        C = stumpff_C(z)
        S = stumpff_S(z)

        if C <= 0:
            return np.inf, None

        y = r1 + r2 + A * (z*S - 1) / np.sqrt(C)

        if y < 0:
            return np.inf, None

        x = np.sqrt(y / C)
        t = (x**3 * S + A*np.sqrt(y)) / np.sqrt(mu)

        return t, y

    z_low = -4*np.pi**2
    z_high = 4*np.pi**2

    for _ in range(200):
        z_mid = 0.5 * (z_low + z_high)
        t_mid, _ = tof_from_z(z_mid)

        if not np.isfinite(t_mid):
            z_low = z_mid
            continue

        if t_mid < tof:
            z_low = z_mid
        else:
            z_high = z_mid

    z = 0.5 * (z_low + z_high)
    _, y = tof_from_z(z)

    f = 1 - y/r1
    g = A * np.sqrt(y/mu)
    gdot = 1 - y/r2

    v1_vec = (r2_vec - f*r1_vec) / g
    v2_vec = (gdot*r2_vec - r1_vec) / g

    return v1_vec, v2_vec


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


def angle_between_vectors(r1, r2):
    cos_angle = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


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

    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def propagate_arc(initial_state, tof, mu, n_steps=3000):
    states = np.zeros((n_steps + 1, 6))
    states[0, :] = initial_state

    dt = tof / n_steps

    for i in range(n_steps):
        states[i+1, :] = rk4_step(states[i, :], dt, mu)

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
# Mission timing
# ============================================================

science_start_epoch = DateTime(2026, 6, 16).to_epoch()

science_duration_days = 170.0
science_duration = science_duration_days * 24 * 3600

departure_epoch = science_start_epoch + science_duration


# ============================================================
# Europa parking orbit
# ============================================================

parking_altitude = 100.0e3
r_parking = radius_europa + parking_altitude

v_circ_europa = np.sqrt(mu_europa / r_parking)
v_esc_europa = np.sqrt(2 * mu_europa / r_parking)


# ============================================================
# Initial moon geometry
# ============================================================

europa_state_dep = get_state("Europa", departure_epoch)
ganymede_state_dep = get_state("Ganymede", departure_epoch)

phase_angle_dep = angle_between_vectors(
    europa_state_dep[:3],
    ganymede_state_dep[:3]
)

distance_dep = np.linalg.norm(
    ganymede_state_dep[:3] - europa_state_dep[:3]
)

print("Mission timing")
print("--------------")
print("Science start: 1 June 2026")
print(f"Science duration: {science_duration_days:.1f} days")
print(f"Disposal departure after science start: {science_duration_days:.1f} days")
print()

print("Moon geometry at disposal departure")
print("-----------------------------------")
print(f"Europa-Ganymede phase angle: {phase_angle_dep:.2f} deg")
print(f"Europa-Ganymede distance: {distance_dep/1000:.0f} km")
print()


# ============================================================
# Direct Lambert search
# ============================================================
# This searches direct transfers that exactly target Ganymede.
# Arrival ΔV is not included because this is disposal by impact.

tof_days_array = np.linspace(5.0, 200.0, 500)

results = []

for tof_days in tof_days_array:

    tof = tof_days * 24 * 3600
    arrival_epoch = departure_epoch + tof

    europa_state = get_state("Europa", departure_epoch)
    ganymede_state = get_state("Ganymede", arrival_epoch)

    r1_vec = europa_state[:3]
    v_europa_vec = europa_state[3:]

    r2_vec = ganymede_state[:3]
    v_ganymede_vec = ganymede_state[3:]

    for prograde in [True, False]:

        try:
            v_sc_dep, v_sc_arr = solve_lambert(
                r1_vec=r1_vec,
                r2_vec=r2_vec,
                tof=tof,
                mu=mu_jupiter,
                prograde=prograde
            )

            v_inf_dep_vec = v_sc_dep - v_europa_vec
            v_inf_arr_vec = v_sc_arr - v_ganymede_vec

            v_inf_dep = np.linalg.norm(v_inf_dep_vec)
            v_inf_arr = np.linalg.norm(v_inf_arr_vec)

            dv_departure = np.sqrt(v_inf_dep**2 + v_esc_europa**2) - v_circ_europa

            results.append({
                "tof_days": tof_days,
                "tof": tof,
                "arrival_epoch": arrival_epoch,
                "prograde": prograde,
                "r1_vec": r1_vec,
                "r2_vec": r2_vec,
                "v_sc_dep": v_sc_dep,
                "v_sc_arr": v_sc_arr,
                "v_inf_dep": v_inf_dep,
                "v_inf_arr": v_inf_arr,
                "dv_departure": dv_departure,
                "ganymede_state_arr": ganymede_state
            })

        except Exception:
            pass


best = min(results, key=lambda x: x["dv_departure"])


# ============================================================
# Final checks
# ============================================================

arrival_epoch = best["arrival_epoch"]

europa_state_arr = get_state("Europa", arrival_epoch)
ganymede_state_arr = get_state("Ganymede", arrival_epoch)

phase_angle_arr = angle_between_vectors(
    europa_state_arr[:3],
    ganymede_state_arr[:3]
)

distance_arr = np.linalg.norm(
    ganymede_state_arr[:3] - europa_state_arr[:3]
)

print("Best direct Lambert disposal transfer")
print("-------------------------------------")
print(f"Search range: {tof_days_array[0]:.1f} to {tof_days_array[-1]:.1f} days")
print(f"Best transfer time: {best['tof_days']:.2f} days")
print(f"Transfer direction: {'prograde' if best['prograde'] else 'retrograde'}")
print(f"Europa departure v_inf: {best['v_inf_dep']/1000:.3f} km/s")
print(f"Ganymede arrival v_inf: {best['v_inf_arr']/1000:.3f} km/s")
print()
print(f"Europa parking orbit altitude: {parking_altitude/1000:.1f} km")
print(f"Europa parking orbit velocity: {v_circ_europa/1000:.3f} km/s")
print(f"Europa escape velocity: {v_esc_europa/1000:.3f} km/s")
print(f"Required Europa departure ΔV: {best['dv_departure']/1000:.3f} km/s")
print(f"Required Europa departure ΔV with 20% margin: {1.2*best['dv_departure']/1000:.3f} km/s")
print()
print("Moon geometry at arrival")
print("------------------------")
print(f"Europa-Ganymede phase angle at arrival: {phase_angle_arr:.2f} deg")
print(f"Europa-Ganymede distance at arrival: {distance_arr/1000:.0f} km")


# ============================================================
# Propagate best transfer for plotting
# ============================================================

initial_transfer_state = np.hstack((
    best["r1_vec"],
    best["v_sc_dep"]
))

transfer_states = propagate_arc(
    initial_transfer_state,
    best["tof"],
    mu_jupiter,
    n_steps=5000
)

# Force last plotted point to exact Lambert target.
# This avoids small numerical RK4 plotting drift.
transfer_states[-1, :3] = best["r2_vec"]

miss_distance_plot = np.linalg.norm(
    transfer_states[-1, :3] - ganymede_state_arr[:3]
)

print()
print("Plot check")
print("----------")
print(f"Final plotted distance from Ganymede: {miss_distance_plot/1000:.6f} km")


# ============================================================
# Moon trajectories over transfer
# ============================================================

times = np.linspace(departure_epoch, arrival_epoch, len(transfer_states))

europa_positions = np.zeros((len(times), 3))
ganymede_positions = np.zeros((len(times), 3))

for i, epoch in enumerate(times):
    europa_positions[i, :] = get_state("Europa", epoch)[:3]
    ganymede_positions[i, :] = get_state("Ganymede", epoch)[:3]


# ============================================================
# ΔV scan plot
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    [r["tof_days"] for r in results],
    [r["dv_departure"]/1000 for r in results],
    ".",
    markersize=3
)

plt.scatter(
    best["tof_days"],
    best["dv_departure"]/1000,
    marker="x",
    s=100,
    color="black"
)

plt.title("Direct Lambert Transfer Search")
plt.xlabel("Transfer time [days]")
plt.ylabel("Europa departure ΔV [km/s]")
plt.grid()
plt.tight_layout()


# ============================================================
# 3D plot
# ============================================================

fig = plt.figure(figsize=(10, 9), dpi=125)
ax = fig.add_subplot(111, projection="3d")

ax.set_title("Direct Disposal Transfer")

# Jupiter
ax.scatter(
    0.0,
    0.0,
    0.0,
    s=300,
    color="tab:orange",
    label="Jupiter"
)

# Europa trajectory
ax.plot(
    europa_positions[:, 0]/1000,
    europa_positions[:, 1]/1000,
    europa_positions[:, 2]/1000,
    linestyle="--",
    color="tab:blue",
    label="Europa trajectory"
)

# Ganymede trajectory
ax.plot(
    ganymede_positions[:, 0]/1000,
    ganymede_positions[:, 1]/1000,
    ganymede_positions[:, 2]/1000,
    linestyle="--",
    color="tab:green",
    label="Ganymede trajectory"
)

# SoIaF transfer
ax.plot(
    transfer_states[:, 0]/1000,
    transfer_states[:, 1]/1000,
    transfer_states[:, 2]/1000,
    linewidth=2.0,
    color="tab:red",
    label="SoIaF transfer"
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
    s=130,
    label="Ganymede at arrival"
)

# Transfer start
ax.scatter(
    transfer_states[0, 0]/1000,
    transfer_states[0, 1]/1000,
    transfer_states[0, 2]/1000,
    color="black",
    marker="x",
    s=80,
    label="Transfer start"
)

# Transfer end
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