import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup, propagation, simulator
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion

spice.load_standard_kernels()

# ---const---
inclination_deg = 90
aop_deg         = 90
n = 29.997


simulation_years       = 5
simulation_start_epoch = DateTime(2020, 1, 1).to_epoch()
simulation_end_epoch   = simulation_start_epoch + simulation_years * constants.JULIAN_YEAR

bodies_to_create = [
    "Europa",
    "Ganymede",
    "Callisto",
    "Io",
    "Jupiter",
]

body_to_propagate = ["SoIaF"]

global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
)

body_settings.add_empty_settings("SoIaF")
body_settings.get("SoIaF").constant_mass = 2000.0

bodies = environment_setup.create_system_of_bodies(
    body_settings
)

accelerations_settings_spacecraft = {
    "Jupiter": [
        propagation_setup.acceleration.point_mass_gravity()
    ],
}

acceleration_settings = {
    "SoIaF": accelerations_settings_spacecraft
}

central_bodies = ["Jupiter"]

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    body_to_propagate,
    central_bodies
)

R_Jupiter = 69911.0e3
mu_jup    = bodies.get("Jupiter").gravitational_parameter


# Used for cross-section point (longitude of ascending node)
_io_s    = spice.get_body_cartesian_state_at_epoch(
    "Io", "Jupiter", global_frame_orientation, "NONE", simulation_start_epoch
)
_io_h    = np.cross(_io_s[:3], _io_s[3:6])
_io_node = np.cross(np.array([0.0, 0.0, 1.0]), _io_h)
raan_deg = float(np.rad2deg(np.arctan2(_io_node[1], _io_node[0]))) % 360.0
print(f"RAAN set to Io ascending node: {raan_deg:.2f}°")

r_Io = 421700000
T_Io = 2*np.pi*np.sqrt(r_Io**(3)/mu_jup)


a_cap = (mu_jup * (n * T_Io / (2 * np.pi))**2)**(1/3)
eccentricity = np.sqrt(1 - r_Io / a_cap)
r_periapsis = a_cap * (1 - eccentricity)

r_apoapsis = a_cap * (1.0 + eccentricity)
T_cap      = 2*np.pi*np.sqrt(a_cap**(3)/mu_jup) / 3600

v_inf      = 1600.0  # m/s
a_hyp      = -mu_jup / v_inf**2
e_hyp      = 1 - r_periapsis / a_hyp
v_peri_hyp = np.sqrt(v_inf**2 + 2*mu_jup/r_periapsis)
v_peri_cap = np.sqrt(mu_jup*(2/r_periapsis - 1/a_cap))
dv_joi     = v_peri_hyp - v_peri_cap

def print_info():
    print(f'Capture orbit period            : {T_cap/24:.2f} days')
    print(f'Capture orbit eccentricity      : {eccentricity:.3f}')
    print(f'Capture orbit perijove          : {r_periapsis/R_Jupiter:.2f} R_Jupiter')
    print(f'Capture orbit semi-major axis   : {a_cap/R_Jupiter:.2f} R_J ')
    print(f'v_inf                           : {v_inf/1e3:.3f} km/s')
    print(f'v at periapsis (hyperbola)      : {v_peri_hyp/1e3:.3f} km/s')
    print(f'v at periapsis (capture orbit)  : {v_peri_cap/1e3:.3f} km/s')
    print(f'JOI dV                          : {dv_joi/1e3:.3f} km/s')

print_info()

# --- Science Orbit

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter     = mu_jup,
    semi_major_axis             = a_cap,
    eccentricity                = eccentricity,
    inclination                 = np.deg2rad(inclination_deg),
    argument_of_periapsis       = np.deg2rad(aop_deg),
    longitude_of_ascending_node = np.deg2rad(raan_deg),
    true_anomaly                = np.deg2rad(0.0)
)

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    600.0,
    propagation_setup.integrator.CoefficientSets.rk_4
)

termination_settings = propagation_setup.propagator.time_termination(
    simulation_end_epoch
)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    body_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings
)

dynamics_simulator = simulator.create_dynamics_simulator(
    bodies,
    propagator_settings
)

states   = result2array(dynamics_simulator.state_history)

# Trimming for 30 flybys 
sc_r     = np.linalg.norm(states[:, 1:4], axis=1)
sc_vr    = np.sum(states[:, 1:4] * states[:, 4:7], axis=1) / sc_r
n_orbits = int(np.sum(np.diff(np.sign(sc_vr)) > 0))

apo_indices = np.where(np.diff(np.sign(sc_vr)) < 0)[0]
states      = states[:apo_indices[29] + 1]   # keep only up to 30th apoapsis

t_start_sc = states[0, 0]
t_end_sc = states[-1, 0]
duration_days = (t_end_sc - t_start_sc) / constants.JULIAN_DAY
print(f'30 flybys duration (starting at periapsis, and finishing at Apoapsis): {duration_days:.2f} days ({duration_days/365.25:.2f} years)')


# --- hyperbolic approach arc 
nu_hyp_start    = np.deg2rad(-150.0)
state_hyp_start = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter     = mu_jup,
    semi_major_axis             = a_hyp,
    eccentricity                = e_hyp,
    inclination                 = np.deg2rad(inclination_deg),
    argument_of_periapsis       = np.deg2rad(aop_deg),
    longitude_of_ascending_node = np.deg2rad(raan_deg),
    true_anomaly                = nu_hyp_start
)
H_start         = np.arcsinh(np.sqrt(e_hyp**2 - 1) * np.sin(nu_hyp_start)
                              / (1 + e_hyp * np.cos(nu_hyp_start)))
tof_hyp         = (H_start - e_hyp * np.sinh(H_start)) / np.sqrt(mu_jup / (-a_hyp)**3)
hyp_start_epoch = simulation_start_epoch - tof_hyp

prop_hyp = propagation_setup.propagator.translational(
    central_bodies, acceleration_models, body_to_propagate,
    state_hyp_start, hyp_start_epoch, integrator_settings,
    propagation_setup.propagator.time_termination(simulation_start_epoch)
)
states_hyp = result2array(simulator.create_dynamics_simulator(bodies, prop_hyp).state_history)

# --- Disposal
r_a_cap = a_cap*(1+eccentricity)
r_a_dis = r_a_cap
r_p_dis = R_Jupiter * 0.99
a_dis = (r_a_dis + r_p_dis) / 2
e_dis = (r_a_dis-r_p_dis) / (r_p_dis + r_a_dis)

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter     = mu_jup,
    semi_major_axis             = a_dis,
    eccentricity                = e_dis, 
    inclination                 = np.deg2rad(inclination_deg),
    argument_of_periapsis       = np.deg2rad(aop_deg),
    longitude_of_ascending_node = np.deg2rad(raan_deg),
    true_anomaly                = np.deg2rad(180.0)
)

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    600.0,
    propagation_setup.integrator.CoefficientSets.rk_4
)

termination_settings = propagation_setup.propagator.dependent_variable_termination(
    propagation_setup.dependent_variable.altitude("SoIaF", "Jupiter"), 
    0.0, True
)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    body_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings
)

dynamics_simulator = simulator.create_dynamics_simulator(
    bodies,
    propagator_settings
)

states_disp   = result2array(dynamics_simulator.state_history)

V_a_cap = np.sqrt(mu_jup / a_cap * (1-eccentricity) / (1+eccentricity))
V_a_dis = np.sqrt(mu_jup / a_dis * (1-e_dis) / (1+e_dis))
dv_dis = V_a_dis - V_a_cap

print(f'The required disposal Delta V is {dv_dis/1000:.3f} km/s')

t_start_sc = states_disp[0, 0]
t_end_sc = states_disp[-1, 0]
duration_days = (t_end_sc - t_start_sc) / constants.JULIAN_DAY
print(f'disposal duration (starting at apoapsis and ending when it crashes with Jupiter): {duration_days:.2f} days ({duration_days/365.25:.2f} years)')


# --- Plots

def moons_pos():
    global moons
    moons = {
        "Io":       {"period_days": 1.769,  "color": "Orange"},
        "Europa":   {"period_days": 3.551,  "color": "Blue"},
        "Ganymede": {"period_days": 7.155,  "color": "Yellow"},
        "Callisto": {"period_days": 16.689, "color": "Grey"},
    }

    global moon_positions
    moon_positions = {}
    for moon_name, props in moons.items():
        t_end = simulation_start_epoch + props["period_days"] * constants.JULIAN_DAY
        times = np.linspace(simulation_start_epoch, t_end, 600)
        moon_positions[moon_name] = np.array([
            spice.get_body_cartesian_state_at_epoch(
                moon_name, "Jupiter", global_frame_orientation, "NONE", t
            )[:3]
            for t in times
        ]) / R_Jupiter

moons_pos()

def plot_system(states, states_hyp, states_disp, moon_positions, moons):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title("Jupiter System — SoIaF JOI + Galilean Moons")

    for moon_name, props in moons.items():
        pos = moon_positions[moon_name]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=moon_name, color=props["color"])

    ax.scatter(0, 0, 0, label="Jupiter", marker="o", color="Brown", s=50)

    theta = np.linspace(0, 2 * np.pi, 300)
    r_Io_RJ = r_Io / R_Jupiter
    ax.plot(r_Io_RJ * np.cos(theta), r_Io_RJ * np.sin(theta),
            np.zeros_like(theta), color="Red", linewidth=0.8,
            linestyle=":", alpha=0.5, label="Io orbit (equatorial)")

    ax.plot(states_hyp[:, 1] / R_Jupiter, states_hyp[:, 2] / R_Jupiter, states_hyp[:, 3] / R_Jupiter,
            color="blue", linewidth=0.8, label="SoIaF (approach)")
    ax.scatter(*states[0, 1:4] / R_Jupiter, color="black", s=40, zorder=6, label="JOI burn")
    ax.plot(states[:, 1] / R_Jupiter, states[:, 2] / R_Jupiter, states[:, 3] / R_Jupiter,
            color="red", linewidth=0.6, label="SoIaF (post-JOI)")
    ax.plot(states_disp[:, 1] / R_Jupiter, states_disp[:, 2] / R_Jupiter, states_disp[:, 3] / R_Jupiter,
            color="green", linewidth=0.6, label="SoIaF (disposal)")

    lim = min(r_apoapsis / R_Jupiter * 1.1, 100.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.legend(fontsize=8)
    ax.set_xlabel("x [R_J]")
    ax.set_ylabel("y [R_J]")
    ax.set_zlabel("z [R_J]")
    plt.tight_layout()
    plt.show()


plot_system(states, states_hyp, states_disp, moon_positions, moons)