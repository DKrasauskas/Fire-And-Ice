# ============================================================
# Standard modules
# ============================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# Tudatpy modules
# ============================================================

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics.propagation_setup import dependent_variable
from tudatpy.dynamics.propagation import create_dependent_variable_dictionary


# ============================================================
# Load SPICE kernels
# ============================================================

spice.load_standard_kernels()


# ============================================================
# Body and frame setup
# ============================================================

bodies_to_create = ["Jupiter", "Europa"]

global_frame_origin = "Europa"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
)

body_settings.add_empty_settings("SoIaF")

# No aerodynamic forces
body_settings.get("SoIaF").aerodynamic_coefficient_settings = (
    environment_setup.aerodynamic_coefficients.constant(
        0.0,
        [0.0, 0.0, 0.0]
    )
)

# No radiation pressure
body_settings.get("SoIaF").radiation_pressure_target_settings = (
    environment_setup.radiation_pressure.cannonball_radiation_target(
        0.0,
        0.0,
        {"Jupiter": ["Europa"]}
    )
)

bodies = environment_setup.create_system_of_bodies(body_settings)

spacecraft_mass = 1633.28
bodies.get("SoIaF").mass = spacecraft_mass


# ============================================================
# Propagation setup
# ============================================================

bodies_to_propagate = ["SoIaF"]
central_bodies = ["Europa"]

acceleration_settings = {
    "SoIaF": {
        "Europa": [
            propagation_setup.acceleration.spherical_harmonic_gravity(5, 5)
        ],
        "Jupiter": [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    }
}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies
)


# ============================================================
# Simulation time
# ============================================================

simulation_start_epoch = DateTime(2026, 6, 1).to_epoch()
simulation_end_epoch = DateTime(2026, 6, 30).to_epoch()


# ============================================================
# Initial 100 km circular polar orbit around Europa
# ============================================================

europa_mu = bodies.get("Europa").gravitational_parameter
europa_radius = bodies.get("Europa").shape_model.average_radius

altitude = 100.0e3
semi_major_axis = europa_radius + altitude
circular_velocity = np.sqrt(europa_mu / semi_major_axis)

# Define polar orbit in Europa-equator frame
position_europa_frame = np.array([
    semi_major_axis,
    0.0,
    0.0
])

velocity_europa_frame = np.array([
    0.0,
    0.0,
    circular_velocity
])

# Rotate Europa-equator frame state into J2000
rotation_matrix_europa_to_j2000 = spice.compute_rotation_matrix_between_frames(
    "IAU_EUROPA",
    "J2000",
    simulation_start_epoch
)

position_j2000 = rotation_matrix_europa_to_j2000 @ position_europa_frame
velocity_j2000 = rotation_matrix_europa_to_j2000 @ velocity_europa_frame

initial_state = np.concatenate((position_j2000, velocity_j2000))


# ============================================================
# Dependent variables
# ============================================================

dependent_variables_to_save = [
    dependent_variable.latitude("SoIaF", "Europa"),
    dependent_variable.longitude("SoIaF", "Europa"),
    dependent_variable.keplerian_state("SoIaF", "Europa"),
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type,
        "SoIaF",
        "Jupiter"
    ),
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type,
        "SoIaF",
        "Europa"
    )
]


# ============================================================
# Integrator and propagator
# ============================================================

termination_condition = propagation_setup.propagator.time_termination(
    simulation_end_epoch
)

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    10.0,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)


# ============================================================
# Run propagation
# ============================================================

dynamics_simulator = simulator.create_dynamics_simulator(
    bodies,
    propagator_settings
)


# ============================================================
# Extract results
# ============================================================

states_array = result2array(
    dynamics_simulator.propagation_results.state_history
)

dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)

relative_time_hours = (
    dep_var_dict.time_history - dep_var_dict.time_history[0]
) / 3600.0


# ============================================================
# Ground track
# ============================================================

latitude = dep_var_dict.asarray(
    dependent_variable.latitude("SoIaF", "Europa")
)

longitude = dep_var_dict.asarray(
    dependent_variable.longitude("SoIaF", "Europa")
)

hours = 24
subset = int(len(relative_time_hours) / 24.0 * hours)

latitude_plot = np.rad2deg(latitude[:subset])
longitude_plot = np.rad2deg(longitude[:subset])

longitude_plot = (longitude_plot + 180.0) % 360.0 - 180.0

plt.figure(figsize=(9, 5))
plt.title(f"{hours} hour ground track of SoIaF around Europa")
plt.scatter(longitude_plot, latitude_plot, s=1)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.yticks(np.arange(-90, 91, 45))
plt.grid()
plt.tight_layout()


# ============================================================
# 3D orbit plot with Europa to scale
# ============================================================

fig = plt.figure(figsize=(8, 8), dpi=150)
ax = fig.add_subplot(111, projection="3d")

ax.set_title("100 km Polar Science Orbit around Europa")

# Plot only ONE orbit
T = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / europa_mu)

one_orbit_mask = (
    states_array[:, 0] - states_array[0, 0]
) <= T

x_sc = states_array[one_orbit_mask, 1]
y_sc = states_array[one_orbit_mask, 2]
z_sc = states_array[one_orbit_mask, 3]

# Europa sphere
u = np.linspace(0.0, 2.0*np.pi, 80)
v = np.linspace(0.0, np.pi, 80)

x_europa = europa_radius * np.outer(np.cos(u), np.sin(v))
y_europa = europa_radius * np.outer(np.sin(u), np.sin(v))
z_europa = europa_radius * np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(
    x_europa,
    y_europa,
    z_europa,
    color="wheat",
    alpha=1.0,
    linewidth=0,
    shade=True
)

# Orbit
orbit_line, = ax.plot(
    x_sc,
    y_sc,
    z_sc,
    color="red",
    linewidth=1.0,
    label="SoIaF orbit"
)

# Equal scaling
plot_radius = semi_major_axis * 1.03

ax.set_xlim(-plot_radius, plot_radius)
ax.set_ylim(-plot_radius, plot_radius)
ax.set_zlim(-plot_radius, plot_radius)

ax.set_box_aspect([1, 1, 1])

# Better viewing angle
ax.view_init(
    elev=35,
    azim=45
)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")

europa_patch = Patch(
    facecolor="wheat",
    edgecolor="wheat",
    label="Europa"
)

ax.legend(
    handles=[orbit_line, europa_patch],
    loc="center left",
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()
plt.show()