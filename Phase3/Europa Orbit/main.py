# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics.propagation_setup import dependent_variable
from tudatpy.dynamics.propagation import create_dependent_variable_dictionary

# Load SPICE kernels
spice.load_standard_kernels()

# Bodies
bodies_to_create = ["Jupiter", "Europa"]

# Propagation frame
global_frame_origin = "Europa"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
)

# Add spacecraft
body_settings.add_empty_settings("SoIaF")

# No aerodynamic interface
reference_area_drag = 0.0
drag_coefficient = 0.0
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag,
    [drag_coefficient, 0.0, 0.0]
)
body_settings.get("SoIaF").aerodynamic_coefficient_settings = aero_coefficient_settings

# No radiation pressure
reference_area_radiation = 0.0
radiation_pressure_coefficient = 0.0
occulting_bodies_dict = {"Jupiter": ["Europa"]}

radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation,
    radiation_pressure_coefficient,
    occulting_bodies_dict
)
body_settings.get("SoIaF").radiation_pressure_target_settings = radiation_pressure_settings

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("SoIaF").mass = 1633.28  # kg

# Propagation setup
bodies_to_propagate = ["SoIaF"]
central_bodies = ["Europa"]

# Acceleration model
accelerations_settings_SoIaF = {
    "Jupiter": [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    "Europa": [
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5)
    ]
}

acceleration_settings = {"SoIaF": accelerations_settings_SoIaF}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies
)

# Simulation epochs
simulation_start_epoch = DateTime(2026, 6, 1).to_epoch()
simulation_end_epoch = DateTime(2026, 6, 30).to_epoch()

# Europa parameters
europa_mu = bodies.get("Europa").gravitational_parameter
europa_radius = bodies.get("Europa").shape_model.average_radius

# Initial orbit parameters
altitude = 100.0e3  # m
semi_major_axis = europa_radius + altitude
circular_velocity = np.sqrt(europa_mu / semi_major_axis)

# Define a circular polar orbit with respect to Europa's equator.
# This is first written in a Europa-equator-aligned frame:
# x-y plane = Europa equatorial plane
# z-axis = Europa north pole
#
# r along Europa equator
# v along Europa spin axis
# This makes the orbit plane contain Europa's rotation axis, so it is polar.
position_europa_equator_frame = np.array([
    semi_major_axis,
    0.0,
    0.0
])

velocity_europa_equator_frame = np.array([
    0.0,
    0.0,
    circular_velocity
])

# Rotate this Europa-equator-aligned state into J2000 for propagation.
# The same rotation is applied to position and velocity on purpose:
# this defines the inertial orbit plane aligned with Europa's pole at the initial epoch.
rotation_matrix_europa_to_j2000 = spice.compute_rotation_matrix_between_frames(
    "IAU_EUROPA",
    "J2000",
    simulation_start_epoch
)

position_j2000 = rotation_matrix_europa_to_j2000 @ position_europa_equator_frame
velocity_j2000 = rotation_matrix_europa_to_j2000 @ velocity_europa_equator_frame

initial_state = np.concatenate((position_j2000, velocity_j2000))

# Dependent variables
dependent_variables_to_save = [
    dependent_variable.total_acceleration("SoIaF"),
    dependent_variable.keplerian_state("SoIaF", "Europa"),
    dependent_variable.latitude("SoIaF", "Europa"),
    dependent_variable.longitude("SoIaF", "Europa"),
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

# Termination condition
termination_condition = propagation_setup.propagator.time_termination(
    simulation_end_epoch
)

# Integrator settings
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

# Propagator settings
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

# Run propagation
dynamics_simulator = simulator.create_dynamics_simulator(
    bodies,
    propagator_settings
)

# Extract results
states_history = dynamics_simulator.propagation_results.state_history
states_array = result2array(states_history)

dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)

relative_time_hours = (
    dep_var_dict.time_history - dep_var_dict.time_history[0]
) / 3600.0

# Ground track
latitude = dep_var_dict.asarray(
    dependent_variable.latitude("SoIaF", "Europa")
)
longitude = dep_var_dict.asarray(
    dependent_variable.longitude("SoIaF", "Europa")
)

hours = 24
subset = int(len(relative_time_hours) / 24 * hours)

latitude = np.rad2deg(latitude[:subset])
longitude = np.rad2deg(longitude[:subset])

# Wrap longitude to [-180, 180]
longitude = (longitude + 180.0) % 360.0 - 180.0

plt.figure(figsize=(9, 5))
plt.title(f"{hours} hour ground track of SoIaF around Europa")
plt.scatter(longitude, latitude, s=1)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()

# 3D trajectory
fig = plt.figure(figsize=(6, 6), dpi=125)
ax = fig.add_subplot(111, projection="3d")
ax.set_title("SoIaF trajectory around Europa")

ax.plot(
    states_array[:, 1],
    states_array[:, 2],
    states_array[:, 3],
    color = "green",
    label="SoIaF",
    linestyle="-."
)

ax.scatter(
    0.0,
    0.0,
    0.0,
    label="Europa",
    marker="o"
)

ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_aspect("equal")

plt.show()

# print properties -----------------------------------------------------------------------------------
a = europa_radius + 100000
T = 2*np.pi*np.sqrt(a**3/europa_mu)
print(f"Orbital period: {T/3600} h ")

n = np.sqrt(europa_mu/a**3)
print(f"Mean motion: {np.rad2deg(n)*3600} deg/h")

v = np.sqrt(europa_mu/a)
print(f"Orbital velocity: {v/1000:.3f} km/s")

print(f"Altitude: 100 km")

print(f"Orbits per day: {24/(T/3600)}")

#### Ground track shift per orbit (longitude shift)
europa_rotation_rate = bodies.get("Europa").rotation_model.body_fixed_angular_velocity
europa_rotation_period = 2*np.pi / europa_rotation_rate

delta_longitude = 360 * T / europa_rotation_period
print(f"Ground track shift per orbit: {delta_longitude} deg")
