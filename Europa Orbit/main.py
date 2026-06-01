# Load standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import element_conversion
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics.propagation_setup import dependent_variable
from tudatpy.dynamics.propagation import create_dependent_variable_dictionary

# Load SPICE kernels
spice.load_standard_kernels()

# Bodies
bodies_to_create = ["Jupiter", "Europa"]

# Reference frame
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
simulation_start_epoch = DateTime(2026, 6, 28).to_epoch()
simulation_end_epoch = DateTime(2026, 6, 29).to_epoch()

# Initial Europa polar orbit
europa_mu = bodies.get("Europa").gravitational_parameter
europa_radius = bodies.get("Europa").shape_model.average_radius

altitude = 100.0e3  # m
semi_major_axis = europa_radius + altitude
eccentricity = 0.0
inclination = np.deg2rad(90.0)
argument_of_periapsis = np.deg2rad(0.0)
raan = np.deg2rad(0.0)
true_anomaly = np.deg2rad(0.0)

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=europa_mu,
    semi_major_axis=semi_major_axis,
    eccentricity=eccentricity,
    inclination=inclination,
    argument_of_periapsis=argument_of_periapsis,
    longitude_of_ascending_node=raan,
    true_anomaly=true_anomaly
)

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

dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars_history)

dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)

relative_time_hours = (
    dep_var_dict.time_history - dep_var_dict.time_history[0]
) / 3600.0

# Total acceleration
total_acceleration = dep_var_dict.asarray(
    dependent_variable.total_acceleration("SoIaF")
)
total_acceleration_norm = np.linalg.norm(total_acceleration, axis=1)

plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on SoIaF over the propagation")
plt.plot(relative_time_hours, total_acceleration_norm)
plt.xlabel("Time [hr]")
plt.ylabel("Total acceleration [m/s$^2$]")
plt.xlim([min(relative_time_hours), max(relative_time_hours)])
plt.grid()
plt.tight_layout()

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