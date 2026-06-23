# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2008, 4, 28).to_epoch()
simulation_end_epoch   = DateTime(2009, 6, 30).to_epoch()

# Define string names for bodies to be created from default.
bodies_to_create = ["Sun", "Jupiter", "Io", "Europa", "Ganymede", "Callisto"]

# Use "Earth"/"J2000" as global frame origin and orientation.
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)



# Create empty body settings for the satellite
body_settings.add_empty_settings("Delfi-C3")
# Create aerodynamic coefficient interface settings
reference_area_drag = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area_drag, [drag_coefficient, 0.0, 0.0]
)

# Add the aerodynamic interface to the body settings
body_settings.get("Delfi-C3").aerodynamic_coefficient_settings = aero_coefficient_settings
# Create radiation pressure settings
reference_area_radiation = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Jupiter"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# Add the radiation pressure interface to the body settings
body_settings.get("Delfi-C3").radiation_pressure_target_settings = vehicle_target_settings
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("Delfi-C3").mass = 2.2 #kg
# Define bodies that are propagated
bodies_to_propagate = ["Delfi-C3"]

# Define central bodies of propagation
central_bodies = ["Jupiter"]
# Define accelerations acting on Delfi-C3 by Sun and Earth.
accelerations_settings_delfi_c3 = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Jupiter=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),

    ],
    Io=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),

    ],
    Europa=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),

    ],
    Ganymede=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),

    ],
    
)

# Create global accelerations settings dictionary.
acceleration_settings = {"Delfi-C3": accelerations_settings_delfi_c3}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)
R_j  =7.1492e7 
earth_gravitational_parameter = bodies.get("Jupiter").gravitational_parameter
initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=earth_gravitational_parameter,
    semi_major_axis= R_j * 5,
    eccentricity=4.03294322e-03,
    inclination=0.0,
    argument_of_periapsis=1.31226971e+00,
    longitude_of_ascending_node=3.82958313e-01,
    true_anomaly=3.07018490e+00,
)


from tudatpy.dynamics.propagation_setup import dependent_variable

# Define list of dependent variables to save
dependent_variables_to_save = [
    dependent_variable.keplerian_state("Io", "Jupiter"),
]

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 100.0
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

# Create propagation settings
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

# Create simulation object and propagate the dynamics
dynamics_simulator = simulator.create_dynamics_simulator(
    bodies, propagator_settings
)


from tudatpy.dynamics.propagation import create_dependent_variable_dictionary

# Extract the resulting state and dependent variable history and convert it to an ndarray
states_history = dynamics_simulator.propagation_results.state_history
states_array = result2array(states_history)
# ! Retrieving dependent variables from the dep_vars_array requires careful indexing
dep_vars_history = dynamics_simulator.propagation_results.dependent_variable_history
dep_vars_array = result2array(dep_vars_history)

# Create dependent variable dictionary
dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)
relative_time_hours = (dep_var_dict.time_history - dep_var_dict.time_history[0])/3600

# numpy array of shape (len(relative_time_hours), 3)
# the dependent variable settings could also be reused from the dependent_variables_to_save list, be careful with the indexing though!
# delfi_total_acceleration = dep_var_dict.asarray(dependent_variables_to_save[0])
# Plot Kepler elements as a function of time
kepler_elements = dep_var_dict.asarray(dependent_variable.keplerian_state("Io", "Jupiter"))
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:,0] / 1e3
ax1.plot(relative_time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:,1]
ax2.plot(relative_time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
ax3.plot(relative_time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:,3])
ax4.plot(relative_time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:,4])
ax5.plot(relative_time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:,5])
ax6.scatter(relative_time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(relative_time_hours), max(relative_time_hours)])
    ax.grid()
plt.tight_layout()
plt.show()