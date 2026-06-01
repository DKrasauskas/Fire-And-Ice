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

# Define string names for bodies to be created from default ----------------------------------------------
bodies_to_create = ["Jupiter", "Europa"]

# Use "Europa"/"J2000" as global frame origin and orientation.
global_frame_origin = "Europa"
global_frame_orientation = "J2000"

# Create default body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create empty body settings for the satellite -------------------------------------------------------------
body_settings.add_empty_settings("SoIaF")

# Create aerodynamic coefficient interface settings --------------------------------------------------------
reference_area_drag = 0 
drag_coefficient = 0
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(reference_area_drag, [drag_coefficient, 0.0, 0.0])

# Add the aerodynamic interface to the body settings
body_settings.get("SoIaF").aerodynamic_coefficient_settings = aero_coefficient_settings

# Create radiation pressure settings -----------------------------------------------------------------------
reference_area_radiation = 0 
radiation_pressure_coefficient = 0
occulting_bodies_dict = dict()
occulting_bodies_dict["Jupiter"] = ["Europa"]
vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict )

# Add the radiation pressure interface to the body settings
body_settings.get("SoIaF").radiation_pressure_target_settings = vehicle_target_settings


# Propagation setup -------------------------------------------------------------------------------------------------------------
bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("SoIaF").mass = 2.2 #kg

# Define bodies that are propagated
bodies_to_propagate = ["SoIaF"]

# Define central bodies of propagation
central_bodies = ["Europa"]

# Create the acceleration model ------------------------------------------------------------------
# Define accelerations acting on SoIaF by Jupiter and Europa 
accelerations_settings_SoIaF = dict(
    Jupiter=[
        # propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Europa=[
        propagation_setup.acceleration.spherical_harmonic_gravity(5, 5),
        # propagation_setup.acceleration.aerodynamic()
    ],)

# Create global accelerations settings dictionary.
acceleration_settings = {"SoIaF": accelerations_settings_SoIaF}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

# Set simulation start and end epochs ------------------------------------------------------------------
simulation_start_epoch = DateTime(2026, 6, 28).to_epoch()
simulation_end_epoch   = DateTime(2026, 6, 29).to_epoch()

# Retrieve the initial state of SoIaF using Two-Line-Elements (TLEs)
SoIaF_tle = environment_setup.ephemeris.sgp4(
    "1 32789U 07021G   08119.60740078 -.00000054  00000-0  00000+0 0  9999",
    "2 32789 098.0082 179.6267 0015321 307.2977 051.0656 14.81417433    68",
)
SoIaF_ephemeris = environment_setup.create_body_ephemeris(SoIaF_tle, "SoIaF")
initial_state = SoIaF_ephemeris.cartesian_state( simulation_start_epoch )

# Define dependent variables to save ------------------------------------------------------------------------
from tudatpy.dynamics.propagation_setup import dependent_variable

# Define list of dependent variables to save
dependent_variables_to_save = [
    dependent_variable.total_acceleration("SoIaF"),
    dependent_variable.keplerian_state("SoIaF", "Europa"),
    dependent_variable.latitude("SoIaF", "Europa"),
    dependent_variable.longitude("SoIaF", "Europa"),
]
acceleration_dependent_variables_to_save = [
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "SoIaF", "Jupiter"
    ),
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "SoIaF", "Europa",
    )
]

dependent_variables_to_save += acceleration_dependent_variables_to_save

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 10.0
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
delfi_total_acceleration = dep_var_dict.asarray(dependent_variable.total_acceleration("SoIaF"))

# the dependent variable settings could also be reused from the dependent_variables_to_save list, be careful with the indexing though!
# delfi_total_acceleration = dep_var_dict.asarray(dependent_variables_to_save[0])

# Total acceleration over time ---------------------------------------------------------------------------------------------------------------------
# Plot total acceleration as function of time
total_acceleration_norm = np.linalg.norm(delfi_total_acceleration, axis=1)

plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on SoIaF over the course of propagation.")
plt.plot(relative_time_hours, total_acceleration_norm)
plt.xlabel('Time [hr]')
plt.ylabel('Total Acceleration [m/s$^2$]')
plt.xlim([min(relative_time_hours), max(relative_time_hours)])
plt.grid()
plt.tight_layout()

# Plot ground track for a period of 3 hours ------------------------------------------------------------------------------------------------------
latitude = dep_var_dict.asarray(dependent_variable.latitude("SoIaF", "Europa"))
longitude = dep_var_dict.asarray(dependent_variable.longitude("SoIaF", "Europa"))
hours = 3
subset = int(len(relative_time_hours) / 24 * hours)
latitude = np.rad2deg(latitude[0: subset])
longitude = np.rad2deg(longitude[0: subset])

plt.figure(figsize=(9, 5))
plt.title("3 hour ground track of Delfi-C3")
plt.scatter(longitude, latitude, s=1)
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()

