# General imports
import copy
import numpy as np
import os
from matplotlib import pyplot as plt

# Tudatpy imports
import tudatpy
from tudatpy.util import result2array
from tudatpy import dynamics
from tudatpy.dynamics import propagation_setup, environment_setup, parameters, parameters_setup, simulator
from tudatpy.astro import polyhedron_utilities
from tudatpy.math import interpolators, root_finders
from tudatpy.astro.time_representation import DateTime

#######################################################################################################################
# Compute unit of length of the CR3BP
def cr3bp_unit_of_length (distance_between_primaries: float) -> float:
    return distance_between_primaries

########################################################################################################################
# Compute unit of time of the CR3BP
def cr3bp_unit_of_time (gravitational_parameter_primary: float,
                        gravitational_parameter_secondary: float,
                        distance_between_primaries: float) -> float:

    mean_motion = np.sqrt((gravitational_parameter_primary + gravitational_parameter_secondary) / \
                          distance_between_primaries ** 3)
    unit = 1/mean_motion

    return unit

########################################################################################################################
# Get full-state rotation matrix from inertial frame to body-fixed frame
def get_inertial_to_body_fixed_full_matrix(bodies: tudatpy.dynamics.environment.SystemOfBodies,
                                           body_name: str,
                                           time: float) -> np.ndarray:

    inertial_to_body_fixed_matrix = bodies.get(body_name).rotation_model.inertial_to_body_fixed_rotation(time)
    inertial_to_body_fixed_matrix_derivative = bodies.get(body_name).rotation_model.time_derivative_inertial_to_body_fixed_rotation(time)

    inertial_to_body_fixed_full_matrix = np.zeros((6, 6))
    inertial_to_body_fixed_full_matrix[0:3,0:3] = inertial_to_body_fixed_matrix
    inertial_to_body_fixed_full_matrix[3:6,0:3] = inertial_to_body_fixed_matrix_derivative
    inertial_to_body_fixed_full_matrix[3:6,3:6] = inertial_to_body_fixed_matrix

    return inertial_to_body_fixed_full_matrix

########################################################################################################################
# Get full-state rotation matrix from body-fixed frame to inertial frame
def get_body_fixed_to_inertial_full_matrix(bodies: tudatpy.dynamics.environment.SystemOfBodies,
                                           body_name: str,
                                           time: float) -> np.ndarray:

    body_fixed_to_inertial_matrix = bodies.get(body_name).rotation_model.body_fixed_to_inertial_rotation(time)
    body_fixed_to_inertial_matrix_derivative = bodies.get(body_name).rotation_model.time_derivative_body_fixed_to_inertial_rotation(time)

    body_fixed_to_inertial_full_matrix = np.zeros((6, 6))
    body_fixed_to_inertial_full_matrix[0:3,0:3] = body_fixed_to_inertial_matrix
    body_fixed_to_inertial_full_matrix[3:6,0:3] = body_fixed_to_inertial_matrix_derivative
    body_fixed_to_inertial_full_matrix[3:6,3:6] = body_fixed_to_inertial_matrix

    return body_fixed_to_inertial_full_matrix

########################################################################################################################
# Conversion of state from inertial to body-fixed frame
def convert_state_history_inertial_to_body_fixed(
        bodies: tudatpy.dynamics.environment.SystemOfBodies,
        body_name: str,
        state_history_inertial: dict) -> dict:

    state_history_body_fixed = copy.deepcopy(state_history_inertial)

    for t in state_history_body_fixed.keys():
        body_state_inertial = bodies.get_body(body_name).state_in_base_frame_from_ephemeris(t)
        rotation_matrix = get_inertial_to_body_fixed_full_matrix(bodies, body_name, t)
        state_history_body_fixed[t] = rotation_matrix @ (state_history_inertial[t] - body_state_inertial)

    return state_history_body_fixed

########################################################################################################################
# Conversion of state from body-fixed to inertial frame
def convert_state_history_body_fixed_to_inertial(
        bodies: tudatpy.dynamics.environment.SystemOfBodies,
        body_name: str,
        state_history_body_fixed: dict) -> dict:

    state_history_inertial = copy.deepcopy(state_history_body_fixed)

    for t in state_history_body_fixed.keys():
        body_state_inertial = bodies.get_body(body_name).state_in_base_frame_from_ephemeris(t)
        rotation_matrix = get_body_fixed_to_inertial_full_matrix(bodies, body_name, t)
        state_history_inertial[t] = rotation_matrix @ state_history_body_fixed[t] + body_state_inertial

    return state_history_inertial

########################################################################################################################
# Conversion state transition matrix from inertial to synodic frame
def convert_stm_history_inertial_to_body_fixed(
        bodies: tudatpy.dynamics.environment.SystemOfBodies,
        body_name: str,
        stm_history_inertial: dict) -> dict:

    stm_history_body_fixed = copy.deepcopy(stm_history_inertial)

    body_fixed_to_inertial_matrix_initial = get_body_fixed_to_inertial_full_matrix(
        bodies, body_name, min(stm_history_inertial.keys()))

    for t in stm_history_inertial.keys():
        inertial_to_body_fixed_matrix_final = get_inertial_to_body_fixed_full_matrix(bodies, body_name, t)
        stm_history_body_fixed[t] = inertial_to_body_fixed_matrix_final @ stm_history_inertial[t] @ body_fixed_to_inertial_matrix_initial

    return stm_history_body_fixed

########################################################################################################################
# Conversion state transition matrix from inertial to synodic frame
def convert_stm_inertial_to_body_fixed(
        bodies: tudatpy.dynamics.environment.SystemOfBodies,
        body_name: str,
        stm_inertial: np.ndarray,
        time_initial: float,
        time_final: float) -> np.ndarray:

    inertial_to_body_fixed_matrix_final = get_inertial_to_body_fixed_full_matrix(bodies, body_name, time_final)
    body_fixed_to_inertial_matrix_initial = get_body_fixed_to_inertial_full_matrix(bodies, body_name, time_initial)

    stm_synodic = inertial_to_body_fixed_matrix_final @ stm_inertial @ body_fixed_to_inertial_matrix_initial

    return stm_synodic

########################################################################################################################
# Create propagator settings for time termination
def create_time_termination_propagator_settings(central_bodies,
                                                acceleration_models,
                                                bodies_to_propagate,
                                                initial_state: np.ndarray,
                                                simulation_start_epoch,
                                                integrator_settings,
                                                simulation_end_epoch: float,
                                                dependent_variables_to_save: list):

    # Create propagation settings.
    termination_settings_time = propagation_setup.propagator.time_termination(
        simulation_end_epoch,
        terminate_exactly_on_final_condition=True)
    current_propagator = propagation_setup.propagator.encke

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings_time,
        propagator=current_propagator,
        output_variables=dependent_variables_to_save)

    return propagator_settings

########################################################################################################################
def create_hybrid_termination_propagator_settings(central_bodies,
                                                  acceleration_models,
                                                  bodies_to_propagate,
                                                  initial_state: np.ndarray,
                                                  simulation_start_epoch,
                                                  integrator_settings,
                                                  dependent_variables_to_save: list,
                                                  name_spacecraft: str,
                                                  name_secondary: str,
                                                  gravitational_parameter_secondary: float,
                                                  volume_secondary: float,
                                                  hybrid_termination_max_distance: float,
                                                  hybrid_termination_max_time: float):

    # Select target value of laplacian
    value = 2 * np.pi
    lower_bound_laplacian = - value * gravitational_parameter_secondary / volume_secondary

    # Create termination condition to detect impact (laplacian of polyhedron)
    termination_variable = propagation_setup.dependent_variable.gravity_field_laplacian_of_potential(
        name_spacecraft, name_secondary)
    root_finder_settings = root_finders.bisection(
        maximum_iteration=10,
        maximum_iteration_handling=root_finders.MaximumIterationHandling.accept_result)
    termination_settings_laplacian = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=termination_variable,
        limit_value=lower_bound_laplacian,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=True,
        termination_root_finder_settings=root_finder_settings)

    # Create termination condition based on maximum distance to secondary
    upper_bound_distance = hybrid_termination_max_distance
    termination_variable = propagation_setup.dependent_variable.relative_distance(name_spacecraft, name_secondary)
    root_finder_settings = root_finders.bisection(
        maximum_iteration=1,
        maximum_iteration_handling = root_finders.MaximumIterationHandling.accept_result)
    termination_settings_distance = propagation_setup.propagator.dependent_variable_termination(
          dependent_variable_settings = termination_variable,
          limit_value=upper_bound_distance,
          use_as_lower_limit=False,
          terminate_exactly_on_final_condition=True,
          termination_root_finder_settings=root_finder_settings)

    # Create termination condition based on propagation time
    termination_settings_time = propagation_setup.propagator.time_termination(
        hybrid_termination_max_time, terminate_exactly_on_final_condition=True)

    termination_conditions_list = [termination_settings_laplacian, termination_settings_distance, termination_settings_time]

    # Create hybrid termination condition
    termination_settings_hybrid = propagation_setup.propagator.hybrid_termination(
        termination_conditions_list, fulfill_single_condition=True)

    # Select propagator
    current_propagator = propagation_setup.propagator.cowell

    hybrid_termination_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings_hybrid,
        propagator=current_propagator,
        output_variables=dependent_variables_to_save)

    return hybrid_termination_propagator_settings


# import requests
# import numpy as np

# # Fetch a mid-family L2 Lyapunov orbit from JPL's catalog
# url = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
# params = {
#     "sys": "earth-moon",
#     "family": "lyapunov",
#     "libr": "2",
# }
# r = requests.get(url, params=params)
# data = r.json()

# # Print system info so we can verify units
# print("System name:", data["system"]["name"])
# print("lunit (km):", data["system"]["lunit"])
# print("tunit (s):", data["system"]["tunit"])
# print("L2 location:", data["system"]["L2"])

# # Pick a mid-family orbit (index ~halfway through)
# fields = data["fields"]  # ["x","y","z","vx","vy","vz","jacobi","period","stability"]
# orbits = data["data"]
# mid = len(orbits) // 2
# orbit = orbits[mid]

# print(f"\nSelected orbit {mid} of {len(orbits)}:")
# for f, v in zip(fields, orbit):
#     print(f"  {f} = {v}")