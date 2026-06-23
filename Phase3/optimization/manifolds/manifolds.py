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

from stuff import *


####################################################################################################################
# Define dimensional model parameters and then make them dimensionless

name_primary = "Earth"
name_secondary = "Moon"
name_spacecraft = "Spacecraft"



gravitational_parameter_primary = 3.986004418e14   # m^3/s^2  (Earth)
gravitational_parameter_secondary = 4.9048695e12   # m^3/s^2  (Moon)
distance_between_primaries = 384400e3    


# Get CR3BP units
tu_cr3bp = cr3bp_unit_of_time(gravitational_parameter_primary, gravitational_parameter_secondary,
                              distance_between_primaries)
lu_cr3bp = cr3bp_unit_of_length(distance_between_primaries)

# Select hybrid termination conditions
hybrid_termination_max_time = 2 * np.pi * tu_cr3bp # 1 revolutions of the secondary
hybrid_termination_max_distance = 50e3 # km

# Make values dimensionless
gravitational_parameter_primary = gravitational_parameter_primary / lu_cr3bp**3 * tu_cr3bp**2
gravitational_parameter_secondary = gravitational_parameter_secondary / lu_cr3bp**3 * tu_cr3bp**2
distance_between_primaries = distance_between_primaries / lu_cr3bp

hybrid_termination_max_time = hybrid_termination_max_time / tu_cr3bp
hybrid_termination_max_distance = hybrid_termination_max_distance / lu_cr3bp

####################################################################################################################
# Define dimensionless model parameters

# State and period of a planar Lyapunov orbit around the L2 point
# Specified in dimensionless coordinates wrt the secondary's body-fixed frame
# Planar Lyapunov orbit around Earth-Moon L2
# Dimensionless CR3BP synodic frame coordinates
initial_state_lpo_body_fixed = np.array([
  9.9954640232520486e-01,
  1.7587679990654670e-26,
  5.2885766349983189e-37,
 -3.6584674737167909e-14,
  1.4454711386969636e+00,
    -2.7075866679100242e-33
])
period_lpo = 6.2683544910132873e+00  # dimensionless CR3BP time units

simulation_start_epoch = DateTime(2000, 1, 1).to_epoch()

manifolds_position_perturbation = 1e-7

# Define settings for manifold propagation
no_manifold_nodes = 50


####################################################################################################################
# Create system of bodies

# Initial state of primary in CR3BP, with primary as propagation/ephemeris origin
initial_state_keplerian_primary = np.zeros(6)
# Initial state of secondary in CR3BP, with primary as propagation/ephemeris origin
initial_state_keplerian_secondary = np.zeros(6)
initial_state_keplerian_secondary[0] = distance_between_primaries

# Frame origin and orientation
global_frame_origin = name_primary
global_frame_orientation = "ECLIPJ2000"

# Define body settings
body_settings = environment_setup.BodyListSettings(global_frame_origin, global_frame_orientation)

# Primary
body_settings.add_empty_settings(name_primary)
# Gravity: point mass
body_settings.get(name_primary).gravity_field_settings = environment_setup.gravity_field.central(
    gravitational_parameter_primary)
# Ephemeris: primary at origin -> constant ephemeris
body_settings.get(name_primary).ephemeris_settings = environment_setup.ephemeris.constant(
    initial_state_keplerian_primary,
    "SSB",
    global_frame_orientation)

# Secondary
body_settings.add_empty_settings(name_secondary)
body_frame_secondary = name_secondary + "_Fixed"
# Gravity: polyhedron
body_settings.get(name_secondary).gravity_field_settings =environment_setup.gravity_field.central(
    gravitational_parameter_secondary)

# Ephemeris: secondary has circular orbit around primary
body_settings.get(name_secondary).ephemeris_settings = environment_setup.ephemeris.keplerian(
    initial_state_keplerian_secondary,
    simulation_start_epoch,
    gravitational_parameter_primary + gravitational_parameter_secondary)
# Rotation model: tidally locked
# Using "simple" instead of "synchronous" model because the latter doesn't have the derivatives of the rotation
# matrix implemented tudat
rotation_rate = 1 / cr3bp_unit_of_time(
    gravitational_parameter_primary, gravitational_parameter_secondary, distance_between_primaries)
body_settings.get(name_secondary).rotation_model_settings = environment_setup.rotation_model.simple(
    base_frame="ECLIPJ2000",
    target_frame=body_frame_secondary,
    initial_orientation=np.eye(3),
    initial_time=simulation_start_epoch,
    rotation_rate=rotation_rate)

# Spacecraft
body_settings.add_empty_settings(name_spacecraft)
body_settings.get(name_spacecraft).constant_mass = 0.0

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

####################################################################################################################
# Create acceleration models

# Define bodies that are propagated.
bodies_to_propagate = [name_spacecraft]
# Define central bodies.
central_bodies = [name_primary]

# Define accelerations acting on spacecraft
acceleration_settings_on_spacecraft = {
    name_primary: [propagation_setup.acceleration.point_mass_gravity()],
    name_secondary: [propagation_setup.acceleration.point_mass_gravity()]
}

# Create global accelerations settings dictionary.
acceleration_settings = {name_spacecraft: acceleration_settings_on_spacecraft}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies)

####################################################################################################################
# Create integrator settings

current_coefficient_set = propagation_setup.integrator.CoefficientSets.rkdp_87
# Define absolute and relative tolerance
current_tolerance = 1e-12
initial_time_step = 1e-6
# Maximum step size: inf; minimum step size: eps
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    initial_time_step, current_coefficient_set, np.finfo(float).eps, np.inf,
    current_tolerance, current_tolerance)

####################################################################################################################
# Select dependent variables

dependent_variables_to_save = []

####################################################################################################################
# Propagate lagrange point orbit with variational equations

# Get initial state in the inertial frame
state_history_lpo_inertial = convert_state_history_body_fixed_to_inertial(
    bodies, name_secondary, {simulation_start_epoch: initial_state_lpo_body_fixed})

# Create propagator settings
time_propagator_settings = create_time_termination_propagator_settings(
    central_bodies, acceleration_models, bodies_to_propagate, state_history_lpo_inertial[simulation_start_epoch],
    simulation_start_epoch, integrator_settings, period_lpo, dependent_variables_to_save)

# Propagate variational equations, propagating just the STM
parameter_settings = parameters_setup.initial_states(time_propagator_settings, bodies)
lpo_single_arc_solver = simulator.create_variational_equations_solver(
        bodies, time_propagator_settings,
        parameters_setup.create_parameter_set(parameter_settings, bodies),
        simulate_dynamics_on_creation=True)

# Retrieve state and STM history and convert them to body-fixed frame
state_history_lpo_inertial = lpo_single_arc_solver.state_history
stm_history_lpo_inertial = lpo_single_arc_solver.state_transition_matrix_history

state_history_lpo_body_fixed = convert_state_history_inertial_to_body_fixed(
    bodies, name_secondary, state_history_lpo_inertial)
stm_history_lpo_body_fixed = convert_stm_history_inertial_to_body_fixed(
    bodies, name_secondary, stm_history_lpo_inertial)

####################################################################################################################
# Propagate the invariant manifolds

lpo_initial_time = min(state_history_lpo_body_fixed.keys())
lpo_final_time = max(state_history_lpo_body_fixed.keys())

# Get monodromy matrix
monodromy_matrix_body_fixed = stm_history_lpo_body_fixed[lpo_final_time]
# Get unstable eigenvector of monodromy matrix
eigenvalues, eigenvectors = np.linalg.eig(monodromy_matrix_body_fixed)
unstable_eigenvector_id = np.argmax(np.abs(eigenvalues))
unstable_eigenvector = eigenvectors[:, unstable_eigenvector_id]

# Create STM interpolator
interpolator_settings = interpolators.lagrange_interpolation(4)
stm_history_lpo_body_fixed_interpolator = interpolators.create_one_dimensional_matrix_interpolator(
    stm_history_lpo_body_fixed, interpolator_settings)
# Create state interpolator
state_history_lpo_body_fixed_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    state_history_lpo_body_fixed, interpolator_settings)

# Loop over manifold nodes and propagate the manifold
manifold_single_arc_solvers = [[],[]]
for manifold_direction_to_propagate in [-1, 1]:
    for i in range(no_manifold_nodes):
        # Deal with i=0 differently to avoid cases with numerical errors in the initial time
        if i == 0:
            current_state = state_history_lpo_body_fixed[lpo_initial_time]
            current_unstable_eigenvector = unstable_eigenvector
        # Compute unstable eigenvector at current node
        else:
            time_since_arc_start = i * (lpo_final_time - lpo_initial_time) / no_manifold_nodes
            current_state = state_history_lpo_body_fixed_interpolator.interpolate(time_since_arc_start)
            current_stm = stm_history_lpo_body_fixed_interpolator.interpolate(time_since_arc_start)
            current_unstable_eigenvector = current_stm @ unstable_eigenvector
            current_unstable_eigenvector = current_unstable_eigenvector / np.linalg.norm(current_unstable_eigenvector)

        # Sanity check
        if not np.all(np.imag(current_unstable_eigenvector) == 0):
            raise RuntimeError("Error when creating manifold initial state: eigenvector has imaginary components")
        else:
            current_unstable_eigenvector = np.real(current_unstable_eigenvector)

        # Compute initial state of manifold
        manifold_initial_state_body_fixed = current_state + manifold_direction_to_propagate * manifolds_position_perturbation / \
                                            np.linalg.norm(current_unstable_eigenvector[0:3]) * current_unstable_eigenvector

        # Get initial state in the inertial frame
        state_history_manifold_inertial = convert_state_history_body_fixed_to_inertial(
            bodies, name_secondary,
            {simulation_start_epoch: manifold_initial_state_body_fixed})

        # Create propagator settings
        simulation_end_epoch = simulation_start_epoch + hybrid_termination_max_time
        termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
        hybrid_propagator_settings =  propagation_setup.propagator.translational(
                central_bodies,
                acceleration_models,
                bodies_to_propagate,
                state_history_manifold_inertial[simulation_start_epoch],
                simulation_start_epoch,
                integrator_settings,
                termination_settings
            )
        # Propagate manifold
        manifold_single_arc_solver = simulator.create_dynamics_simulator(
            bodies, hybrid_propagator_settings)

        if manifold_direction_to_propagate == -1:
            manifold_single_arc_solvers[0].append(manifold_single_arc_solver)
        else:
            manifold_single_arc_solvers[1].append(manifold_single_arc_solver)

####################################################################################################################
# Make plot: x vs y, x vs z

fig, ax = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)

state_history_lpo_body_fixed_array = result2array(state_history_lpo_body_fixed)[:,1:]

# Plot orbit
ax[0].plot(state_history_lpo_body_fixed_array[:, 0] * lu_cr3bp/1e3,
           state_history_lpo_body_fixed_array[:, 1] * lu_cr3bp/1e3, lw=2, zorder=10, label="Orbit")
ax[1].plot(state_history_lpo_body_fixed_array[:, 0] * lu_cr3bp/1e3,
           state_history_lpo_body_fixed_array[:, 2] * lu_cr3bp/1e3, lw=2, zorder=10)

for manifold_branch_id in [0,1]:
    for single_arc_solver in manifold_single_arc_solvers[manifold_branch_id]:

        # Extract manifold state history and convert it to body-fixed frame
        state_history_manifold_inertial = single_arc_solver.state_history
        state_history_manifold_body_fixed = convert_state_history_inertial_to_body_fixed(
            bodies, name_secondary, state_history_manifold_inertial)
        state_history_manifold_body_fixed_array = result2array(state_history_manifold_body_fixed)[:,1:]

        if manifold_branch_id == 0:
            c = "m"
        else:
            c = "r"

        if state_history_manifold_body_fixed_array[-1,1] > 0:
            zorder_y = 6
        else:
            zorder_y = 4

        if state_history_manifold_body_fixed_array[-1,2] > 0:
            zorder_z = 6
        else:
            zorder_z = 4

        # Plot manifold
        ax[0].plot(state_history_manifold_body_fixed_array[:, 0] * lu_cr3bp/1e3,
                   state_history_manifold_body_fixed_array[:, 1] * lu_cr3bp/1e3, lw=0.2, c=c, zorder=zorder_z)
        ax[1].plot(state_history_manifold_body_fixed_array[:, 0] * lu_cr3bp/1e3,
                   state_history_manifold_body_fixed_array[:, 2] * lu_cr3bp/1e3, lw=0.2, c=c, zorder=zorder_y)

# Add label to manifolds
ax[0].plot([np.nan], [np.nan], label="Manifold: -1 branch", c="m", lw=4)
ax[0].plot([np.nan], [np.nan], label="Manifold: +1 branch", c="r", lw=4)

# # Plot the shape of the secondary
# ax[0].tricontourf(vertices_coordinates[:,0] * lu_cr3bp/1e3, vertices_coordinates[:,1] * lu_cr3bp/1e3,
#                   np.zeros(np.shape(vertices_coordinates[:,0])), colors="tab:grey", zorder=5)
# ax[1].tricontourf(vertices_coordinates[:,0] * lu_cr3bp/1e3, vertices_coordinates[:,2] * lu_cr3bp/1e3,
#                   np.zeros(np.shape(vertices_coordinates[:,0])), colors="tab:grey", zorder=5)

for ax_ in ax:
    ax_.set_xlabel('x [km]')
    ax_.grid()
    ax_.set_axisbelow(True)
    ax_.set_aspect('equal')

    ax_.set_xlim(left=ax_.get_xlim()[0] - 1)
    ax_.set_ylim(bottom=ax_.get_ylim()[0] - 1)
    ax_.set_ylim(top=ax_.get_ylim()[1] + 1)

ax[0].legend()
ax[0].set_ylabel('y [km]')
ax[1].set_ylabel('z [km]')
plt.show()