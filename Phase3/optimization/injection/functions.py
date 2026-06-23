import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Tudat imports
import tudatpy
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.dynamics import environment_setup, propagation_setup, propagation, simulator
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.kernel.interface import spice
from injection.mgas.constants import *
from setup import *

min_flyby_altitude = 40000
max_flyby_altitude = 1000000

R = RADIUS_GANYMEDE
fixed_step_size = 100.0
cube = 1000 #e6 # 1000km
REFERENCE_FRAME = 1

def status(diff, bodies, bodies_to_create, body_settings, DURATION):
    simulation_start_epoch = DateTime(2000, 4, 25).to_epoch() +  diff
    simulation_end_epoch   = simulation_start_epoch + DURATION# + .0558 * constants.JULIAN_YEAR# - 60*60 * 29

    system_initial_state_barycentric, system_initial_state_hierarchical, central_bodies_barycentric, central_bodies_hierarchical,  acceleration_models_barycentric, acceleration_models_hierarchical,  termination_settings = initialize_simulation(bodies_to_create, bodies, body_settings, simulation_start_epoch, simulation_end_epoch)
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )


    # Create propagation settings
    for propagation_variant in ["barycentric"]:

        if propagation_variant == "barycentric":
            propagator_settings_barycentric = propagation_setup.propagator.translational(
                central_bodies_barycentric,
                acceleration_models_barycentric,
                bodies_to_create,
                system_initial_state_barycentric,
                simulation_start_epoch,
                integrator_settings,
                termination_settings
            )
        else:
            propagator_settings_hierarchical = propagation_setup.propagator.translational(
                central_bodies_hierarchical,
                acceleration_models_hierarchical,
                bodies_to_create,
                system_initial_state_hierarchical,
                simulation_start_epoch,
                integrator_settings,
                termination_settings
            )
    # Propagate the system of bodies and save the state history (all in one step)

    for propagation_variant in ["barycentric"]:

        if propagation_variant == "barycentric":
            results_barycentric = simulator.create_dynamics_simulator(
                bodies, propagator_settings_barycentric).state_history
            
    # Convert the state dictionary to a multi-dimensional array
    barycentric_system_state_array = result2array(results_barycentric)

    # fig1 = plt.figure(figsize=(8, 8))
    # ax1 = fig1.add_subplot(111, projection='3d')
    # ax1.set_title(f'System state evolution of all bodies w.r.t SSB.')

    jupiter_center =  barycentric_system_state_array[:, 6 * REFERENCE_FRAME + 1 : 6 * REFERENCE_FRAME + 4]
    system_state = jupiter_center

    ganymede_state_x = barycentric_system_state_array[:, 6 * 3 + 1] 
    ganymede_state_y = barycentric_system_state_array[:, 6 * 3 + 2]
    ganymede_state_z = barycentric_system_state_array[:, 6 * 3 + 3]

    ganymede_state = barycentric_system_state_array[-1, 6 * 3 + 1: 6* 3 + 7] - barycentric_system_state_array[-1, 6 * 1 + 1: 6* 1 + 7]
    sc_state = barycentric_system_state_array[-1, 6 * 6 + 1: 6* 6 + 7] - barycentric_system_state_array[-1, 6 * 1 + 1: 6* 1 + 7]

    r_sc  = sc_state[:3]
    v_sc  = sc_state[3:]

    r_ga  = ganymede_state[:3]
    v_ga  = ganymede_state[3:]


    h_sc  = np.cross(r_sc, v_sc)
    h_ga  = np.cross(r_ga, v_ga)

    # Angle between the two orbital planes
    cos_angle = np.dot(h_sc, h_ga) / (np.linalg.norm(h_sc) * np.linalg.norm(h_ga))
    plane_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    h_ga  = np.cross(r_ga, v_ga)
    h_hat = h_ga / np.linalg.norm(h_ga)

    # h_hat = [sin(i)*sin(LAN), -sin(i)*cos(LAN), cos(i)]
    inclination_matched = np.arccos(h_hat[2])
    lan_matched         = np.arctan2(h_hat[0], -h_hat[1])

    # print(f"inclination = {np.degrees(inclination_matched):.4f} deg")
    # print(f"LAN         = {np.degrees(lan_matched):.4f} deg")
    # print(f"h_sc  (normalized) = {h_sc / np.linalg.norm(h_sc)}")
    # print(f"h_ga  (normalized) = {h_ga / np.linalg.norm(h_ga)}")
    # print(f"Plane separation   = {plane_angle:.3f} deg")


    state_x =  system_state[:, 0]
    state_y =  system_state[:, 1]
    state_z =  system_state[:, 2]

    reference_state = np.array([state_x, state_y, state_z])

    ganymede_state = ganymede_state[:3]
    sc_state = sc_state[:3]

    separation = ganymede_state / RADIUS_GANYMEDE - sc_state / RADIUS_GANYMEDE
    #print(np.sqrt(np.dot(separation, separation)))
    return np.sqrt(np.dot(separation, separation))


def optimize_flyby(bodies, bodies_to_create, body_settings, DURATION, diff = 60*60 * 43.07):

    x = []
    y = []
    for i in range(-300, 300):
        separation = status(diff + i * 60 * 60, bodies, bodies_to_create, body_settings, DURATION)
        x.append(separation)
        y.append(i)
        print(f"Stage 1 progress : {(i + 1000) / 20}")
    min_index = np.where(x == np.min(x))[0][0]
    time_minimum = y[min_index] * 60 * 60 + diff 

    x = []
    y = []
    for i in range(-100, 100):
        separation = status(time_minimum + i * 60, bodies, bodies_to_create, body_settings, DURATION)
        x.append(separation)
        y.append(i)
        print(f"Stage 2 progress : {(i + 100) / 2}")
    min_index = np.where(x == np.min(x))[0][0]
    time_minimum = y[min_index] * 60  + time_minimum 

    x = []
    y = []
    for i in range(-100, 100):
        separation = status(time_minimum + i * 6, bodies, bodies_to_create, body_settings, DURATION)
        x.append(separation)
        y.append(i)
        print(f"Stage 3 progress : {(i + 100) / 2}")
    min_index = np.where(x == np.min(x))[0][0]
    time_minimum = y[min_index] * 6  + time_minimum 

    x = np.array(x)
    y = np.array(y)
    valid_flybys = np.where(np.array(x) > 1 + min_flyby_altitude / RADIUS_GANYMEDE)[0][:]
    x = x[valid_flybys]
    y = y[valid_flybys]
    print(1 + max_flyby_altitude / RADIUS_GANYMEDE)
    print(1 + min_flyby_altitude / RADIUS_GANYMEDE)
    valid_flybys = np.where(np.array(x) < 1 + max_flyby_altitude / RADIUS_GANYMEDE)[0][:]

    x = x[valid_flybys]
    y = y[valid_flybys]

    left_flybys = x[:int(len(x) / 2)]
    right_flybys = x[int(len(x)):]
    y_left = y[:int(len(x) / 2)]

    y_left = time_minimum + y_left * 6
    # print(y_left)



    # print(np.min(left_flybys))

    # print(min_index)
    # plt.plot(y_left, left_flybys)
    # plt.show()
    return y_left, left_flybys


def perform_flyby(time, bodies_to_create, bodies, body_settings,DURATION,  n_steps = 1):
    simulation_start_epoch = DateTime(2000, 4, 25).to_epoch() +  np.min(time)
    simulation_end_epoch   = simulation_start_epoch + DURATION * n_steps# - 60*60 * 29


    system_initial_state_barycentric, system_initial_state_hierarchical, central_bodies_barycentric, central_bodies_hierarchical,  acceleration_models_barycentric, acceleration_models_hierarchical,  termination_settings = initialize_simulation(bodies_to_create, bodies, body_settings, simulation_start_epoch, simulation_end_epoch)
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    dependent_variables_names = [
        propagation_setup.dependent_variable.latitude("target_orbit", "Ganymede"),
        propagation_setup.dependent_variable.longitude("target_orbit", "Ganymede"),
        propagation_setup.dependent_variable.altitude("target_orbit", "Ganymede")
    ]

    # Create propagation settings
    for propagation_variant in ["barycentric"]:
        propagator_settings_barycentric = propagation_setup.propagator.translational(
            central_bodies_barycentric,
            acceleration_models_barycentric,
            bodies_to_create,
            system_initial_state_barycentric,
            simulation_start_epoch,
            integrator_settings,
            termination_settings,
            propagation_setup.propagator.cowell, 
            dependent_variables_names
        )
    # Propagate the system of bodies and save the state history (all in one step)

    dynamics_simulator = simulator.create_dynamics_simulator(
    bodies, propagator_settings_barycentric
)

    barycentric_system_state_array = result2array(dynamics_simulator.state_history)
    flyby_array = result2array(dynamics_simulator.dependent_variable_history)
    
            
    # Convert the state dictionary to a multi-dimensional array
    #barycentric_system_state_array = result2array(results_barycentric)
    #flyby_array =     result2array(results_barycentric.dependent_variable_history)
    jupiter_center =  barycentric_system_state_array[:, 6 * REFERENCE_FRAME + 1 : 6 * REFERENCE_FRAME + 4]
    system_state = jupiter_center

    ganymede_state_x = barycentric_system_state_array[:, 6 * 3 + 1] 
    ganymede_state_y = barycentric_system_state_array[:, 6 * 3 + 2]
    ganymede_state_z = barycentric_system_state_array[:, 6 * 3 + 3]

    ganymede_state = barycentric_system_state_array[0, 6 * 3 + 1: 6* 3 + 7] - barycentric_system_state_array[0, 6 * 1 + 1: 6* 1 + 7]
    sc_state = barycentric_system_state_array[-1, 6 * 6 + 1: 6* 6 + 7] - barycentric_system_state_array[-1, 6 * 1 + 1: 6* 1 + 7]

    r_sc  = sc_state[:3]
    v_sc  = sc_state[3:]

    r_ga  = ganymede_state[:3]
    v_ga  = ganymede_state[3:]


    h_sc  = np.cross(r_sc, v_sc)
    h_ga  = np.cross(r_ga, v_ga)

    # Angle between the two orbital planes
    cos_angle = np.dot(h_sc, h_ga) / (np.linalg.norm(h_sc) * np.linalg.norm(h_ga))
    plane_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    h_ga  = np.cross(r_ga, v_ga)
    h_hat = h_ga / np.linalg.norm(h_ga)

    # h_hat = [sin(i)*sin(LAN), -sin(i)*cos(LAN), cos(i)]
    inclination_matched = np.arccos(h_hat[2])
    lan_matched         = np.arctan2(h_hat[0], -h_hat[1])

    # print(f"inclination = {np.degrees(inclination_matched):.4f} deg")
    # print(f"LAN         = {np.degrees(lan_matched):.4f} deg")
    # print(f"h_sc  (normalized) = {h_sc / np.linalg.norm(h_sc)}")
    # print(f"h_ga  (normalized) = {h_ga / np.linalg.norm(h_ga)}")
    # print(f"Plane separation   = {plane_angle:.3f} deg")


    state_x =  system_state[:, 0]
    state_y =  system_state[:, 1]
    state_z =  system_state[:, 2]

    reference_state = np.array([state_x, state_y, state_z])

    ganymede_state = ganymede_state[:3]
    sc_state = sc_state[:3]
    output_items = [
        np.array([(barycentric_system_state_array[:, 6 * 3 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 3 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 3 + 3] - state_z) /R]),
         np.array([(barycentric_system_state_array[:, 6 * 6 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 6 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 6 + 3] - state_z) /R]),
        np.array([(barycentric_system_state_array[:, 6 * 7 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 7 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 7 + 3] - state_z) /R])
    ]
    return output_items, flyby_array

def plot_jovian_system(time, bodies_to_create, bodies, body_settings,DURATION,  n_steps = 1):
    simulation_start_epoch = DateTime(2000, 4, 25).to_epoch() +  np.min(time)
    simulation_end_epoch   = simulation_start_epoch + DURATION * n_steps# - 60*60 * 29


    system_initial_state_barycentric, system_initial_state_hierarchical, central_bodies_barycentric, central_bodies_hierarchical,  acceleration_models_barycentric, acceleration_models_hierarchical,  termination_settings = initialize_simulation(bodies_to_create, bodies, body_settings, simulation_start_epoch, simulation_end_epoch)
    # integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    #     fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    # )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )


    # Create propagation settings
    for propagation_variant in ["barycentric"]:

        if propagation_variant == "barycentric":
            propagator_settings_barycentric = propagation_setup.propagator.translational(
                central_bodies_barycentric,
                acceleration_models_barycentric,
                bodies_to_create,
                system_initial_state_barycentric,
                simulation_start_epoch,
                integrator_settings,
                termination_settings
            )
        else:
            propagator_settings_hierarchical = propagation_setup.propagator.translational(
                central_bodies_hierarchical,
                acceleration_models_hierarchical,
                bodies_to_create,
                system_initial_state_hierarchical,
                simulation_start_epoch,
                integrator_settings,
                termination_settings
            )
    # Propagate the system of bodies and save the state history (all in one step)

    for propagation_variant in ["barycentric"]:

        if propagation_variant == "barycentric":
            results_barycentric = simulator.create_dynamics_simulator(
                bodies, propagator_settings_barycentric).state_history
            
    # Convert the state dictionary to a multi-dimensional array
    barycentric_system_state_array = result2array(results_barycentric)

    jupiter_center =  barycentric_system_state_array[:, 6 * REFERENCE_FRAME + 1 : 6 * REFERENCE_FRAME + 4]
    system_state = jupiter_center

    ganymede_state_x = barycentric_system_state_array[:, 6 * 3 + 1] 
    ganymede_state_y = barycentric_system_state_array[:, 6 * 3 + 2]
    ganymede_state_z = barycentric_system_state_array[:, 6 * 3 + 3]

    ganymede_state = barycentric_system_state_array[0, 6 * 3 + 1: 6* 3 + 7] - barycentric_system_state_array[0, 6 * 1 + 1: 6* 1 + 7]
    sc_state = barycentric_system_state_array[-1, 6 * 6 + 1: 6* 6 + 7] - barycentric_system_state_array[-1, 6 * 1 + 1: 6* 1 + 7]

    r_sc  = sc_state[:3]
    v_sc  = sc_state[3:]

    r_ga  = ganymede_state[:3]
    v_ga  = ganymede_state[3:]


    h_sc  = np.cross(r_sc, v_sc)
    h_ga  = np.cross(r_ga, v_ga)

    # Angle between the two orbital planes
    cos_angle = np.dot(h_sc, h_ga) / (np.linalg.norm(h_sc) * np.linalg.norm(h_ga))
    plane_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    h_ga  = np.cross(r_ga, v_ga)
    h_hat = h_ga / np.linalg.norm(h_ga)

    # h_hat = [sin(i)*sin(LAN), -sin(i)*cos(LAN), cos(i)]
    inclination_matched = np.arccos(h_hat[2])
    lan_matched         = np.arctan2(h_hat[0], -h_hat[1])

    # print(f"inclination = {np.degrees(inclination_matched):.4f} deg")
    # print(f"LAN         = {np.degrees(lan_matched):.4f} deg")
    # print(f"h_sc  (normalized) = {h_sc / np.linalg.norm(h_sc)}")
    # print(f"h_ga  (normalized) = {h_ga / np.linalg.norm(h_ga)}")
    # print(f"Plane separation   = {plane_angle:.3f} deg")


    state_x =  system_state[:, 0]
    state_y =  system_state[:, 1]
    state_z =  system_state[:, 2]

    reference_state = np.array([state_x, state_y, state_z])

    ganymede_state = ganymede_state[:3]
    sc_state = sc_state[:3]
    output_items = [
        np.array([(barycentric_system_state_array[:, 6 * 1 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 1 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 1 + 3] - state_z) /R]),

        np.array([(barycentric_system_state_array[:, 6 * 3 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 3 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 3 + 3] - state_z) /R]),

        np.array([(barycentric_system_state_array[:, 6 * 2 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 2 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 2 + 3] - state_z) /R]),

        np.array([(barycentric_system_state_array[:, 6 * 4 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 4 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 4 + 3] - state_z) /R]),
                
        np.array([(barycentric_system_state_array[:, 6 * 5 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 5 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 5 + 3] - state_z) /R])
    ]
    return output_items
