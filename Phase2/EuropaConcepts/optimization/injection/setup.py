# General imports
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
from tudatpy.kernel.astro import element_conversion

# Pygmo imports
import pygmo as pg
from utils.helpers import *

def create_bodies(FRAME = "J2000"):

    central_body = "Jupiter"

    #values are rughly approx
    minimum_pericenters = {
        "Callisto": 2634.1e3 + 100e3,
        "Ganymede": 2634.1e3 + 100e3, 
        "Europa":   1560.8e3 + 100e3,  
        "Io":       1821.6e3 + 100e3, 
    }

    bodies_to_create = ["Sun", "Jupiter", "Callisto", "Ganymede", "Europa", "Io"]

    global_frame_origin = "SSB"        
    global_frame_orientation = FRAME


    bodies = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation
    )
    # Add empty body settings for body Oumuamua, and add to existing list of settings 
    body_settings.add_empty_settings( "target_orbit" )
    body_settings.add_empty_settings( "target_orbit_2" )

    # Manually create and assign environment model settings to new body settings
    body_settings.get( "target_orbit" ).ephemeris_settings =  environment_setup.ephemeris.custom_ephemeris( 
        spacecraft_state_function, 'Jupiter', FRAME )
    oumuamua_gravitational_parameter = 0.08  # Example value, adjust as needed

    body_settings.get("target_orbit").gravity_field_settings = (
        environment_setup.gravity_field.central(oumuamua_gravitational_parameter)
    )

    body_settings.get("Ganymede").rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Jupiter", global_frame_orientation, "IAU_" + "Ganymede")




    # # Manually create and assign environment model settings to new body settings
    body_settings.get( "target_orbit_2" ).ephemeris_settings =  environment_setup.ephemeris.custom_ephemeris( 
        spacecraft_state_function3, 'Jupiter', FRAME )
    oumuamua_gravitational_parameter = 0.008  # Example value, adjust as needed

    body_settings.get("target_orbit_2").gravity_field_settings = (
        environment_setup.gravity_field.central(oumuamua_gravitational_parameter)
    )
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies_to_create.append("target_orbit")
    bodies_to_create.append("target_orbit_2")
    return bodies, bodies_to_create, body_settings


def initialize_simulation(bodies_to_create, bodies, body_settings, simulation_start_epoch, simulation_end_epoch, ephemeris = None):


    acceleration_dict = {}
    for body_i in bodies_to_create:
        current_accelerations = {}
        for body_j in bodies_to_create:
            if body_i != body_j:
                current_accelerations[body_j] = [
                    propagation_setup.acceleration.point_mass_gravity()
                ]
        acceleration_dict[body_i] = current_accelerations

    
    delta_v_vector = np.array([[0.0],   # x-component [m/s]
                              [0.0],    # y-component
                            [100000.0]])   # z-component

  
    kep_elements = element_conversion.cartesian_to_keplerian(spacecraft_state_function(simulation_start_epoch) , jupiter_gravitational_parameter)
    
    a   = kep_elements[0]   # negative for hyperbolic!
    e   = kep_elements[1]   # > 1 for hyperbolic
    nu0 = kep_elements[5]   # current true anomaly [rad]

    if e > 1.0:
        # ── HYPERBOLIC CASE ──────────────────────────────────────────
        # Hyperbolic eccentric anomaly F from true anomaly
        F0 = 2.0 * np.arctanh(
            np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu0 / 2.0)
        )
        # Hyperbolic mean anomaly
        M0 = e * np.sinh(F0) - F0

        # Mean motion (a is negative, so use |a|)
        n = np.sqrt(jupiter_gravitational_parameter / abs(a)**3)

        # Time since perigee (can be negative if not yet reached, positive if past)
        t_since_perigee = M0 / n

        if t_since_perigee >= 0:
            # Perigee is in the past — the s/c already passed it
            print(f"WARNING: Perigee was {t_since_perigee:.1f} s ago. "
                f"Cannot burn at perigee unless you propagate backwards.")
            # Options:
            #   1. Burn at simulation_start_epoch (right now, closest to perigee)
            #   2. Skip the maneuver
            #   3. Use a different burn point (e.g. specific altitude)
            perigee_epoch = simulation_start_epoch  # fallback: burn immediately
        else:
            # Perigee is still ahead (t_since_perigee < 0 means time until perigee)
            dt_to_perigee = abs(t_since_perigee)
            perigee_epoch = simulation_start_epoch + dt_to_perigee
            print(f"Perigee in {dt_to_perigee:.1f} s (epoch {perigee_epoch:.1f} s)")

    else:
        # ── ELLIPTIC CASE (fallback) ─────────────────────────────────
        E0 = 2.0 * np.arctan2(
            np.sqrt(1.0 - e) * np.sin(nu0 / 2.0),
            np.sqrt(1.0 + e) * np.cos(nu0 / 2.0)
        )
        M0 = (E0 - e * np.sin(E0)) % (2.0 * np.pi)
        n  = np.sqrt(jupiter_gravitational_parameter / a**3)
        remaining_M = (2.0 * np.pi - M0) % (2.0 * np.pi)
        dt_to_perigee = remaining_M / n
        perigee_epoch = simulation_start_epoch + dt_to_perigee
        print(f"Perigee in {dt_to_perigee:.1f} s (epoch {perigee_epoch:.1f} s)")
#

    # Step 1: get angular momentum vector from current state (conserved quantity)
    r_sc = spacecraft_state_function(simulation_start_epoch)[:3]
    v_sc = spacecraft_state_function(simulation_start_epoch)[3:]
    h_vec = np.cross(r_sc, v_sc)           # angular momentum vector [m²/s]
    h_hat = h_vec / np.linalg.norm(h_vec)  # unit normal to orbital plane

    # Step 2: get eccentricity vector (points from focus toward perigee)
    mu = jupiter_gravitational_parameter
    e_vec = np.cross(v_sc, h_vec) / mu - r_sc / np.linalg.norm(r_sc)
    e_hat = e_vec / np.linalg.norm(e_vec)  # unit vector toward perigee

    # Step 3: prograde at perigee = h_hat × e_hat
    #         (perpendicular to both the perigee direction and orbit normal)
    prograde_hat = np.cross(h_hat, e_hat)
    prograde_hat = prograde_hat / np.linalg.norm(prograde_hat)  # normalize (should already be unit)

    # Step 4: scale by desired delta-v magnitude
    delta_v_magnitude = -880.1#-810.02  # [m/s] — adjust as needed
    delta_v_vector = (delta_v_magnitude * prograde_hat).reshape(3, 1)
   # delta_v_vector = np.array([0, 0, 0])
    impulsive_burn_settings = propagation_setup.acceleration.quasi_impulsive_shots_acceleration(
        thrust_mid_times    = [simulation_start_epoch + dt_to_perigee],          # mid-time of the burn [s]
        delta_v_values      = [delta_v_vector],  # list of (3,1) arrays
        total_maneuver_time = 60.0,              # must be > 0
        maneuver_rise_time  = 10.0              # must be > 0 and < total_maneuver_time / 2
    )

    # Slot it into the existing acceleration dict
    if "target_orbit" not in acceleration_dict["target_orbit"]:
        acceleration_dict["target_orbit"]["target_orbit"] = []
    acceleration_dict["target_orbit"]["target_orbit"].append(impulsive_burn_settings)


    # Convert acceleration mappings into acceleration models for both propagation variants
    # Central bodies for barycentric propagation
    central_bodies_barycentric = ["SSB"] * len(bodies_to_create)
    central_bodies_hierarchical = []
    for body_name in bodies_to_create:
        if body_name != "Jupiter":
            central_bodies_hierarchical.append("Jupiter")
        else:
            central_bodies_hierarchical.append("SSB")

    for propagation_variant in ["barycentric", "hierarchical"]:
        central_bodies = central_bodies_barycentric if propagation_variant == "barycentric" else central_bodies_hierarchical

        acceleration_models = propagation_setup.create_acceleration_models(
            body_system=bodies,
            selected_acceleration_per_body=acceleration_dict,
            bodies_to_propagate=bodies_to_create,
            central_bodies=central_bodies
        )

        if propagation_variant == "barycentric":
            acceleration_models_barycentric = acceleration_models
        else:
            acceleration_models_hierarchical = acceleration_models

    # Define the initial state of each body, taking them from SPICE
    for propagation_variant in ["barycentric", "hierarchical"]:
        central_bodies = central_bodies_barycentric if propagation_variant == "barycentric" else central_bodies_hierarchical

        system_initial_state = propagation.get_initial_state_of_bodies(
            bodies_to_propagate=bodies_to_create,
            central_bodies=central_bodies,
            body_system=bodies,
            initial_time=simulation_start_epoch
        )

        if propagation_variant == "barycentric":
            system_initial_state_barycentric = system_initial_state
        else:
            system_initial_state_hierarchical = system_initial_state

    # Create termination settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    return system_initial_state_barycentric, system_initial_state_hierarchical, central_bodies_barycentric, central_bodies_hierarchical,  acceleration_models_barycentric, acceleration_models_hierarchical,  termination_settings
