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

# Pygmo imports
import pygmo as pg
from helpers import *

    
spice.load_standard_kernels()



central_body = "Jupiter"


bodies_to_create = ["Jupiter", "Callisto", "Ganymede", "Europa", "Io"]

global_frame_origin = "Jupiter"        
global_frame_orientation = "J2000"


bodies = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)


#____________________________departure orbits_______________________
departure_semi_major_axis = np.inf
departure_eccentricity = 0

#____________________________insertion orbits_______________________
arrival_semi_major_axis =  np.inf 
arrival_eccentricity = 0.0

FRAME = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation
)

# body_settings = environment_setup.get_default_body_settings(
#     ["Jupiter", "Callisto", "Ganymede", "Europa", "Io"],
#     "SSB", 
#     FRAME
# )

# Add empty body settings for body Oumuamua, and add to existing list of settings 
body_settings.add_empty_settings( "target_orbit" )

# Manually create and assign environment model settings to new body settings
body_settings.get( "target_orbit" ).ephemeris_settings =  environment_setup.ephemeris.custom_ephemeris( 
	spacecraft_state_function, 'Jupiter', FRAME )
oumuamua_gravitational_parameter = 0.08  # Example value, adjust as needed

body_settings.get("target_orbit").gravity_field_settings = (
    environment_setup.gravity_field.central(oumuamua_gravitational_parameter)
)

body_settings.get("Callisto").ephemeris_settings = environment_setup.ephemeris.direct_spice(
    "Jupiter", FRAME
)
body_settings.get("Ganymede").ephemeris_settings = environment_setup.ephemeris.direct_spice(
    "Jupiter", FRAME
)
body_settings.get("Europa").ephemeris_settings = environment_setup.ephemeris.direct_spice(
    "Jupiter", FRAME
)
body_settings.get("Io").ephemeris_settings = environment_setup.ephemeris.direct_spice(
    "Jupiter", FRAME
)

bodies = environment_setup.create_system_of_bodies(body_settings)
bodies_to_create.append("target_orbit")

