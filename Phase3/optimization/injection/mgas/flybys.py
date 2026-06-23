import sys
# Load required standard modules
import os
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment, environment_setup
from tudatpy.dynamics import propagation, propagation_setup, simulator
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array


# Specify which moon is to be considered for the JUICE flybys (can be set to Europa, Ganymede, or Callisto)
flyby_moon = "Ganymede"
if flyby_moon != "Europa" and flyby_moon != "Ganymede" and flyby_moon != "Callisto":
    raise NameError('flyby_moon should be set to Europa, Ganymede, or Callisto.')

# Load spice kernels
path = os.path.abspath('../')
kernels = [path+'/kernels/kernel_juice.bsp',path+'/kernels/kernel_noe.bsp']
spice.load_standard_kernels(kernels)

# Set simulation start and end epochs according to JUICE mission timeline
start_epoch = 32.0 * constants.JULIAN_YEAR
end_epoch = 34.5 * constants.JULIAN_YEAR

# Define default body settings
bodies_to_create = ["Europa", "Ganymede", "Callisto", "Io", "Jupiter", "Sun"]
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin,
                                                            global_frame_orientation)

# Set rotation of flyby moon to synchronous
body_settings.get("Europa").rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Jupiter", global_frame_orientation, "IAU_" + "Europa")



# Create empty settings for JUICE spacecraft
body_settings.add_empty_settings("JUICE")

# Add JUICE mass to body settings
body_settings.get("JUICE").constant_mass = 5.0e3 #kg

# Set empty ephemeris for JUICE
empty_ephemeris_dict = dict()
juice_ephemeris = environment_setup.ephemeris.tabulated(
    empty_ephemeris_dict,
    global_frame_origin,
    global_frame_orientation)
body_settings.get("JUICE").ephemeris_settings = juice_ephemeris