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

R_IO = 	420000000 
R_EU =  664862000 
R_GA =  1071600000 
RADIUS_GANYMEDE = 2634100
RADIIUS_JUPITER = 69911000