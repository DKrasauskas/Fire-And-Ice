import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup, propagation, simulator
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion
from scipy.optimize import brentq

from orbits import *
import helpers as hp

spice.load_standard_kernels()

inclination_deg = 90
aop_deg         = 90
n               = 29.9975

simulation_years       = 2 /365
simulation_start_epoch = DateTime(2020, 1, 1, 14, 30, 0).to_epoch()
simulation_end_epoch   = simulation_start_epoch + simulation_years * constants.JULIAN_YEAR

bodies_to_create  = ["Europa", "Ganymede", "Callisto", "Io", "Jupiter"]
body_to_propagate = ["SoIaF"]

global_frame_origin      = "Jupiter"
global_frame_orientation = "J2000"



body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)
body_settings.add_empty_settings("SoIaF")
body_settings.get("SoIaF").constant_mass = 2000.0
bodies = environment_setup.create_system_of_bodies(body_settings)

mu_jup = bodies.get("Jupiter").gravitational_parameter

_io_s    = spice.get_body_cartesian_state_at_epoch(
    "Io", global_frame_origin, global_frame_orientation, "NONE", simulation_start_epoch
)
_io_h    = np.cross(_io_s[:3], _io_s[3:6])
_io_node = np.cross(np.array([0.0, 0.0, 1.0]), _io_h)
raan_deg = float(np.rad2deg(np.arctan2(_io_node[1], _io_node[0]))) % 360.0
print(f"RAAN: {raan_deg:.2f} deg")

r_Io         = 421700000
T_Io         = 2 * np.pi * np.sqrt(r_Io**3 / mu_jup) * 1.001
a_cap        = (mu_jup * (n * T_Io / (2 * np.pi))**2) ** (1 / 3)
eccentricity = np.sqrt(1 - r_Io / a_cap)
r_periapsis  = a_cap * (1 - eccentricity)
r_apoapsis   = a_cap * (1 + eccentricity)
T_cap   = 2 * np.pi * np.sqrt(a_cap**3 / mu_jup) / 3600  # hours
T_cap_s = T_cap * 3600



DURATION = T_Io *  10.574
R = hp.R

R_perijove = hp.R * 0.5

mu = mu_jup

eccentricity = solve_eccentricity(DURATION * 1.005, R, mu)

a_cap = R / (1 - eccentricity**2)

moon_map = "/home/dominykas/Desktop/io_visual.png"
img = plt.imread(moon_map)

fig, (ax_north, ax_south) = plt.subplots(1, 2, figsize=(8.4, 6), subplot_kw={'projection': 'polar'})
plt.rcParams['axes.grid'] = False  
fig.figimage(img, xo=70, yo=170, zorder=-1)

params = Ephemeris(
    (
        18859035560.264233 - 20000000,
        0.98804852543645449,
        np.deg2rad(89.900551367034282),
        np.deg2rad(86.444023939868416),
        np.deg2rad(106.37893435189432) - np.deg2rad(0.7),
        6.2809071433766599
    )
)
set_spacecraft_ephemeris(params)





params = Ephemeris(
    (
        18859035560.264233- 1030000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.7),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 1000000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.8),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 950000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.9),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)


params = Ephemeris(
    (
        18859035560.264233- 1050000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.6),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)


params = Ephemeris(
    (
        18859035560.264233- 1050000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.5),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)



params = Ephemeris(
    (
        18859035560.264233- 1050000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.4),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 1050000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.3),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 1050000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.2),
        6.2809071433766599 - 0.000305# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 990000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.1),
        6.2809071433766599 - 0.000297# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 900000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.),
        6.2809071433766599 - 0.000285# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)


params = Ephemeris(
    (
        18859035560.264233- 880000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(.9),
        6.2809071433766599 - 0.000275# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 840000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.),
        6.2809071433766599 - 0.000265# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 840000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.),
        6.2809071433766599 - 0.000255# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 840000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(1.),
        6.2809071433766599 - 0.000245# - 0.0001
    )
)
set_spacecraft_ephemeris(params)
plot(ax_north, ax_south)

params = Ephemeris(
    (
        18859035560.264233- 890000000,
        0.98804852543645449,
        np.deg2rad(84.900551367034282),
        np.deg2rad(89.444023939868416),
        np.deg2rad(106.37893435189432) + np.deg2rad(.9),
        6.2809071433766599 - 0.000235# - 0.0001
    )
)

set_spacecraft_ephemeris(params)
plot(ax_north, ax_south, True)


# cb = fig.colorbar(plot_last, ax=[ax_north, ax_south], orientation='horizontal', pad=0.15, shrink=0.6)
# cb.set_label('Altitude (km)')
plt.show()

