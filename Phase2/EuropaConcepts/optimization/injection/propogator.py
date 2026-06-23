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
import utils.helpers as hp
from setup import *
import functions as fn
    
spice.load_standard_kernels()



bodies, bodies_to_create, body_settings = create_bodies()

fn.min_flyby_altitude = 40000
fn.max_flyby_altitude = 1000000


fn.R = RADIUS_GANYMEDE
fn.fixed_step_size = 10.0
fn.cube = 3 #e6 # 1000km
fn.REFERENCE_FRAME = 3
time = get_time_for_anomaly()


#perform_flyby(time1, bodies_to_create1, bodies1,  body_settings1)


# time, altitude      = optimize_flyby(bodies, bodies_to_create, body_settings)
# print(np.min(time))

times = [680814.0, 72354.0, -535992.0, -526116.0, 720444.0, 731040.0, -493968.0, 136590.0, 150822.0, 168738.0]
output = []

ground_track = None
for i in range(9, 10):
    hp.r_periapsis = R_GA * 0.5 + R_GA * i / 20
    bodies, bodies_to_create, body_settings = create_bodies()
    DURATION = get_time_for_anomaly()
    # time, altitude      = optimize_flyby(bodies, bodies_to_create, body_settings, DURATION)
    # times.append(np.min(time))
    time = [times[i]]
    output1, ground_track             = fn.perform_flyby(time,  bodies_to_create, bodies, body_settings,DURATION, n_steps= 2)
    output.append(output1)
print("times :")

#/home/dominykas/Desktop/optimization/injection/ganymede_ground_track.jpg
fig, ax = plt.subplots()
moon_map = "/home/dominykas/Desktop/optimization/injection/ganymede_ground_track.jpg"
img = plt.imread(moon_map)

fig, ax = plt.subplots()
ax.imshow(img, extent=[0, 360, -90, 90])
#ax.imshow(img, extent=[0, 360, -90, 90])
for k in range(1):

    # Resolve 2pi ambiguity for longitude
    for i in range(len(ground_track)):
        if ground_track[i, 2] < 0:
            ground_track[i, 2] = ground_track[i, 2] + 2.0 * np.pi

    plot = ax.scatter(ground_track[:, 2] * 180 / np.pi, ground_track[:, 1] * 180 / np.pi, s=2,
                      c=ground_track[:, 3] / 1e3, cmap='rainbow_r', vmin=0, vmax=5000)
cb = plt.colorbar(plot)
plt.show()

# fig1 = plt.figure(figsize=(8, 8))
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.set_title(f'System state evolution of all bodies w.r.t SSB.')

# print(times)

# #ax1.plot(output[0][0][0], output[0][0][1], output[0][0][2])
# for i, output1 in enumerate(output):
#     k = i + 6
#     perijove = (R_GA * 0.5 + R_GA * k / 20) /RADIIUS_JUPITER
#     ax1.plot(output1[1][0][::10], output1[1][1][::10], output1[1][2][::10], label=f"perijove{perijove} R_j")
#     # ax1.plot(output1[2][0], output1[2][1], output1[2][2])



# # output1 = fn.plot_jovian_system(0, bodies_to_create, bodies, body_settings, 60 *60 *24 * 30, n_steps = 1)
# # for i in output1:
# #      ax1.plot(i[0], i[1], i[2])

# # # Add a legend, labels, and use a tight layout to save space
# ax1.legend()
# ax1.set_xlabel('x [m]')
# ax1.set_xlim([-fn.cube, fn.cube])
# ax1.set_ylabel('y [m]')
# ax1.set_ylim([-fn.cube, fn.cube])
# ax1.set_zlabel('z [m]')
# ax1.set_zlim([-fn.cube, fn.cube])
# plt.tight_layout()
# plt.show()