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
import helpers as hp
from constants import *

class Ephemeris:
    def __init__(self, params):
        self.a, self.e, self.i, self.aop, self.lan, self.M = params
        

def set_spacecraft_ephemeris(ephemeris):
    hp.arg_of_periapsis = ephemeris.aop
    hp.semi_major_axis  = ephemeris.a
    hp.lan              = ephemeris.lan
    hp.Mean             = ephemeris.M
    hp.inclination      = ephemeris.i
    hp.eccentricity     = ephemeris.e

def phase_true_anomaly(raan_deg, eccentricity, T_Io, T_cap_s,  _io_s):
    dt      = (((np.deg2rad(raan_deg + 180.0) - np.arctan2(_io_s[1], _io_s[0])) % (2*np.pi))
               / (2 * np.pi / T_Io))
    E_cross = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)))
    M_0     = (E_cross - eccentricity * np.sin(E_cross) - 2 * np.pi / T_cap_s * dt) % (2 * np.pi)
    E_0     = M_0
    for _ in range(100):
        dE  = (M_0 - E_0 + eccentricity * np.sin(E_0)) / (1 - eccentricity * np.cos(E_0))
        E_0 += dE
        if abs(dE) < 1e-12:
            break
    nu_0 = (2 * np.arctan2(np.sqrt(1 + eccentricity) * np.sin(E_0 / 2),
                            np.sqrt(1 - eccentricity) * np.cos(E_0 / 2))) % (2 * np.pi)
    print(f"Phased true anomaly: {np.rad2deg(nu_0):.2f} deg  (Io at crossing in {dt/3600:.2f} h)")
    return nu_0


def print_info(T_cap, r_periapsis, eccentricity, a_cap):
    print(f"T = {T_cap/24:.2f} days  |  e = {eccentricity:.3f}  |  "
          f"r_peri = {r_periapsis/R_Jupiter:.2f} RJ  |  a = {a_cap/R_Jupiter:.2f} RJ")



def find_flybys(states, global_frame_origin, global_frame_orientation):
    io_pos = np.array([
        spice.get_body_cartesian_state_at_epoch(
            "Io", global_frame_origin, global_frame_orientation, "NONE", t)[:3]
        for t in states[:, 0]
    ])
    dist    = np.linalg.norm(states[:, 1:4] - io_pos, axis=1)
    indices = [
        i for i in range(1, len(dist) - 1)
        if dist[i] < dist[i-1] and dist[i] < dist[i+1] and dist[i] < flyby_threshold
    ]
    print(f"Flybys found (< {flyby_threshold/1e3:.0f} km): {len(indices)}")
    if not indices:
        raise SystemExit("No flybys — increase flyby_threshold.")
    return indices


def run_detailed_propagation(states, flyby_indices, bodies, body_to_propagate, global_frame_origin, global_frame_orientation, simulation_start_epoch):
    accel = propagation_setup.create_acceleration_models(
        bodies,
        {"SoIaF": {"Jupiter": [propagation_setup.acceleration.point_mass_gravity()]}},
        body_to_propagate, ["Io"]
    )
    integrator = propagation_setup.integrator.runge_kutta_fixed_step(
        1, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )
    dep_vars = [
        propagation_setup.dependent_variable.latitude("SoIaF", "Io"),
        propagation_setup.dependent_variable.longitude("SoIaF", "Io"),
        propagation_setup.dependent_variable.altitude("SoIaF", "Io"),
    ]

    arc_settings = []
    flyby_epochs = []
    for idx in flyby_indices:
        t_ca        = states[idx, 0]
        io_state    = spice.get_body_cartesian_state_at_epoch(
            "Io", global_frame_origin, global_frame_orientation, "NONE", t_ca
        )
        sc_state_io = states[idx, 1:7] - io_state
        day         = (t_ca - simulation_start_epoch) / constants.JULIAN_DAY
        print(f"  day {day:6.1f}  dist to Io ≈ {np.linalg.norm(sc_state_io[:3])/1e3:,.0f} km")

        term = propagation_setup.propagator.non_sequential_termination(
            propagation_setup.propagator.time_termination(t_ca + flyby_half_window),
            propagation_setup.propagator.time_termination(t_ca - flyby_half_window)
        )
        arc_settings.append(propagation_setup.propagator.translational(
            ["Io"], accel, body_to_propagate, sc_state_io, t_ca,
            integrator, term, propagation_setup.propagator.cowell, dep_vars
        ))
        flyby_epochs.append(t_ca)

    print("Running detailed propagation...")
    sim = simulator.create_dynamics_simulator(
        bodies, propagation_setup.propagator.multi_arc(arc_settings)
    )
    return sim.propagation_results.single_arc_results, flyby_epochs



def plot_altitude_profiles(arc_results, flyby_epochs, simulation_start_epoch):
    n_flybys = len(arc_results)
    fig, axes = plt.subplots((n_flybys + 1) // 2, 2, figsize=(12, 4 * ((n_flybys + 1) // 2)), squeeze=False)
    axes = axes.flatten()

    for k in range(n_flybys):
        dep    = result2array(arc_results[k].dependent_variable_history)
        st     = result2array(arc_results[k].state_history)
        t_h    = (st[:, 0] - flyby_epochs[k]) / 3600.0
        alt_km = dep[:, 3] / 1e3
        day    = (flyby_epochs[k] - simulation_start_epoch) / constants.JULIAN_DAY

        axes[k].plot(t_h, alt_km)
        axes[k].axvline(0, color="r", linestyle="--", label="CA")
        axes[k].set_title(f"Flyby #{k+1}  (day {day:.1f})")
        axes[k].set_xlabel("Time from CA [h]")
        axes[k].set_ylabel("Altitude [km]")
        axes[k].legend()
        axes[k].grid(True)

    for j in range(n_flybys, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_ground_tracks(arc_results, propagation_results):
    # Plot flybys
    moon_map = "/home/dominykas/Desktop/Fire-And-Ice/Io Flyby/io.jpg"
    img = plt.imread(moon_map)

    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 360, -90, 90])
    for k in range(2):
        dependent_variables = result2array(propagation_results[k].dependent_variable_history)

        # Resolve 2pi ambiguity for longitude
        for i in range(len(dependent_variables)):
            if dependent_variables[i, 2] < 0:
                dependent_variables[i, 2] = dependent_variables[i, 2] + 2.0 * np.pi

        plot = ax.scatter(dependent_variables[:, 2] * 180 / np.pi, dependent_variables[:, 1] * 180 / np.pi, s=2,
                        c=dependent_variables[:, 3] / 1e3, cmap='rainbow_r', vmin=0, vmax=5000)
    cb = plt.colorbar(plot)
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.xticks(np.arange(0, 361, 40))
    plt.yticks(np.arange(-90, 91, 30))
    cb.set_label('Altitude [km]')
    plt.title('SoFI flybys at ' + "Io")
    plt.show()


def plot_3d_trajectory(states):
    ax = plt.figure(figsize=(9, 7)).add_subplot(111, projection="3d")
    #io_pos = io_pos / R_Jupiter
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_io / R_Jupiter * np.cos(theta), R_io / R_Jupiter * np.sin(theta),
            np.zeros(300), "k:", lw=0.8, label="Io orbit")
    ax.scatter(0, 0, 0, s=50, color="orange", label="Jupiter")
    #ax.plot(io_pos[:, 0] , io_pos[:, 1], io_pos[:, 2], color="orange", label="Io")
    ax.plot(states[:, 1] / R_Jupiter, states[:, 2] / R_Jupiter, states[:, 3] / R_Jupiter,
            lw=0.4, label="SoIaF")
    ax.plot(states[:, 7] / R_Jupiter, states[:, 8] / R_Jupiter, states[:, 9] / R_Jupiter,
            lw=0.4, label="SoIaF")

    # for k, idx in enumerate(flyby_indices):
    #     x, y, z = states[idx, 1:4] / R_Jupiter
    #     ax.scatter(x, y, z, s=30, color="r", label="flyby" if k == 0 else None)

    lim = 10# * R_Jupiter * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [RJ]"); ax.set_ylabel("y [RJ]"); ax.set_zlabel("z [RJ]")
    ax.set_title("SoIaF trajectory")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


    # transit time can be computed via:

def solve_eccentricity(delta_t, R, mu):
    def E1(e):
        return 2 * np.arctan(np.sqrt((1 - e) / (1 + e)))

    def transit_time(e):
        e1 = E1(e)
        M1 = e1 - e * np.sin(e1)
        T = 2 * np.pi * np.sqrt((R / (1 - e**2))**3 / mu)
        return T * (1 - M1 / np.pi)

    # Solve transit_time(e) = delta_t
    e_solution = brentq(lambda e: transit_time(e) - delta_t, 1e-6, 1 - 1e-9)
    return e_solution


def plot_jovian_system(time, bodies_to_create, bodies, body_settings,DURATION,  n_steps = 1):
    simulation_start_epoch = DateTime(2000, 4, 25).to_epoch() + time
    simulation_end_epoch   = DateTime(2000, 4, 25).to_epoch() + time + DURATION * n_steps# - 60*60 * 29


    system_initial_state_barycentric, system_initial_state_hierarchical, central_bodies_barycentric, central_bodies_hierarchical,  acceleration_models_barycentric, acceleration_models_hierarchical,  termination_settings = hp.initialize_simulation(bodies_to_create, bodies, body_settings, simulation_start_epoch, simulation_end_epoch)
    # integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    #     fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    # )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    dependent_variables_names = [
        propagation_setup.dependent_variable.latitude("target_orbit", "Io"),
        propagation_setup.dependent_variable.longitude("target_orbit", "Io"),
        propagation_setup.dependent_variable.altitude("target_orbit", "Io")
    ]
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
                termination_settings,
                propagation_setup.propagator.cowell, 
                dependent_variables_names
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
    dynamics_simulator = simulator.create_dynamics_simulator(
    bodies, propagator_settings_barycentric
    )

    barycentric_system_state_array = result2array(dynamics_simulator.state_history)
    flyby_array = result2array(dynamics_simulator.dependent_variable_history)
    
    jupiter_center =  barycentric_system_state_array[:, 6 * REFERENCE_FRAME + 1 : 6 * REFERENCE_FRAME + 4]
    system_state = jupiter_center

    ganymede_state_x = barycentric_system_state_array[:, 6 * 3 + 1] 
    ganymede_state_y = barycentric_system_state_array[:, 6 * 3 + 2]
    ganymede_state_z = barycentric_system_state_array[:, 6 * 3 + 3]

    ganymede_state = barycentric_system_state_array[0, 6 * 5 + 1: 6* 5 + 7] - barycentric_system_state_array[0, 6 * 1 + 1: 6* 1 + 7]
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

    print(f"inclination = {np.degrees(inclination_matched):.4f} deg")
    print(f"LAN         = {np.degrees(lan_matched):.4f} deg")
    print(f"h_sc  (normalized) = {h_sc / np.linalg.norm(h_sc)}")
    print(f"h_ga  (normalized) = {h_ga / np.linalg.norm(h_ga)}")
    print(f"Plane separation   = {plane_angle:.3f} deg")


    state_x =  system_state[:, 0]
    state_y =  system_state[:, 1]
    state_z =  system_state[:, 2]

    reference_state = np.array([state_x, state_y, state_z])
    R = R_io
    ganymede_state = ganymede_state[:3]
    sc_state = sc_state[:3]
    output_items = [
        # np.array([(barycentric_system_state_array[:, 6 * 3 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 3 + 2] - state_y) /R,
        #         (barycentric_system_state_array[:, 6 * 3 + 3] - state_z) /R]),
        #  np.array([(barycentric_system_state_array[:, 6 * 2 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 2 + 2] - state_y) /R,
        #         (barycentric_system_state_array[:, 6 * 2 + 3] - state_z) /R]),
        # np.array([(barycentric_system_state_array[:, 6 * 4 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 4 + 2] - state_y) /R,
        #         (barycentric_system_state_array[:, 6 * 4 + 3] - state_z) /R]),
        np.array([(barycentric_system_state_array[:, 6 * 5 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 5 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 5 + 3] - state_z) /R]),
        np.array([(barycentric_system_state_array[:, 6 * 6 + 1] - state_x) / R , (barycentric_system_state_array[:, 6 * 6 + 2] - state_y) /R,
                (barycentric_system_state_array[:, 6 * 6 + 3] - state_z) /R])
    ]
    return output_items, flyby_array

def plot(ax_north, ax_south, cbar = False):
    bodies, bodies_to_create, body_settings = hp.create_bodies()
    DURATION = 1200
    output1, ground_track = plot_jovian_system(36500, bodies_to_create, bodies, body_settings, DURATION, n_steps = 1)
    print(ground_track)

    ground_track1 = ground_track[:, :]

    ground_track2 = ground_track[-100:, :]
    ground_trackT = [ground_track1]
    # fig1 = plt.figure(figsize=(8, 8))
    # ax1 = fig1.add_subplot(111, projection='3d')
    # ax1.set_title(f'System state evolution of all bodies w.r.t SSB.')


    # cube = 5
    # for i in output1:
    #     ax1.plot(i[0], i[1], i[2])
    #     ax1.scatter(i[0][0], i[1][0], i[2][0])
    #     ax1.scatter(i[0][-1], i[1][-1], i[2][-1], marker="o")
    # ax1.legend()
    # ax1.set_xlabel('x [m]')
    # ax1.set_xlim([-cube, cube])
    # ax1.set_ylabel('y [m]')
    # ax1.set_ylim([-cube, cube])
    # ax1.set_zlabel('z [m]')
    # ax1.set_zlim([-cube, cube])
    # plt.tight_layout()
    # plt.show()


    ax_north.grid(False)
    ax_north.set_axis_off()
    ax_south.set_axis_off()
    ax_north.set_title('Northern Hemisphere', fontsize=12, fontweight='bold', pad=10)
    ax_south.set_title('Southern Hemispherec', fontsize=12, fontweight='bold', pad=10)

    for ground_track1 in ground_trackT:
        correct_indices = np.where(ground_track1[:, 3] / 1e3 < 50000)[0]
        ground_track1 = ground_track1[correct_indices, :]

        ground_track1[:, 2] = ground_track1[:, 2] + np.pi
        for i in range(len(ground_track1)):
            if ground_track1[i, 2] < 0:
                ground_track1[i, 2] = ground_track1[i, 2] + 2.0 * np.pi

        lon_rad = ground_track1[:, 2]
        lat_rad = ground_track1[:, 1]
        lat_deg = lat_rad * 180 / np.pi
        alt_km = ground_track1[:, 3] / 1e3

        north_mask = lat_deg >= 0
        south_mask = lat_deg < 0

        if np.any(north_mask):
            r_north = 90 - lat_deg[north_mask]
            lon_south_flipped = -lon_rad[north_mask] % (2 * np.pi)
            plot_last = ax_north.scatter(lon_rad[north_mask], r_north, s=2,
                                        c=alt_km[north_mask], cmap='rainbow_r', vmin=40, vmax=2000, zorder=20)
            plot_last = ax_south.scatter(lon_south_flipped, r_north, s=2,
                                        c=alt_km[north_mask], cmap='rainbow_r', vmin=40, vmax=2000, zorder=20)

        if np.any(south_mask):
            # Flip longitude direction to account for orientation reversal
            # when projecting south polar view onto north polar axis
            lon_south_flipped = -lon_rad[south_mask] % (2 * np.pi)
            r_south = 90 + lat_deg[south_mask]
            plot_last = ax_north.scatter(lon_south_flipped, r_south, s=2,
                                        c=alt_km[south_mask], cmap='rainbow_r', vmin=40, vmax=2000, zorder=10)
            plot_last = ax_south.scatter(lon_rad[south_mask] , r_south, s=2,
                                        c=alt_km[south_mask], cmap='rainbow_r', vmin=40, vmax=2000, zorder=10)

