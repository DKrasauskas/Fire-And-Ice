import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup, propagation, simulator
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion

spice.load_standard_kernels()

inclination_deg = 90
aop_deg         = 90
n               = 29.9975

simulation_years       = 1
simulation_start_epoch = DateTime(2020, 1, 1).to_epoch()
simulation_end_epoch   = simulation_start_epoch + simulation_years * constants.JULIAN_YEAR

bodies_to_create  = ["Europa", "Ganymede", "Callisto", "Io", "Jupiter"]
body_to_propagate = ["SoIaF"]

global_frame_origin      = "Jupiter"
global_frame_orientation = "J2000"

flyby_threshold   = 1.0e8
flyby_half_window = 2.0 * 3600.0
R_Jupiter         = 69911.0e3

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
T_Io         = 2 * np.pi * np.sqrt(r_Io**3 / mu_jup)
a_cap        = (mu_jup * (n * T_Io / (2 * np.pi))**2) ** (1 / 3)
eccentricity = np.sqrt(1 - r_Io / a_cap)
r_periapsis  = a_cap * (1 - eccentricity)
r_apoapsis   = a_cap * (1 + eccentricity)
T_cap   = 2 * np.pi * np.sqrt(a_cap**3 / mu_jup) / 3600  # hours
T_cap_s = T_cap * 3600


def phase_true_anomaly():
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


def print_info():
    print(f"T = {T_cap/24:.2f} days  |  e = {eccentricity:.3f}  |  "
          f"r_peri = {r_periapsis/R_Jupiter:.2f} RJ  |  a = {a_cap/R_Jupiter:.2f} RJ")

print_info()

initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter     = mu_jup,
    semi_major_axis             = a_cap,
    eccentricity                = eccentricity,
    inclination                 = np.deg2rad(inclination_deg),
    argument_of_periapsis       = np.deg2rad(aop_deg),
    longitude_of_ascending_node = np.deg2rad(raan_deg),
    true_anomaly                = phase_true_anomaly()
)

accelerations_settings_spacecraft = {
    "Jupiter": [propagation_setup.acceleration.point_mass_gravity()]
}
acceleration_settings = {"SoIaF": accelerations_settings_spacecraft}
central_bodies        = ["Jupiter"]

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, body_to_propagate, central_bodies
)

integrator_settings  = propagation_setup.integrator.runge_kutta_fixed_step(
    40.0, propagation_setup.integrator.CoefficientSets.rk_4
)
termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies, acceleration_models, body_to_propagate,
    initial_state, simulation_start_epoch, integrator_settings, termination_settings
)

print("\nRunning coarse propagation...")
dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
states = result2array(dynamics_simulator.state_history)
print(f"  {len(states)} steps recorded.")


def find_flybys(states):
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


flyby_indices = find_flybys(states)


def run_detailed_propagation(states, flyby_indices):
    accel = propagation_setup.create_acceleration_models(
        bodies,
        {"SoIaF": {"Jupiter": [propagation_setup.acceleration.point_mass_gravity()]}},
        body_to_propagate, ["Io"]
    )
    integrator = propagation_setup.integrator.runge_kutta_variable_step(
        initial_time_step=10.0,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rkf_78,
        step_size_control_settings=propagation_setup.integrator.step_size_control_custom_blockwise_scalar_tolerance(
            propagation_setup.integrator.standard_cartesian_state_element_blocks, 1e-12, 1e-12
        ),
        step_size_validation_settings=propagation_setup.integrator.step_size_validation(0.01, 500.0)
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


arc_results, flyby_epochs = run_detailed_propagation(states, flyby_indices)


def plot_altitude_profiles(arc_results, flyby_epochs):
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


def plot_ground_tracks(arc_results):
    all_alts = np.concatenate([
        result2array(arc_results[k].dependent_variable_history)[:, 3]
        for k in range(len(arc_results))
    ])
    vmax = np.percentile(all_alts, 95) / 1e3

    _, ax = plt.subplots(figsize=(12, 5))
    sc = None
    for k in range(len(arc_results)):
        dep    = result2array(arc_results[k].dependent_variable_history)
        lat    = np.rad2deg(dep[:, 1])
        lon    = np.rad2deg(dep[:, 2]) % 360.0
        alt_km = dep[:, 3] / 1e3
        mid    = len(lat) // 2
        sc = ax.scatter(lon, lat, s=4, c=alt_km, cmap="viridis", vmin=0, vmax=vmax)
        ax.scatter(lon[mid], lat[mid], marker="*", s=80, color="r",
                   label="CA" if k == 0 else None)
        ax.annotate(f"#{k+1}", (lon[mid], lat[mid]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7)

    if sc is not None:
        plt.colorbar(sc, ax=ax).set_label("Altitude [km]")

    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title("Io flyby ground tracks")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3d_trajectory(states, flyby_indices):
    ax = plt.figure(figsize=(9, 7)).add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(r_Io / R_Jupiter * np.cos(theta), r_Io / R_Jupiter * np.sin(theta),
            np.zeros(300), "k:", lw=0.8, label="Io orbit")
    ax.scatter(0, 0, 0, s=50, color="orange", label="Jupiter")
    ax.plot(states[:, 1] / R_Jupiter, states[:, 2] / R_Jupiter, states[:, 3] / R_Jupiter,
            lw=0.4, label="SoIaF")

    for k, idx in enumerate(flyby_indices):
        x, y, z = states[idx, 1:4] / R_Jupiter
        ax.scatter(x, y, z, s=30, color="r", label="flyby" if k == 0 else None)

    lim = a_cap / R_Jupiter * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x [RJ]"); ax.set_ylabel("y [RJ]"); ax.set_zlabel("z [RJ]")
    ax.set_title("SoIaF trajectory")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


plot_altitude_profiles(arc_results, flyby_epochs)
plot_ground_tracks(arc_results)
plot_3d_trajectory(states, flyby_indices)
