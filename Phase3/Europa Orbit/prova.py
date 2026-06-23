import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.dynamics.propagation_setup import dependent_variable
from tudatpy.dynamics.propagation import create_dependent_variable_dictionary


spice.load_standard_kernels()

# ============================================================
# Body setup
# ============================================================

bodies_to_create = ["Jupiter", "Europa"]

global_frame_origin = "Europa"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation 
)

body_settings.add_empty_settings("SoIaF")

body_settings.get("SoIaF").aerodynamic_coefficient_settings = (
    environment_setup.aerodynamic_coefficients.constant(
        0.0,
        [0.0, 0.0, 0.0]
    )
)

body_settings.get("SoIaF").radiation_pressure_target_settings = (
    environment_setup.radiation_pressure.cannonball_radiation_target(
        0.0,
        0.0,
        {"Jupiter": ["Europa"]}
    )
)

bodies = environment_setup.create_system_of_bodies(body_settings)
bodies.get("SoIaF").mass = 1633.28


# ============================================================
# Propagation setup
# ============================================================

bodies_to_propagate = ["SoIaF"]
central_bodies = ["Europa"]

acceleration_settings = {
    "SoIaF": {
        "Jupiter": [
            propagation_setup.acceleration.point_mass_gravity()
        ],
        "Europa": [
            propagation_setup.acceleration.spherical_harmonic_gravity(5, 5)
        ]
    }
}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies
)


# ============================================================
# Simulation time
# ============================================================

simulation_start_epoch = DateTime(2026, 6, 1).to_epoch()
simulation_end_epoch = DateTime(2026, 7, 1).to_epoch()   # full 720 h


# ============================================================
# Initial 100 km polar circular orbit
# ============================================================

europa_mu = bodies.get("Europa").gravitational_parameter
europa_radius = bodies.get("Europa").shape_model.average_radius

altitude = 100.0e3
semi_major_axis = europa_radius + altitude
circular_velocity = np.sqrt(europa_mu / semi_major_axis)

position_europa_equator_frame = np.array([
    semi_major_axis,
    0.0,
    0.0
])

velocity_europa_equator_frame = np.array([
    0.0,
    0.0,
    circular_velocity
])

rotation_matrix_europa_to_j2000 = spice.compute_rotation_matrix_between_frames(
    "IAU_EUROPA",
    "J2000",
    simulation_start_epoch
)

position_j2000 = rotation_matrix_europa_to_j2000 @ position_europa_equator_frame
velocity_j2000 = rotation_matrix_europa_to_j2000 @ velocity_europa_equator_frame

initial_state = np.concatenate((position_j2000, velocity_j2000))


# ============================================================
# Dependent variables
# ============================================================

dependent_variables_to_save = [
    dependent_variable.total_acceleration("SoIaF"),
    dependent_variable.keplerian_state("SoIaF", "Europa"),
    dependent_variable.latitude("SoIaF", "Europa"),
    dependent_variable.longitude("SoIaF", "Europa"),
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type,
        "SoIaF",
        "Jupiter"
    ),
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type,
        "SoIaF",
        "Europa"
    )
]


# ============================================================
# Integrator and propagator
# ============================================================

termination_condition = propagation_setup.propagator.time_termination(
    simulation_end_epoch
)

fixed_step_size = 20.0

integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size,
    coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
)

propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)


# ============================================================
# Run propagation
# ============================================================

dynamics_simulator = simulator.create_dynamics_simulator(
    bodies,
    propagator_settings
)


# ============================================================
# Extract results
# ============================================================

states_array = result2array(
    dynamics_simulator.propagation_results.state_history
)

dep_var_dict = create_dependent_variable_dictionary(dynamics_simulator)

time_history = dep_var_dict.time_history
relative_time_hours = (time_history - time_history[0]) / 3600.0

latitude = dep_var_dict.asarray(
    dependent_variable.latitude("SoIaF", "Europa")
)

longitude = dep_var_dict.asarray(
    dependent_variable.longitude("SoIaF", "Europa")
)

latitude_deg = np.rad2deg(latitude)
longitude_deg = np.rad2deg(longitude)
longitude_deg = (longitude_deg + 180.0) % 360.0 - 180.0


# ============================================================
# Ground-track plotting function
# ============================================================

def plot_ground_track(hours):
    mask = relative_time_hours <= hours

    plt.figure(figsize=(9, 5))
    plt.title(f"{hours:.0f} h ground track of SoIaF around Europa")
    plt.scatter(
        longitude_deg[mask],
        latitude_deg[mask],
        s=1
    )
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 91, 45))
    plt.grid()
    plt.tight_layout()


plot_ground_track(24)
plot_ground_track(720)


# ============================================================
# Orbit properties
# ============================================================

T = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / europa_mu)
mean_motion = np.sqrt(europa_mu / semi_major_axis**3)

europa_rotation_period = 3.551 * 24.0 * 3600.0

ground_track_shift = 360.0 * T / europa_rotation_period
orbits_per_earth_day = 24.0 / (T / 3600.0)
orbits_per_europa_day = europa_rotation_period / T

science_duration_days = 170.0
science_duration_seconds = science_duration_days * 24.0 * 3600.0

total_orbits_170_days = science_duration_seconds / T
repeat_cycles_170_days = science_duration_seconds / europa_rotation_period

a_jupiter = dep_var_dict.asarray(
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type,
        "SoIaF",
        "Jupiter"
    )
)

a_europa = dep_var_dict.asarray(
    dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type,
        "SoIaF",
        "Europa"
    )
)

kep = dep_var_dict.asarray(
    dependent_variable.keplerian_state("SoIaF", "Europa")
)

sma = kep[:, 0]
ecc = kep[:, 1]
inc_j2000 = np.rad2deg(kep[:, 2])

rp = sma * (1.0 - ecc)
ra = sma * (1.0 + ecc)

print("Europa science orbit properties")
print("--------------------------------")
print(f"Altitude: {altitude/1000:.1f} km")
print(f"Europa mean radius: {europa_radius/1000:.1f} km")
print(f"Orbital radius / semi-major axis: {semi_major_axis/1000:.1f} km")
print(f"Orbital velocity: {circular_velocity/1000:.3f} km/s")
print(f"Orbital period: {T/3600:.3f} h")
print(f"Mean motion: {np.rad2deg(mean_motion)*3600:.3f} deg/h")
print(f"Orbits per Earth day: {orbits_per_earth_day:.2f}")
print()
print("Ground-track properties")
print("-----------------------")
print(f"Europa rotation period: {europa_rotation_period/(24*3600):.3f} Earth days")
print(f"Ground-track longitude shift per orbit: {ground_track_shift:.3f} deg")
print(f"Orbits per Europa rotation: {orbits_per_europa_day:.2f}")
print(f"Approximate ground-track repeat cycle: {europa_rotation_period/(24*3600):.3f} Earth days")
print()
print("Science-phase coverage over 170 days")
print("------------------------------------")
print(f"Total science duration: {science_duration_days:.1f} days")
print(f"Total number of orbits: {total_orbits_170_days:.0f}")
print(f"Number of Europa rotations / repeat cycles: {repeat_cycles_170_days:.1f}")
print(f"Approximate repeated passes over the same longitude pattern: {repeat_cycles_170_days:.1f}")
print()
print("Latitude coverage in 30-day propagation")
print("---------------------------------------")
print(f"Minimum latitude reached: {np.min(latitude_deg):.2f} deg")
print(f"Maximum latitude reached: {np.max(latitude_deg):.2f} deg")
print()
print("Orbit stability indicators over 30 days")
print("---------------------------------------")
print(f"Initial eccentricity: {ecc[0]:.6f}")
print(f"Final eccentricity: {ecc[-1]:.6f}")
print(f"Maximum eccentricity: {np.max(ecc):.6f}")
print(f"Initial periapsis altitude: {(rp[0]-europa_radius)/1000:.2f} km")
print(f"Final periapsis altitude: {(rp[-1]-europa_radius)/1000:.2f} km")
print(f"Minimum periapsis altitude: {(np.min(rp)-europa_radius)/1000:.2f} km")
print(f"Maximum apoapsis altitude: {(np.max(ra)-europa_radius)/1000:.2f} km")
print()
print("Inclination in J2000 frame")
print("--------------------------")
print(f"Mean J2000 inclination: {np.mean(inc_j2000):.3f} deg")
print(f"J2000 inclination variation: {np.max(inc_j2000)-np.min(inc_j2000):.4f} deg")
print()
print("Acceleration environment")
print("------------------------")
print(f"Mean Europa gravity acceleration: {np.mean(a_europa):.6f} m/s^2")
print(f"Mean Jupiter perturbing acceleration: {np.mean(a_jupiter):.6f} m/s^2")
print(f"Mean Jupiter/Europa acceleration ratio: {100*np.mean(a_jupiter)/np.mean(a_europa):.3f} %")

plt.show()