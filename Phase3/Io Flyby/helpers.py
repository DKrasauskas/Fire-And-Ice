import numpy as np
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy import dynamics
from tudatpy.dynamics import environment_setup, propagation_setup, propagation, simulator
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.astro import element_conversion

def true_to_mean_anomaly(nu, e):
    """
    Convert true anomaly to mean anomaly.
    nu : true anomaly [radians]
    e  : eccentricity [0, 1)
    Returns M in [0, 2π)
    """
    # Step 1: True → Eccentric anomaly (preserving quadrant)
    E = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu / 2),
        np.sqrt(1 + e) * np.cos(nu / 2)
    )

    # Step 2: Eccentric → Mean anomaly (Kepler's equation)
    M = E - e * np.sin(E)

    # Step 3: Normalize to [0, 2π)
    M = M % (2 * np.pi)

    return M


R =  422700000
jupiter_gravitational_parameter = 1.266e17
eccentricity = 0.4
semi_major_axis = R / (1 - eccentricity ** 2)
TRUE_ANOMALY = np.radians(90)
mean_anomaly_at_j2000 = true_to_mean_anomaly(TRUE_ANOMALY, eccentricity)
deltaM = 0
inclination      = np.deg2rad(90 + 25.6773)
arg_of_periapsis = np.radians(90)
lan              = np.radians(2.0157)


semi_major_axis = 18859035560.264233 - 100000000
eccentricity = 0.98804852543645449
inclination = np.deg2rad(89.900551367034282)
arg_of_periapsis = np.deg2rad(86.444023939868416)
lan = np.deg2rad(106.37893435189432) - np.deg2rad(0.6)
Mean =6.2809071433766599


def spacecraft_state_function(current_time):
    if current_time != current_time:  # NaN check
        return np.zeros(6)
    #global M, semi_major_axis
    # --- Mean Anomaly ---
   # mean_motion = np.sqrt(jupiter_gravitational_parameter / abs(semi_major_axis)**3)
    #M = mean_anomaly_at_j2000 + deltaM# - 0.01# + mean_motion * current_time
    M = Mean
    if eccentricity < 1.0:
        M = (M + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi] for elliptic only

    # --- Kepler Solver ---
    if eccentricity < 1.0:
        # Elliptic: solve M = E - e*sin(E)
        E = M if eccentricity < 0.8 else np.pi
        for _ in range(100):
            delta = (E - eccentricity * np.sin(E) - M) / (1.0 - eccentricity * np.cos(E))
            E -= delta
            if abs(delta) < 1e-11:
                break
        cos_E = np.cos(E)
        sin_E = np.sin(E)

        # Perifocal position & velocity
        r = semi_major_axis * (1.0 - eccentricity * cos_E)
        x_perifocal  =  semi_major_axis * (cos_E - eccentricity)
        y_perifocal  =  semi_major_axis * np.sqrt(1.0 - eccentricity**2) * sin_E
        v_factor     =  np.sqrt(jupiter_gravitational_parameter * semi_major_axis) / r
        vx_perifocal =  v_factor * (-sin_E)
        vy_perifocal =  v_factor * np.sqrt(1.0 - eccentricity**2) * cos_E

    # --- Perifocal -> Inertial Rotation ---
    cos_o, sin_o = np.cos(lan),              np.sin(lan)
    cos_w, sin_w = np.cos(arg_of_periapsis), np.sin(arg_of_periapsis)
    cos_i, sin_i = np.cos(inclination),      np.sin(inclination)

    P = np.array([
         cos_o * cos_w - sin_o * sin_w * cos_i,
         sin_o * cos_w + cos_o * sin_w * cos_i,
         sin_w * sin_i
    ])
    Q = np.array([
        -cos_o * sin_w - sin_o * cos_w * cos_i,
        -sin_o * sin_w + cos_o * cos_w * cos_i,
         cos_w * sin_i
    ])

    position_3d = x_perifocal * P + y_perifocal * Q
    velocity_3d = vx_perifocal * P + vy_perifocal * Q
    epsilon = np.deg2rad(25.4392911)  

    Rx_eq = np.array([
        [1,              0,               0],
        [0,  np.cos(epsilon), -np.sin(epsilon)],
        [0,  np.sin(epsilon),  np.cos(epsilon)]
    ])

    position_3d = Rx_eq @ (x_perifocal * P + y_perifocal * Q)
    velocity_3d = Rx_eq @ (vx_perifocal * P + vy_perifocal * Q)

    return np.concatenate([position_3d, velocity_3d])

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
    # Manually create and assign environment model settings to new body settings
    body_settings.get( "target_orbit" ).ephemeris_settings =  environment_setup.ephemeris.custom_ephemeris( 
        spacecraft_state_function, 'Jupiter', FRAME )
    oumuamua_gravitational_parameter = 0.08  # Example value, adjust as needed

    
    body_settings.get("target_orbit").gravity_field_settings = (
        environment_setup.gravity_field.central(oumuamua_gravitational_parameter)
    )

    body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.synchronous(
    "Jupiter", global_frame_orientation, "IAU_" + "Io")


    oumuamua_gravitational_parameter = 0.008  # Example value, adjust as needed

    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies_to_create.append("target_orbit")

    return bodies, bodies_to_create, body_settings

def initialize_simulation(bodies_to_create, bodies, body_settings, simulation_start_epoch, simulation_end_epoch, ephemeris = None):


    acceleration_dict = {}
    for body_i in bodies_to_create:
        current_accelerations = {}
        for body_j in bodies_to_create:
            if body_i != body_j:
                if body_j == "Io":
                    continue
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
        dt_to_perigee = 10000#remaining_M / n
        perigee_epoch = simulation_start_epoch + dt_to_perigee
        print(f"Perigee in {dt_to_perigee:.1f} s (epoch {perigee_epoch:.1f} s)")
#
    dt_to_perigee = 600000
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
    delta_v_magnitude = 9.5#-.900#-810.02  # [m/s] — adjust as needed
    #delta_v_vector = (delta_v_magnitude * prograde_hat).reshape(3, 1)
    
    delta_v_vector = delta_v_magnitude* np.array([1, 0, 0])
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

    bodies_acc = ["Jupiter"]
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

