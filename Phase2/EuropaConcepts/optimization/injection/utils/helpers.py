# General imports
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Tudat imports
import tudatpy
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.dynamics import environment_setup
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime
from tudatpy.kernel.interface import spice

# Pygmo imports
import pygmo as pg

from utils.constants import *

# perigee_radius = R_GA * .9               
# apogee_radius = R_GA * 10                 
# semi_major_axis = (apogee_radius + perigee_radius) / 2.0
# eccentricity = (apogee_radius - perigee_radius) / (apogee_radius + perigee_radius)

v_inf     = 6000   # hyperbolic excess speed (m/s)
r_periapsis =  R_GA * .9 # periapsis radius (your closest approach, e.g. R_GA * 0.9)
jupiter_gravitational_parameter = 1.266e17

# Semi-major axis (negative for hyperbola)

# Eccentricity (always > 1 for hyperbola)


deltaM = 0.1

def spacecraft_state_function(current_time):
    if current_time != current_time:  # NaN check
        return np.zeros(6)

    # --- Orbital Elements ---
#     inclination = 25.5675 deg
# LAN         = -1.4933 deg
    semi_major_axis = -jupiter_gravitational_parameter / v_inf**2
    eccentricity = 1 - r_periapsis / semi_major_axis
    #mean_anomaly_at_j2000 = np.deg2rad(306.88003)
    nu_max = np.arccos(-1.0 / eccentricity)  # asymptote limit
    #print(f"Valid true anomaly range: +/- {np.degrees(nu_max):.2f} deg")

    cos_nu = (1.0 / eccentricity) * ((semi_major_axis * (1.0 - eccentricity**2)) / R_GA - 1.0)

    # Guard: if |cos_nu| > 1, R_GA is outside the physical trajectory
    if abs(cos_nu) > 1.0:
        raise ValueError(f"R_GA={R_GA:.3e} is outside the valid range of this hyperbola. "
                        f"|cos_nu|={abs(cos_nu):.6f}. "
                        f"R_GA must be >= r_periapsis={r_periapsis:.3e}")

    sin_nu = np.sqrt(1 - cos_nu**2)  # ← positive for outbound, negative for inbound
                                    #   set to negative if you're past apoapsis
    cosh_F = (eccentricity + cos_nu) / (1 + eccentricity * cos_nu)
    sinh_F = (np.sqrt(eccentricity**2 - 1) * sin_nu) / (1 + eccentricity * cos_nu)
    F = np.arcsinh(sinh_F)  # or np.log(cosh_F + np.sqrt(cosh_F**2 - 1))

    # Hyperbolic mean anomaly:
    M = eccentricity * np.sinh(F) - F# - 0.01
    mean_anomaly_at_j2000 = M# + 0.01
    inclination      = np.radians(25.5657)
    arg_of_periapsis = np.radians(45.0)
    lan              = np.radians(-1.4933)

    # --- Mean Anomaly ---
    mean_motion = np.sqrt(jupiter_gravitational_parameter / abs(semi_major_axis)**3)
    M = mean_anomaly_at_j2000 + deltaM# - 0.01# + mean_motion * current_time

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

    else:
        # Hyperbolic: solve M = e*sinh(F) - F
        F = np.log(2.0 * abs(M) / eccentricity + 1.8)
        if M < 0:
            F = -F
        for _ in range(100):
            delta = (eccentricity * np.sinh(F) - F - M) / (eccentricity * np.cosh(F) - 1.0)
            F -= delta
            if abs(delta) < 1e-11:
                break
        cos_E = np.cosh(F)
        sin_E = np.sinh(F)

        # Perifocal position & velocity (a is negative, so -a > 0)
       # True anomaly from F
        cos_nu = (np.cosh(F) - eccentricity) / (1.0 - eccentricity * np.cosh(F))
        sin_nu = (np.sqrt(eccentricity**2 - 1.0) * np.sinh(F)) / (1.0 - eccentricity * np.cosh(F))

        # Position directly from true anomaly
        r            = (-semi_major_axis) * (eccentricity**2 - 1.0) / (1.0 + eccentricity * cos_nu)
        x_perifocal  = r * cos_nu
        y_perifocal  = r * sin_nu

        # Angular momentum — the clean way to get velocity in perifocal frame
        h            = np.sqrt(jupiter_gravitational_parameter * (-semi_major_axis) * (eccentricity**2 - 1.0))
        vx_perifocal = -(jupiter_gravitational_parameter / h) * sin_nu
        vy_perifocal =  (jupiter_gravitational_parameter / h) * (eccentricity + cos_nu)

        # Sanity check
        v_visviva = np.sqrt(jupiter_gravitational_parameter * (2.0/r - 1.0/semi_major_axis))
        v_total   = np.sqrt(vx_perifocal**2 + vy_perifocal**2)
       # print(semi_major_axis)

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

    return np.concatenate([position_3d, velocity_3d])

def spacecraft_state_function3(current_time):
    if current_time != current_time:  # NaN check
        return np.zeros(6)
    semi_major_axis = -jupiter_gravitational_parameter / v_inf**2
    eccentricity = 1 - r_periapsis / semi_major_axis
    #mean_anomaly_at_j2000 = np.deg2rad(306.88003)
    nu_max = np.arccos(-1.0 / eccentricity)  # asymptote limit
    #print(f"Valid true anomaly range: +/- {np.degrees(nu_max):.2f} deg")

    cos_nu = (1.0 / eccentricity) * ((semi_major_axis * (1.0 - eccentricity**2)) / R_GA - 1.0)

    # Guard: if |cos_nu| > 1, R_GA is outside the physical trajectory
    if abs(cos_nu) > 1.0:
        raise ValueError(f"R_GA={R_GA:.3e} is outside the valid range of this hyperbola. "
                        f"|cos_nu|={abs(cos_nu):.6f}. "
                        f"R_GA must be >= r_periapsis={r_periapsis:.3e}")

    sin_nu = np.sqrt(1 - cos_nu**2)  # ← positive for outbound, negative for inbound
                                    #   set to negative if you're past apoapsis
    cosh_F = (eccentricity + cos_nu) / (1 + eccentricity * cos_nu)
    sinh_F = (np.sqrt(eccentricity**2 - 1) * sin_nu) / (1 + eccentricity * cos_nu)
    F = np.arcsinh(sinh_F)  # or np.log(cosh_F + np.sqrt(cosh_F**2 - 1))

    # Hyperbolic mean anomaly:
    M = eccentricity * np.sinh(F) - F# - 0.01
    mean_anomaly_at_j2000 = M# + 0.01ot
    # --- Orbital Elements ---
    inclination      = np.radians(25.305)
    arg_of_periapsis = np.radians(45.0)
    lan              = np.radians(0.0)

    # --- Mean Anomaly ---
    mean_motion = np.sqrt(jupiter_gravitational_parameter / abs(semi_major_axis)**3)
    M = 1# mean_anomaly_at_j2000 - 100# - 1# + mean_motion * current_time

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

    else:
        # Hyperbolic: solve M = e*sinh(F) - F
        F = np.log(2.0 * abs(M) / eccentricity + 1.8)
        if M < 0:
            F = -F
        for _ in range(100):
            delta = (eccentricity * np.sinh(F) - F - M) / (eccentricity * np.cosh(F) - 1.0)
            F -= delta
            if abs(delta) < 1e-11:
                break
        cos_E = np.cosh(F)
        sin_E = np.sinh(F)

        # True anomaly from F
        cos_nu = (np.cosh(F) - eccentricity) / (1.0 - eccentricity * np.cosh(F))
        sin_nu = (np.sqrt(eccentricity**2 - 1.0) * np.sinh(F)) / (1.0 - eccentricity * np.cosh(F))

        # Position directly from true anomaly
        r            = (-semi_major_axis) * (eccentricity**2 - 1.0) / (1.0 + eccentricity * cos_nu)
        x_perifocal  = r * cos_nu
        y_perifocal  = r * sin_nu

        # Angular momentum — the clean way to get velocity in perifocal frame
        h            = np.sqrt(jupiter_gravitational_parameter * (-semi_major_axis) * (eccentricity**2 - 1.0))
        vx_perifocal = -(jupiter_gravitational_parameter / h) * sin_nu
        vy_perifocal =  (jupiter_gravitational_parameter / h) * (eccentricity + cos_nu)

        # Sanity check
        v_visviva = np.sqrt(jupiter_gravitational_parameter * (2.0/r - 1.0/semi_major_axis))
        v_total   = np.sqrt(vx_perifocal**2 + vy_perifocal**2)
       # print(semi_major_axis)

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

    return np.concatenate([position_3d, velocity_3d])

def spacecraft_state_function2(current_time):
    # Ensure current_time isn't NaN
    if current_time != current_time:
        return np.zeros(6)

    # --- Keplerian Size & Shape Elements ---
   

    # --- Keplerian Orientation Elements ---
    inclination = np.radians(25.305)      # i
    arg_of_periapsis = np.radians(45.0)  # omega
    lan = np.radians(0.0)                 # Omega

    # Compute orbital period and mean motion (angular velocity 'n')
    orbital_period = 2.0 * np.pi * np.sqrt(semi_major_axis ** 3 / jupiter_gravitational_parameter)
    mean_motion = 2.0 * np.pi / orbital_period

    # 1. Compute current Mean Anomaly (grows perfectly linearly with time)
    M = mean_anomaly_at_j2000 + np.pi#+ mean_motion * current_time
    # Keep M wrapped cleanly between -pi and pi
    M = (M + np.pi) % (2.0 * np.pi) - np.pi

    # 2. Solve Kepler's Equation (M = E - e*sin(E)) via Newton-Raphson
    E = M if eccentricity < 0.8 else np.pi  # Initial guess
    tolerance = 1e-11
    max_iterations = 100
    
    for _ in range(max_iterations):
        delta_E = (E - eccentricity * np.sin(E) - M) / (1.0 - eccentricity * np.cos(E))
        E -= delta_E
        if abs(delta_E) < tolerance:
            break

    # 3. Calculate 2D state in the local orbital plane (Perifocal Frame) using E
    cos_E = np.cos(E)
    sin_E = np.sin(E)
    
    x_perifocal = semi_major_axis * (cos_E - eccentricity)
    y_perifocal = semi_major_axis * np.sqrt(1.0 - eccentricity**2) * sin_E
    
    # Distance from center of focus (Jupiter) to spacecraft
    r = semi_major_axis * (1.0 - eccentricity * cos_E)
    
    # Perifocal velocities
    v_factor = np.sqrt(jupiter_gravitational_parameter * semi_major_axis) / r
    vx_perifocal = v_factor * (-sin_E)
    vy_perifocal = v_factor * (np.sqrt(1.0 - eccentricity**2) * cos_E)

    # 4. Compute 3D Direction Vectors (Unchanged, your rotation logic is perfect)
    cos_o, sin_o = np.cos(lan), np.sin(lan)
    cos_w, sin_w = np.cos(arg_of_periapsis), np.sin(arg_of_periapsis)
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)

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

    # 5. Transform 2D vectors into 3D Inertial space
    position_3d = x_perifocal * P + y_perifocal * Q
    velocity_3d = vx_perifocal * P + vy_perifocal * Q

    return np.concatenate([position_3d, velocity_3d])


def get_time_for_anomaly():
    semi_major_axis = -jupiter_gravitational_parameter / v_inf**2
    eccentricity = 1 - r_periapsis / semi_major_axis
    return deltaM * np.sqrt(np.abs(semi_major_axis ** 3) / jupiter_gravitational_parameter)