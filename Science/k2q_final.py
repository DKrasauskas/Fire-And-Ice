import numpy as np

class Planet:
    def __init__(self):
        pass

class Jupiter(Planet):
    def __init__(self):
        self.M = 1.898125e27
        self.R = 71488000

class Io(Planet):
    def __init__(self):
        self.M = 8.931938e22
        self.R = 71488000
        self.a = 420000000 
    
class Europa(Planet):
    def __init__(self):
        self.M = 4.79984e22
        self.R = 71488000
        self.a = 664862000

G = 6.67e-11
k2q_Europa = 1e-5
k2q_Io = 1e-5 + 1e-6

planet = Jupiter()
moon = Europa()

def get_required_days(planet, moon, sigma_x, delta_k2q):
    drift_bare = (9/2) * G * moon.M / moon.a**3 * (planet.R / moon.a)**5
    sigma_theta = sigma_x / moon.a
    t_seconds = np.sqrt(2 * sigma_theta / (drift_bare * delta_k2q))

    dn_dt = drift_bare * k2q_Io                             # rad/s²
    dn_dt_per_day2 = dn_dt * (86400**2)                         # rad/day²
    delta_theta = 0.5 * drift_bare * delta_k2q * t_seconds**2  # rad
    delta_x = delta_theta * moon.a                              # m

    print(f"dn/dt without k2/Q:  {drift_bare:.4e} rad/s²")
    print(f"dn/dt with k2/Q={k2q_Europa:.0e}):   {dn_dt:.4e} rad/s²  =  {dn_dt_per_day2:.4e} rad/day²")
    print(f"Position precision (sigma_x):      {sigma_x} m")
    print(f"Equivalent TA precision (sigma_θ): {sigma_theta:.4e} rad")
    print(f"Target k2/Q uncertainty:           {delta_k2q:.0e}")
    print(f"TA shift from delta_k2q over t:    {delta_theta:.4e} rad  =  {delta_x:.4e} m along-track")

    return t_seconds / 86400

days = get_required_days(planet, moon, sigma_x = 0.2, delta_k2q=1.3e-4)
print(f"Required observation time:         {days:.1f} days  ({days/365:.2f} years)")