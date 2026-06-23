import numpy as np

class spacecraft():
    def __init__(self, absorptivity, emissivity, radius):
        self.absorptivity = absorptivity
        self.emissivity = emissivity
        self.radius = radius
    
    def _pr_area(self):
        return np.pi * self.radius ** 2
    
    def _su_area(self):
        return 4 * np.pi * self.radius ** 2
    
    def total_abs(self, ir0, pho, flux, albedo):
        if pho:
            sin2 = np.sin(pho) ** 2
        
        if flux: # Solar flux from the sun
            solar = self.absorptivity * flux * self._pr_area()
        else:
            solar = 0
            print('Solar flux is not included')
            
        if pho and flux: # Reflected flux from the planet
            k = 0.664 + 0.521 * pho - 0.203 * pho ** 2
            albedo_pow = flux * albedo * self._pr_area() * self.absorptivity * k * sin2
        else:
            albedo_pow = 0
            print('Albedo effect is not included')
        
        if ir0 and pho: # IR
            ir = ir0 * sin2 * self._pr_area() * self.emissivity
        else:
            ir = 0
            print('IR is not included')

        return solar + albedo_pow + ir

    def total_em(self, t_s, t_e=0):
        return self.emissivity * boltzmann_const * self._su_area() * (t_s ** 4 - t_e ** 4)

    def heater_power(self, t_target, ir0, pho, albedo, flux, t_e=0, ir1=None, pho1=None):
        absorbed = self.total_abs(ir0, pho, flux, albedo)
        if ir1 is not None and pho1 is not None:
            absorbed += ir1 * np.sin(pho1) ** 2 * self._pr_area() * self.emissivity
        return max(0, self.total_em(t_target, t_e) - absorbed)

    def eq_temp(self, ir0, pho, albedo, flux, t_e=0, ir1=None, pho1=None):
        absorbed = self.total_abs(ir0, pho, flux, albedo)
        if ir1 is not None and pho1 is not None:
            absorbed += ir1 * np.sin(pho1) ** 2 * self._pr_area() * self.emissivity
        temp = (absorbed / (self.emissivity * boltzmann_const * self._su_area()) + t_e ** 4) ** 0.25
        return temp
    
    
        
def ang_radius(r_body, altitude): # inputs in m
    return np.arcsin(r_body / (r_body + altitude))

def solar_flux(dist): # input dist in au
    const = 1361.0 # W/m^2 flux at 1 au 
    return const * (1/dist) ** 2

# -- Constants --
boltzmann_const = 5.67e-8 # W/m²K⁴
albedo_ven = 0.75 # bond

r_ven = 6051.8e3
r_jup = 69911e3

min_dist_sun = 0.72 # au
max_dist_sun = 5.2 # au

alt_ven = 350e3
alt_jup = 2e3
alt_eur = 1e5
ir_ven = 160 # W/m^2
ir_jup = 13.6 # W/m^2
# ---


T_target = 275.15  # 2°C in K

both_sc = spacecraft(0.3, 0.8, 1.1)

# Highest temperature
print('---Temperature at the hottest point of the trajectory---')
print(both_sc.eq_temp(ir_ven, ang_radius(r_ven, alt_ven), albedo_ven, solar_flux(min_dist_sun)))


# Io
sc_io = spacecraft(0.3, 0.8, 0.88)

# Lowest temperature
print('---Temperature at the coldest point of the trajectory for Io---')
print(sc_io.eq_temp(ir_jup, ang_radius(r_jup, alt_jup), None, None))

# Europa
temp_eur = 102 # mean temperature in europa [K]
em_eur = 0.94
ir_eur = em_eur * boltzmann_const * temp_eur ** 4
sc_europa = spacecraft(0.3, 0.8, 0.88)

# Lowest temperature
print('---Temperature at the coldest point of the trajectory---')
print(sc_europa.eq_temp(ir_eur, ang_radius(r_jup, alt_eur), None, None))