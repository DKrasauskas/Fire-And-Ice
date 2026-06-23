import numpy as np

mu_sun = 1.32712440042e20

R_ja = 816.363e9
R_jp = 740.595e9
a_j = (R_ja + R_jp ) * 0.5

R_ea = 152.10e9
R_ep = 152.10e9

a_hohman = (R_ja + R_ea) * 0.5

v_jupiter=  np.sqrt( 2 * mu_sun / R_ja - mu_sun / a_j)
v_hohmann =  np.sqrt( 2 * mu_sun / R_ja - mu_sun / a_hohman)

V_infy = v_jupiter - v_hohmann

print(V_infy)

mu_j = 1.26686534e17
R_j =  7.1492e7 
E_orb = V_infy ** 2 /2

r_science_apogee = R_j * 120
r_science_perigee = R_j * 3.5
a_science = (r_science_apogee + r_science_perigee) * 0.5

E_KINETIC = E_orb + mu_j / (r_science_perigee)
E_kinetic_required = -mu_j / (2 * a_science) + mu_j / (r_science_perigee)

v_perigee = np.sqrt(2 * E_KINETIC)
v_perigee_target = np.sqrt(2 * E_kinetic_required)
print(v_perigee)
print(v_perigee_target)
print(v_perigee - v_perigee_target)

#orbit raise:

r_science_target2  = R_j * 3.5
a_new =  (r_science_apogee + r_science_target2) * 0.5

E_kinetic_required = -mu_j / (2 * a_new) + mu_j / (r_science_apogee)
E_kinetic_current  = -mu_j / (2 * a_science) + mu_j / (r_science_apogee)

dv = np.sqrt(E_kinetic_required * 2) - np.sqrt(E_kinetic_current * 2)
print(dv)
deltaV = dv + v_perigee - v_perigee_target
print(deltaV)

#io orbit


r_science_target2  = 	420000000 
a_new =  (r_science_target2 + r_science_target2) * 0.5

E_kinetic_required = -mu_j / (2 * a_new) + mu_j / (r_science_target2)
E_kinetic_current  = -mu_j / (2 * a_science) + mu_j / (r_science_target2)

dv = np.sqrt(E_kinetic_required * 2) 
v_enc =  np.sqrt(E_kinetic_current * 2)
print(dv)
print(v_enc)


R_io = 1821 #km
H_min = R_io + 1000

alpha = np.arcsin(R_io / H_min)
distance = np.cos(alpha) * H_min * 2
print(distance)

print(distance / (30 * 60))
