import numpy as np

mu_io = 6.67e-11 * 8.931938e22

R_io = 1821.6e3
eccentricity = 0.1
a = R_io + 300e3

J2 = 1.83e-3
prec = - 3 / 2  * J2 * (R_io / (a * (1 - eccentricity ** 2))) ** 2 * np.sqrt(mu_io / a ** 3) * 60 * 60 * 24 * 1.79 * 360 / (2 * np.pi)
print(prec)