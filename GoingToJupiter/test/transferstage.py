# Transfer stage
import numpy as np
Isp = 430
g0 = 9.8065


def transfer_mass(massJupiter, dv):
    epower = np.e**(dv/(Isp * g0))
    Mwet = massJupiter * (epower - 1)/(1-0.1*epower)
    return Mwet

dvvalues = 3800 - np.arange(0, 50, 3800)
print(dvvalues)

print(transfer_mass(3715, dvvalues))