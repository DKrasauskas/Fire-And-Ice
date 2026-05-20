import numpy as np
import matplotlib.pyplot as plt

dose_90 = np.array([
1.55E+05,
7.17E+04,
3.62E+04,
2.52E+04,
2.12E+04,
1.96E+04,
1.88E+04,
1.81E+04,
1.77E+04,
1.31E+04,
8.41E+03,
2.63E+03,
])

dose_0 = np.array([
    4.61E+04,
    2.34E+04,
    1.41E+04,
    1.12E+04,
    1.03E+04,
    9.78E+03,
    9.14E+03,
    8.21E+03,
    7.22E+03,
    3.21E+03,
    1.28E+03,
    4.27E+02,
])

shieldin = np.array([
    0.5992,
    1.498,
    2.996,
    4.494,
    5.992,
    7.49,
    8.988,
    10.486,
    11.984,
    17.976,
    23.968,
    29.9601,
])

fig, ax = plt.subplots()
N_orb = 30
# Data lines
ax.plot(shieldin, np.log(dose_90 * 1e-3 * N_orb), label='Dose 90°', color='steelblue')
ax.plot(shieldin, np.log(dose_0  * 1e-3 * N_orb), label='Dose 0°',  color='tomato')

# Reference lines
ax.axhline(y=np.log(25.0),  color='black', linestyle='--', linewidth=1, label='JUNO vault TID')
ax.axvline(x=17.5, color='gray',  linestyle=':',  linewidth=1, label='JUNO radiatino vault shielding')

# Axis labels and title
ax.set_xlabel('Shielding thickness (cm)')
ax.set_ylabel('Dose (mSv)')
ax.set_title('Dose vs. Shielding Thickness')

# Legend and grid
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)

Ti_rho = 4502 #kg / m ^3
SA_vault = 1 #m^2
Titanium_Volume = shieldin * 1e-3 * SA_vault * 6
shielding_mass = Ti_rho * Titanium_Volume
print(shielding_mass)

plt.tight_layout()
plt.show()