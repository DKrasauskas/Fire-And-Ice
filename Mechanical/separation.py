import numpy as np

D = 1
m = 1381
g = 9.81
L = 0.5 - 0.1

in_line = m * 6 * g / (np.pi * D) + 4 * m * 2 * g * L / (np.pi * 1 ** 2)
print(in_line )
print(in_line * np.pi * D)
print(42000 * np.pi * D)



