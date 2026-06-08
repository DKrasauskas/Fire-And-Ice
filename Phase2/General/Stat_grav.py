import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

R_IO    = 1821.6e3     # m
K_IO    = 1e-5         # Kaula constant for Io

R_EUROPA = 1560.8e3    # m
K_EUROPA = 2e-5

# Reference calibration point (Fayolle / Verma)
N_REF   = 46
L_REF   = 8
H_REF   = 25e3  
v_REF = 4e3      
SIGMA_REF = 0.07e-3


L = 8
H = 40e3
v = 30e3
SIGMA = 0.003e-3

def kaula_signal(l, K):
    return K / l**2

def calculate_necessary_flybys_cross_body():
    atten_ref = (R_EUROPA / (R_EUROPA + H_REF)) ** L_REF

    C = (kaula_signal(L_REF, K_EUROPA) * atten_ref * np.sqrt(N_REF)) / (SIGMA_REF * v_REF)

    atten = (R_IO / (R_IO + H)) ** L

    A = kaula_signal(L, K_IO)

    N_exact = ((C * SIGMA * v) / (atten * A)) ** 2

    return int(np.ceil(N_exact)), N_exact

print(calculate_necessary_flybys_cross_body())


