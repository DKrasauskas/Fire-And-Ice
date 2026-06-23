import math

def size_eps(P_nominal_Io, P_nominal_Eu, P_safe_Io, P_safe_Eu):
    t_eclipse_Io   = 4
    t_eclipse_Eu   = 2.9
    DoD            = 0.8
    energy_density = 300
    vol_density    = 780
    S_Jupiter      = 47.13
    dt_eclipse_Eu  = 82.324
    dt_eclipse_Io_PSM = 12 * 24
    eta_cell       = 0.30
    F_eol          = 0.60
    sa_specific_mass = 4.007
    cos10          = math.cos(math.pi / 18)

    E_batt_Io = P_safe_Io * t_eclipse_Io * 1.25 / DoD
    E_batt_Eu = P_safe_Eu * t_eclipse_Eu * 1.25 / DoD
    m_batt_Io = E_batt_Io / energy_density
    m_batt_Eu = E_batt_Eu / energy_density

    P_excess_Io = max(E_batt_Io / 100, E_batt_Io * DoD / dt_eclipse_Io_PSM)
    P_excess_Eu = max(E_batt_Eu / 100, E_batt_Eu * DoD / dt_eclipse_Eu)

    A_Io = (P_nominal_Io + P_excess_Io) / cos10 / (S_Jupiter * eta_cell * F_eol)
    A_Eu = (P_nominal_Eu + P_excess_Eu) / cos10 / (S_Jupiter * eta_cell * F_eol)

    m_sa_Io = A_Io * sa_specific_mass
    m_sa_Eu = A_Eu * sa_specific_mass

    return {
        "m_batt_Io": m_batt_Io, "m_batt_Eu": m_batt_Eu,
        "m_sa_Io":   m_sa_Io,   "m_sa_Eu":   m_sa_Eu,
        "A_Io": A_Io, "A_Eu": A_Eu,
        "E_batt_Io": E_batt_Io, "E_batt_Eu": E_batt_Eu,
    }


if __name__ == "__main__":
    r = size_eps(484.05, 492.35, 351.40, 366.40)
    print(r)