import math
import numpy as np

def size_adcs(wet_mass):

    a, b, c     = 1, 1.7, 1.7
    rad         = 3.1
    offset_wall = 3.5
    wall_offset = 0.9
    m_side      = 122
    d           = offset_wall + wall_offset
    par_axis    = m_side * d**2

    m_body = wet_mass - m_side * 2

    Ixx_sa = 2 * (par_axis + 0.5  * m_side * rad**2)
    Iyy_sa = 2 * (par_axis + 0.25 * m_side * rad**2)
    Izz_sa = 2 * (par_axis + 0.25 * m_side * rad**2)

    Ixx = Ixx_sa + 1/12 * m_body * (b**2 + c**2)
    Iyy = Iyy_sa + 1/12 * m_body * (a**2 + c**2)
    Izz = Izz_sa + 1/12 * m_body * (a**2 + b**2)

    r_eu_sc    = 1_660_800
    mu_Eu      = 3202.721e9
    R_Eu       = 6.71e8
    M_J        = 1.55e20
    slew_deg   = 30.0
    slew_time  = 700.0
    mom_window = 86400
    dump_time  = 10.0
    mom_arm    = 1.5
    isp        = 220.0
    pulse_n    = 264.0055
    dry_rcs    = 6.0
    g0         = 9.80665

    T_gg = 3 * mu_Eu * abs(Ixx - Izz) * np.sin(2 * np.pi / 180) / (2 * r_eu_sc**3)
    T_m  = 1.0 * (2 * M_J / R_Eu**3)
    T_D  = max(T_gg, T_m)

    theta  = math.radians(slew_deg)
    T_slew = 4 * Ixx * theta / slew_time**2
    H_req  = T_D * mom_window
    T_rw   = T_slew * 1.5
    H_rw   = H_req  * 1.2
    m_rw_4 = 4 * 0.4 * H_rw**0.6  # frozen

    F_rcs = H_rw / (dump_time * mom_arm) / 2
    I_rcs = F_rcs * dump_time * pulse_n * 2
    mp    = I_rcs / (isp * g0)
    m_rcs = mp * 3 + dry_rcs
    m_act = m_rw_4 + m_rcs

    return {"m_rcs": m_rcs, "m_act": m_act}


if __name__ == "__main__":
    r = size_adcs(1148.2)
    print(r)