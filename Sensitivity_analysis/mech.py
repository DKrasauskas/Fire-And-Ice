import numpy as np

# Material 
E          = 71.7e9
rho_mat    = 2810.0
sigma_y    = 503e6
nu         = 0.33
SF_yield   = 1.25
SF_ult     = 1.50
margin_factor  = 1.3
gamma_buckling = 0.5

# Loads
a_axial_max   = 6.0 * 9.81
a_lateral_max = 2.0 * 9.81

# Frequency requirements
f_req_axial   = 35.0
f_req_lateral = 10.0
f_req_min_res = 25.0

# Geometry (fixed)
R_cyl_top  = 0.295 / 2;  L_cyl_top = 1.0;  deck_top_side = 1.7
R_cyl_bot  = 0.264 / 2;  L_cyl_bot = 0.6;  deck_bot_side = 1.7
R_cyl_mid  = 1.0   / 2;  L_cyl_mid = 0.4

t_shell    = 0.002

#  Sandwich panel
h_core       = 0.020
rho_core     = 50.0
t_face_min   = 0.0003
t_sweep_max  = 0.005
n_sweep      = 10000
t_min_gauge  = 0.003
t_round      = 0.0005


# Panel helpers 
def areal_density(t_face):
    return 2 * rho_mat * t_face + rho_core * h_core

def bending_stiffness(t_face):
    d = h_core + t_face
    return E * t_face * d**2 / 2

def panel_frequency(t_face, a, b):
    D  = bending_stiffness(t_face)
    ms = areal_density(t_face)
    return (np.pi**2 / 2) * np.sqrt(D / ms) * (1/a**2 + 1/b**2)

def size_deck(a, b):
    t_sweep = np.linspace(t_face_min, t_sweep_max, n_sweep)
    freqs   = np.array([panel_frequency(t, a, b) for t in t_sweep])
    def min_t(f_req):
        idx = np.where(freqs >= f_req)[0]
        return t_sweep[idx[0]] if len(idx) else None
    candidates = [t for t in (min_t(f_req_axial), min_t(f_req_lateral), min_t(f_req_min_res)) if t is not None]
    if not candidates:
        raise ValueError("no thickness satisfies freq reqs")
    t_face = max(candidates)
    return t_face, panel_frequency(t_face, a, b), areal_density(t_face)

def round_up(t, step=t_round):
    return np.ceil(t / step) * step

def size_cylinder_combined(F_axial, M_bend, R, L):
    sigma_allow = sigma_y / (SF_yield * margin_factor)
    Nx_axial    = F_axial / (2 * np.pi * R)
    Nx_bend     = M_bend  / (np.pi * R**2)
    Nx_total    = Nx_axial + Nx_bend
    t_stress    = Nx_total / sigma_allow
    N_required  = Nx_total * SF_ult * margin_factor
    C           = np.sqrt(3 * (1 - nu**2))
    t_buckle    = np.sqrt((N_required * R * C) / (gamma_buckling * E))
    t_cyl       = round_up(max(t_stress, t_buckle, t_min_gauge))
    sigma_peak  = Nx_total / t_cyl
    return t_cyl, sigma_peak

def cylinder_mass(R, t, L):
    return rho_mat * 2 * np.pi * R * t * L

def deck_mass(side, t_face):
    return side**2 * areal_density(t_face)

def shell_mass(side, height, t):
    area = 4*(side*height) + 2*(side*side)
    return rho_mat * area * t


# Main sizing function
def size_structure(m_top_payload, m_bottom_payload, n_iter=5, verbose=False):
    # Deck sizing (geometry-only, independent of payload mass)
    t_face_top, f_top, _ = size_deck(deck_top_side, deck_top_side)
    t_face_bot, f_bot, _ = size_deck(deck_bot_side, deck_bot_side)

    m_deck_top  = deck_mass(deck_top_side, t_face_top)
    m_deck_bot  = deck_mass(deck_bot_side, t_face_bot)
    m_shell_top = shell_mass(deck_top_side, L_cyl_top, t_shell)
    m_shell_bot = shell_mass(deck_bot_side, L_cyl_bot + L_cyl_mid, t_shell)

    m_struct_top = m_struct_mid = m_struct_bot = 0.0

    for i in range(n_iter):
        # Top cylinder
        m_above_top = m_top_payload + m_deck_top + m_shell_top + m_struct_top
        t_cyl_top, _ = size_cylinder_combined(
            m_above_top * a_axial_max,
            m_above_top * a_lateral_max * L_cyl_top,
            R_cyl_top, L_cyl_top)
        m_cyl_top = cylinder_mass(R_cyl_top, t_cyl_top, L_cyl_top)

        # Mid (connecting) cylinder
        m_above_mid = m_above_top + m_cyl_top + m_struct_mid
        t_cyl_mid, _ = size_cylinder_combined(
            m_above_mid * a_axial_max,
            m_above_mid * a_lateral_max * (L_cyl_top + L_cyl_mid),
            R_cyl_mid, L_cyl_mid)
        m_cyl_mid = cylinder_mass(R_cyl_mid, t_cyl_mid, L_cyl_mid)

        # Bottom cylinder
        m_above_bot = (m_above_mid + m_cyl_mid
                       + m_bottom_payload + m_deck_bot + m_shell_bot + m_struct_bot)
        t_cyl_bot, sigma_bot = size_cylinder_combined(
            m_above_bot * a_axial_max,
            m_above_bot * a_lateral_max * (L_cyl_top + L_cyl_mid + L_cyl_bot),
            R_cyl_bot, L_cyl_bot)
        m_cyl_bot = cylinder_mass(R_cyl_bot, t_cyl_bot, L_cyl_bot)

        m_struct_top = m_cyl_top
        m_struct_mid = m_cyl_mid
        m_struct_bot = m_cyl_bot

        if verbose:
            print(f"  iter {i+1}: top={m_cyl_top:.2f} mid={m_cyl_mid:.2f} bot={m_cyl_bot:.2f} kg")

    # Mass rollup
    m_primary_upper = m_deck_top + m_cyl_top
    m_primary_inter = m_cyl_mid
    m_primary_lower = m_deck_bot + m_cyl_bot

    misc_frac = 0.20
    m_misc_upper = misc_frac * m_primary_upper
    m_misc_inter = misc_frac * m_primary_inter
    m_misc_lower = misc_frac * m_primary_lower

    m_total_upper = m_primary_upper + m_misc_upper + m_shell_top
    m_total_inter = m_primary_inter + m_misc_inter
    m_total_lower = m_primary_lower + m_misc_lower + m_shell_bot

    # inter counted twice (shared between both SC)
    m_total_struct = m_total_upper + 2*m_total_inter + m_total_lower

    m_struct_europa = m_total_upper + m_total_inter
    m_struct_io     = m_total_lower + m_total_inter

    return {
        # Per-SC structural mass
        "m_struct_europa": m_struct_europa,
        "m_struct_io":     m_struct_io,
        "m_total_struct":  m_total_struct,
        # Components
        "m_deck_top":   m_deck_top,
        "m_deck_bot":   m_deck_bot,
        "m_cyl_top":    m_cyl_top,
        "m_cyl_mid":    m_cyl_mid,
        "m_cyl_bot":    m_cyl_bot,
        "m_shell_top":  m_shell_top,
        "m_shell_bot":  m_shell_bot,
        # Thicknesses [mm]
        "t_cyl_top_mm":   t_cyl_top  * 1e3,
        "t_cyl_mid_mm":   t_cyl_mid  * 1e3,
        "t_cyl_bot_mm":   t_cyl_bot  * 1e3,
        "t_face_top_mm":  t_face_top * 1e3,
        "t_face_bot_mm":  t_face_bot * 1e3,
    }


# ── Standalone run ────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    g0     = 9.80665
    margin = 1.2

    dv_europa  = 1859.0;  dry_europa = 702.42;  isp_europa = 333.0
    dv_io      = 1358.0;  dry_io     = 746.64;  isp_io     = 333.0

    def prop_mass(dry, dv, isp):
        mr   = np.exp(dv / (isp * g0)) - 1
        prop = dry * mr * margin
        return prop, dry + prop

    prop_eu, wet_eu = prop_mass(dry_europa, dv_europa, isp_europa)
    prop_io, wet_io = prop_mass(dry_io,     dv_io,     isp_io)

    print(f"wet Europa = {wet_eu:.2f} kg,  wet Io = {wet_io:.2f} kg")

    res = size_structure(wet_eu, wet_io, verbose=True)

    print("\n--- structural mass summary ---")
    print(f"Europa structure: {res['m_struct_europa']:.2f} kg")
    print(f"Io structure:     {res['m_struct_io']:.2f} kg")
    print(f"Total structure:  {res['m_total_struct']:.2f} kg")
    print(f"\nCylinder thicknesses:")
    print(f"  Europa cyl : {res['t_cyl_top_mm']:.2f} mm")
    print(f"  Mid cyl    : {res['t_cyl_mid_mm']:.2f} mm")
    print(f"  Io cyl     : {res['t_cyl_bot_mm']:.2f} mm")
    print(f"  Europa deck: {res['t_face_top_mm']:.2f} mm face sheet")
    print(f"  Io deck    : {res['t_face_bot_mm']:.2f} face sheet")
