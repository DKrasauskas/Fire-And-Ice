import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# INPUTS — edit these
# ─────────────────────────────────────────────────────────────────────────────

# --- Material: Al 7075-T6 ---
E       = 71.7e9        # Young's modulus [Pa]
rho_mat = 2810.0        # Density [kg/m³]
sigma_y = 503e6         # Yield strength [Pa]
nu      = 0.33          # Poisson's ratio

# --- Safety factors (ECSS-E-ST-32) ---
SF_yield   = 1.25
SF_ult     = 1.50

# --- Spacecraft mass ---
m_sc = 800.0            # Total spacecraft mass [kg]  ← update when known

# --- Launch loads (Neutron launcher, quasi-static) ---
a_axial   = 6.0 * 9.81  # Axial acceleration [m/s²]   (REQ-MEC-01)
a_lateral = 2.0 * 9.81  # Lateral acceleration [m/s²] (REQ-MEC-02)

# --- Frequency requirements ---
f_req_axial   = 25.0    # Hz  (REQ-MEC-03)
f_req_lateral = 10.0    # Hz  (REQ-MEC-04)
f_req_min     = 35.0    # Hz  (REQ-MEC-05, minimum resonance)

# --- Spacecraft bus geometry ---
bus_side   = 1.2        # Side length of square bus [m]
bus_height = 1.5        # Height of bus [m]

# --- Honeycomb panel parameters ---
# Panel modelled as simply-supported flat plate
# Face sheet thickness t_face on each side, core height h_core
h_core     = 0.020      # Core height [m]   (20 mm typical)
rho_core   = 50.0       # Honeycomb core density [kg/m³]  (Al honeycomb)
t_face_min = 0.0003     # Minimum face sheet thickness [m] (0.3 mm)

# --- Central cylinder (thrust tube) ---
R_cyl = 0.20            # Cylinder radius [m]
L_cyl = bus_height      # Cylinder length [m]

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — HONEYCOMB PANEL SIZING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART 1 — HONEYCOMB SANDWICH PANEL")
print("=" * 65)

# Panel dimensions (largest panel = full bus side × height)
a = bus_side    # panel width  [m]
b = bus_height  # panel height [m]

# Effective bending stiffness D of sandwich panel:
# D = E * t_face * (h_core + t_face)^2 / 2  (thin face sheet approx)
# Fundamental frequency of simply-supported plate:
# f = (pi^2 / (2 * a^2)) * sqrt(D / (rho_s))
# where rho_s = areal mass density [kg/m²]

# Areal density as function of face sheet thickness t
def areal_density(t_face):
    return 2 * rho_mat * t_face + rho_core * h_core  # kg/m²

def bending_stiffness(t_face):
    d = h_core + t_face  # distance between face sheet centroids
    return E * t_face * d**2 / 2  # N·m

def panel_frequency(t_face, a, b):
    D    = bending_stiffness(t_face)
    m_s  = areal_density(t_face)
    # Simply supported: f = (pi²/2) * sqrt(D/m_s) * (1/a² + 1/b²)
    f = (np.pi**2 / 2) * np.sqrt(D / m_s) * (1/a**2 + 1/b**2)
    return f

# Sweep face sheet thickness to find minimum satisfying frequency
t_sweep = np.linspace(t_face_min, 0.005, 10000)
freqs   = np.array([panel_frequency(t, a, b) for t in t_sweep])

# Find minimum t satisfying each frequency requirement
def min_t_for_freq(f_req):
    idx = np.where(freqs >= f_req)[0]
    if len(idx) == 0:
        return None
    return t_sweep[idx[0]]

t_axial   = min_t_for_freq(f_req_axial)
t_lateral = min_t_for_freq(f_req_lateral)
t_min_res = min_t_for_freq(f_req_min)

print(f"\nPanel dimensions: {a*1000:.0f} x {b*1000:.0f} mm")
print(f"Core height:      {h_core*1000:.0f} mm")
print(f"\nFace sheet thickness required:")
print(f"  Axial   freq ≥ {f_req_axial:.0f} Hz  → t_face ≥ {t_axial*1000:.2f} mm" if t_axial else f"  Axial   freq requirement not achievable in sweep range")
print(f"  Lateral freq ≥ {f_req_lateral:.0f} Hz  → t_face ≥ {t_lateral*1000:.2f} mm" if t_lateral else f"  Lateral freq requirement not achievable")
print(f"  Min res  freq ≥ {f_req_min:.0f} Hz  → t_face ≥ {t_min_res*1000:.2f} mm" if t_min_res else f"  Min resonance requirement not achievable")

t_face_req = max(filter(None, [t_axial, t_lateral, t_min_res]))
f_achieved = panel_frequency(t_face_req, a, b)

print(f"\n→ DESIGN face sheet thickness: {t_face_req*1000:.2f} mm (frequency-driven)")
print(f"  Achieved frequency:           {f_achieved:.1f} Hz")
print(f"  Panel areal density:          {areal_density(t_face_req):.2f} kg/m²")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — LAUNCH LOAD STRESS CHECK ON PANELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART 2 — STRESS CHECK UNDER LAUNCH LOADS")
print("=" * 65)

F_axial   = m_sc * a_axial    # Total axial force [N]
F_lateral = m_sc * a_lateral  # Total lateral force [N]

A_panel_total = 4 * bus_side * 2 * t_face_req  # total face sheet area [m²]

sigma_axial   = F_axial   / A_panel_total
sigma_lateral = F_lateral / A_panel_total

print(f"\nTotal axial force:      {F_axial/1e3:.1f} kN")
print(f"Total lateral force:    {F_lateral/1e3:.1f} kN")
print(f"Total face sheet area:  {A_panel_total*1e4:.1f} cm²")
print(f"\nAxial stress:           {sigma_axial/1e6:.2f} MPa")
print(f"Lateral stress:         {sigma_lateral/1e6:.2f} MPa")
print(f"Allowable (yield/SF):   {sigma_y/SF_yield/1e6:.1f} MPa")

if sigma_axial < sigma_y / SF_yield:
    print(f"  ✓ Axial stress OK   (margin: {(sigma_y/SF_yield/sigma_axial - 1)*100:.0f}%)")
else:
    print(f"  ✗ Axial stress EXCEEDS allowable — increase t_face")

if sigma_lateral < sigma_y / SF_yield:
    print(f"  ✓ Lateral stress OK (margin: {(sigma_y/SF_yield/sigma_lateral - 1)*100:.0f}%)")
else:
    print(f"  ✗ Lateral stress EXCEEDS allowable — increase t_face")

print("\n" + "=" * 65)
print("PART 3 — CENTRAL CYLINDER (THRUST TUBE)")
print("=" * 65)

t_cyl_stress = F_axial / (2 * np.pi * R_cyl * sigma_y / SF_yield)

f_cyl_axial = (1 / (2 * L_cyl)) * np.sqrt(E / rho_mat)

# Euler buckling of cylinder under axial load
# t_cyl_buckle from: F_crit = 2*pi*E*t²/sqrt(3*(1-nu²)) ≥ F_axial * SF
t_cyl_buckle = np.sqrt(F_axial * SF_ult * np.sqrt(3*(1-nu**2)) / (2 * np.pi * E))

t_cyl = max(t_cyl_stress, t_cyl_buckle, 0.001)  # minimum 1 mm

print(f"\nCylinder radius: {R_cyl*1000:.0f} mm,  length: {L_cyl*1000:.0f} mm")
print(f"\nWall thickness required:")
print(f"  From stress check:   {t_cyl_stress*1000:.2f} mm")
print(f"  From buckling check: {t_cyl_buckle*1000:.2f} mm")
print(f"\n→ DESIGN wall thickness: {t_cyl*1000:.2f} mm")
print(f"  Cylinder axial freq: {f_cyl_axial:.1f} Hz")

sigma_cyl = F_axial / (2 * np.pi * R_cyl * t_cyl)
print(f"  Cylinder stress:     {sigma_cyl/1e6:.2f} MPa  (allowable: {sigma_y/SF_yield/1e6:.1f} MPa)")

print("\n" + "=" * 65)
print("PART 4 — STRUCTURAL MASS ESTIMATE")
print("=" * 65)

panel_area_total = 4 * (bus_side * bus_height) + 2 * (bus_side * bus_side)
m_panels = panel_area_total * areal_density(t_face_req)

# Central cylinder
m_cyl = rho_mat * 2 * np.pi * R_cyl * t_cyl * L_cyl

vault_side = 0.60
t_vault    = 0.020  # from shielding analysis
vault_area = 6 * vault_side**2
m_vault    = rho_mat * vault_area * t_vault

m_primary = m_panels + m_cyl + m_vault
m_misc    = 0.15 * m_primary
m_total   = m_primary + m_misc

print(f"\nPanel area total:  {panel_area_total:.2f} m²")
print(f"Panel mass:        {m_panels:.1f} kg")
print(f"Cylinder mass:     {m_cyl:.1f} kg")
print(f"Vault mass:        {m_vault:.1f} kg")
print(f"Misc (15%):        {m_misc:.1f} kg")
print(f"\n→ TOTAL structural mass: {m_total:.1f} kg")
print(f"  Structure fraction:    {m_total/m_sc*100:.1f}% of s/c mass")
print(f"  (Typical deep space: 15–25%)")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Panel face sheet thickness : {t_face_req*1000:.2f} mm  (freq-driven)")
print(f"  Panel frequency achieved   : {f_achieved:.1f} Hz")
print(f"  Cylinder wall thickness    : {t_cyl*1000:.2f} mm")
print(f"  Total structural mass      : {m_total:.1f} kg")
print(f"  All stress checks          : {'PASS' if sigma_axial < sigma_y/SF_yield and sigma_cyl < sigma_y/SF_yield else 'FAIL'}")