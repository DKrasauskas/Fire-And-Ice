import numpy as np
import matplotlib.pyplot as plt

BETA    = 0.37
THETA   = 85.2
T_MIN   = 5     # years
T_MAX   = 9    # years

def reliability(t, beta, theta):
    return np.exp(-np.power(t / theta, beta))

t_range  = np.linspace(0.5, 30, 500)
R_curve  = reliability(t_range, BETA, THETA)

t_band   = np.linspace(T_MIN, T_MAX, 300)
R_band   = reliability(t_band, BETA, THETA)
R_min_pt = reliability(T_MIN, BETA, THETA)
R_max_pt = reliability(T_MAX, BETA, THETA)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_range, R_curve * 100, color='#2171b5', lw=2)
ax.plot(t_band, R_band * 100, color='#e6550d', lw=3, zorder=3,
        label=f'Mission window {T_MIN}–{T_MAX} yr  '
              f'(R: {R_max_pt*100:.1f}% – {R_min_pt*100:.1f}%)')
ax.axvline(T_MIN, color='#e6550d', lw=0.9, ls='--')
ax.axvline(T_MAX, color='#e6550d', lw=0.9, ls='--')
ax.set_xlabel('t (years)', fontsize=12)
ax.set_ylabel('R(t) (%)', fontsize=12)
ax.set_title(
    rf'Reliability  $R(t) = \exp[-(t/\hat{{\theta}})^{{\hat{{\beta}}}}]$'
    f'\n' + rf'$\hat{{\beta}}={BETA},\ \hat{{\theta}}={THETA}$ yr',
    fontsize=12
)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('weibull_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to weibull_plot.png")
