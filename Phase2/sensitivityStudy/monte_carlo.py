import numpy as np
import matplotlib.pyplot as plt

options       = ["A: Europa 1 sc; Orbit Io",
                 "B: Europa 2 sc; Orbit Io",
                 "C: Europa 1 sc; Flyby Io",
                 "D: Europa 2 sc; Flyby Io"]
short_options = ["A", "B", "C", "D"]

criteria = [
    {"name": "Science return",     "weight": 0.25, "var": 10,  "w_lo": -0.15, "w_hi": +0.05,
     "scores": [3.75, 4.5,  3.125, 3.875]},
    {"name": "System performance", "weight": 0.15, "var": 15,  "w_lo":  0.00, "w_hi": +0.10,
     "scores": [3.4,  1.5,  5.0,   3.5  ]},
    {"name": "Risk",               "weight": 0.15, "var": 30,  "w_lo":  0.00, "w_hi": +0.10,
     "scores": [2.0,  1.0,  5.0,   4.0  ]},
    {"name": "Cost",               "weight": 0.25, "var": 20,  "w_lo": -0.15, "w_hi": +0.05,
     "scores":  [5.0,  2.2,  4.0,   2.2  ]},
    {"name": "Schedule",           "weight": 0.10, "var": 25,  "w_lo":  0.00, "w_hi": +0.10,
     "scores": [3.5,  3.0,  3.5,   2.5  ]},
    {"name": "Sustainability",     "weight": 0.10, "var": 35,  "w_lo":  0.00, "w_hi": +0.05,
     "scores": [4.5,  1.0,  5.0,   2.0  ]},
]

n_opt  = 4
n_crit = 6
COLORS = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
N      = 1_000_000
np.random.seed(42)

baseline   = np.array([sum(c["weight"] * c["scores"][i] for c in criteria) for i in range(n_opt)])
winner_idx = int(np.argmax(baseline))

w_min  = np.array([c["weight"] + c["w_lo"] for c in criteria])
w_max  = np.array([c["weight"] + c["w_hi"] for c in criteria])
w_mode = np.array([c["weight"]             for c in criteria])

print("Weight distributions (triangular):")
for k, c in enumerate(criteria):
    print(f"  {c['name']:<22}  range [{w_min[k]:.2f}, {w_max[k]:.2f}]  mode={w_mode[k]:.2f}")
print()

sampled_weights = np.column_stack([
    np.random.triangular(w_min[k], w_mode[k], w_max[k], size=N)
    for k in range(n_crit)
])
sampled_weights /= sampled_weights.sum(axis=1, keepdims=True)

sampled_scores = np.zeros((N, n_crit, n_opt))
for k, c in enumerate(criteria):
    mean = np.array(c["scores"])
    std  = mean * c["var"] / 100
    sampled_scores[:, k, :] = np.random.normal(mean, std, size=(N, n_opt))

s_max = sampled_scores.max(axis=2, keepdims=True)
scale = np.where(s_max > 5, 5 / s_max, 1.0)
sampled_scores *= scale

sampled_totals = np.einsum('nk,nki->ni', sampled_weights, sampled_scores)

sampled_ranks = np.argsort(np.argsort(-sampled_totals, axis=1), axis=1) + 1
p_first       = np.mean(sampled_ranks == 1, axis=0)
p_c_first     = p_first[winner_idx]

competitors = [j for j in range(n_opt) if j != winner_idx]
gap_samples = sampled_totals[:, winner_idx] - sampled_totals[:, competitors].max(axis=1)

rank_order = np.argsort(-baseline)

print("=" * 60)
print(f"  Monte Carlo  N={N:,}  |  scores: 1σ=var%  |  weights: triangular\n")
print(f"  {'Option':<30}  {'Baseline':>8}  {'P(1st) %':>9}")
print("-" * 60)
for i in rank_order:
    print(f"  {options[i]:<30}  {baseline[i]:>8.4f}  {p_first[i]*100:>8.2f}%")
print("-" * 60)
print(f"\n  P(C stays 1st) = {p_c_first*100:.2f}%")
print(f"  P(rank flip)   = {(1-p_c_first)*100:.2f}%")
print(f"  Mean gap       = {gap_samples.mean():.4f}")
print(f"  5th pct gap    = {np.percentile(gap_samples, 5):.4f}")
print("=" * 60)

title = f"N={N:,} simulations"
flip_mask = gap_samples < 0

fig1, ax_g = plt.subplots(figsize=(13, 5))
fig1.suptitle(title, fontsize=12, fontweight='bold')

bins = np.linspace(gap_samples.min(), gap_samples.max(), 150)
ax_g.hist(gap_samples[~flip_mask], bins=bins, color='lightgreen', alpha=0.8,
          label=f"C stays 1st  ({(~flip_mask).mean()*100:.1f}%)")
ax_g.hist(gap_samples[flip_mask],  bins=bins, color='salmon', alpha=0.8,
          label=f"Rank flip  ({flip_mask.mean()*100:.1f}%)")
ax_g.axvline(0, color='black', lw=1.5, ls='--', zorder=5)
ax_g.axvline(gap_samples.mean(), color='green', lw=1.2, ls=':',
             label=f"Mean gap = {gap_samples.mean():.3f}")
ax_g.set_xlabel("Gap: Option C total − best rival total", fontsize=10)
ax_g.set_ylabel("Count", fontsize=10)
ax_g.set_title("Distribution of margin between C and best rival each run", fontsize=9)
ax_g.legend(fontsize=8.5, framealpha=0.9)
ax_g.grid(alpha=0.2)

fig1.tight_layout()
fig1.savefig('monte_carlo_gap.png', dpi=150, bbox_inches='tight')

fig2, ax_p = plt.subplots(figsize=(13, 4))
fig2.suptitle(title, fontsize=12, fontweight='bold')

bars = ax_p.barh(
    [f"Option {short_options[i]}" for i in rank_order],
    [p_first[i] * 100 for i in rank_order],
    color='steelblue', alpha=0.7, height=0.5
)
for bar, i in zip(bars, rank_order):
    ax_p.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
              f"{p_first[i]*100:.2f}%", va='center', fontsize=9,
              color='black', fontweight='bold')

ax_p.set_xlabel("Probability of ranking 1st (%)", fontsize=10)
ax_p.set_xlim(0, 115)
ax_p.grid(axis='x', alpha=0.2)

fig2.tight_layout()
fig2.savefig('monte_carlo_prob.png', dpi=150, bbox_inches='tight')

plt.show()
print("Plots saved to monte_carlo_gap.png, monte_carlo_prob.png")
