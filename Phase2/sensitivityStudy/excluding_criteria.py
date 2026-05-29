import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

options = ["A: Europa 1 sc; Orbit Io",
           "B: Europa 2 sc; Orbit Io",
           "C: Europa 1 sc; Flyby Io",
           "D: Europa 2 sc; Flyby Io"]

short_options = ["A", "B", "C", "D"]

criteria = [
    {"name": "Science return",     "weight": 0.25, "scores": [3.75, 4.5,  3.125, 3.875]},
    {"name": "System performance", "weight": 0.15, "scores": [3.4,  1.5,  5.0,   3.5  ]},
    {"name": "Risk",               "weight": 0.15, "scores": [2.0,  1.0,  5.0,   4.0  ]},
    {"name": "Cost",               "weight": 0.25, "scores": [5.0,  2.2,  4.0,   2.2  ]},
    {"name": "Schedule",           "weight": 0.10, "scores": [3.5,  3.0,  3.5,   2.5  ]},
    {"name": "Sustainability",     "weight": 0.10, "scores": [4.5,  1.0,  5.0,   2.0  ]},
]

EXCLUDE = {"Science return", "Cost", "Schedule", "Sustainability"}

n_opt = len(options)

def compute_scores(weights):
    return [sum(w * c["scores"][i] for w, c in zip(weights, criteria))
            for i in range(n_opt)]

baseline_w = [c["weight"] for c in criteria]
scenarios = [("Baseline", baseline_w)]
for k, c in enumerate(criteria):
    if c["name"] not in EXCLUDE:
        continue
    w = [baseline_w[j] if j != k else 0.0 for j in range(len(criteria))]
    total = sum(w)
    w = [x / total for x in w]
    scenarios.append((f"No\n{c['name']}", w))

x = np.arange(len(scenarios))
all_scores = np.array([compute_scores(w) for _, w in scenarios])

COLORS = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']

fig, ax_score = plt.subplots(figsize=(10, 5))
fig.suptitle("Sensitivity Analysis — Excluding Criteria One at a Time",
             fontsize=13, fontweight='bold', y=0.98)

for i in range(n_opt):
    ax_score.plot(x, all_scores[:, i], color=COLORS[i], lw=2, marker='o',
                  markersize=7, label=options[i], zorder=3)
    for xi, score in enumerate(all_scores[:, i]):
        ax_score.annotate(f"{score:.2f}", (xi, score),
                          textcoords="offset points", xytext=(0, 7),
                          ha='center', fontsize=7.5, color=COLORS[i])

ax_score.axvline(0.5, color='lightgray', lw=1, ls='--')
ax_score.set_ylabel("Weighted score", fontsize=11)
ax_score.set_ylim(1.5, 5.5)
ax_score.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax_score.grid(True, alpha=0.2)
ax_score.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax_score.set_xticks(x)
ax_score.set_xticklabels([label for label, _ in scenarios], fontsize=9)

plt.tight_layout()
plt.savefig('excluding_criteria.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to excluding_criteria.png")
