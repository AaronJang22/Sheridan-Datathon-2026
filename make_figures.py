"""Results figures for the proton-storm forecasting write-up (forward split, test 2014-2020)."""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "/home/claude/figs"; os.makedirs(OUT, exist_ok=True)

# Colab results table (forward split: train <= 2013, test 2014 - Mar 2020, 14 storms)
rows = [  # H, model, n_samples, PR_AUC, POD, median_lead_h, episodes_per_month, episode_precision
    (1, "flux_only", 210678, .440, 1, .958, 7.827, .025), (1, "persistence", 210678, .338, 1, .958, .748, .259),
    (1, "granite_ttm_finetuned", 210678, .269, 1, .958, .748, .259), (1, "flux+solar_wind", 210678, .159, 1, .958, 8.090, .024),
    (1, "granite_ttm_zeroshot", 210678, .131, 1, .917, .734, .264),
    (6, "flux_only", 210174, .283, 1, 4.375, 6.235, .040), (6, "granite_ttm_finetuned", 210174, .230, 1, 4.958, .653, .340),
    (6, "persistence", 210174, .206, 1, 4.792, .916, .258), (6, "granite_ttm_zeroshot", 210174, .141, 1, 5.500, .833, .300),
    (6, "flux+solar_wind", 210174, .138, 1, 3.750, 4.082, .061),
    (24, "flux_only", 210055, .181, 1, 10.417, 2.112, .171), (24, "persistence", 210055, .151, 1, 12.250, 1.167, .333),
    (24, "granite_ttm_finetuned", 210055, .148, 1, 10.542, 1.056, .289), (24, "granite_ttm_zeroshot", 210055, .113, 1, 12.250, .931, .433),
    (24, "flux+solar_wind", 210055, .112, 1, 11.292, 2.640, .116),
]
R = pd.DataFrame(rows, columns=["H", "model", "n", "PR_AUC", "POD", "lead", "eps_month", "prec"])
R["months"] = R.n * 15 / 60 / 24 / 30.4                     # same month length as the pipeline
R["false_alarms"] = R.eps_month * R.months * (1 - R.prec)
R["years"] = R.months / 12

NAME = {"persistence": "Persistence baseline", "granite_ttm_zeroshot": "Granite TTM, zero-shot",
        "granite_ttm_finetuned": "Granite TTM, fine-tuned", "flux_only": "Boosting, flux only",
        "flux+solar_wind": "Boosting, flux + solar wind"}
COL = {"persistence": "#1baf7a", "granite_ttm_zeroshot": "#eb6834", "granite_ttm_finetuned": "#2a78d6",
       "flux_only": "#a3a29c", "flux+solar_wind": "#a3a29c"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

def title(fig, t, s):
    fig.text(0.02, 0.965, t, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, s, fontsize=10.5, color=INK2, va="top")

def rbar(ax, y, w, h, color, horiz=True):
    from matplotlib.patches import FancyBboxPatch
    if horiz:
        p = FancyBboxPatch((0, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.0", fc=color, ec=SURF, lw=2)
    ax.add_patch(p)

# ---------- Fig 1: headline, 6 h false alarms per model
d = R[R.H == 6].sort_values("false_alarms", ascending=False).reset_index(drop=True)
yrs = d.years.iloc[0]
fig, ax = plt.subplots(figsize=(10, 5.2)); fig.subplots_adjust(left=0.27, right=0.95, top=0.78, bottom=0.18)
title(fig, "Fine-tuned Granite cut false alarms by a third at the 6-hour horizon",
      f"False storm alarms over the {yrs:.0f}-year test period (2014–2020). Every model caught all 14 storms.")
for i, r in d.iterrows():
    ax.barh(i, r.false_alarms, height=0.62, color=COL[r.model], edgecolor=SURF, linewidth=2)
    ax.text(r.false_alarms + 6, i, f"{r.false_alarms:.0f}   ·   {r.lead:.1f} h median warning",
            va="center", fontsize=10.5, color=INK)
ax.set_yticks(range(len(d))); ax.set_yticklabels([NAME[m] for m in d.model], fontsize=11, color=INK)
ax.set_xlim(0, d.false_alarms.max() * 1.45); ax.set_xlabel("False alarms (alarm episodes not followed by a storm)")
ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
fig.text(0.02, 0.03, "Alarm threshold set so each model is on alert during 1% of quiet time. Forward split: trained ≤2013.",
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/fig1_false_alarms_6h.png", dpi=200); plt.close(fig)

# ---------- Fig 2: false alarms by horizon, persistence vs Granite
models = ["persistence", "granite_ttm_zeroshot", "granite_ttm_finetuned"]
H = [1, 6, 24]; w = 0.26
fig, ax = plt.subplots(figsize=(10, 5.2)); fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
title(fig, "Granite's advantage depends on the forecast horizon",
      "False alarms over the test period by horizon. At 1 h fine-tuned Granite gives the same alarms as persistence.")
for j, m in enumerate(models):
    v = [R[(R.H == h) & (R.model == m)].false_alarms.item() for h in H]
    x = np.arange(len(H)) + (j - 1) * w
    ax.bar(x, v, width=w, color=COL[m], edgecolor=SURF, linewidth=2, label=NAME[m])
    for xi, vi in zip(x, v):
        ax.text(xi, vi + 1.2, f"{vi:.0f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(range(len(H))); ax.set_xticklabels([f"{h} h ahead" for h in H], fontsize=11, color=INK)
ax.set_ylabel("False alarms"); ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0); ax.set_ylim(0, 66); ax.set_yticks(range(0, 61, 10))
ax.legend(frameon=False, loc="upper left", fontsize=10, ncol=3, bbox_to_anchor=(0, 1.12))
fig.text(0.02, 0.03, "Boosting models omitted (≈130–570 false alarms). All models caught 14/14 storms at every horizon.",
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/fig2_false_alarms_by_horizon.png", dpi=200); plt.close(fig)

# ---------- Fig 3: trade-off at 6 h, warning time vs false alarms per year
d = R[R.H == 6].copy(); d["fa_yr"] = d.false_alarms / d.years
fig, ax = plt.subplots(figsize=(10, 5.6)); fig.subplots_adjust(left=0.09, right=0.96, top=0.79, bottom=0.17)
title(fig, "Warning time vs. false alarms, 6-hour horizon",
      "Top-left is better: more warning before a storm, fewer false alarms per year.")
off = {"persistence": (8, -14), "granite_ttm_zeroshot": (8, 6), "granite_ttm_finetuned": (8, 6),
       "flux_only": (-10, 8), "flux+solar_wind": (8, -14)}
for _, r in d.iterrows():
    ax.scatter(r.fa_yr, r.lead, s=110, color=COL[r.model], edgecolor=SURF, linewidth=2, zorder=3)
    dx, dy = off[r.model]
    ax.annotate(NAME[r.model], (r.fa_yr, r.lead), xytext=(dx, dy), textcoords="offset points",
                ha="right" if dx < 0 else "left", fontsize=10.5, color=INK)
ax.set_xscale("log"); ax.set_xlim(3, 120); ax.set_ylim(3.3, 6.0)
ax.set_xticks([3, 5, 10, 20, 50, 100]); ax.set_xticklabels(["3", "5", "10", "20", "50", "100"])
ax.set_xlabel("False alarms per year (log scale)"); ax.set_ylabel("Median warning before onset (h)")
ax.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
fig.text(0.02, 0.03, "14 test storms; differences between the top three are a handful of alarms and come from one training run.",
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/fig3_tradeoff_6h.png", dpi=200); plt.close(fig)

print(R[["H", "model", "false_alarms", "lead"]].round(1).to_string(index=False))
