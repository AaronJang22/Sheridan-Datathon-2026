"""Four-tier proton radiation alerts on the forward test period (2014 - Mar 2020).

  GREEN   quiet
  YELLOW  watch: storm likely within hours. Raised by observed flux >= 1 pfu, or (with --scores) by the Granite
          6 h forecast crossing its alarm threshold. Turns ON after 15 min of trigger (3 x 5-min readings, NOAA-style
          sustain rule) and OFF only after 2 h without trigger, so it does not flicker.
  ORANGE  storm: observed flux >= 10 pfu (NOAA S1-S2) - heightened monitoring, review EVA plans
  RED     EVA hazard: observed flux >= 1000 pfu (NOAA S3+, "radiation hazard avoidance recommended for astronauts on EVA")

Orange and red follow NOAA thresholds on observed flux; the model's job is the early YELLOW watch.
With --scores, also reports (exploratory) how early Granite's 24 h forecast reached 1000 pfu before each RED onset.

Usage:
  python tier_alerts.py OMNI.csv                                                   # observed-flux trigger only
  python tier_alerts.py OMNI.csv --scores out/granite_ttm_finetuned_scores.csv.gz  # Granite-raised yellow
Writes out/tier_episodes_<trigger>.csv and figs/fig4_tiers_<trigger>.png
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import Patch
import proton_forecast as pf

YEL, ORA, RED = 1.0, 10.0, 1000.0
ON_N, OFF_N = 3, 24                                   # 15 min on, 2 h off (5-min steps)
NAMES = ["green", "yellow", "orange", "red"]


def tiers(x, gflag=None):
    """0 green, 1 yellow, 2 orange, 3 red from 5-min flux x (NaN keeps previous state); gflag: Granite yellow trigger"""
    out = np.zeros(len(x), np.int8); on = False; up = down = 0
    for i, v in enumerate(x):
        trig = (not np.isnan(v) and v >= YEL) or (gflag is not None and gflag[i])
        if not np.isnan(v) or gflag is not None:
            up, down = (up + 1, 0) if trig else (0, down + 1)
            if not on and up >= ON_N: on = True
            if on and down >= OFF_N: on = False
        if not np.isnan(v) and v >= ORA:
            on = True; out[i] = 3 if v >= RED else 2
        else:
            out[i] = 1 if on else 0
    return out


def yellow_episodes(t, tr):
    """alerts starting from GREEN; escalated if they reach ORANGE; lead = start -> first ORANGE"""
    rows, n, i = [], len(tr), 0
    while i < n:
        if tr[i] == 1 and (i == 0 or tr[i - 1] == 0):
            j = i
            while j + 1 < n and tr[j + 1] != 0: j += 1
            seg = tr[i:j + 1]; o = np.flatnonzero(seg >= 2); r = np.flatnonzero(seg == 3)
            rows.append((t[i], t[j], len(o) > 0, (t[i + o[0]] - t[i]).total_seconds() / 3600 if len(o) else np.nan,
                         int(seg.max()), (t[i + r[0]] - t[i + o[0]]).total_seconds() / 3600 if len(r) else np.nan))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows, columns=["start", "end", "reached_orange", "lead_to_orange_h", "max_tier",
                                       "orange_to_red_h"])


def granite_flags(index, scores_path):
    S = pd.read_csv(scores_path, parse_dates=["time"]).set_index("time").sort_index()
    S = S.reindex(index, method="ffill", limit=2)               # 15-min forecasts held for the next two 5-min steps
    yflag = (S.max_lp_6h >= S.thr_6h).fillna(False).values
    redfc = (S.max_lp_24h >= np.log10(RED)).fillna(False).values
    return yflag, redfc, 10 ** S.thr_6h.dropna().iloc[0]


def red_forecast_check(t, x, redfc):
    """exploratory: for each RED onset, how early did the 24 h forecast first say >= 1000 pfu (within prior 48 h)?"""
    red = np.where(np.isnan(x), False, x >= RED)
    onset = [i for i in np.flatnonzero(red) if not red[max(0, i - 576):i].any()]     # no RED in prior 48 h
    rows = []
    for i in onset:
        w = np.flatnonzero(redfc[max(0, i - 576):i])
        rows.append((t[i], (t[i] - t[max(0, i - 576) + w[0]]).total_seconds() / 3600 if len(w) else np.nan))
    # forecast-RED episodes (gaps <= 1 h merged) not followed by an actual RED within 24 h
    f = np.flatnonzero(redfc); false_ = total = 0
    if len(f):
        brk = np.flatnonzero(np.diff(f) > 12)
        for a, b in zip(np.r_[f[0], f[brk + 1]], np.r_[f[brk], f[-1]]):
            total += 1
            false_ += not red[a:min(len(red), b + 288)].any()
    return pd.DataFrame(rows, columns=["red_onset", "forecast_lead_h"]), total, false_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--scores", default=None)
    ap.add_argument("--out", default="out"); ap.add_argument("--figs", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figs, exist_ok=True)
    trig = "granite" if a.scores else "observed"

    p = pf.load(a.csv).proton_flux_10mev
    test = p.loc["2014-01-01":]; x = test.values; t = test.index
    yflag = redfc = None
    if a.scores:
        yflag, redfc, thr_pfu = granite_flags(t, a.scores)
        print(f"YELLOW trigger: observed >= 1 pfu OR Granite 6 h forecast peak >= {thr_pfu:.2f} pfu")
    else:
        print("YELLOW trigger: observed flux >= 1 pfu")
    tr = tiers(x, yflag)
    valid = ~np.isnan(x); yrs = valid.sum() * 5 / 60 / 24 / 365.25
    print(f"test period: {yrs:.1f} years of valid data")
    for k, nm in enumerate(NAMES):
        h = ((tr == k) & valid).sum() * 5 / 60
        print(f"  {nm:6s} {h:7.0f} h  ({h / yrs:6.1f} h/yr, {100 * h / (valid.sum() * 5 / 60):.2f}% of time)")

    E = yellow_episodes(t, tr)
    esc, ne = E[E.reached_orange], E[~E.reached_orange]
    print(f"\nYELLOW alerts from quiet: {len(E)}  -> ORANGE: {len(esc)} ({100 * len(esc) / max(len(E), 1):.0f}%)  "
          f"-> no storm: {len(ne)} ({len(ne) / yrs:.1f}/yr)")
    if len(esc):
        print(f"  warning yellow->orange: median {esc.lead_to_orange_h.median():.1f} h, quartiles "
              f"{esc.lead_to_orange_h.quantile(.25):.1f}-{esc.lead_to_orange_h.quantile(.75):.1f} h")
    reds = E[E.max_tier == 3]
    for r in reds.itertuples():
        print(f"  RED (>=1000 pfu) in alert starting {r.start}: orange gave {r.orange_to_red_h:.1f} h before red")
    E.to_csv(f"{a.out}/tier_episodes_{trig}.csv", index=False)

    if redfc is not None:
        RF, tot, fal = red_forecast_check(t, x, redfc)
        print(f"\n[exploratory] Granite 24 h forecast reaching >=1000 pfu: {tot} forecast episodes, {fal} not followed by RED")
        for r in RF.itertuples():
            lead = "never forecast" if np.isnan(r.forecast_lead_h) else f"forecast {r.forecast_lead_h:.1f} h ahead"
            print(f"  RED onset {r.red_onset}: {lead}")
        print("  (only a handful of RED events: treat as anecdotal)")
    figure(p, E, trig, a)


def figure(p, E, trig, a):
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    COL = {1: "#fab219", 2: "#ec835a", 3: "#d03b3b"}; ALPHA = {1: 0.24, 2: 0.16, 3: 0.42}; LINE = "#2a78d6"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})
    panels = [("Slow rise, S1–S2 storm: Sept 2014", "2014-09-09 12:00", "2014-09-12 12:00"),
              ("Fast rise, S3 storm: Sept 2017", "2017-09-09 12:00", "2017-09-12 12:00")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.87, top=0.72, bottom=0.14, wspace=0.08)
    src = "Granite forecast or observed flux" if trig == "granite" else "observed flux"
    fig.text(0.02, 0.965, "Four-tier radiation alerts", fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, f"≥10 MeV proton flux. Yellow watch raised by {src}; orange and red follow NOAA S1 and S3 thresholds.",
             fontsize=10.5, color=INK2, va="top")
    yflag_all = None
    if trig == "granite":
        yflag_all, _, _ = granite_flags(p.index, a.scores)
    for ax, (ttl, s0, s1) in zip(axes, panels):
        i0, i1 = p.index.get_indexer([pd.Timestamp(s0), pd.Timestamp(s1)])
        lo = max(0, i0 - 2000)
        tr = tiers(p.values[lo:i1 + 1], None if yflag_all is None else yflag_all[lo:i1 + 1])[i0 - lo:]
        w = p.iloc[i0:i1 + 1]; tw = pd.Series(tr, index=w.index)
        for _, seg in tw.groupby((tw != tw.shift()).cumsum()):
            lvl = int(seg.iloc[0])
            if lvl: ax.axvspan(seg.index[0], seg.index[-1] + pd.Timedelta("5min"), color=COL[lvl], alpha=ALPHA[lvl], lw=0)
        ax.plot(w.index, w.values, color=LINE, lw=2)
        for y in (YEL, ORA, 100, RED):
            ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)))
        ep = E[(E.start <= w.index[-1]) & (E.end >= w.index[0]) & E.reached_orange]
        if len(ep):
            o = ep.iloc[-1]; on = o.start + pd.Timedelta(hours=o.lead_to_orange_h)
            ax.annotate("", xy=(on, 0.15), xytext=(o.start, 0.15), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
            ax.text(o.start - pd.Timedelta("2h"), 0.15, f"{o.lead_to_orange_h:.1f} h watch ", ha="right", va="center",
                    fontsize=10.5, color=INK)
        ax.set_title(ttl, loc="left", fontsize=12, color=INK, pad=8)
        ax.set_yscale("log"); ax.set_ylim(0.1, 5000)
        ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].set_ylabel("Proton flux ≥10 MeV (pfu, log)")
    for y, lab in [(YEL, "1 pfu"), (ORA, "10 pfu  S1"), (100, "100 pfu  S2"), (RED, "1000 pfu  S3")]:
        axes[1].text(1.02, y, lab, transform=axes[1].get_yaxis_transform(), va="center", fontsize=9.5, color=INK2)
    axes[0].legend(handles=[Patch(color=COL[1], alpha=0.5, label="YELLOW: watch"),
                            Patch(color=COL[2], alpha=0.4, label="ORANGE: storm, 10–1000 pfu"),
                            Patch(color=COL[3], alpha=0.4, label="RED: EVA hazard, ≥1000 pfu")],
                   frameon=False, loc="upper left", fontsize=10, ncol=3, bbox_to_anchor=(0, 1.24))
    fig.text(0.02, 0.03, "Yellow turns on after 15 min of trigger and off after 2 h without. Arrow: watch time before the storm (orange).",
             fontsize=9, color=INK2)
    out = f"{a.figs}/fig4_tiers_{trig}.png"
    fig.savefig(out, dpi=200); print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
