#!/usr/bin/env python3
"""
Part 1: does a solar radiation storm come before a geomagnetic storm?

NOTE: reconstructed after the fact. The original analysis script was lost; this script was written to
reproduce the numbers in the report and prints each one next to the report's value.

Definitions (the ones that reproduce the report):
  radiation storm      >=10 MeV proton flux >= 10 pfu (NOAA S1). A new storm starts when flux reaches
                       10 pfu after at least 4 days without doing so -> 33 storms, Jan 2010 - Mar 2020.
  geomagnetic storm    a day whose minimum SYM-H is <= -100 nT (strong) or <= -50 nT (moderate).
  "followed within w"  a geomagnetic-storm day on any of the w calendar days AFTER the day the radiation
                       storm began (days d+1 .. d+w).
  random baseline      the same check from 3,000 random days, Jan 2010 - Mar 2020 (numpy default_rng, seed 1);
                       the exact rate over every day is printed too, since a random draw varies.

Usage: python storm_lag.py OMNI.csv [--out out]
"""
import argparse, os, math
import numpy as np, pandas as pd

S1, GAP_DAYS = 10.0, 4
THR = {"strong": -100, "moderate": -50}
WINDOWS = (2, 3, 4, 5)
REPORT = {"strong": {"after": [15, 21, 24, 24], "random": [2, 3, 4, 5]},
          "moderate": {"after": [48, 55, 58, 67], "random": [14, 18, 23, 26]}}


def radiation_storms(p):
    above = np.flatnonzero((p >= S1).fillna(False).values)
    starts, last = [], -10 ** 9
    for i in above:
        if i - last > GAP_DAYS * 288:            # 288 five-minute steps per day
            starts.append(i)
        last = i
    t = p.index[starts]
    nxt = list(t[1:]) + [p.index[-1]]
    peaks = [p.loc[a:b].max() for a, b in zip(t, nxt)]
    return pd.DataFrame({"start": t, "peak_pfu": np.round(peaks, 1)})


def followed(days, dmin, w, thr):
    """for each day d: is there a day in d+1..d+w with daily min SYM-H <= thr? (NaN if window incomplete)"""
    v = dmin.values; pos = dmin.index.get_indexer(days)
    return np.array([np.nanmin(v[i + 1:i + w + 1]) <= thr if i + w < len(v) else np.nan for i in pos])


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out", default="out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv, parse_dates=["datetime_utc"]).set_index("datetime_utc")
    end = df.proton_flux_10mev.last_valid_index()                   # proton data end: 4 Mar 2020
    p = df.proton_flux_10mev.loc[:end]
    dmin = df.sym_h_nT.resample("1D").min()                         # SYM-H continues past 2020; windows can use it
    ev = radiation_storms(p)
    days = ev.start.dt.floor("D")
    print(f"Radiation storms (>=10 pfu, >= {GAP_DAYS} days apart), {p.index[0].date()} to {end.date()}: {len(ev)}  [report: 33]")

    rng = np.random.default_rng(1)
    all_days = pd.date_range(p.index[0].floor("D"), end.floor("D"), freq="D")
    rdays = all_days[rng.choice(len(all_days), 3000)]

    print("\nShare followed by a geomagnetic storm (after a radiation storm / at random times):")
    for kind, thr in THR.items():
        print(f"  {kind} (SYM-H <= {thr} nT)")
        for j, w in enumerate(WINDOWS):
            af = round(100 * np.nanmean(followed(days, dmin, w, thr)))
            rd = round(100 * np.nanmean(followed(rdays, dmin, w, thr)))
            ex = 100 * np.nanmean(followed(all_days, dmin, w, thr))
            r = REPORT[kind]
            print(f"    {w} days: {af:3d}% vs {rd:2d}%  (every day: {ex:.1f}%)   [report: {r['after'][j]}% vs {r['random'][j]}%]")

    f4 = followed(days, dmin, 4, THR["strong"]).astype(bool)
    k, n = int(f4.sum()), len(ev)
    ex4 = 100 * np.nanmean(followed(all_days, dmin, 4, THR["strong"]))
    lo, hi = wilson(k, n)
    print(f"\nStrong storm within 4 days: {k} of {n} ({100 * k / n:.0f}%), 95% CI {100 * lo:.0f}-{100 * hi:.0f}%   "
          f"[report: 24%; limits text says 9 storms, ~12-38%]")
    print(f"No strong storm within 4 days: {n - k} of {n}   [report: 25 of 33]")
    print(f"Ratio vs random: {100 * k / n / 4:.1f}x vs the sampled 4%, {100 * k / n / ex4:.1f}x vs every day ({ex4:.1f}%)   [report: 6x]")

    v = dmin.values; lags = []
    for d, hit in zip(days, f4):
        if hit:
            i = dmin.index.get_loc(d); lags.append(int(np.flatnonzero(v[i + 1:i + 5] <= THR["strong"])[0]) + 1)
    print(f"Median head start of the radiation over the strong storm: {np.median(lags):.0f} days (n={len(lags)})   [report: 2 days]")

    daily = pd.DataFrame({"pmax": p.resample("1D").max(), "symh_min": dmin.loc[:end]}).dropna()
    r = np.corrcoef(daily.pmax, -daily.symh_min)[0, 1]
    print(f"Same-day correlation, daily max proton flux vs storm strength (-min SYM-H): r = {r:.2f}   [report: 0.11]")

    print("\nCase studies (daily max proton flux, daily min SYM-H):")
    for a_, b_ in (("2012-03-06", "2012-03-11"), ("2017-09-04", "2017-09-13")):
        t = pd.DataFrame({"proton_max_pfu": p.loc[a_:b_].resample("1D").max().round(1), "symh_min_nT": dmin.loc[a_:b_]})
        print(t.to_string(), "\n")

    ev["strong_storm_within_4d"] = f4
    ev.to_csv(os.path.join(a.out, "radiation_storms_part1.csv"), index=False)
    print("wrote", os.path.join(a.out, "radiation_storms_part1.csv"))


if __name__ == "__main__":
    main()
