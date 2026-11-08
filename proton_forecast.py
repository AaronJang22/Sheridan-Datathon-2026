#!/usr/bin/env python3
"""
Solar energetic proton (>=10 MeV) onset forecasting from the OMNI 5-min file.

Task   : at time t (flux currently < 10 pfu and no S1 in the previous 24 h),
         will a NEW S1 event (>=10 pfu) begin within the next H hours? H = 1, 6, 24.
Models : persistence (score = current flux), HGB on flux-history features,
         HGB on flux + solar-wind features. Climatology = base rate (Brier reference).
Eval   : (a) leave-one-year-out, whole-year blocks with a 3-day purge on each side,
         (b) strict forward split: train <= 2013, test >= 2014.
         Sample metrics (AUC, PR-AUC, Brier skill) + event metrics at a fixed 1% false-positive
         rate (event detection, lead time, alarm episodes/month, episode precision).

Usage: python3 -I proton_forecast.py OMNI.csv OUT_DIR
Any other forecaster (e.g. IBM Granite) can be scored by adding a column to `scores` in run_split.
"""
import sys, os, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

S1 = 10.0          # pfu, NOAA S1 threshold (>=10 MeV)
QUIET = 288        # 24 h of 5-min steps
STEP = 3           # forecast every 15 min
HORIZONS = [1, 6, 24]
PURGE = pd.Timedelta(days=3)
FPR = 0.01

# ---------------------------------------------------------------- data
def load(path):
    df = pd.read_csv(path, parse_dates=["datetime_utc"]).set_index("datetime_utc")
    return df.loc[: df.proton_flux_10mev.last_valid_index()]       # proton data end 2020-03

def csum_window(flag, lo, hi):
    """sum of flag over steps (t+lo .. t+hi) inclusive, clipped to array bounds"""
    c = np.concatenate([[0], np.cumsum(flag.astype(np.int64))])
    n = len(flag); t = np.arange(n)
    a = np.clip(t + lo, 0, n); b = np.clip(t + hi + 1, 0, n)
    return c[b] - c[a]

def features(df):
    p = df.proton_flux_10mev
    lp = np.log10(p)
    F = pd.DataFrame(index=df.index)
    F["lp"] = lp
    for w in (3, 12, 72, 288):
        F[f"lp_max_{w}"] = lp.rolling(w, min_periods=1).max()
    for d in (3, 12, 36, 72):
        F[f"lp_d{d}"] = lp - lp.shift(d)
    F["lp_acc"] = (lp - lp.shift(3)) - (lp.shift(3) - lp.shift(6))
    flux_cols = list(F.columns)
    sw = pd.DataFrame(index=df.index)
    for c in ["speed_km_s", "density_cm3", "pressure_nPa", "bt_nT", "bz_gsm_nT", "efield_mV_m", "sym_h_nT", "ae_nT"]:
        sw[c] = df[c]
    sw["speed_mean_72"] = df.speed_km_s.rolling(72, min_periods=12).mean()
    sw["speed_max_288"] = df.speed_km_s.rolling(288, min_periods=36).max()
    sw["speed_d72"] = df.speed_km_s - df.speed_km_s.shift(72)
    sw["dens_max_72"] = df.density_cm3.rolling(72, min_periods=12).max()
    sw["bt_max_288"] = df.bt_nT.rolling(288, min_periods=36).max()
    sw["symh_min_288"] = df.sym_h_nT.rolling(288, min_periods=36).min()
    sw["ae_max_72"] = df.ae_nT.rolling(72, min_periods=12).max()
    return pd.concat([F, sw], axis=1), flux_cols

def labels_and_events(df):
    p = df.proton_flux_10mev.values
    valid = ~np.isnan(p)
    cross = np.where(valid, p >= S1, False)
    prior = csum_window(cross, -QUIET, -1)                       # S1 samples in previous 24 h
    onset = cross & (prior == 0)
    eligible = valid & (p < S1) & (csum_window(cross, -QUIET, 0) == 0)
    Y, OK = {}, {}
    for H in HORIZONS:
        n = H * 12
        Y[H] = csum_window(cross, 1, n) > 0
        OK[H] = csum_window(valid, 1, n) >= 0.8 * n              # >=80% of future window observed (or crossing seen)
        OK[H] |= Y[H]
    oi = np.flatnonzero(onset)
    bounds = np.append(oi, len(p))
    ev = pd.DataFrame({"onset": df.index[oi]})
    ev["peak_pfu"] = [np.nanmax(p[bounds[i]:bounds[i + 1]]) for i in range(len(oi))]
    ev["sev"] = pd.cut(ev.peak_pfu, [10, 100, 1000, 1e9], right=False, labels=["S1", "S2", "S3+"])
    return Y, OK, eligible, ev

# ---------------------------------------------------------------- evaluation
def event_metrics(t, y, score, onsets, sev, H):
    """fixed-FPR operating point; returns dict + per-event table"""
    thr = np.quantile(score[~y], 1 - FPR)
    alarm = score >= thr
    Hd = pd.Timedelta(hours=H)
    rows = []
    for o, s in zip(onsets, sev):
        m = (t >= o - Hd) & (t < o)
        if not m.any():
            continue                                             # onset not forecastable (e.g. ongoing storm tail)
        hit = m & alarm
        lead = (o - t[hit].min()).total_seconds() / 3600 if hit.any() else np.nan
        rows.append((o, s, bool(hit.any()), lead))
    E = pd.DataFrame(rows, columns=["onset", "sev", "detected", "lead_h"])
    # alarm episodes
    ta = t[alarm]
    ep_id = np.cumsum(np.r_[True, np.diff(ta.values).astype("timedelta64[m]").astype(int) > 60]) if len(ta) else np.array([])
    n_ep, true_ep = 0, 0
    for e in np.unique(ep_id):
        te = ta[ep_id == e]
        n_ep += 1
        true_ep += bool(((onsets >= te.min()) & (onsets <= te.max() + Hd)).any())
    months = len(t) * STEP * 5 / 60 / 24 / 30.4
    out = dict(thr=thr, n_events=len(E), POD=E.detected.mean() if len(E) else np.nan,
               median_lead_h=E.lead_h.median(), lead_ge_1h=(E.lead_h >= 1).mean() if len(E) else np.nan,
               episodes_per_month=n_ep / months, episode_precision=true_ep / n_ep if n_ep else np.nan)
    return out, E

def hgb():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=3,
                                          min_samples_leaf=100, l2_regularization=1.0, random_state=0)

def run_split(name, X, flux_cols, Y, OK, elig, test_mask_fn, train_mask_fn, onsets, sev, idx_all, times):
    """test_mask_fn(fold) / train_mask_fn(fold) -> boolean over idx_all; folds iterate"""
    res_rows, ev_tabs = [], []
    for H in HORIZONS:
        base = elig[idx_all] & OK[H][idx_all]
        preds = {k: np.full(len(idx_all), np.nan) for k in ["persistence", "flux_only", "flux+solar_wind"]}
        clim = np.full(len(idx_all), np.nan)
        for fold in test_mask_fn.folds:
            te = base & test_mask_fn(fold)
            tr = base & train_mask_fn(fold)
            ytr = Y[H][idx_all][tr]
            clim[te] = ytr.mean()
            preds["persistence"][te] = X["lp"].values[idx_all][te]
            for key, cols in [("flux_only", flux_cols), ("flux+solar_wind", list(X.columns))]:
                m = hgb().fit(X[cols].values[idx_all][tr], ytr)
                preds[key][te] = m.predict_proba(X[cols].values[idx_all][te])[:, 1]
        sel = ~np.isnan(clim)
        y = Y[H][idx_all][sel]; t = times[idx_all][sel]
        for key, sc in preds.items():
            s = sc[sel]; ok = ~np.isnan(s)
            auc = roc_auc_score(y[ok], s[ok]); ap = average_precision_score(y[ok], s[ok])
            bss = np.nan
            if key != "persistence":
                bs = np.mean((s[ok] - y[ok]) ** 2); bc = np.mean((clim[sel][ok] - y[ok]) ** 2)
                bss = 1 - bs / bc
            em, E = event_metrics(t[ok], y[ok], s[ok], onsets, sev, H)
            res_rows.append(dict(split=name, H_h=H, model=key, n_samples=int(ok.sum()), n_pos=int(y[ok].sum()),
                                 AUC=auc, PR_AUC=ap, base_rate=y[ok].mean(), Brier_skill=bss, **em))
            E.insert(0, "model", key); E.insert(0, "H_h", H); E.insert(0, "split", name)
            ev_tabs.append(E)
    return pd.DataFrame(res_rows), pd.concat(ev_tabs)

class Folds:
    def __init__(self, folds, fn):
        self.folds, self._fn = folds, fn
    def __call__(self, fold):
        return self._fn(fold)

def main(path, out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = load(path)
    X, flux_cols = features(df)
    Y, OK, elig, ev = labels_and_events(df)
    ev.to_csv(f"{out}/events.csv", index=False)
    print(f"{len(df):,} rows {df.index[0]} -> {df.index[-1]};  events: {ev.sev.value_counts().to_dict()}")
    idx_all = np.arange(0, len(df), STEP)
    times = df.index
    yr = times.year.values[idx_all]
    tt = times[idx_all]
    onsets, sev = pd.DatetimeIndex(ev.onset), ev.sev.values

    # (a) leave-one-year-out, 2018-2020 pooled (no events, only negatives)
    groups = [(y, y) for y in range(2010, 2018)] + [(2018, 2020)]
    def span(g):
        return pd.Timestamp(f"{g[0]}-01-01"), pd.Timestamp(f"{g[1]}-12-31 23:59")
    def te_fn(g):
        a, b = span(g); return (tt >= a) & (tt <= b)
    def tr_fn(g):
        a, b = span(g); return (tt < a - PURGE) | (tt > b + PURGE)
    loyo = Folds(groups, te_fn); loyo_tr = Folds(groups, tr_fn)
    r1, e1 = run_split("leave-one-year-out", X, flux_cols, Y, OK, elig, loyo, loyo_tr, onsets, sev, idx_all, times)
    print(f"LOYO done {time.time()-t0:.0f}s")

    # (b) strict forward split
    cut = pd.Timestamp("2014-01-01")
    fw = Folds(["fwd"], lambda g: tt >= cut); fw_tr = Folds(["fwd"], lambda g: tt < cut - PURGE)
    r2, e2 = run_split("forward (train<=2013, test>=2014)", X, flux_cols, Y, OK, elig, fw, fw_tr,
                       onsets[onsets >= cut], sev[onsets >= cut], idx_all, times)
    R = pd.concat([r1, r2]); E = pd.concat([e1, e2])
    R.to_csv(f"{out}/results.csv", index=False); E.to_csv(f"{out}/per_event.csv", index=False)
    pd.set_option("display.width", 250, "display.float_format", lambda v: f"{v:.3f}")
    print(R.drop(columns=["thr"]).to_string(index=False))
    # detection by severity at the fixed-FPR point
    S = (E.groupby(["split", "H_h", "model", "sev"], observed=True).detected.agg(["mean", "count"]).reset_index())
    S.to_csv(f"{out}/pod_by_severity.csv", index=False)
    print(S[S.H_h == 6].to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
