#!/usr/bin/env python3
"""
IBM Granite TinyTimeMixer (TTM r2) for solar proton event forecasting.
Plugs into proton_forecast.py: same data, labels, eligibility, forward split
(train <= 2013, test >= 2014) and event metrics, so results line up with results.csv.

How Granite is used
  1. Forecast log10(>=10 MeV proton flux) on a 15-min grid:
     context 512 steps (5.3 days) -> prediction 96 steps (24 h).
  2. Event score for horizon H = max forecast log-flux over the next H hours
     ("does the forecast cross 10 pfu?"). Scored exactly like persistence.

Modes
  default      zero-shot: pretrained TTM, no training
  --finetune   fine-tune the TTM head on training years (<= 2013), backbone frozen
  --smoke      random-init TTM on a small subset; checks the plumbing only (no download)

Install : pip install granite-tsfm          (pulls torch + transformers)
Run     : python granite_ttm.py OMNI.csv OUT_DIR [--finetune]
Needs proton_forecast.py in the same folder. CPU is fine (TTM is ~1M parameters).
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score, average_precision_score
import proton_forecast as pf

CTX, PRED = 512, 96                       # 15-min steps
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
CUT = pd.Timestamp("2014-01-01")
LOG_S1 = np.log10(pf.S1)


def load_ttm(smoke):
    if smoke:
        from tsfm_public import TinyTimeMixerConfig, TinyTimeMixerForPrediction
        cfg = TinyTimeMixerConfig(context_length=CTX, prediction_length=PRED, num_input_channels=1,
                                  patch_length=64, patch_stride=64)
        return TinyTimeMixerForPrediction(cfg)
    try:
        from tsfm_public.toolkit.get_model import get_model
        return get_model(MODEL_ID, context_length=CTX, prediction_length=PRED)
    except Exception as e:                # older granite-tsfm: weights live on a revision branch
        print("get_model failed, using revision branch:", e)
        from tsfm_public import TinyTimeMixerForPrediction
        return TinyTimeMixerForPrediction.from_pretrained(MODEL_ID, revision="512-96-r2")


@torch.no_grad()
def forecast(model, s, origins, bs=512):
    """origins: indices k on the 15-min grid; context = s[k-CTX+1 .. k]; returns [n, PRED]"""
    W = sliding_window_view(s, CTX)
    out = np.empty((len(origins), PRED), np.float32)
    model.eval()
    for i in range(0, len(origins), bs):
        k = origins[i:i + bs]
        x = torch.from_numpy(W[k - CTX + 1].copy()).unsqueeze(-1).to(DEV)   # [B, CTX, 1]
        out[i:i + bs] = model(past_values=x).prediction_outputs[:, :PRED, 0].cpu().numpy()
    return out


def finetune(model, s, tt, epochs=3, n=40000, lr=1e-3, bs=256, seed=0):
    """Train the head on <=2013 windows; half of each epoch drawn from windows whose
    next 24 h reach >=1 pfu, so the rare ramps are not drowned out by quiet time."""
    rng = np.random.default_rng(seed)
    W, Wf = sliding_window_view(s, CTX), sliding_window_view(s, PRED)
    last = np.flatnonzero(tt < CUT - pf.PURGE).max() - PRED
    k_all = np.arange(CTX - 1, last)
    active = Wf[k_all + 1].max(1) >= 0.0
    k_act, k_quiet = k_all[active], k_all[~active]
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for ep in range(epochs):
        k = np.concatenate([rng.choice(k_act, n // 2), rng.choice(k_quiet, n // 2)])
        rng.shuffle(k)
        tot = 0.0
        for i in range(0, len(k), bs):
            kk = k[i:i + bs]
            x = torch.from_numpy(W[kk - CTX + 1].copy()).unsqueeze(-1).to(DEV)
            y = torch.from_numpy(Wf[kk + 1].copy()).unsqueeze(-1).to(DEV)
            loss = model(past_values=x, future_values=y).loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(kk)
        print(f"  epoch {ep + 1}/{epochs}  train MSE {tot / len(k):.4f}")
    return model


def export_replay(df, ev, tt, test_k, F, thr6, tag, path, before="2.5D", after="1.5D"):
    """Replay file for the console page: for each test storm, 5-min observed log10 flux over
    [onset - before, onset + after] and, if F is given, the model's 96-step (24 h) forecast made at
    every 15-min time in that window (forecast k covers origin + 15 min * (1..96))."""
    import json
    p = df.proton_flux_10mev
    row = {k: r for r, k in enumerate(test_k)}
    kpos = pd.Series(np.arange(len(tt)), index=tt)

    def window(t0, t1):
        fl = np.log10(p.loc[t0:t1 - pd.Timedelta("5min")].values)
        item = dict(t0=str(t0), flux=[None if np.isnan(v) else round(float(v), 5) for v in fl])
        if F is not None:
            ks = kpos.loc[t0:t1 - pd.Timedelta("15min")].values
            item["fc"] = [[round(float(v), 2) for v in F[row[k]]] if k in row else None for k in ks]
            if thr6 is not None:   # exact 6 h alarm decision per forecast (rounded forecasts can flip it)
                item["g6"] = "".join("1" if (k in row and F[row[k], :24].max() >= thr6) else "0" for k in ks)
        return item

    storms = []
    for o, pk, sv in zip(ev.onset, ev.peak_pfu, ev.sev):
        if o < CUT:
            continue
        t0 = (o - pd.Timedelta(before)).floor("15min"); t1 = (o + pd.Timedelta(after)).floor("15min")
        storms.append(dict(kind="storm", onset=str(o), peak_pfu=round(float(pk), 1), sev=str(sv), **window(t0, t1)))

    # every watch (yellow) that never became a storm, using the same tier rules as tier_alerts.py
    watches = []
    if F is not None and thr6 is not None:
        import tier_alerts as ta
        q = p.loc[CUT:]; t = q.index; x = q.values
        m6 = pd.Series(F[:, :24].max(1), index=tt[test_k])
        g = (m6.reindex(t, method="ffill", limit=2) >= thr6).values         # NaN compares False
        tr, trO = ta.tiers(x, g), ta.tiers(x)
        E = ta.yellow_episodes(t, tr)
        for e in E[~E.reached_orange].itertuples():
            si = t.get_loc(e.start)
            t0 = (e.start - pd.Timedelta("18h")).floor("15min")
            t1 = min(e.end + pd.Timedelta("18h"), e.start + pd.Timedelta("3D")).floor("15min")
            watches.append(dict(kind="watch", start=str(e.start), end=str(e.end),
                                raised_by="READYation" if trO[si] == 0 else "flux",
                                duration_h=round((e.end - e.start).total_seconds() / 3600 + 5 / 60, 2),
                                peak_pfu=round(float(np.nanmax(q.loc[e.start:e.end].values)), 2),
                                **window(t0, t1)))
        print(f"watches that did not become storms: {len(watches)} "
              f"({sum(w['raised_by'] == 'READYation' for w in watches)} raised by READYation first)")
    out = dict(model=tag, thr_6h_log10=None if thr6 is None else round(float(thr6), 4), step_flux_min=5,
               step_fc_min=15, fc_steps=PRED, storms=storms, watches=watches)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"saved replay file -> {path}  ({len(storms)} storms, {len(watches)} no-storm watches, "
          f"{os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replay", action="store_true", help="also write out/readyation_replay.json for the console page")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    torch.manual_seed(0)

    df = pf.load(a.csv)
    Y, OK, elig, ev = pf.labels_and_events(df)
    idx = np.arange(0, len(df), pf.STEP)                  # 15-min grid, same as proton_forecast.py
    tt = df.index[idx]
    lp = np.log10(df.proton_flux_10mev).iloc[idx]
    s = lp.ffill().fillna(lp.median()).values.astype(np.float32)   # past-only fill, no leakage

    test_k = np.flatnonzero(tt >= CUT)                    # forecast every test time (needed for the alert tiers);
    if a.smoke:                                           # evaluation below still uses only eligible times
        test_k = test_k[::50]
    el = elig[idx][test_k]
    onsets = pd.DatetimeIndex(ev.onset); sev = ev.sev.values
    keep = onsets >= CUT; onsets, sev = onsets[keep], sev[keep]

    model = load_ttm(a.smoke).to(DEV)
    print("device:", DEV)
    tag = "granite_ttm_zeroshot"
    if a.finetune:
        model = finetune(model, s, tt, **(dict(epochs=1, n=1024) if a.smoke else {}))
        tag = "granite_ttm_finetuned"
    print(f"forecasting {len(test_k):,} origins ...")
    F = forecast(model, s, test_k)

    rows = []
    for H in pf.HORIZONS:
        m = OK[H][idx][test_k] & el
        y = Y[H][idx][test_k][m]; t = tt[test_k][m]
        scores = {"persistence": s[test_k][m], tag: F[m, :H * 4].max(1)}
        for name, sc in scores.items():
            em, _ = pf.event_metrics(t, y, sc, onsets, sev, H)
            rows.append(dict(split="forward (train<=2013, test>=2014)", H_h=H, model=name,
                             n_samples=int(m.sum()), n_pos=int(y.sum()),
                             AUC=roc_auc_score(y, sc), PR_AUC=average_precision_score(y, sc),
                             forecast_crosses_S1=float((sc >= LOG_S1).mean()) if name != "persistence" else np.nan,
                             **em))
    R = pd.DataFrame(rows)
    R.to_csv(f"{a.out}/{tag}_results.csv", index=False)
    # per-time forecasts for tier_alerts.py: max forecast log10 flux over the next 1/6/24 h,
    # plus the 6 h alarm threshold (1% false-positive point) used for the YELLOW tier
    thr6 = R[(R.model == tag) & (R.H_h == 6)].thr.item()
    pd.DataFrame({"time": tt[test_k], "max_lp_1h": F[:, :4].max(1), "max_lp_6h": F[:, :24].max(1),
                  "max_lp_24h": F[:, :96].max(1), "thr_6h": thr6}).to_csv(f"{a.out}/{tag}_scores.csv.gz", index=False)
    print(f"saved per-time forecasts -> {a.out}/{tag}_scores.csv.gz")
    if a.replay:
        export_replay(df, ev, tt, test_k, F, thr6, tag, f"{a.out}/readyation_replay.json")
    pd.set_option("display.width", 250, "display.float_format", lambda v: f"{v:.3f}")
    print(R.drop(columns=["thr"]).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s" + ("   [SMOKE TEST: random weights, numbers meaningless]" if a.smoke else ""))


if __name__ == "__main__":
    main()
