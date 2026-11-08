#!/usr/bin/env python3
"""
Rebuild nasa_omni_solar_wind_storms_5min_2010-2026.csv from NASA's raw OMNI 5-minute files.

NOTE: reconstructed after the fact. The original download/cleaning steps were not saved. This script
follows NASA's published OMNI high-resolution format; run it with --compare to check the result
against the team's original CSV before relying on it.

Source: NASA SPDF, OMNI high-resolution 5-min ASCII files, one per year:
        https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min<YEAR>.asc
        Format: https://omniweb.gsfc.nasa.gov/html/omni_min_data.html  (49 words per 5-min record)

What it does
  - downloads each year (cached in data/raw/), splits each record into its 49 words
  - keeps 13 quantities and renames them (table below)
  - turns NASA's fill values (9999.99, 99999.9, ...) into blanks
  - keeps NASA's own number formatting (e.g. "24946." and "0.60"), as the original CSV did
  - writes data/omni_5min_rebuilt.csv (it never overwrites the original CSV)

Usage
  python data/build_omni_csv.py                                  # 2010 through the current year -> data/omni_5min_rebuilt.csv
  python data/build_omni_csv.py --compare nasa_omni_solar_wind_storms_5min_2010-2026.csv
The rebuilt file can be passed to every script in place of the original CSV.
"""
import argparse, os, sys, urllib.request
from datetime import datetime, timedelta

URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_5min{year}.asc"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "omni_5min_rebuilt.csv")      # never the original file name

# (output column, word number in NASA's format (1-based), fill value)
COLUMNS = [
    ("bt_nT",             14, 9999.99),    # field magnitude average, nT
    ("bz_gsm_nT",         19, 9999.99),    # Bz, GSM
    ("by_gsm_nT",         18, 9999.99),    # By, GSM
    ("speed_km_s",        22, 99999.9),    # flow speed
    ("density_cm3",       26, 999.99),     # proton density
    ("temp_K",            27, 9999999.),   # temperature
    ("pressure_nPa",      28, 99.99),      # flow pressure
    ("efield_mV_m",       29, 999.99),     # electric field
    ("ae_nT",             38, 99999),      # AE index
    ("al_nT",             39, 99999),      # AL index
    ("sym_h_nT",          42, 99999),      # SYM-H index
    ("pcn",               45, 999.99),     # polar cap (north) index
    ("proton_flux_10mev", 47, 99999.99),   # GOES proton flux > 10 MeV, 1/(cm2 sr s)
]
NWORDS = 49


def fetch(year, cache):
    path = os.path.join(cache, f"omni_5min{year}.asc")
    if not os.path.exists(path):
        print(f"downloading {year} ...", flush=True)
        req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "omni-csv-builder"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def rows(path):
    with open(path) as f:
        for n, line in enumerate(f, 1):
            w = line.split()
            if not w:
                continue
            if len(w) != NWORDS:
                raise SystemExit(f"{path} line {n}: expected {NWORDS} words, found {len(w)}. "
                                 "NASA's format may have changed; check the format page in this script's header.")
            yr, doy, hr, mn = (int(x) for x in w[:4])
            t = datetime(yr, 1, 1) + timedelta(days=doy - 1, hours=hr, minutes=mn)
            vals = []
            for _, word, fill in COLUMNS:
                tok = w[word - 1]
                vals.append("" if abs(float(tok) - fill) < 1e-6 else tok)
            yield t.strftime("%Y-%m-%d %H:%M"), vals


def build(years, cache, out):
    os.makedirs(cache, exist_ok=True)
    n = 0
    with open(out, "w", newline="") as f:
        f.write("datetime_utc," + ",".join(c for c, _, _ in COLUMNS) + "\n")
        for y in years:
            for t, vals in rows(fetch(y, cache)):
                f.write(t + "," + ",".join(vals) + "\n"); n += 1
    print(f"wrote {out}  ({n:,} rows)")


def compare(new, old):
    """compare two CSVs over the times they share: same blanks, same values"""
    import pandas as pd
    a = pd.read_csv(new, index_col=0); b = pd.read_csv(old, index_col=0)
    common = a.index.intersection(b.index)
    print(f"rows: rebuilt {len(a):,}, original {len(b):,}, shared {len(common):,} "
          f"({b.index[0]} to {common[-1] if len(common) else '-'})")
    if list(a.columns) != list(b.columns):
        print("columns differ:", list(a.columns), "vs", list(b.columns)); return False
    a, b = a.loc[common], b.loc[common]
    ok = True
    for c in a.columns:
        blanks = (a[c].isna() != b[c].isna()).sum()
        diff = (a[c] - b[c]).abs().max()
        good = blanks == 0 and (diff == 0 or diff != diff)
        ok &= good
        print(f"  {c:18s} blank mismatches {blanks:7d}   max difference {diff:g}   {'OK' if good else 'DIFFERENT'}")
    print("MATCH: the rebuilt CSV equals the original on all shared rows" if ok else
          "Differences found: the original CSV was built differently in the columns marked DIFFERENT")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=datetime.utcnow().year)
    ap.add_argument("--cache", default=os.path.join(HERE, "raw"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--compare", metavar="ORIGINAL_CSV", help="check the rebuilt CSV against the original")
    a = ap.parse_args()
    out = a.out
    if a.compare and os.path.abspath(a.compare) == os.path.abspath(out):
        raise SystemExit("--out and --compare point to the same file; choose a different --out")
    build(range(a.start, a.end + 1), a.cache, out)
    if a.compare:
        sys.exit(0 if compare(out, a.compare) else 1)


if __name__ == "__main__":
    main()
