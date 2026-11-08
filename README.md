# READYation

**Early warning for solar radiation storms, using IBM Granite time-series forecasting.**
Sheridan Datathon 2026 · NASA track · IBM tools

Astronauts are put at risk by solar **radiation** (energetic protons), not by geomagnetic storms. NOAA declares a solar radiation storm once ≥10 MeV proton flux reaches 10 pfu (S1). READYation forecasts that moment, so a crew gets a **watch** while the flux is still rising.

READYation is **IBM Granite TinyTimeMixer r2** (`ibm-granite/granite-timeseries-ttm-r2`), fine-tuned on the GOES proton record from NASA's OMNI dataset. It reads the last 5.3 days of flux in 15-minute steps and forecasts the next 24 hours.

## Results

Trained on 2010–2013, tested on 2014 – March 2020 (data the model never saw, 14 radiation storms). Every model was set to the same alarm rule: on alert during 1% of quiet time. Numbers are from the run in `results/` (laptop CPU).

| 6-hour horizon | Storms caught | Median warning | False alarms (6 yrs) |
|---|---|---|---|
| **READYation (Granite, fine-tuned)** | 14 / 14 | 5.0 h | **41** |
| Granite, not fine-tuned | 14 / 14 | 5.5 h | 42 |
| Persistence (current flux) | 14 / 14 | 4.8 h | 49 |
| Gradient boosting + solar wind | 14 / 14 | 3.8 h | 276 |
| Gradient boosting, flux only | 14 / 14 | 4.4 h | 431 |

**Four-tier alerts.** READYation raises only the yellow watch; orange and red follow NOAA thresholds on measured flux.

| Tier | Rule | Meaning |
|---|---|---|
| 🟢 Green | quiet | no action |
| 🟡 Yellow | READYation's 6 h forecast crosses its alarm level, or flux holds ≥1 pfu for 15 min (clears after 2 h without a trigger) | watch: storm likely within hours |
| 🟠 Orange | flux ≥10 pfu (NOAA S1–S2) | storm: heightened monitoring, review EVA plans |
| 🔴 Red | flux ≥1,000 pfu (NOAA S3+) | EVA hazard (NOAA: avoidance recommended for astronauts on EVA) |

| Watch raised by | Watches | Became a storm | No storm | Median watch before storm |
|---|---|---|---|---|
| Flux ≥1 pfu only | 28 | 13 | 2.4 / yr | 2.2 h |
| **READYation + flux** | 35 | 13 | 3.6 / yr | **3.8 h** |

READYation lengthened the watch on 4 of the 13 storms, including two that flux alone caught with under an hour to spare (Dec 2014: 0.6 h → 10.6 h; Sept 2017: 0.2 h → 3.1 h). Of the 22 watches that did not become storms, 7 were near misses (3–9 pfu), 10 were smaller rises to 1–3 pfu, and 5 never reached 1 pfu.

## Repository

| File | What it does |
|---|---|
| `proton_forecast.py` | Data loading, storm labels, baselines (persistence, gradient boosting), evaluation |
| `granite_ttm.py` | READYation: Granite TTM r2 zero-shot or fine-tuned, scored on the same rows; `--replay` exports data for the console |
| `tier_alerts.py` | Four-tier alert rules, statistics and timeline chart |
| `make_figures.py` | Results charts in `figs/` |
| `proton_granite_colab.ipynb` | Google Colab version of the pipeline |
| `web/radiation_comes_first.html` | Project report (Part 1 analysis + READYation) |
| `web/readyation_console.html` | Replay console: real storms and no-storm watches through READYation |
| `web/console_template.html`, `web/build_console.py` | Rebuild the console from a new `readyation_replay.json` |
| `data/build_omni_csv.py` | Rebuild the OMNI CSV from NASA's raw 5-minute files |
| `results/` | Result tables from the run behind the README: baselines, Granite (zero-shot and fine-tuned), per-storm lead times, tier episodes, Part 1 storm list |
| `requirements.txt` | Python packages |
| `storm_lag.py` | Part 1: are radiation storms followed by geomagnetic storms? Reproduces the report's Part 1 numbers |
| `genes/process_genes.py` | Part 1 gene panel: extracts the 7 radiation-response genes from GeneLab OSD-118 |
| `genes/data/` | GeneLab expression table (`all_genes.csv`) and the processed panel (`processed_gene_dataset.csv`) |
| `figs/` | Charts used in the report |

## Data

The input is NASA OMNI 5-minute data, 2010–2026 (`nasa_omni_solar_wind_storms_5min_2010-2026.csv`, 139 MB). It is **not in this repository** because GitHub rejects files over 100 MB. Rebuild it from NASA's raw files (needs internet; about 17 yearly downloads):

```
python data/build_omni_csv.py                     # writes data/omni_5min_rebuilt.csv; use it as %CSV%
python data/build_omni_csv.py --compare nasa_omni_solar_wind_storms_5min_2010-2026.csv   # check against the original
```

The original download steps were not saved; `build_omni_csv.py` was reconstructed from NASA's published OMNI format. Run `--compare` against the team's original file to confirm they match.

- Proton flux (≥10 MeV) comes from NOAA GOES satellites in geostationary orbit, and is not time-shifted.
- Solar wind comes from ACE and Wind at L1, time-shifted to Earth's bow shock.
- Proton flux in OMNI ends on 4 March 2020, so all results cover 2010–2020.

## Radiation before geomagnetic storms (Part 1)

`storm_lag.py` finds the 33 radiation storms in 2010 – Mar 2020 (≥10 pfu, at least 4 days apart) and checks how often a geomagnetic storm (daily minimum SYM-H ≤ −100 nT strong, ≤ −50 nT moderate) follows within 2–5 days, compared with 3,000 random days. It prints each result next to the report's value.

```
python storm_lag.py %CSV%
```

| Result | Value |
|---|---|
| Strong storm within 4 days of a radiation storm | 8 of 33 (24%, 95% CI 13–41%) vs 3.6% of all days: about 6× |
| Radiation storms with no strong storm within 4 days | 25 of 33 |
| Median head start of the radiation | 2 days |
| Same-day correlation (daily max flux vs storm strength) | r = 0.11 |

Like the gene script, this was **reconstructed** after the original was lost; it reproduces every Part 1 number in the report.

## Gene panel (Part 1)

`genes/process_genes.py` extracts seven radiation-response genes (TP53, CDKN1A, ATM, IL6, ICAM1, SOD2, BAX) from NASA GeneLab **OSD-118**: Illumina BeadChip signal for 47,322 probes in 16 human fibroblast samples. It writes `genes/data/processed_gene_dataset.csv` and prints each gene's strongest-probe mean signal, the values shown in the report.

```
python genes/process_genes.py --check     # rebuild and compare with the committed file
```

The original processing script was lost. This one was **reconstructed** afterwards and reproduces the committed file byte for byte, including every value in the report. The dataset does not say which samples flew, so no flight-vs-ground comparison is made.

## Run it

Requires Python 3.11–3.13 (granite-tsfm does not support 3.14 yet). On Windows:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set CSV=nasa_omni_solar_wind_storms_5min_2010-2026.csv

python proton_forecast.py %CSV% out                     # baselines (~2 min)
python granite_ttm.py %CSV% out                         # READYation zero-shot
python granite_ttm.py %CSV% out --finetune --replay     # READYation fine-tuned + console data
python tier_alerts.py %CSV%                             # tiers, flux trigger only
python tier_alerts.py %CSV% --scores out\granite_ttm_finetuned_scores.csv.gz   # tiers with READYation
```

To rebuild the replay console after a new run: `python web/build_console.py out\readyation_replay.json`.

On macOS/Linux use `source venv/bin/activate` and `$CSV`. The first Granite run downloads the model weights from Hugging Face. Everything runs on CPU; a GPU is used automatically if available.

Outputs go to `out/` (CSV results, per-event lead times, forecasts, `readyation_replay.json`) and `figs/` (charts).

## Limitations

- **Fine-tuning varies between machines.** Granite without fine-tuning gives identical results everywhere, but fine-tuning does not: the run in `results/` (laptop CPU) gives 41 false alarms at 6 h, while an earlier Colab GPU run gave 31. Both beat persistence (49); fine-tuning's extra benefit over the untuned model (42) is not established.
- **Small sample.** 14 test storms; differences of a few alarms over six years are within run-to-run variation.
- **Fast storms give little warning.** Storms from flares well connected to Earth rise within minutes. Earlier warning needs flare X-ray and CME inputs (GOES XRS, NASA DONKI), not proton flux alone.
- **Flux, not dose.** READYation forecasts flux at geostationary orbit. Dose inside a spacecraft depends on shielding and on higher-energy protons; NASA bases Artemis shelter decisions on onboard dose.
- **Near-Earth only.** GOES measures near Earth, a reasonable proxy for cislunar space but not for Mars transit.
- **One solar cycle.** 2010–2020 (Cycle 24) was a relatively weak cycle. Newer GOES-16/18 sensors differ from GOES-13 and would need recalibration before live use.

Datathon project, not an operational tool.

## Credits

- Data: NASA OMNI (omniweb.gsfc.nasa.gov); GOES proton flux from NOAA.
- Model: IBM Granite TinyTimeMixer r2 via [granite-tsfm](https://github.com/ibm-granite/granite-tsfm).
- Radiation levels and wording: NOAA Space Weather Scales.
