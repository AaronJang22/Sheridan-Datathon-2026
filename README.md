# READYation

**Early warning for solar radiation storms, using IBM Granite time-series forecasting.**
Sheridan Datathon 2026 · NASA track · IBM tools

Astronauts are put at risk by solar **radiation** (energetic protons), not by geomagnetic storms. NOAA declares a solar radiation storm once ≥10 MeV proton flux reaches 10 pfu (S1). READYation forecasts that moment, so a crew gets a **watch** while the flux is still rising.

READYation is **IBM Granite TinyTimeMixer r2** (`ibm-granite/granite-timeseries-ttm-r2`), fine-tuned on the GOES proton record from NASA's OMNI dataset. It reads the last 5.3 days of flux in 15-minute steps and forecasts the next 24 hours.

## Results

Trained on 2010–2013, tested on 2014 – March 2020 (data the model never saw, 14 radiation storms). Every model was set to the same alarm rule: on alert during 1% of quiet time.

| 6-hour horizon | Storms caught | Median warning | False alarms (6 yrs) |
|---|---|---|---|
| **READYation (Granite, fine-tuned)** | 14 / 14 | 5.0 h | **31** |
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
| `requirements.txt` | Python packages |
| `figs/` | Charts used in the report |

## Data

The input is NASA OMNI 5-minute data, 2010–2026 (`nasa_omni_solar_wind_storms_5min_2010-2026.csv`, 139 MB). It is **not in this repository** because GitHub rejects files over 100 MB. Download 5-minute (HRO) data from [OMNIWeb](https://omniweb.gsfc.nasa.gov/form/omni_min_def.html), or ask the team for the file.

- Proton flux (≥10 MeV) comes from NOAA GOES satellites in geostationary orbit, and is not time-shifted.
- Solar wind comes from ACE and Wind at L1, time-shifted to Earth's bow shock.
- Proton flux in OMNI ends on 4 March 2020, so all results cover 2010–2020.

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

On macOS/Linux use `source venv/bin/activate` and `$CSV`. The first Granite run downloads the model weights from Hugging Face. Everything runs on CPU; a GPU is used automatically if available.

Outputs go to `out/` (CSV results, per-event lead times, forecasts, `readyation_replay.json`) and `figs/` (charts).

## Limitations

- **Small sample.** 14 test storms and one training run. The 37% cut in false alarms is about 18 alarms over six years and should be confirmed with more seeds and more storms.
- **Fast storms give little warning.** Storms from flares well connected to Earth rise within minutes. Earlier warning needs flare X-ray and CME inputs (GOES XRS, NASA DONKI), not proton flux alone.
- **Flux, not dose.** READYation forecasts flux at geostationary orbit. Dose inside a spacecraft depends on shielding and on higher-energy protons; NASA bases Artemis shelter decisions on onboard dose.
- **Near-Earth only.** GOES measures near Earth, a reasonable proxy for cislunar space but not for Mars transit.
- **One solar cycle.** 2010–2020 (Cycle 24) was a relatively weak cycle. Newer GOES-16/18 sensors differ from GOES-13 and would need recalibration before live use.

Datathon project, not an operational tool.

## Credits

- Data: NASA OMNI (omniweb.gsfc.nasa.gov); GOES proton flux from NOAA.
- Model: IBM Granite TinyTimeMixer r2 via [granite-tsfm](https://github.com/ibm-granite/granite-tsfm).
- Radiation levels and wording: NOAA Space Weather Scales.
