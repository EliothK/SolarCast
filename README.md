## Solar GHI Forecasting Pipeline

SolarCast forecasts Global Horizontal Irradiance (GHI) 6 to 48 hours ahead at a site, plus daily totals 1 to 48 weeks ahead.

The short-range models (XGBoost and an LSTM, combined in a weighted ensemble) take Open-Meteo weather forecasts and the exact sun position at the target hour, and learn to correct the forecast against NSRDB satellite measurements.
They predict the clear-sky index (GHI divided by clear-sky GHI), which removes the daily and seasonal cycle the models would otherwise spend their capacity on.
`predict_today.py` then fetches the live Open-Meteo forecast and prints a GHI forecast for every horizon.

____

## Results (Bismarck, ND)

Test RMSE in W/m^2 on held-out 2024 data, all hours including night (full table with MAE, MSE, R^2, MAPE and bias in `outputs/metrics_short_range.csv`):

| Horizon | Ensemble | Open-Meteo forecast alone | Old model, as run live | Old model, NSRDB inputs |
|---|---|---|---|---|
| 6 h | **59.1** | 77.9 | 98.0 | 90.4 |
| 12 h | **59.4** | 77.9 | 102.6 | 95.8 |
| 24 h | **58.0** | 60.5 | 82.5 | 75.1 |
| 48 h | **59.2** | 68.7 | 80.4 | 75.2 |

- "Old model" is the pre-refactor XGBoost (24 h of NSRDB history, no forecast input), retrained on the same years. "As run live" feeds it Open-Meteo data, which is what the old `predict_today.py` did; "NSRDB inputs" is what the old notebooks reported but cannot be had in real time.
- 24/48 h rows are scored on forecasts really issued 1/2 days earlier (Open-Meteo previous runs, Aug to Dec 2024), so they are not directly comparable with the 6/12 h rows (Dec 2023 to Dec 2024).
- 6/12 h forecasts come from Open-Meteo's historical-forecast archive, which uses the newest model run for each hour; live 6/12 h forecasts can be a few hours older, so expect somewhat higher live error.
- The daily and long-range models (7 to 336 days) do no better than a day-of-year climatology baseline; see `outputs/metrics_long_range.csv`.

____

## How it works

- **Inputs:** Open-Meteo hourly data (GHI, DNI, DHI, cloud cover by layer, temperature, humidity, dew point, wind). Training uses Open-Meteo's historical-forecast archive (2018 on) and, for 24/48 h, its previous-runs archive (2024 on), so the models see the same kind of input in training and live.
- **Truth:** NSRDB hourly GHI, used only as the training target.
- **Timestamps:** everything is keyed on UTC hour-ending labels. Open-Meteo stamps each hour's mean at the end of the hour; NSRDB stamps it at the half hour, so 10:30 maps to 11:00. Sun position is computed at mid-hour.
- **Features per sample:** the last 24 h of Open-Meteo data, the forecast for the target hour and its neighbours, and target-hour zenith, clear-sky GHI and solar time.
- **Splits:** time-ordered 70/15/15 with a 48 h gap at each boundary. For 24/48 h, the 2024 previous-runs data is split in time into extra training rows (weighted x5), a validation slice and the test slice.
- **Ensemble:** XGBoost and LSTM weighted by inverse validation MSE per horizon (weights in `artifacts/bundle.json`). Prophet is no longer used for 6 to 48 h; it scored 75% worse than XGBoost there.

____

## Requirements

### API keys

| Service | Purpose | Where to get it |
|---|---|---|
| NRL NSRDB | Historical truth data | https://developer.nlr.gov/signup/ |

Create a `.env` file in the project root:

```
NRL_API_KEY=your_nrl_api_key_here
EMAIL=your_email@example.com
```

Open-Meteo is free and needs no key.

### Software

Python 3.11 (TensorFlow compatibility).

```bash
pip install -r requirements.txt
```

> Prophet needs `cmdstan`; if `pip install prophet` fails, use `conda install -c conda-forge prophet`.

### Hardware

A CPU is enough. An NVIDIA GPU speeds up XGBoost and the LSTM; the full pipeline with tuning takes about 4 minutes on a GTX 1080 Ti once data is downloaded.

____

## Project structure

```
project_root/
├── .env                   # API keys (create this; not committed)
├── run_pipeline.py        # Download data, tune, train, evaluate
├── predict_today.py       # Live 6-48 h forecast
├── solarcast/             # All pipeline logic
│   ├── config.py          # Constants, horizons, paths, site.json
│   ├── sources.py         # Open-Meteo and NSRDB downloads (cached per year, UTC)
│   ├── solar.py           # Sun position and clear-sky GHI
│   ├── features.py        # Feature building and clear-sky index
│   ├── models.py          # XGBoost, LSTM, ensemble, saved bundle
│   ├── train.py           # Splits, training, test evaluation
│   ├── evaluate.py        # Metrics and the old-model baseline
│   ├── tune.py            # Optuna, Keras Tuner and Prophet grid searches
│   ├── longrange.py       # Daily XGBoost and Prophet models
│   ├── predict.py         # Live inference
│   └── plots.py           # Result charts
├── notebooks/
│   ├── results.ipynb      # Plots and tables from the trained models
│   └── legacy/            # The original six notebooks (superseded, kept for reference)
├── tests/                 # pytest suite (runs offline)
├── data/                  # Auto-created: cached downloads (not committed)
├── artifacts/             # Auto-created: trained models (not committed)
└── outputs/               # Auto-created: metrics CSVs and plots
```

____

## Instructions to run

**1. Set up the environment**

```bash
git clone <repo-url>
cd <project-root>
python3.11 -m venv solar_env
source solar_env/bin/activate   # macOS/Linux
# solar_env\Scripts\activate    # Windows
pip install -r requirements.txt
```

**2. Add your API credentials** (see `.env` above)

**3. Run the pipeline**

```bash
python run_pipeline.py                                   # Bismarck, ND
python run_pipeline.py --tune                            # also search hyperparameters for every model (saved and reused)
python run_pipeline.py --lat 39.7392 --lon -104.9903     # Denver, CO > locations/39p7392_W104p9903/
```

| Argument | Default | Description |
|---|---|---|
| `--lat` / `--lon` | Bismarck, ND | Site coordinates |
| `--years` | `2018-2024` | Years for the 6-48 h models; include 2024 so 24/48 h get real day-ahead forecasts |
| `--daily-years` | `2015-2024` | NSRDB years for the daily models |
| `--tune` | off | Search hyperparameters before training (see below) |
| `--trials` | `30` | Optuna trials for each XGBoost search |
| `--lstm-trials` | `10` | Keras Tuner trials for the LSTM (about 1 min each on a GPU) |
| `--skip-lstm` | off | XGBoost only (faster, no TensorFlow) |
| `--skip-long-range` | off | Skip the daily models |
| `--output-dir` | project root or `locations/<site>/` | Where `data/`, `artifacts/` and `outputs/` go |

`--tune` runs four searches, each scored on the validation split, and saves the winners to `artifacts/<name>_params.json`; later runs without `--tune` reuse them:

| Model | Search | Saved as | Trial log |
|---|---|---|---|
| XGBoost 6-48 h | Optuna (TPE) | `xgb_params.json` | `outputs/tuning_xgb_trials.csv` |
| LSTM 6-48 h | Keras Tuner random search (layer sizes, dropout, dense units, learning rate, batch size) | `lstm_params.json` | `outputs/tuning_lstm_trials.csv` |
| XGBoost 7/14 d | Optuna (TPE) | `xgb_daily_params.json` | `outputs/tuning_xgb_daily_trials.csv` |
| Prophet 4-48 weeks | Grid of 32 combinations, parallel | `prophet_long_params.json` | `outputs/tuning_prophet_long_grid.csv` |

Downloads are cached in `data/raw/`, so re-runs only train. The pipeline prints the test metrics table at the end and writes `artifacts/site.json` recording which site the models belong to.

**4. Look at the results**

Open `notebooks/results.ipynb` for plots (`outputs/rmse_by_horizon.png`, `outputs/week_h6.png`, `outputs/week_h48.png`, `outputs/xgb_feature_importance_h6.png`).

**5. Run the tests**

```bash
pytest tests
```

____

### Forecast now (`predict_today.py`)

Models are site-specific: train a site with `run_pipeline.py` before forecasting it.

```bash
python predict_today.py                                  # Bismarck, every horizon, all models
python predict_today.py --model xgboost --horizon 24
python predict_today.py --lat 40.7128 --lon -74.0060     # after: python run_pipeline.py --lat 40.7128 --lon -74.0060
```

| Argument | Default | Description |
|---|---|---|
| `--lat` / `--lon` | Bismarck, ND | Site coordinates |
| `--model` | `all` | `xgboost`, `lstm`, or `all` (adds the ensemble) |
| `--horizon` | every horizon | `6`, `12`, `24` or `48` hours ahead |
| `--artifacts-dir` | auto | Defaults to `artifacts/` for Bismarck and `locations/<site>/artifacts/` for other sites |
| `--allow-site-mismatch` | off | Run even if the models were trained for a different site (refused by default) |

**Example output:**

```
Solar GHI Forecast - 2026-09-24
Location: lat=46.69115, lon=-100.83192
Artifacts: artifacts

Fetching weather data from Open-Meteo
Latest complete hour: 2026-09-24 19:00 CDT

Horizon  Target time                         XGBoost                 LSTM             Ensemble  Open-Meteo forecast            Clear sky
+6       2026-09-25 01:00      night (GHI = 0 W/m^2)
+12      2026-09-25 07:00      night (GHI = 0 W/m^2)
+24      2026-09-25 19:00                 73.2 W/m^2           55.1 W/m^2           64.6 W/m^2           90.0 W/m^2          109.2 W/m^2
+48      2026-09-26 19:00                111.6 W/m^2          115.3 W/m^2          113.3 W/m^2          124.0 W/m^2          103.6 W/m^2
```

Values are the mean GHI over the hour ending at the target time, in the site's local time.

____

## Troubleshooting

**NSRDB returns HTTP 400 "Data processing failure" or 429/503:** the API is rate-limited (one request per 10 s) and sometimes down. Downloads are cached per year in `data/raw/`, so re-running resumes. If an old local-time `data/nsrdb_raw.csv` from the legacy notebooks exists, it is converted to UTC automatically instead.

**LSTM crashes with a cuDNN `DoRnnForward` error:** the LSTM is built with `use_cudnn=False` for this reason; if you add LSTM layers, keep that setting.

**TensorFlow not detecting the GPU:** make sure `nvidia-smi` shows it and follow the [TensorFlow GPU guide](https://www.tensorflow.org/install/pip#linux_1). Everything also runs on CPU.

**Forecast for a new city refuses to run:** train that site first with `python run_pipeline.py --lat <lat> --lon <lon>`.
