
## Solar GHI Forecasting Pipeline

This project trains machine learning models (LSTM, XGBooost, and Prophet) to forecast Global Hoirzonltal Irradiance (GHI) at any geographic location using 10 years of historical solar and meteorological data from NRL's National Solar Radiation Database (NSRDB). Once trained, the pipeline runs a prediction script ("predict_today.py") that fetches live weather data from Open-Meteo and outputs a GHI forecast for any horizon (6h or 12h ahead).

____

## Requirements

### API Keys
Service | Purpose | Where to get it
NRL NSRDB | Historical solar data (notebook 1) | https://developer.nlr.gov/signup/

Create a '.env' file in the project root before running anything:
```
NRL_API_KEY=your_nrl_api_key_here
EMAIL=your_email@example.com
```

> `predict_today.py` fetches live forecasts from Open-Meteo, which is free and requires no API key.

____

### Hardware
Component | Minium | Recommended |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU training works) | NVIDIA GPU with CUDA 11.8+ (speeds up LSTM training significantly) |
| Disk | 2 GB free | 5 GB free| 

> Hyperparameter tuning (notebook 6) with Optuna (50 XGBoost trials) and Keras Tuner (20 LSTM trials) is the most resource intensive step. On CPU alone, expect 2-4 hours. A GPU reduces LSTM tuning to under 30 minutes. This step can be skipped with `--skip-tuning`.

____

### Software

**Python version:** 3.11 (required - TensorFlow/Keras compatibility)

**Package manager:** `pip` or `conda`

**Core dependencies:**
| Package | Purpose |
| `tensorflow >= 2.13` | LSTM model training and inference |
| `xgboost >= 2.0` | XGBoost model training and inference |
| `prophet >= 1.1` | Facebook Prophet time-series model |
| `scikit-learn >= 1.3` | Preprocessing, metrics, cross-validation |
| `pandas >= 2.0` | Data manipulation |
| `numpy >= 1.24` | Numerical operations |
| `requests >= 2.31` | API calls (NSRDB, Open-Meteo) |
| `joblib >= 1.3` | Model serialization |
| `python-dotenv >= 1.0` | Loading `.env` credentials |
| `nbformat >= 5.9` | Notebook patching in `run_pipeline.py` |
| `jupyter >= 1.0` | Notebook execution via `nbconvert` |
| `matplotlib >= 3.7` | Evaluation and CV plots |
| `optuna >= 3.3` | XGBoost hyperparameter tuning |
| `keras-tuner >= 1.4` | LSTM hyperparameter tuning |
| `pvlib >= 0.10` | Solar zenith calculation (optional but recommended) |

**Install all dependencies:**

```bash
pip install tensorflow xgboost prophet scikit-learn pandas numpy requests joblib python-dotenv nbformat jupyter matplotlib optuna keras-tuner pvlib
```

Or use the provided requirements file if available:

```bash
pip install -r requirements.txt
```

> **Prophet note:** Prophet requires `cmdstan` and `pystan`. On some systems you may also need `conda install -c conda-forge prophet` instead of `pip install prophet`.

----

### Operating System

The pipeline is tested on:
- **Linux** (Ubuntu 20.04+) - fully supported
- **macOS** (12+) - fully supported
- **Windows 10/11** - supported; use PowerShell or WSL2 for best results

----

## Project Structure

```
project_root/
├── .env    # API keys (create this - not committed to git)
├── 1_data_acq.ipynb    # Step 1: Download NSRDB historical data
├── 2_preprocessing.ipynb   # Step 2: Clean, engineer features, scale
├── 3_modeling.ipynb    # Step 3: Train LSTM, XGBoost, Prophet
├── 4_evaluation.ipynb  # Step 4: Evaluate models on test set
├── 5_cross_validation.ipynb    # Step 5: Time-series cross-validation
├── 6_hyperparameter_tuning.ipynb # Step 6: Tune models with Optuna/KerasTuner
├── predict_today.py    # Live inference script
├── run_pipeline.py # One-command pipeline runner
├── utils.py    # Shared constants and helper functions
├── data/   # Auto-created: raw and processed CSVs
├── artifacts/  # Auto-created: saved models and scalers
└── outputs/    # Auto-created: plots and evaluation reports
```

----

## Instructions to Run

### Option A - Full pipeline via `run_pipeline.py` (recommended)

This script patches and executes all six notebooks in order for any location.

**1. Set up your environment**

```bash
# Clone the repo and navigate to the project root
git clone <repo-url>
cd <project-root>

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv solar_env
source solar_env/bin/activate   # macOS/Linux
# solar_env\Scripts\activate    # Windows

# Install dependencies
pip install tensorflow xgboost prophet scikit-learn pandas numpy requests joblib python-dotenv nbformat jupyter matplotlib optuna keras-tuner pvlib
```

**2. Add your API credentials**

```bash
# Create .env in the project root
echo "NRL_API_KEY=your_key_here" > .env
echo "EMAIL=your_email@example.com" >> .env
```

**3. Run the pipeline**

Default location (Bismarck, ND - the pre-configured site):

```bash
python run_pipeline.py
```

Custom location (any lat/lon):

```bash
python run_pipeline.py --lat 39.7392 --lon -104.9903  #Denver, CO
```

Skip data acquisition if `data/nsrdb_raw.csv` is already downloaded:

```bash
python run_pipeline.py --skip-data-acq
```

Skip hyperparameter tuning to save several hours of compute:

```bash
python run_pipeline.py --skip-tuning
```

All options combined:

```bash
python run_pipeline.py \
  --lat 34.0522 --lon -118.2437 \
  --years 2018 2019 2020 2021 2022 2023 2024 \
  --skip-data-acq \
  --skip-tuning \
  --output-dir /path/to/custom/output
```

**`run_pipeline.py` arguments:**

| Argument | Default | Description |
|---|---|---|
| `--lat` | `46.69115` | Site latitude |
| `--lon` | `-100.83192` | Site longitude |
| `--years` | `2015–2024` | Training years (space-separated) |
| `--skip-data-acq` | off | Skip notebook 1 if raw data already exists |
| `--skip-tuning` | off | Skip notebook 6 (saves 2–4 hours) |
| `--output-dir` | project root | Override output path for data/, artifacts/, outputs/ |

----

### Option B - Run notebooks manually (step by step)

Open and run each notebook in order in Jupyter Lab or VS Code:

```bash
jupyter lab
```

| Notebook | What it does | Key outputs |
|---|---|---|
| `1_data_acq.ipynb` | Downloads hourly NSRDB data (2015–2024) via NRL API | `data/nsrdb_raw.csv` |
| `2_preprocessing.ipynb` | Cleans data, engineers cyclical features, fits and applies MinMaxScaler, splits train/val/test | `data/nsrdb_preprocessed.csv`, `data/training_scaled.csv`, `data/val_scaled.csv`, `data/test_scaled.csv`, `artifacts/minmax_scaler.pkl` |
| `3_modeling.ipynb` | Trains LSTM (short-horizon), XGBoost (short/medium/long horizons), and Prophet models | `artifacts/lstm_best.keras`, `artifacts/xgb_models.pkl`, `artifacts/prophet_h6.pkl`, `artifacts/prophet_h12.pkl` |
| `4_evaluation.ipynb` | Computes MAE, RMSE, R^2, MAPE on the held-out test set; produces residual and actual-vs-predicted plots | `outputs/` (evaluation charts) |
| `5_cross_validation.ipynb` | Time-series cross-validation (5-fold) for XGBoost and LSTM | `outputs/` (CV results) |
| `6_hyperparameter_tuning.ipynb` | Optuna (XGBoost), Keras Tuner (LSTM), grid search (Prophet) | `artifacts/xgb_tuned_models.pkl`, `artifacts/lstm_tuned.keras`, `artifacts/prophet_tuned_h*.pkl` |

> Each notebook has a **CONFIGURATION cell at the top** (cell 0). If you change the data or output paths in notebook 1, update the matching paths in notebooks 2–6 to stay consistent.

----

### Predict today's GHI (`predict_today.py`)

After the pipeline completes and artifacts are saved, run live inference for any location. No NRL API key is needed - live weather is fetched from Open-Meteo for free.

**Basic usage (Bismarck, ND default):**

```bash
python predict_today.py
```

**Custom location and model:**

```bash
python predict_today.py --lat 40.7128 --lon -74.0060 --model xgboost --horizon 6
```

**Run all three models and show an ensemble:**

```bash
python predict_today.py --lat 40.7128 --lon -74.0060 --model all --horizon 12
```

**`predict_today.py` arguments:**

| Argument | Default | Description |
|---|---|---|
| `--lat` | `46.69115` | Site latitude |
| `--lon` | `-100.83192` | Site longitude |
| `--model` | `xgboost` | Model to use: `xgboost`, `lstm`, `prophet`, or `all` |
| `--horizon` | `6` | Forecast horizon: `6` or `12` hours ahead |
| `--artifacts-dir` | `artifacts/` | Path to trained models and scalers |
| `--albedo` | `0.20` | Surface albedo (0–1). Open-Meteo does not provide this; supply a site-specific value for better accuracy |

**Example output:**

```
Solar GHI Prediction - 2025-06-15
Location: lat=40.7128, lon=-74.006
Horizon: +6 hours
Model(s): all

Fetching weather data from Open-Meteo
96 hourly rows (2025-06-13 00:00 > 2025-06-15 23:00)
Computing solar zenith angle
Engineering features (via utils.py < 2_preprocessing.ipynb)

Last observed: 2025-06-15 10:00
Predicting at: 2025-06-15 16:00 (+6h)

XGBOOST       523.4 W/m^2
LSTM          491.7 W/m^2
PROPHET       508.2 W/m^2

ENSEMBLE      507.8 W/m^2 (mean of 3 models)
```

> If the target forecast time falls after sunset, the script will print `GHI = 0.0 W/m^2 (nighttime)` and exit without running any model - GHI is physically zero when the sun is below the horizon.

----

## Troubleshooting

**NRL API returns a 429 or 503:** The NSRDB API rate-limits to one request per 10 seconds per key. The data acquisition notebook already includes `time.sleep(10)` between years. If errors persist, re-run notebook 1 - it will pick up where it left off if you save partial results.

**`prophet` installation fails:** Try `conda install -c conda-forge prophet` instead of pip. Prophet depends on `cmdstan`, which pip sometimes fails to build from source on Windows.

**TensorFlow not detecting GPU:** Ensure you have CUDA 11.8+ and cuDNN 8.6+ installed and that `nvidia-smi` shows your GPU. See the [TensorFlow GPU guide](https://www.tensorflow.org/install/pip#linux_1).

**`pvlib` not installed:** The `predict_today.py` script will fall back to an approximate astronomical formula for solar zenith angle. Install `pvlib` for better accuracy: `pip install pvlib`.

**Pipeline halts at a notebook:** The error will be printed to the console. Fix the issue in the notebook directly, then re-run `run_pipeline.py` with `--skip-data-acq` to avoid re-downloading data.
