import argparse
import sys
import numpy as np
import pandas as pd
import requests
import joblib

from datetime import date
from pathlib import Path

# All preprocessing and inference helpers come from utils.py, which was extracted from 2_preprocessing.ipynb, 3_modeling.ipynb, and 4_evaluation.ipynb
from utils import (FEATURE_ORDER, PROPHET_REGRESSORS, WINDOW_SIZE, engineer_features, build_latest_window, inverse_ghi,)

# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict today's GHI at any location using the trained models.")
    p.add_argument("--lat",  type=float, default=46.69115, help="Site latitude  (default: 46.69115 - Bismarck, ND)")
    p.add_argument("--lon",  type=float, default=-100.83192, help="Site longitude (default: -100.83192 - Bismarck, ND)")
    p.add_argument("--model", choices=["lstm", "xgboost", "prophet", "all"], default="xgboost", help="Which model(s) to run (default: xgboost)")
    p.add_argument("--horizon", type=int, choices=[6, 12], default=6, help="Forecast horizon in hours ahead (default: 6)")
    p.add_argument("--artifacts-dir", default="artifacts", help="Directory containing trained models and scalers (default: artifacts/)")
    p.add_argument("--albedo", type=float, default=0.20, help="Surface albedo 0–1 (default: 0.20). Open-Meteo does not provide this; supply a site-specific value for better accuracy.")
    return p.parse_args()

# OPEN-METEO DATA FETCH
# Variable mapping mirrors the NSRDB attributes requested in 1_data_acq.ipynb
OPEN_METEO_VARS = (
    "shortwave_radiation," # > ghi
    "direct_normal_irradiance," # > dni
    "diffuse_radiation," # > dhi
    "temperature_2m," # > temperature
    "relative_humidity_2m," # > relative_humidity
    "dew_point_2m," # > dew_point
    "wind_speed_10m," # > wind_speed  (m/s matches NSRDB)
    "wind_direction_10m" # > wind_direction
)

RENAME_MAP = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "dew_point_2m": "dew_point",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}

def fetch_open_meteo(lat: float, lon: float) -> pd.DataFrame:
    #Fetch the last 2 days + today from Open-Meteo (free, no API key).
    #Returns a DataFrame with the same column names as 1_data_acq.ipynb produces after 2_preprocessing.ipynb standardises them.
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": OPEN_METEO_VARS,
            "wind_speed_unit": "ms",    # m/s - matches NSRDB
            "timezone": "auto",
            "past_days": 2, # ensure a full 24-h window is available
            "forecast_days": 2,
        },  
        timeout=30,
    )
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "datetime"
    return df.rename(columns=RENAME_MAP)

# SOLAR ZENITH (not provided by Open-Meteo; computed here)
def add_solar_zenith(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    # Compute apparent solar zenith angle and attach it as 'solar_zenith_angle'. 2_preprocessing.ipynb receives this column from NSRDB directly
    # We must derive it here from the site coordinates. Uses pvlib when available; falls back to a simplified astronomical formula.
    try:
        import pvlib
        loc = pvlib.location.Location(latitude=lat, longitude=lon)
        sol_pos = loc.get_solarposition(df.index)
        df["solar_zenith_angle"] = sol_pos["apparent_zenith"].values
        print("Solar zenith: computed via pvlib")
    except ImportError:
        print("pvlib not installed - using approximate zenith formula (run 'pip install pvlib' for better accuracy).")
        lat_r = np.radians(lat)
        doy = df.index.dayofyear.values
        decl = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
        hour_frac = df.index.hour.values + df.index.minute.values / 60.0
        hour_angle = np.radians((hour_frac - 12) * 15)
        cos_z = (np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle))
        df["solar_zenith_angle"] = np.degrees(np.arccos(np.clip(cos_z, -1.0, 1.0)))
    return df

# MODEL INFERENCE
# Each function loads the artifact saved by the corresponding notebook, runs prediction, and inverse-transforms via inverse_ghi() from utils.py.
def predict_xgboost(X: np.ndarray, horizon: int, arts: Path, scaler) -> float:
    #Load xgb_tuned_models.pkl (6_hyperparameter_tuning.ipynb) or xgb_models.pkl (3_modeling.ipynb) and return a W/m^2 prediction.
    for fname in ("xgb_tuned_models.pkl", "xgb_models.pkl"):
        path = arts / fname
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No XGBoost model found in {arts}")

    models = joblib.load(path)
    if horizon not in models:
        raise KeyError(f"No h={horizon}h model in {path}. Available horizons: {list(models.keys())}")

    pred_scaled = models[horizon].predict(X.reshape(1, -1))
    return float(inverse_ghi(pred_scaled, scaler)[0])

def predict_lstm(X: np.ndarray, horizon: int, arts: Path, scaler) -> float:
    #Load lstm_tuned.keras (6_hyperparameter_tuning.ipynb) or lstm_best.keras (3_modeling.ipynb) and return a W/m^2 prediction.
    import tensorflow as tf

    for fname in ("lstm_tuned.keras", "lstm_best.keras", "lstm_final.keras"):
        path = arts / fname
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No LSTM model found in {arts}")

    model = tf.keras.models.load_model(str(path))
    pred_scaled = model.predict(X, verbose=0)   # shape: (1, 2) > [h=6, h=12]
    h_idx = [6, 12].index(horizon)
    return float(inverse_ghi(pred_scaled[:, h_idx], scaler)[0])

def predict_prophet(df_raw: pd.DataFrame, horizon: int, arts: Path) -> float:
    #Load prophet_tuned_h{n}.pkl (6_hyperparameter_tuning.ipynb) or prophet_h{n}.pkl (3_modeling.ipynb) and return a W/m^2 prediction.
    for fname in (f"prophet_tuned_h{horizon}.pkl", f"prophet_h{horizon}.pkl"):
        path = arts / fname
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No Prophet h={horizon}h model in {arts}")

    model = joblib.load(path)
    row = df_raw.iloc[[-1]][PROPHET_REGRESSORS].copy()
    row["ds"] = row.index
    forecast = model.predict(row[["ds"] + PROPHET_REGRESSORS])
    return float(max(0.0, forecast["yhat"].iloc[0]))

# MAIN
def main() -> None:
    args = parse_args()
    arts = Path(args.artifacts_dir)

    print(f"Solar GHI Prediction - {date.today().isoformat()}")
    print(f"Location: lat={args.lat}, lon={args.lon}")
    print(f"Horizon: +{args.horizon} hours")
    print(f"Model(s): {args.model}")
    print(f"Artifacts: {arts}\n")

    # Verify the scaler exists (proxy for "pipeline has been trained")
    scaler_path = arts / "minmax_scaler.pkl"
    if not scaler_path.exists():
        sys.exit(f"Scaler not found at {scaler_path}.\nRun the pipeline first:\npython run_pipeline.py --lat {args.lat} --lon {args.lon}")
    # Scaler was saved by 2_preprocessing.ipynb
    scaler = joblib.load(scaler_path)

    # Fetch and prepare data
    print("Fetching weather data from Open-Meteo")
    df = fetch_open_meteo(args.lat, args.lon)
    print(f"{len(df)} hourly rows ({df.index.min().strftime('%Y-%m-%d %H:%M')} > {df.index.max().strftime('%Y-%m-%d %H:%M')})")

    print("Computing solar zenith angle")
    df = add_solar_zenith(df, args.lat, args.lon)

    # engineer_features() replicates 2_preprocessing.ipynb feature engineering
    print("Engineering features (via utils.py < 2_preprocessing.ipynb)")
    df = engineer_features(df, albedo=args.albedo)

    # Trim to hours up to the current hour so no future data leaks into the window
    now = (pd.Timestamp.now(tz=df.index.tz) if df.index.tz else pd.Timestamp.now()).floor("h")
    df_past = df[df.index <= now].copy()

    # Scale using the scaler fitted in 2_preprocessing.ipynb (no re-fitting).
    df_scaled = pd.DataFrame(scaler.transform(df_past[FEATURE_ORDER]), index=df_past.index, columns=FEATURE_ORDER,)

    # build_latest_window() replicates build_sequences() from 3_modeling.ipynb
    X = build_latest_window(df_scaled, window=WINDOW_SIZE)
    target_ts = df_past.index[-1] + pd.Timedelta(hours=args.horizon)

    print(f"\nLast observed: {df_past.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"Predicting at: {target_ts.strftime('%Y-%m-%d %H:%M')} (+{args.horizon}h)\n")

    # Nighttime guard: if the sun is below the horizon at the target time, GHI is physically zero — no need to run the model.
    target_df = pd.DataFrame(index=pd.DatetimeIndex([target_ts]))
    target_df = add_solar_zenith(target_df.assign(ghi=0), args.lat, args.lon)
    target_zenith = float(target_df["solar_zenith_angle"].iloc[0])
    if target_zenith >= 90.0:
        print(f"Solar zenith at target time = {target_zenith:.1f}deg - sun is below the horizon.")
        print("GHI = 0.0 W/m^2 (nighttime - no model inference needed)")
        return

    # Inference
    models_to_run = (["lstm", "xgboost", "prophet"] if args.model == "all" else [args.model])
    results = {}

    for name in models_to_run:
        try:
            if name == "xgboost":
                pred = predict_xgboost(X, args.horizon, arts, scaler)
            elif name == "lstm":
                pred = predict_lstm(X, args.horizon, arts, scaler)
            elif name == "prophet":
                pred = predict_prophet(df_past, args.horizon, arts)
            pred = max(0.0, pred)   # GHI is physically non-negative
            results[name] = pred
            print(f"{name.upper():<12} {pred:7.1f} W/m^2")
        except Exception as exc:
            print(f"{name.upper():<12} ERROR - {exc}")

    if len(results) > 1:
        ensemble = float(np.mean(list(results.values())))
        print(f"\n{'ENSEMBLE':<12} {ensemble:7.1f} W/m^2 (mean of {len(results)} models)")

    print(f"\nNote: surface_albedo fixed at {args.albedo}. Pass --albedo <value> for a site-specific estimate.")

if __name__ == "__main__":
    main()