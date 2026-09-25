import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C
from .models import xgb_device
from .solar import solar_geometry

# MAPE is only defined where the truth is clearly above zero; night and twilight hours would divide by ~0
MAPE_MIN_GHI = 50.0

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    lit = y_true > MAPE_MIN_GHI
    return {
        "MAE (W/m^2)": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE (W/m^2)": float(np.sqrt(mse)),
        "R^2": r2_score(y_true, y_pred),
        f"MAPE (%, GHI>{MAPE_MIN_GHI:.0f})": float(np.mean(np.abs(y_pred[lit] - y_true[lit]) / y_true[lit]) * 100),
        "Bias (W/m^2)": float(np.mean(y_pred - y_true)),
    }

def metrics_table(truth: np.ndarray, preds: dict[str, np.ndarray], daytime: np.ndarray, horizon: int) -> pd.DataFrame:
    # One row per model: all-hours metrics (comparable with the old notebooks) plus daytime-only RMSE and skill against the raw weather forecast
    rows = []
    for name, pred in preds.items():
        row = {"Horizon": f"{horizon}h", "Model": name, **regression_metrics(truth, pred)}
        row["Daytime RMSE (W/m^2)"] = float(np.sqrt(mean_squared_error(truth[daytime], pred[daytime])))
        rows.append(row)
    df = pd.DataFrame(rows)
    if "Open-Meteo forecast" in preds:
        ref = df.loc[df["Model"] == "Open-Meteo forecast", "RMSE (W/m^2)"].iloc[0]
        df["Skill vs forecast"] = 1 - df["RMSE (W/m^2)"] / ref
    return df

# OLD RECIPE BASELINE
# Re-creates the pre-refactor short-horizon XGBoost (3.modeling.ipynb): 24 h window of 17 features, raw GHI target, no forecast input.
# It is trained on NSRDB inputs as before, then scored twice: with NSRDB inputs (what the old notebooks reported) and with Open-Meteo inputs (what predict_today.py actually fed it).
OLD_FEATURES = [
    "ghi", "dni", "dhi", "temperature", "relative_humidity", "dew_point", "wind_speed", "wind_direction", "surface_albedo",
    "solar_zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos", "is_daytime",
]

def old_recipe_frame(src: pd.DataFrame, lat: float, lon: float, from_open_meteo: bool) -> pd.DataFrame:
    df = src.copy()
    if from_open_meteo:
        # Old live code: fixed albedo, zenith computed at the hour label instead of mid-hour
        df["surface_albedo"] = 0.20
        df["solar_zenith_angle"] = solar_geometry(df.index + pd.Timedelta(minutes=30), lat, lon)["zenith"].to_numpy()
    idx = df.index
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * idx.dayofyear / 365.25), np.cos(2 * np.pi * idx.dayofyear / 365.25)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * idx.month / 12), np.cos(2 * np.pi * idx.month / 12)
    df["is_daytime"] = ((df["ghi"] > 0) & (df["solar_zenith_angle"] < 90)).astype(int)
    return df[OLD_FEATURES]

def old_recipe_windows(frame: pd.DataFrame, issue_pos: np.ndarray) -> np.ndarray:
    from .features import sequence_windows
    values = frame.to_numpy(dtype="float32")
    return sequence_windows(values, issue_pos).reshape(len(issue_pos), -1)

def train_old_recipe(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=20, eval_metric="rmse", random_state=C.RANDOM_SEED, device=xgb_device(),
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model
