import numpy as np
import pandas as pd

from .config import CS_MIN, KT_MAX, WINDOW_SIZE
from .solar import clear_sky_index, solar_geometry

# A sample is issued at hour-ending label t and predicts GHI at t+h.
# Past features use only rows <= t; target features are Open-Meteo's forecast for t+h plus the (exactly known) sun position at t+h.
# Training and live inference both build features from Open-Meteo, so the models see the same kind of input in both; NSRDB is used only as the training target.

# Columns seen for each past hour
PAST_COLS = [
    "kt", "ghi", "dni", "dhi", "cloud_cover", "cloud_low", "cloud_mid", "cloud_high",
    "temperature", "relative_humidity", "dew_point", "wind_speed", "wind_sin", "wind_cos", "cos_zenith",
]

# Forecast columns taken at the target hour
FORECAST_COLS = [
    "kt", "ghi", "dni", "dhi", "cloud_cover", "cloud_low", "cloud_mid", "cloud_high",
    "temperature", "relative_humidity", "dew_point", "wind_speed",
]

# Sun position at the target hour
GEOMETRY_COLS = ["zenith", "cos_zenith", "cs_ghi", "cs_dni", "solar_hour_sin", "solar_hour_cos", "doy_sin", "doy_cos"]

# Past lags (hours before t) given to XGBoost; the LSTM gets the full window instead
XGB_LAGS = [0, 1, 2, 3, 6, 12, 23]

def prepare_hourly(om: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    # Open-Meteo frame > continuous hourly frame with solar geometry and clear-sky index
    full = pd.date_range(om.index.min(), om.index.max(), freq="h", name="datetime")
    df = om.reindex(full).interpolate(method="time", limit=3)
    df = df.join(solar_geometry(df.index, lat, lon))
    df["kt"] = clear_sky_index(df["ghi"], df["cs_ghi"], CS_MIN, KT_MAX)
    df.loc[df["ghi"].isna(), "kt"] = np.nan
    wind = np.radians(df["wind_direction"])
    df["wind_sin"], df["wind_cos"] = np.sin(wind), np.cos(wind)
    return df

def forecast_frame(forecast_raw: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    # Put an alternative Open-Meteo forecast (e.g. previous runs) on hourly's index, with its own clear-sky index
    fc = forecast_raw.reindex(hourly.index)[[c for c in forecast_raw.columns if c in hourly.columns]].copy()
    fc["kt"] = clear_sky_index(fc["ghi"], hourly["cs_ghi"], CS_MIN, KT_MAX)
    fc.loc[fc["ghi"].isna(), "kt"] = np.nan
    return fc

def target_features(hourly: pd.DataFrame, horizon: int, forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    # Features describing the target hour t+h, indexed by issue time t.
    # forecast: optional frame (same index as hourly) holding forecasts issued further ahead, e.g. Open-Meteo previous runs.
    fc = hourly if forecast is None else forecast
    fut = hourly.shift(-horizon)
    fc_fut = fc.shift(-horizon)
    parts = [
        fc_fut[FORECAST_COLS].add_prefix("fc_"),
        fut[GEOMETRY_COLS].add_prefix("tgt_"),
        pd.DataFrame({
            # Neighbouring forecast hours soften timing errors in the weather model
            "fc_kt_prev": fc["kt"].shift(-horizon + 1),
            "fc_kt_next": fc["kt"].shift(-horizon - 1),
            "fc_cloud_mean3": fc["cloud_cover"].rolling(3, center=True).mean().shift(-horizon),
            "horizon": float(horizon),
        }, index=hourly.index),
    ]
    return pd.concat(parts, axis=1)

def past_lag_features(hourly: pd.DataFrame) -> pd.DataFrame:
    parts = [hourly[PAST_COLS].shift(lag).add_suffix(f"_lag{lag}") for lag in XGB_LAGS]
    parts.append(pd.DataFrame({
        "kt_mean24": hourly["kt"].rolling(WINDOW_SIZE, min_periods=1).mean(),
        "cloud_mean24": hourly["cloud_cover"].rolling(WINDOW_SIZE, min_periods=1).mean(),
    }, index=hourly.index))
    return pd.concat(parts, axis=1)

def build_tabular(hourly: pd.DataFrame, horizon: int, forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    # One row per issue time: past lags + target-hour forecast and geometry (XGBoost input)
    return past_lag_features(hourly).join(target_features(hourly, horizon, forecast))

def target_kt(hourly: pd.DataFrame, truth_ghi: pd.Series, horizon: int) -> pd.DataFrame:
    # Training target: true clear-sky index at t+h, indexed by issue time t.
    # 'usable' marks samples where the sun is high enough for kt to be learned.
    truth = truth_ghi.reindex(hourly.index)
    cs = hourly["cs_ghi"]
    out = pd.DataFrame(index=hourly.index)
    out["truth_ghi"] = truth.shift(-horizon)
    out["cs_ghi"] = cs.shift(-horizon)
    out["kt"] = clear_sky_index(out["truth_ghi"], out["cs_ghi"], CS_MIN, KT_MAX)
    out.loc[out["truth_ghi"].isna(), "kt"] = np.nan
    out["usable"] = (out["cs_ghi"] > CS_MIN) & out["kt"].notna()
    return out

def sequence_windows(values: np.ndarray, issue_pos: np.ndarray, window: int = WINDOW_SIZE) -> np.ndarray:
    # values: (hours, features); issue_pos: row positions of issue times.
    # Returns (samples, window, features) holding rows issue-window+1 .. issue.
    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)  # (hours-window+1, features, window)
    return windows[issue_pos - window + 1].transpose(0, 2, 1)

def kt_to_ghi(kt_pred: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    # Clear-sky index prediction > GHI (W/m^2).
    # target must hold tgt_cs_ghi and fc_ghi for the same rows.
    cs = target["tgt_cs_ghi"].to_numpy()
    fc = np.clip(target["fc_ghi"].to_numpy(), 0, None)
    ghi = np.clip(kt_pred, 0, KT_MAX) * cs
    # Sun too low for kt: fall back to the raw forecast, capped at clear sky
    low = cs <= CS_MIN
    ghi[low] = np.minimum(np.nan_to_num(fc[low]), cs[low])
    return ghi
