import sys
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ordered feature list as produced by 2_preprocessing.ipynb after column standardisation and feature engineering.
# The MinMaxScaler in artifacts/minmax_scaler.pkl was fitted on data in exactly this order.
FEATURE_ORDER = [
    "ghi", "dni", "dhi", "temperature", "relative_humidity", "dew_point", "wind_speed", "wind_direction", "surface_albedo",
    "solar_zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos", "is_daytime",
]

# Forecast horizons (3_modeling.ipynb CONFIGURATION cell)
WINDOW_SIZE = 24 # hours - used by LSTM and short horizon XGBoost
DAILY_WINDOW = 30 # days - used by medium horizon XGBoost

# Forecast horizons (3_modeling.ipynb CONFIGURATION cell)
SHORT_HORIZONS = [6, 12]    # hours
MEDIUM_HORIZONS = [7, 14]   # days
LONG_HORIZONS = [28, 56, 84, 168, 336]  # days

# Regressors passed to Prophet (3_modeling.ipynb)
PROPHET_REGRESSORS = ["temperature", "relative_humidity", "wind_speed", "solar_zenith_angle", "surface_albedo",]

# LSTM architecture (3_modeling.ipynb CONFIGURATION cell)
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2
DENSE_UNITS = 32

# Feature engineering (source: 2_preprocessing.ipynb)
def engineer_features(df: pd.DataFrame, albedo: float = 0.20) -> pd.DataFrame:
    df = df.copy()

    if "surface_albedo" not in df.columns:
        df["surface_albedo"] = albedo

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["is_daytime"] = ((df["ghi"] > 0) & (df["solar_zenith_angle"] < 90)).astype(int)

    return df

# Sliding window sequence builder (source: 3_modeling.ipynb)
def build_sequences(data: np.ndarray, window: int, horizons: list, tgt_idx: int,) -> tuple[np.ndarray, dict]:
    X, ys = [], {h: [] for h in horizons}
    max_h = max(horizons)
    for i in range(window, len(data) - max_h + 1):
        X.append(data[i - window: i, :])
        for h in horizons:
            ys[h].append(data[i + h - 1, tgt_idx])
    return np.array(X), {h: np.array(v) for h, v in ys.items()}

def build_latest_window(df_scaled: pd.DataFrame, window: int = WINDOW_SIZE) -> np.ndarray:
    missing = [c for c in FEATURE_ORDER if c not in df_scaled.columns]
    if missing:
        sys.exit(f"Missing features in scaled DataFrame: {missing}")

    arr = df_scaled[FEATURE_ORDER].values
    if len(arr) < window:
        sys.exit(f"Need at least {window} rows to build the input window; only {len(arr)} rows available.")
    return arr[-window:, :].reshape(1, window, -1)

# Inverse transform (source: 4_evaluation.ipynb - inverse_ghi/regression_metrics)
def inverse_ghi(scaled_arr: np.ndarray, scaler, feature_cols: list = FEATURE_ORDER) -> np.ndarray:
    tgt_i = feature_cols.index("ghi")
    dummy = np.zeros((len(scaled_arr), len(feature_cols)))
    dummy[:, tgt_i] = scaled_arr
    return scaler.inverse_transform(dummy)[:, tgt_i]

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mask = y_true > 1   # exclude nighttime zeros to avoid divide-by-zero
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.sum() > 0 else float("nan")
    )
    return {"MAE (W/m^2)": mae, "RMSE (W/m^2)": rmse, "R^2": r2, "MAPE (%)": mape}