from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import regression_metrics
from .features import sequence_windows
from .models import train_xgb

# Daily models ported from 3.modeling.ipynb: XGBoost for 7/14 days ahead and Prophet for 4-48 weeks ahead.
# Both use NSRDB only (there is no weather forecast that far out), aggregated to local solar days.

PROPHET_REGRESSORS = ["temperature", "relative_humidity", "wind_speed", "solar_zenith_angle", "surface_albedo"]

def daily_frame(nsrdb: pd.DataFrame, lon: float) -> pd.DataFrame:
    # Hourly NSRDB (UTC) > one row per local solar day; GHI/DNI/DHI summed to daily energy (Wh/m^2)
    df = nsrdb.copy()
    df.index = (df.index - pd.Timedelta(minutes=30) + pd.Timedelta(hours=lon / 15)).tz_localize(None)
    df["daylight_hours"] = ((df["ghi"] > 0) & (df["solar_zenith_angle"] < 90)).astype(int)
    daily = df.resample("D").agg({
        "ghi": "sum", "dni": "sum", "dhi": "sum", "temperature": "mean", "relative_humidity": "mean",
        "dew_point": "mean", "wind_speed": "mean", "wind_direction": "mean", "surface_albedo": "mean",
        "solar_zenith_angle": "mean", "daylight_hours": "sum",
    })
    # Drop partial days at either end of the record
    daily = daily.iloc[1:-1]
    doy = daily.index.dayofyear
    daily["doy_sin"], daily["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    daily["month_sin"], daily["month_cos"] = np.sin(2 * np.pi * daily.index.month / 12), np.cos(2 * np.pi * daily.index.month / 12)
    return daily.interpolate(limit=3)

@dataclass
class DailyData:
    daily: pd.DataFrame
    train_end: int
    val_end: int

    @classmethod
    def from_nsrdb(cls, nsrdb: pd.DataFrame, lon: float) -> "DailyData":
        daily = daily_frame(nsrdb, lon)
        n = len(daily)
        return cls(daily, int(n * C.TRAIN_FRAC), int(n * (C.TRAIN_FRAC + C.VAL_FRAC)))

    def xgb_split(self, h: int, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        # 30-day windows issued in [lo, hi - h), target = daily GHI h days after the window
        values = self.daily.to_numpy(dtype="float32")
        pos = np.arange(max(lo, C.DAILY_WINDOW - 1), hi - h)
        return sequence_windows(values, pos, C.DAILY_WINDOW).reshape(len(pos), -1), self.daily["ghi"].to_numpy()[pos + h]

    def prophet_frame(self, h: int) -> pd.DataFrame:
        # Target shifted h days ahead, same-day regressors
        frame = self.daily[PROPHET_REGRESSORS].copy()
        frame["y"] = self.daily["ghi"].shift(-h)
        frame["ds"] = self.daily.index
        return frame

def train_xgb_daily(dd: DailyData, h: int, params: dict | None = None):
    X_tr, y_tr = dd.xgb_split(h, 0, dd.train_end)
    X_va, y_va = dd.xgb_split(h, dd.train_end, dd.val_end)
    return train_xgb(X_tr, y_tr, X_va, y_va, params or C.XGB_DAILY_PARAMS)

def fit_prophet(train_df: pd.DataFrame, params: dict | None = None):
    import logging
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, **(params or C.PROPHET_LONG_PARAMS))
    for col in PROPHET_REGRESSORS:
        m.add_regressor(col)
    m.fit(train_df)
    return m

def prophet_predict(m, df: pd.DataFrame) -> np.ndarray:
    return np.clip(m.predict(df[["ds"] + PROPHET_REGRESSORS])["yhat"].to_numpy(), 0, None)

def train_long_range(nsrdb: pd.DataFrame, lon: float, artifacts, outputs, xgb_params: dict | None = None,
                     prophet_params: dict | None = None, dd: DailyData | None = None) -> pd.DataFrame:
    from prophet.serialize import model_to_json

    dd = dd or DailyData.from_nsrdb(nsrdb, lon)
    daily, train_end, val_end, n = dd.daily, dd.train_end, dd.val_end, len(dd.daily)
    print(f"Daily rows: {n:,} ({daily.index.min():%Y-%m-%d} > {daily.index.max():%Y-%m-%d})")
    rows = []

    # XGBOOST (medium range): 30-day window, one model per horizon
    for h in C.MEDIUM_HORIZONS:
        model = train_xgb_daily(dd, h, xgb_params)
        model.save_model(artifacts / f"xgb_daily_h{h}d.json")
        X_te, y_te = dd.xgb_split(h, val_end, n)
        pred = np.clip(model.predict(X_te), 0, None)
        clim = _climatology(daily.iloc[:train_end], daily.index[val_end: n - h] + pd.Timedelta(days=h))
        rows.append({"Horizon": f"{h}d", "Model": "XGBoost daily", **regression_metrics(y_te, pred)})
        rows.append({"Horizon": f"{h}d", "Model": "Climatology", **regression_metrics(y_te, clim)})
        print(f"XGBoost daily h={h}d: best iteration {model.best_iteration}")

    # PROPHET (long range)
    for h in C.LONG_HORIZONS:
        frame = dd.prophet_frame(h)
        fit_df = frame.iloc[:train_end].dropna()
        m = fit_prophet(fit_df, prophet_params)
        (artifacts / f"prophet_long_h{h}d.json").write_text(model_to_json(m))
        test = frame.iloc[val_end:].dropna()
        clim = _climatology(daily.iloc[:train_end], test["ds"] + pd.Timedelta(days=h))
        rows.append({"Horizon": f"{h}d", "Model": "Prophet long", **regression_metrics(test["y"].to_numpy(), prophet_predict(m, test))})
        rows.append({"Horizon": f"{h}d", "Model": "Climatology", **regression_metrics(test["y"].to_numpy(), clim)})
        print(f"Prophet long h={h}d trained on {len(fit_df):,} days")

    # Daily values are energy, not power
    metrics = pd.DataFrame(rows).round(4)
    metrics.columns = [c.replace("W/m^2", "Wh/m^2/day").replace("GHI>50", "GHI>50 Wh/m^2/day") for c in metrics.columns]
    metrics.to_csv(outputs / "metrics_long_range.csv", index=False)
    return metrics

def _climatology(train_daily: pd.DataFrame, target_days: pd.DatetimeIndex) -> np.ndarray:
    # Baseline: mean training GHI for the same day of year (smoothed over +-7 days)
    by_doy = train_daily["ghi"].groupby(train_daily.index.dayofyear).mean().reindex(range(1, 367))
    smooth = pd.concat([by_doy.iloc[-7:], by_doy, by_doy.iloc[:7]]).rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
    return smooth.to_numpy()[pd.DatetimeIndex(target_days).dayofyear - 1]
