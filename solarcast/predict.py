from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_tabular, kt_to_ghi, prepare_hourly, target_features
from .models import Bundle, combine
from .sources import fetch_open_meteo_live

def issue_position(hourly: pd.DataFrame, now: pd.Timestamp | None = None) -> int:
    # Row of the latest complete hour: label T covers (T-1h, T], so it is complete once T has passed
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    pos = hourly.index.searchsorted(now, side="right") - 1
    if pos < 0:
        raise ValueError(f"No Open-Meteo data at or before {now}")
    return int(pos)

def predict_frame(hourly: pd.DataFrame, bundle: Bundle, pos: int, horizons: list[int], use_lstm: bool = True) -> pd.DataFrame:
    # One row per horizon: model predictions (W/m^2), ensemble, raw forecast and clear sky at the target hour
    rows = []
    for h in horizons:
        if pos + h >= len(hourly):
            raise ValueError(f"Open-Meteo forecast does not reach +{h} h")
        X = build_tabular(hourly, h)[bundle.xgb_features].iloc[[pos]]
        tf = target_features(hourly, h).iloc[[pos]]
        preds = {"XGBoost": kt_to_ghi(bundle.xgb_models[h].predict(X), tf)}
        if use_lstm and bundle.lstm is not None:
            seq = bundle.lstm_inputs.sequences(hourly, np.array([pos]))
            preds["LSTM"] = kt_to_ghi(bundle.lstm.predict([seq, bundle.lstm_inputs.static(tf)], verbose=0).ravel(), tf)
        row = {"horizon": h, "target_time": hourly.index[pos + h]}
        row.update({name: float(max(0.0, p[0])) for name, p in preds.items()})
        row["Ensemble"] = float(max(0.0, combine(preds, bundle.weights[h])[0]))
        row["Open-Meteo forecast"] = float(max(0.0, np.nan_to_num(tf["fc_ghi"].iloc[0])))
        row["Clear sky"] = float(tf["tgt_cs_ghi"].iloc[0])
        row["zenith"] = float(tf["tgt_zenith"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")

def predict_live(lat: float, lon: float, artifacts: Path, horizons: list[int], use_lstm: bool = True) -> tuple[pd.DataFrame, pd.Timestamp, str]:
    bundle = Bundle.load(artifacts, with_lstm=use_lstm)
    om, tz = fetch_open_meteo_live(lat, lon, past_days=2, forecast_days=3)
    hourly = prepare_hourly(om, lat, lon)
    pos = issue_position(hourly)
    return predict_frame(hourly, bundle, pos, horizons, use_lstm), hourly.index[pos], tz
