import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

from . import config as C
from .features import PAST_COLS, sequence_windows

def xgb_device() -> str:
    # CUDA_VISIBLE_DEVICES="" forces CPU (the test suite sets it)
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    return "cuda" if xgb.build_info().get("USE_CUDA") and shutil.which("nvidia-smi") else "cpu"

# XGBOOST (one model per horizon, predicts the clear-sky index)
def train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray, params: dict | None = None,
              sample_weight: np.ndarray | None = None) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        **(params or C.XGB_PARAMS),
        early_stopping_rounds=C.XGB_EARLY_STOPPING,
        eval_metric="rmse",
        random_state=C.RANDOM_SEED,
        device=xgb_device(),
    )
    model.fit(X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_va, y_va)], verbose=False)
    # Predict on CPU: inputs are small pandas frames and a GPU booster would warn about copying them over
    model.set_params(device="cpu")
    return model

# LSTM (one model for all horizons: a past-window branch plus a target-hour branch)
@dataclass
class LstmInputs:
    # Scalers are fitted on training rows only
    seq_scaler: MinMaxScaler
    static_scaler: MinMaxScaler
    static_cols: list[str]

    def sequences(self, hourly: pd.DataFrame, issue_pos: np.ndarray) -> np.ndarray:
        values = np.nan_to_num(self.seq_scaler.transform(hourly[PAST_COLS]), nan=0.0).astype("float32")
        return sequence_windows(values, issue_pos)

    def static(self, target: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(self.static_scaler.transform(target[self.static_cols]), nan=0.0).astype("float32")

def fit_lstm_inputs(hourly_train: pd.DataFrame, target_train: pd.DataFrame) -> LstmInputs:
    static_cols = list(target_train.columns)
    return LstmInputs(
        seq_scaler=MinMaxScaler().fit(hourly_train[PAST_COLS]),
        static_scaler=MinMaxScaler().fit(target_train[static_cols]),
        static_cols=static_cols,
    )

def build_lstm(n_past: int, n_static: int, params: dict | None = None):
    # params: keys of config.LSTM_PARAMS (batch_size is used by train_lstm, not here)
    import keras
    from keras import layers

    p = {**C.LSTM_PARAMS, **(params or {})}
    seq_in = keras.Input(shape=(C.WINDOW_SIZE, n_past), name="past")
    static_in = keras.Input(shape=(n_static,), name="target_hour")
    # use_cudnn=False: the cuDNN LSTM kernel fails on this project's GPU (GTX 1080 Ti), as it did in 3.modeling.ipynb
    x = layers.LSTM(p["units_1"], return_sequences=True, use_cudnn=False)(seq_in)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.LSTM(p["units_2"], use_cudnn=False)(x)
    x = layers.Dropout(p["dropout"])(x)
    x = layers.Concatenate()([x, static_in])
    x = layers.Dense(p["dense_units"], activation="relu")(x)
    x = layers.Dense(max(p["dense_units"] // 2, 8), activation="relu")(x)
    out = layers.Dense(1, name="kt")(x)
    model = keras.Model([seq_in, static_in], out)
    model.compile(optimizer=keras.optimizers.Adam(p["learning_rate"]), loss="mse", metrics=["mae"])
    return model

def train_lstm(train: tuple, val: tuple, sample_weight: np.ndarray | None = None, verbose: int = 2, params: dict | None = None):
    # train/val: ((seq, static), y)
    import keras
    keras.utils.set_random_seed(C.RANDOM_SEED)
    (seq_tr, st_tr), y_tr = train
    p = {**C.LSTM_PARAMS, **(params or {})}
    model = build_lstm(seq_tr.shape[2], st_tr.shape[1], p)
    history = model.fit(
        [seq_tr, st_tr], y_tr, sample_weight=sample_weight,
        validation_data=([val[0][0], val[0][1]], val[1]),
        epochs=C.LSTM_EPOCHS, batch_size=p["batch_size"], verbose=verbose,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.LSTM_PATIENCE, restore_best_weights=True)],
    )
    return model, history.history

# ENSEMBLE
def inverse_error_weights(val_mse: dict[str, float]) -> dict[str, float]:
    # Weight each model by 1/MSE on the validation split, normalised to sum to 1
    inv = {name: 1.0 / max(mse, 1e-9) for name, mse in val_mse.items()}
    total = sum(inv.values())
    return {name: w / total for name, w in inv.items()}

def combine(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names = [n for n in preds if n in weights]
    total = sum(weights[n] for n in names)
    return sum(preds[n] * weights[n] for n in names) / total

# BUNDLE (everything predict_today.py needs, saved in artifacts/)
@dataclass
class Bundle:
    horizons: list[int]
    xgb_features: list[str]
    xgb_models: dict[int, xgb.XGBRegressor] = field(default_factory=dict)
    lstm: object | None = None
    lstm_inputs: LstmInputs | None = None
    weights: dict[int, dict[str, float]] = field(default_factory=dict)

    def save(self, artifacts: Path) -> None:
        for h, model in self.xgb_models.items():
            model.save_model(artifacts / f"xgb_h{h}.json")
        if self.lstm is not None:
            self.lstm.save(artifacts / "lstm.keras")
            joblib.dump(self.lstm_inputs, artifacts / "lstm_inputs.pkl")
        meta = {
            "horizons": self.horizons,
            "xgb_features": self.xgb_features,
            "weights": {str(h): w for h, w in self.weights.items()},
            "has_lstm": self.lstm is not None,
        }
        (artifacts / "bundle.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, artifacts: Path, with_lstm: bool = True) -> "Bundle":
        meta = json.loads((artifacts / "bundle.json").read_text())
        bundle = cls(horizons=meta["horizons"], xgb_features=meta["xgb_features"],
                     weights={int(h): w for h, w in meta["weights"].items()})
        for h in bundle.horizons:
            model = xgb.XGBRegressor()
            model.load_model(artifacts / f"xgb_h{h}.json")
            bundle.xgb_models[h] = model
        if with_lstm and meta["has_lstm"]:
            import keras
            bundle.lstm = keras.models.load_model(artifacts / "lstm.keras")
            # use_cudnn is not saved with the model; re-apply it (see build_lstm)
            for layer in bundle.lstm.layers:
                if isinstance(layer, keras.layers.LSTM):
                    layer.use_cudnn = False
            bundle.lstm_inputs = joblib.load(artifacts / "lstm_inputs.pkl")
        return bundle
