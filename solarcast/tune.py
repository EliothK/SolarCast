import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .models import train_xgb
from .train import HorizonData, LstmData, _stack

# Hyperparameter searches, ported from 6.hyperparameter_tuning.ipynb.
# Each returns the best settings; run_pipeline.py saves them to artifacts/<name>_params.json and training reuses them.
# The short-range Prophet search is not ported: Prophet is no longer used for 6-48 h.

# XGBOOST (short range) via Optuna
def tune_xgb(data: dict[int, HorizonData], horizons: list[int], n_trials: int = 30, outputs: Path | None = None) -> dict:
    # One shared parameter set, scored as mean validation RMSE of the clear-sky index over the given horizons
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        scores = []
        for h in horizons:
            hd = data[h]
            X_tr, y_tr, w_tr = _stack(hd, hd.train)
            X_va, y_va = hd.X["archive"].iloc[hd.early_stop], hd.tgt["kt"].to_numpy()[hd.early_stop]
            model = train_xgb(X_tr, y_tr, X_va, y_va, params, sample_weight=w_tr)
            scores.append(float(np.sqrt(np.mean((model.predict(X_va) - y_va) ** 2))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    if outputs is not None:
        study.trials_dataframe().to_csv(outputs / "tuning_xgb_trials.csv", index=False)
    print(f"XGBoost: best mean validation kt RMSE {study.best_value:.4f}")
    return {**C.XGB_PARAMS, "n_estimators": 2000, **study.best_params}

# LSTM via Keras Tuner (RandomSearch, same search space as the notebook)
def tune_lstm(ld: LstmData, max_trials: int = 10, workdir: Path | None = None, outputs: Path | None = None) -> dict:
    import keras
    import keras_tuner as kt

    from .models import build_lstm

    (seq_tr, st_tr), y_tr = ld.train
    (seq_va, st_va), y_va = ld.val

    class LstmHyperModel(kt.HyperModel):
        def build(self, hp):
            return build_lstm(seq_tr.shape[2], st_tr.shape[1], {
                "units_1": hp.Int("units_1", 64, 256, step=64),
                "units_2": hp.Int("units_2", 32, 128, step=32),
                "dropout": hp.Float("dropout", 0.1, 0.5, step=0.1),
                "dense_units": hp.Int("dense_units", 16, 128, step=16),
                "learning_rate": hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"),
            })

        def fit(self, hp, model, *args, **kwargs):
            return model.fit(*args, batch_size=hp.Choice("batch_size", [64, 128, 256, 512]), **kwargs)

    keras.utils.set_random_seed(C.RANDOM_SEED)
    tuner = kt.RandomSearch(
        LstmHyperModel(), objective="val_loss", max_trials=max_trials, seed=C.RANDOM_SEED,
        directory=str(workdir or Path("outputs") / "kt_lstm"), project_name="lstm_kt", overwrite=True,
    )
    tuner.search(
        [seq_tr, st_tr], y_tr, sample_weight=ld.weights, validation_data=([seq_va, st_va], y_va),
        epochs=C.LSTM_EPOCHS, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.LSTM_PATIENCE, restore_best_weights=True)],
    )
    trials = tuner.oracle.get_best_trials(num_trials=max_trials)
    if outputs is not None:
        pd.DataFrame([{**t.hyperparameters.values, "val_loss": t.score} for t in trials]).to_csv(outputs / "tuning_lstm_trials.csv", index=False)
    best = tuner.get_best_hyperparameters(1)[0].values
    print(f"LSTM: best val_loss {trials[0].score:.5f} with {best}")
    return {**C.LSTM_PARAMS, **best}

# XGBOOST DAILY (7/14 d) via Optuna
def tune_xgb_daily(dd, n_trials: int = 30, outputs: Path | None = None) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        from .longrange import train_xgb_daily
        scores = []
        for h in C.MEDIUM_HORIZONS:
            model = train_xgb_daily(dd, h, params)
            X_va, y_va = dd.xgb_split(h, dd.train_end, dd.val_end)
            scores.append(float(np.sqrt(np.mean((model.predict(X_va) - y_va) ** 2))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    if outputs is not None:
        study.trials_dataframe().to_csv(outputs / "tuning_xgb_daily_trials.csv", index=False)
    print(f"XGBoost daily: best mean validation RMSE {study.best_value:.1f} Wh/m^2/day")
    return {"n_estimators": 1000, **study.best_params}

# PROPHET LONG RANGE via grid search (same grid as the notebook), run in parallel
def _prophet_combo_mae(dd, params: dict, horizons: list[int]) -> float:
    from .longrange import fit_prophet, prophet_predict
    maes = []
    for h in horizons:
        frame = dd.prophet_frame(h)
        m = fit_prophet(frame.iloc[: dd.train_end].dropna(), params)
        val = frame.iloc[dd.train_end: dd.val_end].dropna()
        maes.append(float(np.mean(np.abs(prophet_predict(m, val) - val["y"].to_numpy()))))
    return float(np.mean(maes))

def tune_prophet_long(dd, grid: dict | None = None, horizons: list[int] | None = None, n_jobs: int = -1, outputs: Path | None = None) -> dict:
    from joblib import Parallel, delayed
    grid = grid or C.PROPHET_GRID
    horizons = horizons or C.PROPHET_TUNE_HORIZONS
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    print(f"Prophet: {len(combos)} combinations x {len(horizons)} horizons")
    maes = Parallel(n_jobs=n_jobs)(delayed(_prophet_combo_mae)(dd, p, horizons) for p in combos)
    results = pd.DataFrame(combos).assign(mean_val_mae=maes).sort_values("mean_val_mae", ignore_index=True)
    if outputs is not None:
        results.to_csv(outputs / "tuning_prophet_long_grid.csv", index=False)
    best = results.iloc[0].drop("mean_val_mae").to_dict()
    print(f"Prophet long: best mean validation MAE {results['mean_val_mae'].iloc[0]:.1f} Wh/m^2/day with {best}")
    return best
