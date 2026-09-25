from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import metrics_table, old_recipe_frame, old_recipe_windows, train_old_recipe
from .features import build_tabular, forecast_frame, kt_to_ghi, prepare_hourly, target_features, target_kt
from .models import Bundle, combine, fit_lstm_inputs, inverse_error_weights, train_lstm, train_xgb
from .sources import fetch_nsrdb, fetch_open_meteo_archive, fetch_open_meteo_previous_runs

# Two kinds of target-hour forecast feed the models:
# "archive": Open-Meteo's historical-forecast archive (2018 on). It stitches the newest run for every hour, so its forecasts are short-lead whatever the horizon.
# "lead": Open-Meteo previous runs (2024-01-19 on), forecasts really issued 1 or 2 days ahead. Live 24/48 h forecasts look like this.
# Horizons >= 24 h train on both (lead rows upweighted) and are tested on lead data only; 6/12 h use the archive throughout.

@dataclass
class Splits:
    # Issue-time row positions for each split
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

def make_splits(n: int, gap: int = max(C.HORIZONS), window: int = C.WINDOW_SIZE) -> Splits:
    # Time-ordered 70/15/15 split of issue times.
    # A gap of `gap` hours before each boundary keeps every target inside its own split.
    train_end, val_end = int(n * C.TRAIN_FRAC), int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    return Splits(
        train=np.arange(window - 1, train_end - gap),
        val=np.arange(train_end, val_end - gap),
        test=np.arange(val_end, n - gap),
    )

def split_positions(pos: np.ndarray, train_frac: float, val_frac: float, gap: int) -> Splits:
    # Time-ordered split of an arbitrary sorted set of positions, dropping `gap` hours before each boundary
    if len(pos) == 0:
        return Splits(pos, pos, pos)
    t_cut = pos[int(len(pos) * train_frac)]
    v_cut = pos[min(int(len(pos) * (train_frac + val_frac)), len(pos) - 1)]
    return Splits(
        train=pos[pos < t_cut - gap],
        val=pos[(pos >= t_cut) & (pos < v_cut - gap)],
        test=pos[pos >= v_cut],
    )

def load_training_data(paths: C.Paths, lat: float, lon: float, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Open-Meteo archive (inputs) and NSRDB (truth), both on UTC hour-ending labels
    om = fetch_open_meteo_archive(lat, lon, years, paths.raw)
    nsrdb = fetch_nsrdb(lat, lon, years, paths.raw, legacy_csv=paths.data / "nsrdb_raw.csv")
    hourly = prepare_hourly(om, lat, lon)
    nsrdb = nsrdb.reindex(hourly.index)
    return hourly, nsrdb

def _where(pos: np.ndarray, mask: pd.Series | np.ndarray) -> np.ndarray:
    # Positions in pos where mask is true
    return pos[np.asarray(mask)[pos]]

@dataclass
class HorizonData:
    h: int
    tgt: pd.DataFrame
    X: dict = field(default_factory=dict)  # source > tabular features
    tf: dict = field(default_factory=dict)  # source > target-hour features
    train: list = field(default_factory=list)  # [(source, rows, weight)]
    early_stop: np.ndarray | None = None  # archive validation rows (usable)
    weight_src: str = "archive"
    weight_rows: np.ndarray | None = None  # rows used for ensemble weights (all with truth)
    test_src: str = "archive"
    test_rows: np.ndarray | None = None

def prepare_horizons(hourly: pd.DataFrame, nsrdb: pd.DataFrame, splits: Splits, lead_frames: dict[int, pd.DataFrame]) -> dict[int, HorizonData]:
    out = {}
    for h in C.HORIZONS:
        tgt = target_kt(hourly, nsrdb["ghi"], h)
        usable, has_truth = tgt["usable"].to_numpy(), tgt["truth_ghi"].notna().to_numpy()
        hd = HorizonData(h=h, tgt=tgt)
        hd.X["archive"], hd.tf["archive"] = build_tabular(hourly, h), target_features(hourly, h)
        hd.train.append(("archive", _where(splits.train, usable), 1.0))
        hd.early_stop = _where(splits.val, usable)
        hd.weight_rows = _where(splits.val, has_truth)
        hd.test_rows = _where(splits.test, has_truth)

        lead = lead_frames.get(h // 24) if h >= 24 else None
        if lead is not None:
            hd.X["lead"], hd.tf["lead"] = build_tabular(hourly, h, lead), target_features(hourly, h, lead)
            ok = lead["ghi"].shift(-h).notna().to_numpy() & has_truth
            ok[len(ok) - h:] = False
            # Only lead rows from the archive test period on, so lead test rows can never repeat hours the archive trained or validated on
            ok[: splits.test[0]] = False
            ls = split_positions(np.where(ok)[0], C.LEAD_TRAIN_FRAC, C.LEAD_VAL_FRAC, gap=h)
            hd.train.append(("lead", _where(ls.train, usable), C.LEAD_WEIGHT))
            # Too little lead data for a slice: keep the archive rows for it
            if len(ls.val):
                hd.weight_src, hd.weight_rows = "lead", ls.val
            if len(ls.test):
                hd.test_src, hd.test_rows = "lead", ls.test
        out[h] = hd
    return out

def _stack(hd: HorizonData, parts: list) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = pd.concat([hd.X[src].iloc[rows] for src, rows, _ in parts])
    y = np.concatenate([hd.tgt["kt"].to_numpy()[rows] for _, rows, _ in parts])
    w = np.concatenate([np.full(len(rows), wt) for _, rows, wt in parts])
    return X, y, w

@dataclass
class Prepared:
    hourly: pd.DataFrame
    nsrdb: pd.DataFrame
    splits: Splits
    data: dict[int, HorizonData]

def prepare_training(paths: C.Paths, lat: float, lon: float, years: list[int]) -> Prepared:
    hourly, nsrdb = load_training_data(paths, lat, lon, years)
    splits = make_splits(len(hourly))
    print(f"Hours: {len(hourly):,} ({hourly.index.min():%Y-%m-%d} > {hourly.index.max():%Y-%m-%d})")
    for name in ("train", "val", "test"):
        pos = getattr(splits, name)
        print(f"Archive {name}: {hourly.index[pos[0]]:%Y-%m-%d} > {hourly.index[pos[-1]]:%Y-%m-%d} ({len(pos):,} issue times)")

    lead_frames = {}
    for day in sorted({h // 24 for h in C.HORIZONS if h >= 24}):
        prev = fetch_open_meteo_previous_runs(lat, lon, day, years, paths.raw)
        if len(prev):
            lead_frames[day] = forecast_frame(prev, hourly)
    if not lead_frames:
        print("WARNING: no Open-Meteo previous runs for these years (they start 2024-01-19). 24/48 h models are trained and tested on short-lead archive forecasts only, so their scores are optimistic.")

    return Prepared(hourly, nsrdb, splits, prepare_horizons(hourly, nsrdb, splits, lead_frames))

@dataclass
class LstmData:
    inputs: object  # models.LstmInputs, fitted on training rows
    train: tuple  # ((seq, static), y)
    val: tuple
    weights: np.ndarray

def lstm_datasets(prep: Prepared) -> LstmData:
    # Stacked LSTM training/validation arrays over all horizons (shared by training and the Keras Tuner search)
    hourly, splits, data = prep.hourly, prep.splits, prep.data
    train_hours = hourly.iloc[: splits.train[-1] + 1]
    train_tf = pd.concat([data[h].tf["archive"].iloc[data[h].train[0][1]] for h in C.HORIZONS])
    inputs = fit_lstm_inputs(train_hours, train_tf)

    def lstm_set(parts_by_h: dict[int, list]):
        seqs, statics, ys, ws = [], [], [], []
        for h, parts in parts_by_h.items():
            for src, rows, wt in parts:
                seqs.append(inputs.sequences(hourly, rows))
                statics.append(inputs.static(data[h].tf[src].iloc[rows]))
                ys.append(data[h].tgt["kt"].to_numpy()[rows])
                ws.append(np.full(len(rows), wt))
        return (np.concatenate(seqs), np.concatenate(statics)), np.concatenate(ys).astype("float32"), np.concatenate(ws).astype("float32")

    seq_st_tr, y_tr, w_tr = lstm_set({h: hd.train for h, hd in data.items()})
    seq_st_va, y_va, _ = lstm_set({h: [("archive", hd.early_stop, 1.0)] for h, hd in data.items()})
    return LstmData(inputs, (seq_st_tr, y_tr), (seq_st_va, y_va), w_tr)

def train_short_range(prep: Prepared, paths: C.Paths, lat: float, lon: float, use_lstm: bool = True,
                      xgb_params: dict | None = None, lstm_params: dict | None = None, lstm_verbose: int = 2) -> pd.DataFrame:
    hourly, nsrdb, splits, data = prep.hourly, prep.nsrdb, prep.splits, prep.data
    bundle = Bundle(horizons=C.HORIZONS, xgb_features=list(data[C.HORIZONS[0]].X["archive"].columns))

    # XGBOOST
    for h, hd in data.items():
        X_tr, y_tr, w_tr = _stack(hd, hd.train)
        X_va, y_va = hd.X["archive"].iloc[hd.early_stop], hd.tgt["kt"].to_numpy()[hd.early_stop]
        bundle.xgb_models[h] = train_xgb(X_tr, y_tr, X_va, y_va, xgb_params, sample_weight=w_tr)
        sizes = ", ".join(f"{src} {len(rows):,}" for src, rows, _ in hd.train)
        print(f"XGBoost h={h}h: trained on {sizes}; best iteration {bundle.xgb_models[h].best_iteration}")

    # LSTM (one model stacked over horizons)
    if use_lstm:
        ld = lstm_datasets(prep)
        lstm, history = train_lstm(ld.train, ld.val, sample_weight=ld.weights, verbose=lstm_verbose, params=lstm_params)
        bundle.lstm, bundle.lstm_inputs = lstm, ld.inputs
        print(f"LSTM: stopped after {len(history['loss'])} epochs, best val_loss {min(history['val_loss']):.5f}")

    def predict_models(hd: HorizonData, src: str, rows: np.ndarray) -> dict[str, np.ndarray]:
        return model_predictions(bundle, hourly, hd, src, rows)

    # ENSEMBLE WEIGHTS: inverse validation MSE (lead-matched validation for 24/48 h)
    for h, hd in data.items():
        truth = hd.tgt["truth_ghi"].to_numpy()[hd.weight_rows]
        preds = predict_models(hd, hd.weight_src, hd.weight_rows)
        bundle.weights[h] = inverse_error_weights({n: float(np.mean((p - truth) ** 2)) for n, p in preds.items()})
        print(f"Ensemble weights h={h}h ({hd.weight_src} validation): " + ", ".join(f"{n} {w:.2f}" for n, w in bundle.weights[h].items()))

    bundle.save(paths.artifacts)

    # TEST EVALUATION against baselines, on each horizon's test rows
    nsrdb_old = old_recipe_frame(nsrdb, lat, lon, from_open_meteo=False)
    om_old = old_recipe_frame(hourly, lat, lon, from_open_meteo=True)
    old_ok = nsrdb_old.notna().all(axis=1).rolling(C.WINDOW_SIZE).min().fillna(0).astype(bool).to_numpy()
    tables = []
    for h, hd in data.items():
        # Old recipe: the pre-refactor model, trained on NSRDB windows with a raw GHI target
        y_old = nsrdb["ghi"].shift(-h).to_numpy()
        ok = old_ok & ~np.isnan(y_old)
        tr_o, va_o = _where(splits.train, ok), _where(splits.val, ok)
        old = train_old_recipe(old_recipe_windows(nsrdb_old, tr_o), y_old[tr_o], old_recipe_windows(nsrdb_old, va_o), y_old[va_o])

        te = hd.test_rows
        truth = hd.tgt["truth_ghi"].to_numpy()[te]
        tf = hd.tf[hd.test_src].iloc[te]
        preds = predict_models(hd, hd.test_src, te)
        if len(preds) > 1:
            preds["Ensemble"] = combine(preds, bundle.weights[h])
        preds["Open-Meteo forecast"] = np.clip(np.nan_to_num(tf["fc_ghi"].to_numpy()), 0, None)
        preds["Old recipe (NSRDB inputs)"] = np.clip(old.predict(old_recipe_windows(nsrdb_old.fillna(0), te)), 0, None)
        preds["Old recipe (Open-Meteo inputs)"] = np.clip(old.predict(old_recipe_windows(om_old.fillna(0), te)), 0, None)

        table = metrics_table(truth, preds, tf["tgt_zenith"].to_numpy() < 90, h)
        table.insert(1, "Forecast inputs", hd.test_src)
        table.insert(2, "Test period", f"{hourly.index[te[0]]:%Y-%m-%d} > {hourly.index[te[-1]]:%Y-%m-%d}")
        table.insert(3, "Test hours", len(te))
        tables.append(table)

    metrics = pd.concat(tables, ignore_index=True).round(4)
    metrics.to_csv(paths.outputs / "metrics_short_range.csv", index=False)
    return metrics

def model_predictions(bundle: Bundle, hourly: pd.DataFrame, hd: HorizonData, src: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    X, tf = hd.X[src].iloc[rows], hd.tf[src].iloc[rows]
    preds = {"XGBoost": kt_to_ghi(bundle.xgb_models[hd.h].predict(X), tf)}
    if bundle.lstm is not None:
        seq, st = bundle.lstm_inputs.sequences(hourly, rows), bundle.lstm_inputs.static(tf)
        preds["LSTM"] = kt_to_ghi(bundle.lstm.predict([seq, st], verbose=0, batch_size=4096).ravel(), tf)
    return preds

def test_predictions(prep: Prepared, bundle: Bundle, h: int) -> pd.DataFrame:
    # Test-split predictions for one horizon, indexed by target time (for plots)
    hd = prep.data[h]
    rows = hd.test_rows
    preds = model_predictions(bundle, prep.hourly, hd, hd.test_src, rows)
    if len(preds) > 1:
        preds["Ensemble"] = combine(preds, bundle.weights[h])
    preds["Open-Meteo forecast"] = np.clip(np.nan_to_num(hd.tf[hd.test_src]["fc_ghi"].to_numpy()[rows]), 0, None)
    out = pd.DataFrame(preds, index=prep.hourly.index[rows + h])
    out.insert(0, "Truth", hd.tgt["truth_ghi"].to_numpy()[rows])
    return out

