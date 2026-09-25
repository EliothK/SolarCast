import json
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from solarcast import config as C
from solarcast import train as T
from solarcast.features import prepare_hourly
from solarcast.models import Bundle, fit_lstm_inputs, train_lstm, train_xgb

# End-to-end tests on a small synthetic site (no network)

@pytest.fixture
def prep(site_data, tmp_path, monkeypatch):
    om, nsrdb = site_data
    monkeypatch.setattr(T, "load_training_data", lambda paths, lat, lon, years: (prepare_hourly(om, lat, lon), nsrdb.reindex(prepare_hourly(om, lat, lon).index)))
    # Previous runs = a slightly worse copy of the archive
    monkeypatch.setattr(T, "fetch_open_meteo_previous_runs", lambda lat, lon, day, years, raw: om.assign(ghi=om["ghi"] * (1 + 0.1 * day)))
    paths = C.Paths(tmp_path)
    paths.makedirs()
    return paths, T.prepare_training(paths, C.DEFAULT_LAT, C.DEFAULT_LON, [2024])

def test_prepare_uses_lead_data_for_long_horizons(prep):
    _, p = prep
    for h in C.HORIZONS:
        hd = p.data[h]
        sources = [src for src, _, _ in hd.train]
        if h >= 24:
            # Weights use lead validation when there are enough lead hours (fallback covered by its own test)
            assert sources == ["archive", "lead"] and hd.test_src == "lead" and hd.weight_src in ("lead", "archive")
            assert hd.train[1][2] == C.LEAD_WEIGHT
        else:
            assert sources == ["archive"] and hd.test_src == "archive"
        # Test rows never overlap training rows
        assert not set(hd.test_rows) & set(np.concatenate([rows for _, rows, _ in hd.train]))

def test_small_lead_slices_fall_back_to_archive(site_data, tmp_path, monkeypatch):
    # Lead data covering only the last few days leaves no lead validation rows: weights must come from the archive instead of NaN
    om, nsrdb = site_data
    hourly = prepare_hourly(om, C.DEFAULT_LAT, C.DEFAULT_LON)
    monkeypatch.setattr(T, "load_training_data", lambda *a: (hourly, nsrdb.reindex(hourly.index)))
    monkeypatch.setattr(T, "fetch_open_meteo_previous_runs", lambda lat, lon, day, years, raw: om.iloc[-150:])
    paths = C.Paths(tmp_path)
    paths.makedirs()
    p = T.prepare_training(paths, C.DEFAULT_LAT, C.DEFAULT_LON, [2024])
    assert p.data[48].weight_src == "archive" and len(p.data[48].weight_rows) > 0
    T.train_short_range(p, paths, C.DEFAULT_LAT, C.DEFAULT_LON, use_lstm=False)
    weights = json.loads((paths.artifacts / "bundle.json").read_text())["weights"]
    assert all(np.isfinite(w) for per_h in weights.values() for w in per_h.values())

def test_train_short_range_end_to_end(prep):
    paths, p = prep
    metrics = T.train_short_range(p, paths, C.DEFAULT_LAT, C.DEFAULT_LON, use_lstm=False)
    assert (paths.artifacts / "bundle.json").exists()
    assert (paths.outputs / "metrics_short_range.csv").exists()
    assert set(metrics["Horizon"]) == {f"{h}h" for h in C.HORIZONS}
    for col in ("MAE (W/m^2)", "MSE", "RMSE (W/m^2)", "R^2", "MAPE (%, GHI>50)", "Bias (W/m^2)", "Skill vs forecast"):
        assert metrics[col].notna().all()
    # The synthetic forecast is biased +5 %: the model should learn to beat it
    xgb6 = metrics.query("Horizon == '6h' and Model == 'XGBoost'")["RMSE (W/m^2)"].iloc[0]
    fc6 = metrics.query("Horizon == '6h' and Model == 'Open-Meteo forecast'")["RMSE (W/m^2)"].iloc[0]
    assert xgb6 < fc6
    bundle = Bundle.load(paths.artifacts)
    preds = T.test_predictions(p, bundle, 6)
    assert {"Truth", "XGBoost", "Open-Meteo forecast"} <= set(preds.columns)

def test_bundle_roundtrip_with_lstm(prep, tmp_path):
    _, p = prep
    hd = p.data[6]
    rows = hd.train[0][1][:300]
    X, y = hd.X["archive"].iloc[rows], hd.tgt["kt"].to_numpy()[rows]
    inputs = fit_lstm_inputs(p.hourly, hd.tf["archive"].iloc[rows])
    seq, st = inputs.sequences(p.hourly, rows), inputs.static(hd.tf["archive"].iloc[rows])
    assert seq.shape == (len(rows), C.WINDOW_SIZE, len(inputs.seq_scaler.feature_names_in_))

    import solarcast.config as cfg
    old_epochs, cfg.LSTM_EPOCHS = cfg.LSTM_EPOCHS, 1
    try:
        lstm, history = train_lstm(((seq, st), y), ((seq, st), y), verbose=0)
    finally:
        cfg.LSTM_EPOCHS = old_epochs
    bundle = Bundle(horizons=[6], xgb_features=list(X.columns), xgb_models={6: train_xgb(X, y, X, y)},
                    lstm=lstm, lstm_inputs=inputs, weights={6: {"XGBoost": 0.6, "LSTM": 0.4}})
    bundle.save(tmp_path)
    loaded = Bundle.load(tmp_path)
    np.testing.assert_allclose(loaded.xgb_models[6].predict(X), bundle.xgb_models[6].predict(X), rtol=1e-6)
    np.testing.assert_allclose(loaded.lstm.predict([seq, st], verbose=0), lstm.predict([seq, st], verbose=0), rtol=1e-5)
    assert loaded.weights == {6: {"XGBoost": 0.6, "LSTM": 0.4}}
    # The cuDNN kernel crashes on the project GPU; the setting must survive save/load
    import keras
    lstm_layers = [l for l in loaded.lstm.layers if isinstance(l, keras.layers.LSTM)]
    assert lstm_layers and all(l.use_cudnn is False for l in lstm_layers)
    assert json.loads((tmp_path / "bundle.json").read_text())["has_lstm"] is True

def test_predict_frame_on_trained_bundle(prep):
    from solarcast.predict import predict_frame
    paths, p = prep
    T.train_short_range(p, paths, C.DEFAULT_LAT, C.DEFAULT_LON, use_lstm=False)
    bundle = Bundle.load(paths.artifacts)
    pos = len(p.hourly) - 100
    table = predict_frame(p.hourly, bundle, pos, C.HORIZONS, use_lstm=False)
    assert list(table.index) == C.HORIZONS
    assert (table[["XGBoost", "Ensemble", "Open-Meteo forecast"]] >= 0).all().all()
    night = table[table["zenith"] >= 90]
    assert (night["XGBoost"] == 0).all()
    with pytest.raises(ValueError, match="does not reach"):
        predict_frame(p.hourly, bundle, len(p.hourly) - 10, [48], use_lstm=False)

def test_tune_xgb_returns_usable_params(prep):
    from solarcast.tune import tune_xgb
    _, p = prep
    params = tune_xgb(p.data, [6], n_trials=2)
    assert {"learning_rate", "max_depth", "subsample"} <= set(params)
    hd = p.data[6]
    X, y, w = T._stack(hd, hd.train)
    train_xgb(X, y, X, y, params, sample_weight=w)

def test_params_roundtrip(tmp_path):
    assert C.load_params(tmp_path, "xgb") is None
    C.save_params(tmp_path, "xgb", {"max_depth": 4})
    assert C.load_params(tmp_path, "xgb") == {"max_depth": 4}
    assert (tmp_path / "xgb_params.json").exists()

def test_build_lstm_uses_params():
    from solarcast.models import build_lstm
    model = build_lstm(5, 3, {"units_1": 64, "units_2": 32, "dense_units": 16, "dropout": 0.3, "learning_rate": 0.01})
    import keras
    lstms = [l for l in model.layers if isinstance(l, keras.layers.LSTM)]
    assert [l.units for l in lstms] == [64, 32]
    assert all(l.use_cudnn is False for l in lstms)
    assert abs(float(model.optimizer.learning_rate) - 0.01) < 1e-9

def test_tune_lstm_returns_full_params(prep, tmp_path, monkeypatch):
    from solarcast.tune import tune_lstm
    _, p = prep
    monkeypatch.setattr(C, "LSTM_EPOCHS", 1)
    params = tune_lstm(T.lstm_datasets(p), max_trials=2, workdir=tmp_path / "kt", outputs=tmp_path)
    assert set(C.LSTM_PARAMS) <= set(params)
    assert params["batch_size"] in (64, 128, 256, 512)
    trials = pd.read_csv(tmp_path / "tuning_lstm_trials.csv")
    assert len(trials) == 2 and trials["val_loss"].notna().all()

def test_tune_lstm_raises_when_every_trial_fails(prep, tmp_path, monkeypatch):
    import solarcast.models as M
    from solarcast.tune import tune_lstm
    _, p = prep
    monkeypatch.setattr(C, "LSTM_EPOCHS", 1)
    # Keras Tuner builds once while setting up the search space; let that succeed and fail every trial's build
    real_build, calls = M.build_lstm, []
    def broken(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("broken model")
        return real_build(*a, **k)
    monkeypatch.setattr(M, "build_lstm", broken)
    with pytest.raises(RuntimeError, match="Every Keras Tuner trial failed"):
        tune_lstm(T.lstm_datasets(p), max_trials=2, workdir=tmp_path / "kt")

def test_train_short_range_uses_lstm_params(prep, monkeypatch):
    import solarcast.train as TT
    paths, p = prep
    seen = {}
    def fake_train_lstm(train, val, sample_weight=None, verbose=2, params=None):
        seen["params"] = params
        raise RuntimeError("stop after capturing params")
    monkeypatch.setattr(TT, "train_lstm", fake_train_lstm)
    with pytest.raises(RuntimeError, match="stop after"):
        T.train_short_range(p, paths, C.DEFAULT_LAT, C.DEFAULT_LON, use_lstm=True, lstm_params={"units_1": 64})
    assert seen["params"] == {"units_1": 64}

def test_plots_render(prep):
    from solarcast import plots
    paths, p = prep
    metrics = T.train_short_range(p, paths, C.DEFAULT_LAT, C.DEFAULT_LON, use_lstm=False)
    bundle = Bundle.load(paths.artifacts)
    assert plots.rmse_by_horizon(metrics, ["XGBoost", "Open-Meteo forecast"]).get_title(loc="left")
    week = T.test_predictions(p, bundle, 6).iloc[:168]
    assert len(plots.week_timeseries(week, ["XGBoost"], "week").get_lines()) == 2
    assert plots.feature_importance(bundle.xgb_models[6], top=5).patches

def test_run_pipeline_cli_defaults(monkeypatch):
    import run_pipeline
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])
    args = run_pipeline.parse_args()
    assert args.years == C.DEFAULT_YEARS and not args.tune and not args.skip_lstm

def test_run_pipeline_main_writes_site(prep, monkeypatch):
    import run_pipeline
    paths, p = prep
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--output-dir", str(paths.root), "--skip-lstm", "--skip-long-range"])
    monkeypatch.setattr(T, "prepare_training", lambda *a, **k: p)
    run_pipeline.main()
    assert C.read_site(paths.artifacts)["years"] == C.DEFAULT_YEARS

def test_run_pipeline_tune_saves_and_reuses_all_params(prep, site_data, monkeypatch):
    # --tune runs every tuner and saves its params; the next run without --tune feeds them to training
    import run_pipeline
    import solarcast.longrange as L
    import solarcast.sources as S
    from solarcast import tune
    paths, p = prep
    _, nsrdb = site_data
    monkeypatch.setattr(T, "prepare_training", lambda *a, **k: p)
    monkeypatch.setattr(S, "fetch_nsrdb", lambda *a, **k: nsrdb)
    monkeypatch.setattr(tune, "tune_xgb", lambda *a, **k: {"max_depth": 3})
    monkeypatch.setattr(tune, "tune_lstm", lambda *a, **k: {"units_1": 64})
    monkeypatch.setattr(tune, "tune_xgb_daily", lambda *a, **k: {"max_depth": 2})
    monkeypatch.setattr(tune, "tune_prophet_long", lambda *a, **k: {"seasonality_mode": "additive"})
    used = {}
    monkeypatch.setattr(T, "train_short_range", lambda prep, paths, lat, lon, use_lstm, xgb_params, lstm_params: used.update(xgb=xgb_params, lstm=lstm_params) or pd.DataFrame(columns=["Horizon", "Forecast inputs", "Model", "MAE (W/m^2)", "RMSE (W/m^2)", "R^2", "MAPE (%, GHI>50)", "Bias (W/m^2)", "Skill vs forecast"]))
    monkeypatch.setattr(L, "train_long_range", lambda *a, xgb_params, prophet_params, dd: used.update(xgb_daily=xgb_params, prophet=prophet_params) or pd.DataFrame())

    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--output-dir", str(paths.root), "--tune"])
    run_pipeline.main()
    for name in ("xgb", "lstm", "xgb_daily", "prophet_long"):
        assert (paths.artifacts / f"{name}_params.json").exists()

    used.clear()
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--output-dir", str(paths.root)])
    run_pipeline.main()
    assert used == {"xgb": {"max_depth": 3}, "lstm": {"units_1": 64}, "xgb_daily": {"max_depth": 2}, "prophet": {"seasonality_mode": "additive"}}

def test_predict_today_main_prints_table(prep, monkeypatch, capsys):
    import predict_today
    import solarcast.predict as P
    paths, p = prep
    T.train_short_range(p, paths, C.DEFAULT_LAT, C.DEFAULT_LON, use_lstm=False)
    C.write_site(paths.artifacts, C.DEFAULT_LAT, C.DEFAULT_LON, [2024])
    table = P.predict_frame(p.hourly, Bundle.load(paths.artifacts), len(p.hourly) - 100, C.HORIZONS, use_lstm=False)
    monkeypatch.setattr(P, "predict_live", lambda *a, **k: (table, p.hourly.index[-100], "America/Chicago"))
    monkeypatch.setattr(sys, "argv", ["predict_today.py", "--artifacts-dir", str(paths.artifacts), "--model", "xgboost"])
    predict_today.main()
    out = capsys.readouterr().out
    assert "+6" in out and "+48" in out and "W/m^2" in out

def test_predict_today_without_models_exits(tmp_path, monkeypatch):
    import predict_today
    monkeypatch.setattr(sys, "argv", ["predict_today.py", "--artifacts-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="No trained models"):
        predict_today.main()
