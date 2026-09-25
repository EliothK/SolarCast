import numpy as np
import pandas as pd
import pytest

from solarcast import config as C
from solarcast.evaluate import old_recipe_frame, old_recipe_windows
from solarcast.longrange import _climatology, daily_frame, train_long_range
from tests.conftest import synthetic_site

def test_daily_frame_sums_energy_per_solar_day(site_data):
    _, nsrdb = site_data
    daily = daily_frame(nsrdb, C.DEFAULT_LON)
    # Interior days only; each day's GHI is the sum of its hours (Wh/m^2)
    day = daily.index[5]
    shifted = nsrdb.copy()
    shifted.index = (shifted.index - pd.Timedelta(minutes=30) + pd.Timedelta(hours=C.DEFAULT_LON / 15)).tz_localize(None)
    assert np.isclose(daily.loc[day, "ghi"], shifted.loc[str(day.date()), "ghi"].sum())
    assert 0 < daily["daylight_hours"].max() <= 24

def test_climatology_is_seasonal():
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    train = pd.DataFrame({"ghi": 1000 + 500 * np.sin(2 * np.pi * (idx.dayofyear - 80) / 365.25)}, index=idx)
    clim = _climatology(train, pd.DatetimeIndex(["2023-06-21", "2023-12-21"]))
    assert clim[0] > 1400 and clim[1] < 600

def test_train_long_range_writes_models_and_metrics(tmp_path, monkeypatch):
    import solarcast.config as cfg
    monkeypatch.setattr(cfg, "LONG_HORIZONS", [28])
    _, nsrdb = synthetic_site(hours=24 * 500, start="2022-01-01")
    metrics = train_long_range(nsrdb, C.DEFAULT_LON, tmp_path, tmp_path)
    assert (tmp_path / "xgb_daily_h7d.json").exists() and (tmp_path / "prophet_long_h28d.json").exists()
    assert set(metrics["Model"]) == {"XGBoost daily", "Climatology", "Prophet long"}
    assert metrics["RMSE (Wh/m^2/day)"].notna().all()
    assert not any("W/m^2)" in c for c in metrics.columns)

@pytest.fixture
def daily_data():
    from solarcast.longrange import DailyData
    _, nsrdb = synthetic_site(hours=24 * 500, start="2022-01-01")
    return DailyData.from_nsrdb(nsrdb, C.DEFAULT_LON)

def test_daily_splits_and_windows(daily_data):
    dd = daily_data
    assert 0 < dd.train_end < dd.val_end < len(dd.daily)
    X, y = dd.xgb_split(7, 0, dd.train_end)
    assert X.shape[1] == C.DAILY_WINDOW * dd.daily.shape[1]
    # First sample: window ends at day DAILY_WINDOW-1, target 7 days later
    assert y[0] == dd.daily["ghi"].iloc[C.DAILY_WINDOW - 1 + 7]
    frame = dd.prophet_frame(28)
    assert frame["y"].iloc[0] == dd.daily["ghi"].iloc[28]

def test_tune_xgb_daily(daily_data, tmp_path):
    from solarcast.tune import tune_xgb_daily
    params = tune_xgb_daily(daily_data, n_trials=2, outputs=tmp_path)
    assert {"learning_rate", "max_depth", "gamma"} <= set(params)
    assert len(pd.read_csv(tmp_path / "tuning_xgb_daily_trials.csv")) == 2

def test_tune_prophet_long_picks_lowest_mae(daily_data, tmp_path):
    from solarcast.tune import tune_prophet_long
    grid = {"changepoint_prior_scale": [0.01, 0.5], "seasonality_prior_scale": [1.0], "seasonality_mode": ["additive"]}
    best = tune_prophet_long(daily_data, grid=grid, horizons=[28], n_jobs=1, outputs=tmp_path)
    results = pd.read_csv(tmp_path / "tuning_prophet_long_grid.csv")
    assert len(results) == 2 and results["mean_val_mae"].is_monotonic_increasing
    assert best["changepoint_prior_scale"] == results["changepoint_prior_scale"].iloc[0]
    assert set(best) == set(grid)

def test_train_long_range_uses_given_params(daily_data, tmp_path, monkeypatch):
    import solarcast.config as cfg
    import solarcast.longrange as L
    monkeypatch.setattr(cfg, "LONG_HORIZONS", [28])
    seen, xgb_seen = [], []
    real_fit, real_xgb = L.fit_prophet, L.train_xgb
    monkeypatch.setattr(L, "fit_prophet", lambda df, params=None: seen.append(params) or real_fit(df, params))
    monkeypatch.setattr(L, "train_xgb", lambda *a: xgb_seen.append(a[4]) or real_xgb(*a))
    L.train_long_range(None, C.DEFAULT_LON, tmp_path, tmp_path, xgb_params={**C.XGB_DAILY_PARAMS, "max_depth": 2},
                       prophet_params={**C.PROPHET_LONG_PARAMS, "seasonality_mode": "additive"}, dd=daily_data)
    assert seen == [{**C.PROPHET_LONG_PARAMS, "seasonality_mode": "additive"}]
    assert [p["max_depth"] for p in xgb_seen] == [2] * len(C.MEDIUM_HORIZONS)

def test_old_recipe_frame_from_open_meteo(hourly):
    frame = old_recipe_frame(hourly, C.DEFAULT_LAT, C.DEFAULT_LON, from_open_meteo=True)
    assert (frame["surface_albedo"] == 0.2).all()
    X = old_recipe_windows(frame, np.array([30, 40]))
    assert X.shape == (2, C.WINDOW_SIZE * 17)
