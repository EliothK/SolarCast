import numpy as np
import pandas as pd

from solarcast.config import CS_MIN, WINDOW_SIZE
from solarcast.features import (PAST_COLS, build_tabular, forecast_frame, kt_to_ghi, past_lag_features, sequence_windows,
                                target_features, target_kt)

def test_past_features_ignore_future(hourly):
    # Changing everything after issue time t must not change the past features at t
    t = 60
    before = past_lag_features(hourly).iloc[t]
    changed = hourly.copy()
    changed.iloc[t + 1:, :] = changed.iloc[t + 1:, :] * 3 + 7
    after = past_lag_features(changed).iloc[t]
    pd.testing.assert_series_equal(before, after)

def test_target_features_read_target_hour(hourly):
    h, t = 6, 50
    tf = target_features(hourly, h)
    assert tf["fc_ghi"].iloc[t] == hourly["ghi"].iloc[t + h]
    assert tf["tgt_cs_ghi"].iloc[t] == hourly["cs_ghi"].iloc[t + h]
    assert tf["horizon"].iloc[t] == h

def test_forecast_override_changes_only_forecast_columns(hourly, om_frame):
    h, t = 24, 40
    alt = forecast_frame(om_frame.assign(ghi=om_frame["ghi"] * 0.5), hourly)
    base, over = build_tabular(hourly, h), build_tabular(hourly, h, alt)
    assert over["fc_ghi"].iloc[t] == hourly["ghi"].iloc[t + h] * 0.5
    past_cols = [c for c in base.columns if "_lag" in c]
    pd.testing.assert_frame_equal(base[past_cols], over[past_cols])
    assert over["tgt_zenith"].iloc[t] == base["tgt_zenith"].iloc[t]

def test_target_kt_uses_truth_at_target_hour(hourly):
    truth = hourly["cs_ghi"] * 0.8
    tgt = target_kt(hourly, truth, 6)
    usable = tgt["usable"].to_numpy()
    assert usable.any()
    np.testing.assert_allclose(tgt["kt"].to_numpy()[usable], 0.8)
    assert (tgt.loc[tgt["cs_ghi"] <= CS_MIN, "usable"] == False).all()

def test_sequence_windows_end_at_issue_time():
    values = np.arange(100, dtype=float).reshape(50, 2)
    win = sequence_windows(values, np.array([23, 40]), WINDOW_SIZE)
    assert win.shape == (2, WINDOW_SIZE, 2)
    np.testing.assert_array_equal(win[0, -1], values[23])
    np.testing.assert_array_equal(win[1, 0], values[40 - WINDOW_SIZE + 1])

def test_lstm_window_matches_past_cols(hourly):
    win = sequence_windows(hourly[PAST_COLS].to_numpy(), np.array([30]))
    np.testing.assert_array_equal(win[0, -1], hourly[PAST_COLS].iloc[30].to_numpy())

def test_kt_to_ghi_falls_back_to_forecast_at_low_sun():
    target = pd.DataFrame({"tgt_cs_ghi": [800.0, 10.0, 0.0], "fc_ghi": [500.0, 30.0, 5.0]})
    ghi = kt_to_ghi(np.array([0.5, 0.9, 0.9]), target)
    np.testing.assert_allclose(ghi, [400.0, 10.0, 0.0])
