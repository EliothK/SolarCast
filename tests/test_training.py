import numpy as np
import pandas as pd

from solarcast.config import HORIZONS, location_dir
from solarcast.evaluate import regression_metrics
from solarcast.models import combine, inverse_error_weights
from solarcast.train import make_splits, split_positions

def test_splits_are_ordered_and_gapped():
    s = make_splits(10_000)
    gap = max(HORIZONS)
    assert s.train[-1] + gap < s.val[0]
    assert s.val[-1] + gap < s.test[0]
    # No target (issue + horizon) of one split lands in the next
    assert s.train[-1] + gap < s.val[0] and s.test[-1] + gap < 10_000

def test_split_positions_respects_gap():
    pos = np.arange(1000, 3000)
    s = split_positions(pos, 0.5, 0.1, gap=48)
    assert s.train.max() + 48 < s.val.min()
    assert s.val.max() + 48 < s.test.min()
    assert len(s.test) > 0

def test_inverse_error_weights_favour_lower_error():
    w = inverse_error_weights({"a": 100.0, "b": 400.0})
    assert abs(sum(w.values()) - 1) < 1e-12
    assert abs(w["a"] - 0.8) < 1e-12

def test_combine_uses_weights():
    out = combine({"a": np.array([10.0]), "b": np.array([20.0])}, {"a": 0.75, "b": 0.25})
    assert out[0] == 12.5

def test_regression_metrics_report_all_measures():
    m = regression_metrics(np.array([100.0, 200.0, 0.0]), np.array([110.0, 190.0, 0.0]))
    assert set(m) >= {"MAE (W/m^2)", "MSE", "RMSE (W/m^2)", "R^2", "MAPE (%, GHI>50)", "Bias (W/m^2)"}
    assert m["MSE"] == np.mean([100.0, 100.0, 0.0])
    assert abs(m["MAPE (%, GHI>50)"] - 7.5) < 1e-9

def test_location_dir_format():
    assert str(location_dir(34.05, -118.25)) == "locations/34p0500_W118p2500"
    assert str(location_dir(-33.87, 151.21)) == "locations/S33p8700_151p2100"
