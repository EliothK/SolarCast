import argparse
import json

import pandas as pd
import pytest

import predict_today
from solarcast.config import DEFAULT_LAT, DEFAULT_LON
from solarcast.predict import issue_position

def test_site_mismatch_exits(tmp_path):
    (tmp_path / "site.json").write_text(json.dumps({"lat": DEFAULT_LAT, "lon": DEFAULT_LON}))
    with pytest.raises(SystemExit):
        predict_today.check_site(tmp_path, 40.7128, -74.006, allow_mismatch=False)

def test_site_mismatch_allowed(tmp_path, capsys):
    (tmp_path / "site.json").write_text(json.dumps({"lat": DEFAULT_LAT, "lon": DEFAULT_LON}))
    predict_today.check_site(tmp_path, 40.7128, -74.006, allow_mismatch=True)
    assert "WARNING" in capsys.readouterr().out

def test_missing_site_json_means_default_site(tmp_path):
    predict_today.check_site(tmp_path, DEFAULT_LAT, DEFAULT_LON, allow_mismatch=False)

def test_artifacts_dir_follows_site():
    args = argparse.Namespace(artifacts_dir=None, lat=40.7128, lon=-74.006)
    assert str(predict_today.resolve_artifacts_dir(args)) == "locations/40p7128_W74p0060/artifacts"
    args = argparse.Namespace(artifacts_dir=None, lat=DEFAULT_LAT, lon=DEFAULT_LON)
    assert str(predict_today.resolve_artifacts_dir(args)) == "artifacts"

def test_issue_position_is_latest_complete_hour(hourly):
    now = hourly.index[30] + pd.Timedelta(minutes=45)
    assert issue_position(hourly, now) == 30
