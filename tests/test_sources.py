import numpy as np
import pandas as pd
import pytest

from solarcast import sources
from solarcast.config import DEFAULT_LAT, DEFAULT_LON

class FakeResponse:
    def __init__(self, payload=None, text="", status=200):
        self._payload, self.text, self.status_code = payload, text, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

def _om_payload(times, suffix=""):
    return {"hourly": {"time": times, **{f"{v}{suffix}": [1.0] * len(times) for v in sources.OPEN_METEO_VARS}}, "timezone": "America/Chicago"}

def test_archive_download_is_cached_per_year(tmp_path, monkeypatch):
    calls = []
    def fake_get(url, params, timeout):
        calls.append(params["start_date"])
        return FakeResponse(_om_payload(["2024-01-01T00:00", "2024-01-01T01:00"]))
    monkeypatch.setattr(sources.requests, "get", fake_get)
    first = sources.fetch_open_meteo_archive(DEFAULT_LAT, DEFAULT_LON, [2024], tmp_path)
    second = sources.fetch_open_meteo_archive(DEFAULT_LAT, DEFAULT_LON, [2024], tmp_path)
    assert calls == ["2024-01-01"]
    assert list(first.columns) == list(sources.OPEN_METEO_VARS.values())
    assert str(first.index.tz) == "UTC" and first.equals(second)

def test_previous_runs_strip_suffix_and_skip_early_years(tmp_path, monkeypatch):
    requested = []
    def fake_get(url, params, timeout):
        requested.append(params["hourly"])
        return FakeResponse(_om_payload(["2024-02-01T00:00"], suffix="_previous_day2"))
    monkeypatch.setattr(sources.requests, "get", fake_get)
    df = sources.fetch_open_meteo_previous_runs(DEFAULT_LAT, DEFAULT_LON, 2, [2022, 2023, 2024], tmp_path)
    assert len(requested) == 1 and "shortwave_radiation_previous_day2" in requested[0]
    assert "ghi" in df.columns and len(df) == 1

def test_previous_runs_empty_before_2024(tmp_path):
    df = sources.fetch_open_meteo_previous_runs(DEFAULT_LAT, DEFAULT_LON, 1, [2020], tmp_path)
    assert df.empty and "ghi" in df.columns

def test_live_fetch_returns_timezone(monkeypatch):
    monkeypatch.setattr(sources.requests, "get", lambda url, params, timeout: FakeResponse(_om_payload([1717236000, 1717239600])))
    df, tz = sources.fetch_open_meteo_live(DEFAULT_LAT, DEFAULT_LON)
    assert tz == "America/Chicago" and len(df) == 2

NSRDB_CSV = """Source,Location ID
NSRDB,1
Year,Month,Day,Hour,Minute,GHI,DNI,DHI,Temperature,Relative Humidity,Dew Point,Wind Speed,Wind Direction,Surface Albedo,Solar Zenith Angle
2024,6,1,10,30,5,1,4,20,50,10,3,180,0.2,80
2024,6,1,11,30,50,10,40,21,49,10,3,180,0.2,70
"""

def test_nsrdb_fetch_parses_and_relabels(tmp_path, monkeypatch):
    monkeypatch.setenv("NRL_API_KEY", "key")
    monkeypatch.setattr(sources.requests, "get", lambda url, params, timeout: FakeResponse(text=NSRDB_CSV))
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    df = sources.fetch_nsrdb(DEFAULT_LAT, DEFAULT_LON, [2024], tmp_path)
    assert list(df.index) == list(pd.DatetimeIndex(["2024-06-01 11:00", "2024-06-01 12:00"], tz="UTC"))
    assert df["ghi"].tolist() == [5, 50] and "temperature" in df.columns

def test_nsrdb_fetch_raises_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NRL_API_KEY", "key")
    monkeypatch.setattr(sources.requests, "get", lambda url, params, timeout: FakeResponse(text="Data processing failure.", status=400))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        sources.fetch_nsrdb(DEFAULT_LAT, DEFAULT_LON, [2024], tmp_path)

def test_nsrdb_fetch_needs_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("NRL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NRL_API_KEY"):
        sources.fetch_nsrdb(DEFAULT_LAT, DEFAULT_LON, [2024], tmp_path)

def _local_time_nsrdb(offset: int) -> pd.DataFrame:
    # NSRDB-style frame in local standard time (UTC - offset) with a correct zenith column
    import pvlib
    utc = pd.date_range("2024-01-01 00:30", periods=24 * 20, freq="h")
    zen = pvlib.location.Location(DEFAULT_LAT, DEFAULT_LON).get_solarposition(utc.tz_localize("UTC"))["apparent_zenith"].to_numpy()
    local = utc - pd.Timedelta(hours=offset)
    return pd.DataFrame({"GHI": np.clip(90 - zen, 0, None) * 10, "Solar Zenith Angle": zen}, index=pd.DatetimeIndex(local, name="datetime"))

def test_infer_utc_offset_recovers_known_offset():
    df = _local_time_nsrdb(7).rename(columns={"Solar Zenith Angle": "solar_zenith_angle"})
    assert sources.infer_utc_offset(df, DEFAULT_LAT, DEFAULT_LON) == 7

def test_legacy_csv_is_split_into_utc_caches(tmp_path):
    legacy = tmp_path / "nsrdb_raw.csv"
    _local_time_nsrdb(7).to_csv(legacy)
    years = sources.split_legacy_nsrdb(legacy, DEFAULT_LAT, DEFAULT_LON, tmp_path)
    assert years == [2024]
    cached = pd.read_csv(tmp_path / "nsrdb_2024.csv", index_col="datetime", parse_dates=True)
    assert cached.index[0] == pd.Timestamp("2024-01-01 00:30") and "ghi" in cached.columns

def test_fetch_nsrdb_uses_legacy_when_caches_missing(tmp_path, monkeypatch):
    legacy = tmp_path / "nsrdb_raw.csv"
    _local_time_nsrdb(7).to_csv(legacy)
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: pytest.fail("API must not be called"))
    df = sources.fetch_nsrdb(DEFAULT_LAT, DEFAULT_LON, [2024], tmp_path, legacy_csv=legacy)
    assert df.index[0] == pd.Timestamp("2024-01-01 01:00", tz="UTC")
