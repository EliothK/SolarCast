import numpy as np
import pandas as pd

from solarcast.config import CS_MIN, DEFAULT_LAT, DEFAULT_LON, KT_MAX
from solarcast.solar import clear_sky_index, solar_geometry
from solarcast.sources import _open_meteo_frame, nsrdb_to_hour_ending

def test_nsrdb_half_hour_maps_to_hour_ending_label():
    idx = pd.DatetimeIndex(["2024-06-01 10:30", "2024-06-01 11:30"], tz="UTC")
    out = nsrdb_to_hour_ending(pd.DataFrame({"ghi": [1, 2]}, index=idx))
    assert list(out.index) == list(pd.DatetimeIndex(["2024-06-01 11:00", "2024-06-01 12:00"], tz="UTC"))

def test_open_meteo_unixtime_and_iso_agree():
    hourly = {name: [1.0, 2.0] for name in ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "cloud_cover", "cloud_cover_low",
              "cloud_cover_mid", "cloud_cover_high", "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m", "wind_direction_10m"]}
    iso = _open_meteo_frame({"hourly": {"time": ["2024-06-01T10:00", "2024-06-01T11:00"], **hourly}})
    unix = _open_meteo_frame({"hourly": {"time": [1717236000, 1717239600], **hourly}})
    assert iso.index.equals(unix.index)
    assert str(iso.index.tz) == "UTC"

def test_geometry_uses_mid_hour():
    # The label 19:00 UTC covers 18:00-19:00, so its zenith must equal the sun's position at 18:30
    label = pd.DatetimeIndex(["2024-06-21 19:00"], tz="UTC")
    mid = pd.DatetimeIndex(["2024-06-21 18:30"], tz="UTC")
    at_label = solar_geometry(label, DEFAULT_LAT, DEFAULT_LON)["zenith"].iloc[0]
    at_mid_as_label = solar_geometry(mid + pd.Timedelta(minutes=30), DEFAULT_LAT, DEFAULT_LON)["zenith"].iloc[0]
    assert at_label == at_mid_as_label

def test_solar_noon_near_local_solar_time():
    idx = pd.date_range("2024-06-21", periods=24, freq="h", tz="UTC")
    geo = solar_geometry(idx, DEFAULT_LAT, DEFAULT_LON)
    noon_label = geo["zenith"].idxmin()
    # Solar noon at -100.8 lon is about 18:43 UTC, so the hour (18:00, 19:00] has the highest sun
    assert noon_label.hour == 19

def test_clear_sky_index_masks_low_sun_and_clips():
    kt = clear_sky_index(np.array([10.0, 500.0, 900.0]), np.array([CS_MIN - 1, 1000.0, 500.0]), CS_MIN, KT_MAX)
    assert kt[0] == 0.0
    assert kt[1] == 0.5
    assert kt[2] == KT_MAX
