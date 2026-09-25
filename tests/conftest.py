import os
import sys
from pathlib import Path

# Tests run on CPU so they are deterministic and never compete with a training run for GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solarcast.config import DEFAULT_LAT, DEFAULT_LON
from solarcast.sources import OPEN_METEO_VARS

@pytest.fixture
def om_frame() -> pd.DataFrame:
    # Synthetic Open-Meteo frame: 5 days of hourly UTC data with a daytime GHI bump
    idx = pd.date_range("2024-06-01", periods=120, freq="h", tz="UTC", name="datetime")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(0, 100, size=(len(idx), len(OPEN_METEO_VARS))), index=idx, columns=list(OPEN_METEO_VARS.values()))
    solar_hour = (idx.hour + DEFAULT_LON / 15) % 24
    df["ghi"] = np.clip(np.sin((solar_hour - 6) / 12 * np.pi), 0, None) * 800
    return df

@pytest.fixture
def hourly(om_frame):
    from solarcast.features import prepare_hourly
    return prepare_hourly(om_frame, DEFAULT_LAT, DEFAULT_LON)

def synthetic_site(hours: int = 3000, start: str = "2024-01-01"):
    # Open-Meteo-like inputs and NSRDB-like truth long enough for splits, gaps and windows.
    # Truth = clear sky x a cloud factor the forecast partly knows, so models have something to learn.
    from solarcast.solar import solar_geometry
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="datetime")
    rng = np.random.default_rng(1)
    geo = solar_geometry(idx, DEFAULT_LAT, DEFAULT_LON)
    cloud = pd.Series(rng.uniform(0, 1, hours), index=idx).rolling(6, min_periods=1).mean()
    om = pd.DataFrame(rng.uniform(0, 100, size=(hours, len(OPEN_METEO_VARS))), index=idx, columns=list(OPEN_METEO_VARS.values()))
    om["cloud_cover"] = cloud * 100
    om["ghi"] = geo["cs_ghi"] * (1 - 0.7 * cloud) * 1.05
    truth = geo["cs_ghi"] * (1 - 0.7 * cloud)
    nsrdb = pd.DataFrame({
        "ghi": truth, "dni": truth * 0.8, "dhi": truth * 0.2, "temperature": om["temperature"],
        "relative_humidity": om["relative_humidity"], "dew_point": om["dew_point"], "wind_speed": om["wind_speed"],
        "wind_direction": om["wind_direction"], "surface_albedo": 0.2, "solar_zenith_angle": geo["zenith"],
    }, index=idx)
    return om, nsrdb

@pytest.fixture
def site_data():
    return synthetic_site()
