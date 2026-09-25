import os
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# All frames returned here are indexed by UTC hour-ending labels (tz-aware).
# Label T covers the hour (T-1h, T]: this is how Open-Meteo stamps its hourly radiation (mean of the preceding hour).
# NSRDB hourly values are stamped at the half hour (e.g. 10:30), which is the middle of the Open-Meteo hour labelled 11:00.

# OPEN-METEO
OPEN_METEO_VARS = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "cloud_cover": "cloud_cover",
    "cloud_cover_low": "cloud_low",
    "cloud_cover_mid": "cloud_mid",
    "cloud_cover_high": "cloud_high",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "dew_point_2m": "dew_point",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}

ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_URL = "https://api.open-meteo.com/v1/forecast"

def _open_meteo_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    times = hourly.pop("time")
    idx = (pd.to_datetime(times, unit="s", utc=True) if isinstance(times[0], (int, float))
           else pd.to_datetime(times).tz_localize("UTC"))
    df = pd.DataFrame(hourly, index=pd.DatetimeIndex(idx, name="datetime"))
    return df.rename(columns=OPEN_METEO_VARS)[list(OPEN_METEO_VARS.values())].astype(float)

def fetch_open_meteo_archive(lat: float, lon: float, years: list[int], raw_dir: Path) -> pd.DataFrame:
    # Historical forecasts: the same model output the live API serves, archived hourly.
    # One cached CSV per year so an interrupted download resumes where it stopped.
    frames = []
    for year in sorted(years):
        cache = raw_dir / f"openmeteo_{year}.csv"
        if not cache.exists():
            print(f"Open-Meteo archive {year} >>>", end=" ", flush=True)
            resp = requests.get(ARCHIVE_URL, params={
                "latitude": lat, "longitude": lon,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly": ",".join(OPEN_METEO_VARS), "wind_speed_unit": "ms", "timezone": "UTC",
            }, timeout=300)
            resp.raise_for_status()
            _open_meteo_frame(resp.json()).to_csv(cache)
            print("done")
        frames.append(pd.read_csv(cache, index_col="datetime", parse_dates=True))
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated()]

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

def fetch_open_meteo_previous_runs(lat: float, lon: float, day: int, years: list[int], raw_dir: Path) -> pd.DataFrame:
    # Forecasts issued `day` days before each hour (lead 24*day to 24*day+23 h), available from 2024-01-19.
    # The historical-forecast archive stitches the newest run for every hour, so its "48 h" forecasts are really short-lead ones.
    # These frames give 24 h and 48 h horizons an honest evaluation.
    frames = []
    for year in sorted(y for y in years if y >= 2024):
        cache = raw_dir / f"openmeteo_prev_day{day}_{year}.csv"
        if not cache.exists():
            print(f"Open-Meteo previous runs (day {day}) {year} >>>", end=" ", flush=True)
            resp = requests.get(PREVIOUS_RUNS_URL, params={
                "latitude": lat, "longitude": lon,
                "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                "hourly": ",".join(f"{v}_previous_day{day}" for v in OPEN_METEO_VARS),
                "wind_speed_unit": "ms", "timezone": "UTC",
            }, timeout=300)
            resp.raise_for_status()
            payload = resp.json()
            payload["hourly"] = {k.replace(f"_previous_day{day}", ""): v for k, v in payload["hourly"].items()}
            _open_meteo_frame(payload).to_csv(cache)
            print("done")
        frames.append(pd.read_csv(cache, index_col="datetime", parse_dates=True))
    if not frames:
        return pd.DataFrame(columns=list(OPEN_METEO_VARS.values()))
    return pd.concat(frames).sort_index()

def fetch_open_meteo_live(lat: float, lon: float, past_days: int = 2, forecast_days: int = 3) -> tuple[pd.DataFrame, str]:
    # Returns the hourly frame and the site's IANA timezone name (for display only)
    resp = requests.get(LIVE_URL, params={
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(OPEN_METEO_VARS), "wind_speed_unit": "ms",
        "timezone": "auto", "timeformat": "unixtime",
        "past_days": past_days, "forecast_days": forecast_days,
    }, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return _open_meteo_frame(payload), payload.get("timezone", "UTC")

# NSRDB (ground truth)
NSRDB_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
NSRDB_ATTRIBUTES = "ghi,dni,dhi,air_temperature,relative_humidity,dew_point,wind_speed,wind_direction,surface_albedo,solar_zenith_angle"

def nsrdb_to_hour_ending(df: pd.DataFrame) -> pd.DataFrame:
    # Shift half-hour stamps to the hour-ending label of the hour they sit in (10:30 > 11:00)
    df = df.copy()
    df.index = (df.index + pd.Timedelta(minutes=30)).ceil("h")
    return df[~df.index.duplicated()]

def infer_utc_offset(df: pd.DataFrame, lat: float, lon: float) -> int:
    # Hours to add to local standard time to get UTC, found by matching NSRDB's own solar zenith column against pvlib
    import pvlib
    sample = df.iloc[:: max(1, len(df) // 2000)]
    loc = pvlib.location.Location(lat, lon)
    errors = {}
    for offset in range(-14, 13):
        utc = sample.index + pd.Timedelta(hours=offset)
        zen = loc.get_solarposition(utc.tz_localize("UTC"))["apparent_zenith"].to_numpy()
        errors[offset] = abs(zen - sample["solar_zenith_angle"].to_numpy()).mean()
    return min(errors, key=errors.get)

def split_legacy_nsrdb(legacy_csv: Path, lat: float, lon: float, raw_dir: Path) -> list[int]:
    # One-time migration of the old local-time data/nsrdb_raw.csv (from 1.data_acq.ipynb) into per-year UTC caches
    df = pd.read_csv(legacy_csv, index_col="datetime", parse_dates=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"air_temperature": "temperature"})
    offset = infer_utc_offset(df, lat, lon)
    df.index = df.index + pd.Timedelta(hours=offset)
    print(f"Converted {legacy_csv} from local standard time to UTC (UTC = local {offset:+d} h)")
    years = sorted(df.index.year.unique())
    for year, part in df.groupby(df.index.year):
        cache = raw_dir / f"nsrdb_{year}.csv"
        if not cache.exists():
            part.to_csv(cache)
    return years

def fetch_nsrdb(lat: float, lon: float, years: list[int], raw_dir: Path, legacy_csv: Path | None = None) -> pd.DataFrame:
    # Hourly NSRDB in UTC, one cached CSV per year. Needs NRL_API_KEY and EMAIL in .env.
    # If the API fails and legacy_csv (old local-time download for the same site) exists, it is converted instead.
    from dotenv import load_dotenv
    load_dotenv()
    api_key, email = os.getenv("NRL_API_KEY"), os.getenv("EMAIL")

    missing = [y for y in years if not (raw_dir / f"nsrdb_{y}.csv").exists()]
    if missing and legacy_csv is not None and legacy_csv.exists():
        split_legacy_nsrdb(legacy_csv, lat, lon, raw_dir)

    frames = []
    for year in sorted(years):
        cache = raw_dir / f"nsrdb_{year}.csv"
        if not cache.exists():
            if not api_key:
                raise RuntimeError("NRL_API_KEY is not set; add it to .env (see README)")
            print(f"NSRDB {year} >>>", end=" ", flush=True)
            resp = requests.get(NSRDB_URL, params={
                "api_key": api_key, "email": email, "wkt": f"POINT({lon} {lat})", "names": year,
                "interval": 60, "attributes": NSRDB_ATTRIBUTES, "utc": "true", "leap_day": "true",
                "full_name": "Student", "affiliation": "WGU", "reason": "Academic research", "mailing_list": "false",
            }, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"NSRDB {year}: HTTP {resp.status_code}: {resp.text[:200]}")
            # NSRDB returns 2 metadata rows before the column names row
            df = pd.read_csv(StringIO(resp.text), skiprows=2)
            df.index = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]].rename(columns=str.lower)).rename("datetime")
            df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"])
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            df = df.rename(columns={"air_temperature": "temperature"})
            df.to_csv(cache)
            print(f"{len(df)} rows")
            time.sleep(10)  # NSRDB allows one request per 10 s per key
        frames.append(pd.read_csv(cache, index_col="datetime", parse_dates=True))
    df = pd.concat(frames).sort_index()
    df.index = df.index.tz_localize("UTC")
    return nsrdb_to_hour_ending(df)
