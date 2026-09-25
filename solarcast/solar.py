import numpy as np
import pandas as pd
import pvlib

def solar_geometry(index: pd.DatetimeIndex, lat: float, lon: float, altitude: float = 0.0) -> pd.DataFrame:
    # Zenith and clear-sky irradiance for hour-ending labels.
    # Each label T stands for the hour (T-1h, T], so the sun is evaluated at T-30min, the middle of that hour.
    # This matches both Open-Meteo's hourly means and NSRDB's half-hour stamps.
    mid = index - pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(lat, lon, altitude=altitude)
    pos = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen", solar_position=pos)

    out = pd.DataFrame(index=index)
    out["zenith"] = pos["apparent_zenith"].to_numpy()
    out["cos_zenith"] = np.cos(np.radians(out["zenith"])).clip(lower=0)
    out["cs_ghi"] = cs["ghi"].to_numpy()
    out["cs_dni"] = cs["dni"].to_numpy()

    # Local solar time (not clock time): noon is when the sun is highest, independent of time zones and DST
    solar_hour = (mid.hour + mid.minute / 60 + lon / 15) % 24
    out["solar_hour_sin"] = np.sin(2 * np.pi * solar_hour / 24)
    out["solar_hour_cos"] = np.cos(2 * np.pi * solar_hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * mid.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * mid.dayofyear / 365.25)
    return out

def clear_sky_index(ghi: pd.Series | np.ndarray, cs_ghi: pd.Series | np.ndarray, cs_min: float, kt_max: float) -> np.ndarray:
    # GHI / clear-sky GHI, set to 0 where the sun is too low for the ratio to mean anything
    ghi, cs_ghi = np.asarray(ghi, dtype=float), np.asarray(cs_ghi, dtype=float)
    kt = np.divide(ghi, cs_ghi, out=np.zeros_like(ghi), where=cs_ghi > cs_min)
    return np.clip(kt, 0.0, kt_max)
