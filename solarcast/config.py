import json
from dataclasses import dataclass
from pathlib import Path

# Default site (Bismarck, ND): its data and artifacts live in the project root
DEFAULT_LAT = 46.69115
DEFAULT_LON = -100.83192

# Years with both NSRDB truth and Open-Meteo historical forecasts (the archive has data from 2018)
DEFAULT_YEARS = list(range(2018, 2025))

# Short-range forecast horizons in hours ahead
HORIZONS = [6, 12, 24, 48]

# Hours of Open-Meteo history each sample sees
WINDOW_SIZE = 24

# Below this clear-sky GHI (W/m^2) the clear-sky index is too noisy to predict; the raw forecast is used instead
CS_MIN = 20.0

# Upper clip for the clear-sky index (cloud enhancement can briefly push GHI above clear sky)
KT_MAX = 1.5

# Time-ordered train/val/test fractions, separated by a gap of max(HORIZONS) hours so no target crosses a split
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Lead-matched data (Open-Meteo previous runs, from 2024-01-19) for horizons >= 24 h.
# Its issue times are split in time into extra training rows, a validation slice (ensemble weights) and an honest test slice.
LEAD_TRAIN_FRAC = 0.50
LEAD_VAL_FRAC = 0.10
# Training weight of a lead-matched row relative to an archive row (chosen on 2024 data: x5 and x20 scored alike)
LEAD_WEIGHT = 5.0

RANDOM_SEED = 42

# Default XGBoost settings (replaced by tuned values when artifacts/xgb_params.json exists)
XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}
XGB_EARLY_STOPPING = 50

# Default LSTM settings (replaced by tuned values when artifacts/lstm_params.json exists)
LSTM_PARAMS = {
    "units_1": 128,
    "units_2": 64,
    "dense_units": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 256,
}
LSTM_EPOCHS = 60
LSTM_PATIENCE = 8

# Default settings for the daily models (replaced by tuned values when artifacts/xgb_daily_params.json or prophet_long_params.json exist)
XGB_DAILY_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8}
PROPHET_LONG_PARAMS = {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0, "seasonality_mode": "multiplicative"}

# Prophet grid searched by --tune (from 6.hyperparameter_tuning.ipynb), scored on these horizons (days)
PROPHET_GRID = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    "seasonality_mode": ["additive", "multiplicative"],
}
PROPHET_TUNE_HORIZONS = [28, 84, 336]

# Long-range (daily) models, ported from 3.modeling.ipynb
DAILY_WINDOW = 30
MEDIUM_HORIZONS = [7, 14]
LONG_HORIZONS = [28, 56, 84, 168, 336]

def location_dir(lat: float, lon: float) -> Path:
    # Output directory for non-default sites
    # 34.05, -118.25 > locations/34p0500_W118p2500
    # -33.87, 151.21 > locations/S33p8700_151p2100
    def fmt(val: float, neg_prefix: str) -> str:
        prefix = neg_prefix if val < 0 else ""
        return f"{prefix}{abs(val):.4f}".replace(".", "p")

    return Path("locations") / f"{fmt(lat, 'S')}_{fmt(lon, 'W')}"

def is_default_site(lat: float, lon: float) -> bool:
    return abs(lat - DEFAULT_LAT) < 1e-4 and abs(lon - DEFAULT_LON) < 1e-4

@dataclass
class Paths:
    root: Path

    @classmethod
    def for_site(cls, lat: float, lon: float, output_dir: str | None = None) -> "Paths":
        if output_dir:
            return cls(Path(output_dir))
        return cls(Path(".") if is_default_site(lat, lon) else location_dir(lat, lon))

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    def makedirs(self) -> None:
        for d in (self.raw, self.artifacts, self.outputs):
            d.mkdir(parents=True, exist_ok=True)

def write_site(artifacts: Path, lat: float, lon: float, years: list[int]) -> None:
    site = {"lat": lat, "lon": lon, "years": sorted(years)}
    (artifacts / "site.json").write_text(json.dumps(site, indent=2))

def save_params(artifacts: Path, name: str, params: dict) -> None:
    # Tuned settings, e.g. name="xgb" > artifacts/xgb_params.json
    (artifacts / f"{name}_params.json").write_text(json.dumps(params, indent=2))

def load_params(artifacts: Path, name: str) -> dict | None:
    path = artifacts / f"{name}_params.json"
    return json.loads(path.read_text()) if path.exists() else None

def read_site(artifacts: Path) -> dict:
    site_path = artifacts / "site.json"
    if site_path.exists():
        return json.loads(site_path.read_text())
    # Artifacts from before site.json existed were all trained on the default site
    return {"lat": DEFAULT_LAT, "lon": DEFAULT_LON}
