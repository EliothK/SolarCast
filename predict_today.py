import argparse
import sys
from datetime import date
from pathlib import Path

from solarcast import config as C

# Max distance (degrees lat/lon) between the requested site and the site the models were trained on
SITE_TOLERANCE_DEG = 0.1

# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast GHI 6-48 h ahead at a site the models were trained for.")
    p.add_argument("--lat", type=float, default=C.DEFAULT_LAT, help=f"Site latitude (default: {C.DEFAULT_LAT} - Bismarck, ND)")
    p.add_argument("--lon", type=float, default=C.DEFAULT_LON, help=f"Site longitude (default: {C.DEFAULT_LON} - Bismarck, ND)")
    p.add_argument("--model", choices=["xgboost", "lstm", "all"], default="all", help="Which model(s) to run; 'all' also prints the weighted ensemble (default: all)")
    p.add_argument("--horizon", type=int, choices=C.HORIZONS, default=None, help="Forecast horizon in hours ahead (default: every trained horizon)")
    p.add_argument("--artifacts-dir", default=None, help="Directory containing trained models (default: artifacts/ for Bismarck, locations/<site>/artifacts/ for other sites)")
    p.add_argument("--allow-site-mismatch", action="store_true", help="Run even if the artifacts were trained for a different location")
    return p.parse_args()

# SITE CHECK
# Models are site-specific: running Bismarck-trained models on another city gives silently wrong forecasts.
def resolve_artifacts_dir(args: argparse.Namespace) -> Path:
    if args.artifacts_dir:
        return Path(args.artifacts_dir)
    return C.Paths.for_site(args.lat, args.lon).artifacts

def check_site(arts: Path, lat: float, lon: float, allow_mismatch: bool) -> None:
    site = C.read_site(arts)
    if abs(site["lat"] - lat) <= SITE_TOLERANCE_DEG and abs(site["lon"] - lon) <= SITE_TOLERANCE_DEG:
        return
    msg = (f"Artifacts in {arts} were trained for lat={site['lat']}, lon={site['lon']}, "
           f"but you asked for lat={lat}, lon={lon}.")
    if allow_mismatch:
        print(f"WARNING: {msg} Continuing (--allow-site-mismatch).\n")
        return
    sys.exit(f"{msg}\nTrain models for this site first:\npython run_pipeline.py --lat {lat} --lon {lon}\n"
             "or pass --allow-site-mismatch to use them anyway.")

# MAIN
def main() -> None:
    args = parse_args()
    arts = resolve_artifacts_dir(args)

    print(f"Solar GHI Forecast - {date.today().isoformat()}")
    print(f"Location: lat={args.lat}, lon={args.lon}")
    print(f"Artifacts: {arts}\n")

    if not (arts / "bundle.json").exists():
        sys.exit(f"No trained models found in {arts} (bundle.json missing).\nRun the pipeline first:\npython run_pipeline.py --lat {args.lat} --lon {args.lon}")
    check_site(arts, args.lat, args.lon, args.allow_site_mismatch)

    # Imported here so --help and the checks above stay fast (TensorFlow loads slowly)
    from solarcast.predict import predict_live

    horizons = [args.horizon] if args.horizon else C.HORIZONS
    print("Fetching weather data from Open-Meteo")
    table, issued, tz = predict_live(args.lat, args.lon, arts, horizons, use_lstm=args.model in ("lstm", "all"))
    print(f"Latest complete hour: {issued.tz_convert(tz):%Y-%m-%d %H:%M %Z}\n")

    columns = {"xgboost": ["XGBoost"], "lstm": ["LSTM"], "all": ["XGBoost", "LSTM", "Ensemble"]}[args.model]
    columns = [c for c in columns if c in table.columns] + ["Open-Meteo forecast", "Clear sky"]
    print(f"{'Horizon':<9}{'Target time':<22}" + "".join(f"{c:>21}" for c in columns))
    for h, row in table.iterrows():
        target = row["target_time"].tz_convert(tz).strftime("%Y-%m-%d %H:%M")
        if row["zenith"] >= 90:
            print(f"+{h:<8}{target:<22}{'night (GHI = 0 W/m^2)':>21}")
            continue
        print(f"+{h:<8}{target:<22}" + "".join(f"{row[c]:>15.1f} W/m^2" for c in columns))

    print("\nValues are mean GHI over the hour ending at the target time.")
    if args.model == "all":
        print("Ensemble = XGBoost and LSTM weighted by inverse validation error (see artifacts/bundle.json).")

if __name__ == "__main__":
    main()
