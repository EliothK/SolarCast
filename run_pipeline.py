import argparse
import time

import pandas as pd

from solarcast import config as C

# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download data, train and evaluate the GHI forecasting models for one site.")
    p.add_argument("--lat", type=float, default=C.DEFAULT_LAT, help=f"Site latitude (default: {C.DEFAULT_LAT} - Bismarck, ND)")
    p.add_argument("--lon", type=float, default=C.DEFAULT_LON, help=f"Site longitude (default: {C.DEFAULT_LON} - Bismarck, ND)")
    p.add_argument("--years", type=int, nargs="+", default=C.DEFAULT_YEARS,
        help="Years for the 6-48 h models (default: 2018-2024). Needs NSRDB and Open-Meteo archive data; include 2024 so 24/48 h models get real day-ahead forecasts.")
    p.add_argument("--daily-years", type=int, nargs="+", default=list(range(2015, 2025)), help="NSRDB years for the 7-336 day models (default: 2015-2024)")
    p.add_argument("--tune", action="store_true",
        help="Search hyperparameters before training: Optuna for both XGBoost models, Keras Tuner for the LSTM, a grid for Prophet. Results are saved to artifacts/*_params.json and reused by later runs.")
    p.add_argument("--trials", type=int, default=30, help="Optuna trials per XGBoost search with --tune (default: 30)")
    p.add_argument("--lstm-trials", type=int, default=10, help="Keras Tuner trials with --tune (default: 10, about 1 min each on a GPU)")
    p.add_argument("--skip-lstm", action="store_true", help="Train XGBoost only (much faster, no TensorFlow needed)")
    p.add_argument("--skip-long-range", action="store_true", help="Skip the daily 7-336 day models")
    p.add_argument("--output-dir", default=None,
        help="Override the output directory for data/, artifacts/, outputs/. Defaults to the project root for Bismarck, or locations/{lat}_{lon}/ for any other site.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    paths = C.Paths.for_site(args.lat, args.lon, args.output_dir)
    paths.makedirs()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)

    print("Solar GHI Forecasting Pipeline")
    print(f"Location: lat={args.lat}, lon={args.lon}")
    print(f"Years: {min(args.years)}-{max(args.years)} (6-48 h), {min(args.daily_years)}-{max(args.daily_years)} (daily)")
    print(f"Output dir: {paths.root.resolve()}\n")
    t0 = time.time()

    from solarcast.train import lstm_datasets, prepare_training, train_short_range
    prep = prepare_training(paths, args.lat, args.lon, args.years)

    if args.tune:
        from solarcast import tune
        print(f"\nTuning XGBoost ({args.trials} trials)")
        C.save_params(paths.artifacts, "xgb", tune.tune_xgb(prep.data, [h for h in (6, 24) if h in prep.data], args.trials, paths.outputs))
        if not args.skip_lstm:
            print(f"\nTuning LSTM ({args.lstm_trials} trials)")
            C.save_params(paths.artifacts, "lstm", tune.tune_lstm(lstm_datasets(prep), args.lstm_trials, paths.outputs / "kt_lstm", paths.outputs))
    params = {name: C.load_params(paths.artifacts, name) for name in ("xgb", "lstm", "xgb_daily", "prophet_long")}
    for name, p in params.items():
        if p and not args.tune:
            print(f"Using tuned {name} settings from {paths.artifacts / f'{name}_params.json'}")

    print("\nShort range (6-48 h)")
    short = train_short_range(prep, paths, args.lat, args.lon, use_lstm=not args.skip_lstm,
                              xgb_params=params["xgb"], lstm_params=params["lstm"])

    long = None
    if not args.skip_long_range:
        from solarcast.longrange import DailyData, train_long_range
        from solarcast.sources import fetch_nsrdb
        print("\nLong range (7-336 days)")
        nsrdb = fetch_nsrdb(args.lat, args.lon, args.daily_years, paths.raw, legacy_csv=paths.data / "nsrdb_raw.csv")
        dd = DailyData.from_nsrdb(nsrdb, args.lon)
        if args.tune:
            from solarcast import tune
            print(f"\nTuning daily XGBoost ({args.trials} trials)")
            params["xgb_daily"] = tune.tune_xgb_daily(dd, args.trials, paths.outputs)
            C.save_params(paths.artifacts, "xgb_daily", params["xgb_daily"])
            print("\nTuning long-range Prophet (grid search)")
            params["prophet_long"] = tune.tune_prophet_long(dd, outputs=paths.outputs)
            C.save_params(paths.artifacts, "prophet_long", params["prophet_long"])
        long = train_long_range(nsrdb, args.lon, paths.artifacts, paths.outputs,
                                xgb_params=params["xgb_daily"], prophet_params=params["prophet_long"], dd=dd)

    # Written only on success so a failed run never relabels older artifacts
    C.write_site(paths.artifacts, args.lat, args.lon, args.years)

    # SUMMARY
    m, s = divmod(int(time.time() - t0), 60)
    print(f"\nPipeline complete in {m}m {s}s")
    cols = ["Horizon", "Forecast inputs", "Model", "MAE (W/m^2)", "RMSE (W/m^2)", "R^2", "MAPE (%, GHI>50)", "Bias (W/m^2)", "Skill vs forecast"]
    print(f"\nShort-range test metrics ({paths.outputs / 'metrics_short_range.csv'}):")
    print(short[cols].to_string(index=False))
    if long is not None:
        print(f"\nLong-range test metrics ({paths.outputs / 'metrics_long_range.csv'}):")
        print(long.to_string(index=False))

    predict_cmd = f"python predict_today.py --lat {args.lat} --lon {args.lon}"
    if args.output_dir:
        predict_cmd += f" --artifacts-dir {paths.artifacts}"
    print(f"\nTo forecast GHI now:\n{predict_cmd}")

if __name__ == "__main__":
    main()
