import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat

# CLI
DEFAULT_LAT = 46.69115
DEFAULT_LON = -100.83192

NOTEBOOKS = [
    ("1.data_acq.ipynb", "Step 1 - Data Acquisition"),
    ("2.preprocessing.ipynb", "Step 2 - Preprocessing"),
    ("3.modeling.ipynb", "Step 3 - Model Training"),
    ("4.evaluation.ipynb", "Step 4 - Evaluation"),
    ("5.cross_validation.ipynb", "Step 5 - Cross-Validation"),
    ("6.hyperparameter_tuning.ipynb", "Step 6 - Hyperparameter Tuning"),
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full GHI forecasting pipeline for any location.")
    p.add_argument("--lat", type=float, default=DEFAULT_LAT, help=f"Site latitude (default: {DEFAULT_LAT} - Bismarck, ND)")
    p.add_argument("--lon", type=float, default=DEFAULT_LON, help=f"Site longitude (default: {DEFAULT_LON} - Bismarck, ND)")
    p.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2025)), help="Training years, space-separated (default: 2015 - 2024)")
    p.add_argument("--skip-data-acq", action="store_true", help="Skip 1_data_acq.ipynb if raw data is already downloaded")
    p.add_argument("--skip-tuning", action="store_true",help="Skip 6_hyperparameter_tuning.ipynb (can take several hours)")
    p.add_argument("--output-dir", default=None,
        help="Override the output directory for data/, artifacts/, outputs/. Defaults to the project root for Bismarck, or locations/{lat}_{lon}/ for any other site."
    )
    return p.parse_args()

# DIRECTORY NAMING
def _location_dir(lat: float, lon: float) -> Path:
    # Convert lat/lon to a filesystem safe directory name
    # 34.05, -118.25 > locations/34p0500_W118p2500
    # -33.87, 151.21 > locations/S33p8700_151p2100
    def fmt(val: float, neg_prefix: str) -> str:
        prefix = neg_prefix if val < 0 else ""
        return f"{prefix}{abs(val):.4f}".replace(".", "p")

    return Path("locations") / f"{fmt(lat, 'S')}_{fmt(lon, 'W')}"

# NOTEBOOK PATCHING
def _temp_nb(nb: nbformat.NotebookNode, original: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, dir=original.parent)
    nbformat.write(nb, tmp.name)
    tmp.close()
    return Path(tmp.name)

def _patch_notebook_1(nb_path: Path, lat: float, lon: float, years: list, output_dir: Path) -> Path:
    #Patch 1.data_acq.ipynb: update LAT, LONG, YEARS, and OUTPUT_PATH so data is fetched for the requested location and saved under output_dir.
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    src = nb.cells[0]["source"]

    sy = sorted(years)
    years_repr = (f"list(range({sy[0]},{sy[-1] + 1}))" if sy == list(range(sy[0], sy[-1] + 1)) else str(sy))
    raw_csv = str(output_dir / "data" / "nsrdb_raw.csv").replace("\\", "/")

    src = re.sub(r"LAT\s*=\s*[\d.\-]+", f"LAT = {lat}", src)
    src = re.sub(r"LONG\s*=\s*[\d.\-]+", f"LONG = {lon}", src)
    src = re.sub(r"YEARS\s*=\s*list\(range\([^)]+\)\)", f"YEARS = {years_repr}", src)
    src = re.sub(r'OUTPUT_PATH\s*=\s*"[^"]+"', f'OUTPUT_PATH = "{raw_csv}"', src)

    nb.cells[0]["source"] = src
    return _temp_nb(nb, nb_path)

def _patch_notebook_paths(nb_path: Path, output_dir: Path) -> Path:
    #Patch notebooks 2–6: redirect all relative data/, artifacts/, and outputs/ paths in the CONFIGURATION cell to output_dir."
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    src = nb.cells[0]["source"]
    replacements = {
        "data": str(output_dir / "data").replace("\\", "/"),
        "artifacts": str(output_dir / "artifacts").replace("\\", "/"),
        "outputs": str(output_dir / "outputs").replace("\\", "/"),
    }
    for prefix, full_path in replacements.items():
        src = re.sub(rf'"({prefix}/)', rf'"{full_path}/', src)
        src = re.sub(rf"'({prefix}/)", rf"'{full_path}/", src)

    nb.cells[0]["source"] = src
    return _temp_nb(nb, nb_path)

# NOTEBOOK EXECUTION
def _run_notebook(nb_path: Path, label: str) -> bool:
    # Execute a notebook in-place via nbconvert.
    # Returns True on success, False on failure.
    print(f"\n{label}\n{nb_path.name}")
    t0 = time.time()

    result = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=3600", str(nb_path),])

    m, s = divmod(int(time.time() - t0), 60)
    if result.returncode == 0:
        print(f"\nCompleted in {m}m {s}s")
        return True
    print(f"\nFailed after {m}m {s}s (exit code {result.returncode})")
    return False

# MAIN
def main() -> None:
    args = parse_args()

    is_default = (abs(args.lat - DEFAULT_LAT) < 1e-4 and abs(args.lon - DEFAULT_LON) < 1e-4)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else Path(".").resolve()
            if is_default else _location_dir(args.lat, args.lon).resolve()
    )

    for sub in ("data", "artifacts", "outputs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).parent.resolve()

    print(f"Solar GHI Forecasting Pipeline")
    print(f"Location: lat={args.lat}, lon={args.lon}")
    print(f"Years: {min(args.years)}–{max(args.years)}")
    print(f"Output dir: {output_dir}")
    print(f"Notebooks:")
    for nb, label in NOTEBOOKS:
        print(f"{nb}")

    failed = None
    skipped = []

    for nb_filename, label in NOTEBOOKS:
        if nb_filename == "1.data_acq.ipynb" and args.skip_data_acq:
            skipped.append(label)
            print(f"\n- Skipping {label} (--skip-data-acq)")
            continue
        if nb_filename == "6.hyperparameter_tuning.ipynb" and args.skip_tuning:
            skipped.append(label)
            print(f"\n- Skipping {label} (--skip-tuning)")
            continue

        nb_path = project_root / nb_filename
        if not nb_path.exists():
            print(f"\nWARNING: {nb_filename} not found - skipping.")
            skipped.append(label)
            continue

        # Patch the config cell for non-default locations
        if not is_default or args.output_dir:
            patched = (
                _patch_notebook_1(nb_path, args.lat, args.lon, args.years, output_dir)
                if nb_filename == "1_data_acq.ipynb" else _patch_notebook_paths(nb_path, output_dir)
            )
        else:
            patched = nb_path   # default site - run notebooks as is

        success = _run_notebook(patched, label)

        if patched != nb_path and patched.exists():
            patched.unlink()

        if not success:
            failed = label
            print(f"\nPipeline halted at '{label}'.\nFix the error above, then re-run. If data is already downloaded, add --skip-data-acq to skip notebook 1.")
            break

    # SUMMARY
    if failed is None:
        print("Pipeline complete!")
        print(f"\nArtifacts: {output_dir / 'artifacts'}")
        print(f"Outputs: {output_dir / 'outputs'}")
        if skipped:
            print(f"\nSkipped: {', '.join(skipped)}")
        predict_cmd = f"python predict_today.py --lat {args.lat} --lon {args.lon}"
        if not is_default:
            predict_cmd += f"--artifacts-dir {output_dir / 'artifacts'}"
        print(f"\nTo predict today's GHI:\n{predict_cmd}")
    else:
        print(f"Pipeline failed at: {failed}")

if __name__ == "__main__":
    main()
