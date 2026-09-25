from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

def test_workflow_runs_suite_on_push_and_pr():
    wf = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML reads the bare key `on` as True
    triggers = wf.get("on", wf.get(True))
    assert {"push", "pull_request"} <= set(triggers)
    steps = wf["jobs"]["pytest"]["steps"]
    runs = " ".join(s.get("run", "") for s in steps)
    assert "pip install -r requirements.txt" in runs
    assert "pytest tests" in runs

def test_workflow_python_matches_readme():
    wf = yaml.safe_load(WORKFLOW.read_text())
    setup = next(s for s in wf["jobs"]["pytest"]["steps"] if s.get("uses", "").startswith("actions/setup-python"))
    assert setup["with"]["python-version"] == "3.11"
    assert "Python 3.11" in (ROOT / "README.md").read_text()

def test_requirements_cover_every_third_party_import():
    # CI installs only requirements.txt, so every package the code and tests import must be listed
    import ast
    import sys
    listed = {line.split(">=")[0].split("==")[0].strip().lower().replace("_", "-")
              for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")}
    # Import name > pip name where they differ
    pip_name = {"sklearn": "scikit-learn", "dotenv": "python-dotenv", "keras_tuner": "keras-tuner", "yaml": "pyyaml", "keras": "tensorflow"}
    local = {"solarcast", "tests", "predict_today", "run_pipeline"}
    missing = set()
    for path in [*ROOT.glob("*.py"), *(ROOT / "solarcast").glob("*.py"), *(ROOT / "tests").glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in local:
                    continue
                if pip_name.get(top, top).lower().replace("_", "-") not in listed:
                    missing.add(f"{top} ({path.name})")
    assert not missing, f"Imported but not in requirements.txt: {sorted(missing)}"
