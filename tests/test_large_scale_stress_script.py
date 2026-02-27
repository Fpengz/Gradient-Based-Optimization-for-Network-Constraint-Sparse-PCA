from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_large_scale_stress_outputs_schema(tmp_path: Path):
    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/run_large_scale_stress.py",
        "--output-dir",
        str(outdir),
        "--n-features-grid",
        "50,60",
        "--n-repeats",
        "1",
        "--max-iter",
        "10",
        "--n-samples",
        "80",
    ]
    subprocess.run(cmd, check=True)
    dirs = sorted(outdir.glob("large-scale-stress-*"))
    assert dirs, "Expected large-scale stress output directory"
    latest = dirs[-1]
    config = json.loads((latest / "config.json").read_text(encoding="utf-8"))
    assert config["n_features_grid"] == [50, 60]
    summary = pd.read_csv(latest / "summary.csv")
    for col in [
        "pg_residual_last_mean",
        "pg_residual_ratio_mean",
        "objective_monotone_rate",
    ]:
        assert col in summary.columns
