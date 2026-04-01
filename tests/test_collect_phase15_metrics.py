from __future__ import annotations

import csv
import json
import importlib.util
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_phase15_metrics_filters_and_summarizes(monkeypatch, tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "collect_phase15_metrics.py"
    spec = importlib.util.spec_from_file_location("collect_phase15_metrics", script_path)
    assert spec is not None and spec.loader is not None
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)

    monkeypatch.setattr(collector, "ROOT", tmp_path)

    good_run = tmp_path / "results" / "trackB" / "phase1_5" / "seed0" / "lambda2_0p10" / "decoy_high"
    bad_run = tmp_path / "results" / "trackB" / "phase1_5" / "seed1" / "lambda2_0p20" / "decoy_high"
    missing_artifacts_run = tmp_path / "results" / "trackB" / "phase1_5" / "seed2" / "lambda2_0p30" / "decoy_high"

    _write_json(
        good_run / "artifacts.json",
        {
            "config": {
                "track": "B",
                "phase": 1.5,
                "graph_family": "chain",
                "support_type": "connected_disjoint",
                "seed": 0,
                "lambda2": 0.1,
                "decoy_intensity": "high",
                "output_dir": "results/trackB/phase1_5/seed0/lambda2_0p10/decoy_high",
            }
        },
    )
    _write_json(
        good_run / "metrics.json",
        {
            "Proposed": {
                "support_metrics": {
                    "union": {"f1": 0.8, "precision": 0.9, "recall": 0.7},
                    "per_component": {
                        "0": {"f1": 0.5},
                        "1": {"f1": 0.75},
                        "2": {"f1": 1.0},
                    },
                },
                "shared_explained_variance": 1.23,
            }
        },
    )

    _write_json(
        bad_run / "artifacts.json",
        {
            "config": {
                "track": "A",
                "phase": 1.5,
                "graph_family": "chain",
                "support_type": "connected_disjoint",
                "seed": 1,
                "lambda2": 0.2,
                "decoy_intensity": "high",
                "output_dir": "results/trackB/phase1_5/seed1/lambda2_0p20/decoy_high",
            }
        },
    )
    _write_json(
        bad_run / "metrics.json",
        {
            "Proposed": {
                "support_metrics": {
                    "union": {"f1": 0.1, "precision": 0.2, "recall": 0.3},
                    "per_component": {"0": {"f1": 0.1}},
                }
            }
        },
    )

    (missing_artifacts_run / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        missing_artifacts_run / "metrics.json",
        {
            "Proposed": {
                "support_metrics": {
                    "union": {"f1": 0.4, "precision": 0.5, "recall": 0.6},
                    "per_component": {"0": {"f1": 0.4}},
                }
            }
        },
    )

    collector.main()

    out_path = tmp_path / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
    assert out_path.exists()

    rows = list(csv.DictReader(out_path.open("r", encoding="utf-8")))
    assert len(rows) == 1

    row = rows[0]
    assert row["track"] == "B"
    assert row["phase"] == "1.5"
    assert row["method"] == "Proposed"
    assert row["support_f1"] == "0.8"
    assert row["component_f1_min"] == "0.5"
    assert row["component_f1_median"] == "0.75"
    assert row["component_f1_mean"] == "0.75"
    assert row["shared_explained_variance"] == "1.23"
