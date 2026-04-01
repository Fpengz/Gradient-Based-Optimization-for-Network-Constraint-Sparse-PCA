from __future__ import annotations

import csv
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

component_f1_summary = import_module("grpca_gd.analysis.component_metrics").component_f1_summary

FIELDNAMES = [
    "experiment",
    "output_dir",
    "track",
    "phase",
    "graph_family",
    "support_type",
    "decoy_intensity",
    "seed",
    "lambda2",
    "method",
    "status",
    "support_f1",
    "support_precision",
    "support_recall",
    "graph_smoothness_norm",
    "graph_smoothness_raw",
    "shared_explained_variance",
    "connect_num_components",
    "connect_largest_ratio",
    "component_f1_min",
    "component_f1_median",
    "component_f1_mean",
]


def _results_root(root: Path | None = None) -> Path:
    if root is None:
        root = ROOT
    return root / "results" / "trackB" / "phase1_5"


def _output_path(root: Path | None = None) -> Path:
    return _results_root(root) / "aggregated_phase15.csv"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_runs(results_root: Path) -> Iterable[Dict[str, object]]:
    if not results_root.exists():
        return iter(())

    for metrics_path in results_root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        artifacts_path = run_dir / "artifacts.json"
        if not artifacts_path.exists():
            continue

        artifacts = _load_json(artifacts_path)
        cfg = artifacts.get("config", {})
        if cfg.get("track") != "B" or float(cfg.get("phase", 0.0)) != 1.5:
            continue

        yield {
            "cfg": cfg,
            "metrics": _load_json(metrics_path),
            "artifacts": artifacts,
            "output_dir": cfg.get("output_dir", str(run_dir)),
        }


def _rows_for_run(run: Dict[str, object]) -> List[Dict[str, object]]:
    cfg = run["cfg"]
    metrics = run["metrics"]
    artifacts = run["artifacts"]
    output_dir = run["output_dir"]
    status = artifacts.get("status", artifacts.get("manifest", {}).get("status"))

    rows: List[Dict[str, object]] = []
    for method, payload in metrics.items():
        support_metrics = payload.get("support_metrics", {})
        union = support_metrics.get("union", {})
        summary = component_f1_summary(support_metrics.get("per_component", {}))
        connectivity = payload.get("support_connectivity_union", {})
        rows.append(
            {
                "experiment": "trackB_phase15",
                "output_dir": output_dir,
                "track": cfg.get("track"),
                "phase": cfg.get("phase"),
                "graph_family": cfg.get("graph_family"),
                "support_type": cfg.get("support_type"),
                "decoy_intensity": cfg.get("decoy_intensity"),
                "seed": cfg.get("seed"),
                "lambda2": cfg.get("lambda2"),
                "method": method,
                "status": status,
                "support_f1": union.get("f1"),
                "support_precision": union.get("precision"),
                "support_recall": union.get("recall"),
                "graph_smoothness_norm": payload.get("graph_smoothness_norm_trueL"),
                "graph_smoothness_raw": payload.get("graph_smoothness_raw_trueL"),
                "shared_explained_variance": payload.get("shared_explained_variance"),
                "connect_num_components": connectivity.get("num_components"),
                "connect_largest_ratio": connectivity.get("largest_component_ratio"),
                **summary,
            }
        )
    return rows


def collect_rows(results_root: Path | None = None) -> List[Dict[str, object]]:
    if results_root is None:
        results_root = _results_root()
    rows: List[Dict[str, object]] = []
    for run in _iter_runs(results_root):
        rows.extend(_rows_for_run(run))
    return rows


def main() -> None:
    output_path = _output_path()
    rows = collect_rows(output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
