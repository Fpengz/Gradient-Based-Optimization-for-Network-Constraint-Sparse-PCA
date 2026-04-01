from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _experiment_label(cfg: Dict, output_dir: str) -> str:
    block = cfg.get("block")
    if block:
        return str(block)
    support_type = cfg.get("support_type")
    if support_type == "connected_disjoint":
        return "graph_aligned"
    if "sweep" in output_dir:
        return "sweep"
    return "main"


def _write_phase2_outputs(rows: List[Dict[str, object]], root: Path) -> None:
    if not rows:
        return
    phase2_dir = root / "results" / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    all_path = phase2_dir / "all_metrics.csv"
    with all_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    by_experiment: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        label = str(row.get("experiment", "unknown"))
        by_experiment.setdefault(label, []).append(row)
    for label, subset in by_experiment.items():
        block_dir = phase2_dir / label
        block_dir.mkdir(parents=True, exist_ok=True)
        out_path = block_dir / "metrics_summary.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=subset[0].keys())
            writer.writeheader()
            writer.writerows(subset)


def _iter_runs() -> Iterable[Dict]:
    for metrics_path in OUTPUTS.rglob("metrics.json"):
        run_dir = metrics_path.parent
        artifacts_path = run_dir / "artifacts.json"
        if not artifacts_path.exists():
            continue
        artifacts = _load_json(artifacts_path)
        cfg = artifacts.get("config", {})
        metrics = _load_json(metrics_path)
        output_dir = cfg.get("output_dir", str(run_dir))
        yield {
            "cfg": cfg,
            "metrics": metrics,
            "output_dir": output_dir,
        }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for run in _iter_runs():
        cfg = run["cfg"]
        metrics = run["metrics"]
        output_dir = run["output_dir"]
        experiment = _experiment_label(cfg, output_dir)
        for method, payload in metrics.items():
            support = payload.get("support_metrics", {}).get("union", {})
            per_comp = payload.get("support_metrics", {}).get("per_component", {})
            per_comp_f1 = [
                v.get("f1", 0.0) for v in per_comp.values() if isinstance(v, dict)
            ]
            per_comp_mean = (
                float(sum(per_comp_f1) / len(per_comp_f1)) if per_comp_f1 else None
            )
            conn = payload.get("support_connectivity_union", {})
            rows.append(
                {
                    "experiment": experiment,
                    "block": cfg.get("block"),
                    "config_name": cfg.get("config_name"),
                    "output_dir": output_dir,
                    "graph_family": cfg.get("graph_family"),
                    "support_type": cfg.get("support_type"),
                    "seed": cfg.get("seed"),
                    "lambda2": cfg.get("lambda2"),
                    "method": method,
                    "corruption_type": cfg.get("corruption_type"),
                    "corruption_level": cfg.get("corruption_level"),
                    "grid_rows": cfg.get("grid_rows"),
                    "grid_cols": cfg.get("grid_cols"),
                    "k": cfg.get("r"),
                    "prior_graph_state": cfg.get("prior_graph_state"),
                    "dataset_name": cfg.get("dataset_name"),
                    "support_f1": support.get("f1"),
                    "support_precision": support.get("precision"),
                    "support_recall": support.get("recall"),
                    "per_component_f1_mean": per_comp_mean,
                    "orthogonality_error": payload.get("orthogonality_error"),
                    "graph_smoothness_norm": payload.get("graph_smoothness_norm_trueL"),
                    "graph_smoothness_raw": payload.get("graph_smoothness_raw_trueL"),
                    "shared_explained_variance": payload.get("shared_explained_variance"),
                    "connect_num_components": conn.get("num_components"),
                    "connect_largest_ratio": conn.get("largest_component_ratio"),
                }
        )

    out_path = ROOT / "results" / "metrics_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    _write_phase2_outputs(rows, ROOT)


if __name__ == "__main__":
    main()
