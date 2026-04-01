from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOTS = [
    ROOT / "outputs",
    ROOT / "results",
]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _experiment_label(cfg: Dict, output_dir: str) -> str:
    if cfg.get("dataset_type") == "real":
        return "real_data"
    support_type = cfg.get("support_type")
    if support_type == "connected_disjoint":
        return "graph_aligned"
    if "sweep" in output_dir:
        return "sweep"
    return "main"


def _iter_runs() -> Iterable[Dict]:
    for root in OUTPUT_ROOTS:
        if not root.exists():
            continue
        for metrics_path in root.rglob("metrics.json"):
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
            conn = payload.get("support_connectivity_union", {})
            rows.append(
                {
                    "experiment": experiment,
                    "output_dir": output_dir,
                    "graph_family": cfg.get("graph_family"),
                    "support_type": cfg.get("support_type"),
                    "track": cfg.get("track"),
                    "phase": cfg.get("phase"),
                    "decoy_intensity": cfg.get("decoy_intensity"),
                    "seed": cfg.get("seed"),
                    "lambda2": cfg.get("lambda2"),
                    "method": method,
                    "support_f1": support.get("f1"),
                    "support_precision": support.get("precision"),
                    "support_recall": support.get("recall"),
                    "graph_smoothness_norm": payload.get("graph_smoothness_norm_trueL"),
                    "graph_smoothness_raw": payload.get("graph_smoothness_raw_trueL"),
                    "shared_explained_variance": payload.get("shared_explained_variance"),
                    "sparsity_fraction": payload.get("sparsity_fraction"),
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


if __name__ == "__main__":
    main()
