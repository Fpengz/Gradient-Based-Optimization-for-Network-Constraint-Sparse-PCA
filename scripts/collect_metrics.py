from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

CANONICAL_FIELDS = [
    "dataset",
    "graph_family",
    "artifact_id",
    "artifact_version",
    "data_source",
    "prep_config_hash",
    "eval_protocol_id",
    "method",
    "method_version",
    "seed",
    "rank",
    "lambda1",
    "lambda2",
    "rho",
    "corruption_type",
    "corruption_level",
    "graph_used_id",
    "graph_reference_id",
    "explained_variance",
    "smoothness_used_graph",
    "smoothness_reference_graph",
    "runtime_sec",
    "iterations",
    "nnz_loadings",
    "sparsity_ratio",
    "final_objective",
    "final_coupling_gap",
    "final_orthogonality_defect",
    "stop_reason",
    "convergence_flag",
    "support_precision",
    "support_recall",
    "support_f1",
    "n_samples",
    "n_features",
]


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _experiment_label(cfg: Dict, output_dir: str) -> str:
    support_type = cfg.get("support_type")
    if support_type == "connected_disjoint":
        return "graph_aligned"
    if "sweep" in output_dir:
        return "sweep"
    return "main"


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
            "artifacts": artifacts,
            "output_dir": output_dir,
        }


def main() -> None:
    rows: List[Dict[str, object]] = []
    for run in _iter_runs():
        cfg = run["cfg"]
        metrics = run["metrics"]
        artifacts = run["artifacts"]
        output_dir = run["output_dir"]
        experiment = _experiment_label(cfg, output_dir)
        for method, payload in metrics.items():
            support = payload.get("support_metrics", {}).get("union", {})
            conn = payload.get("support_connectivity_union", {})
            row = {
                "experiment": experiment,
                "output_dir": output_dir,
                "graph_smoothness_norm": payload.get("graph_smoothness_norm_trueL"),
                "graph_smoothness_raw": payload.get("graph_smoothness_raw_trueL"),
                "shared_explained_variance": payload.get(
                    "shared_explained_variance", payload.get("explained_variance")
                ),
                "connect_num_components": conn.get("num_components"),
                "connect_largest_ratio": conn.get("largest_component_ratio"),
            }
            for field in CANONICAL_FIELDS:
                if field == "method":
                    row[field] = payload.get(field, method)
                elif field in {"graph_family", "dataset"}:
                    row[field] = payload.get(field, cfg.get(field))
                elif field in {
                    "support_precision",
                    "support_recall",
                    "support_f1",
                }:
                    row[field] = payload.get(field, support.get(field.split("_")[1]))
                else:
                    row[field] = payload.get(field, cfg.get(field))

            if row["artifact_id"] is None:
                row["artifact_id"] = artifacts.get("manifest", {}).get("artifact_id")

            rows.append(row)

    out_path = ROOT / "results" / "metrics_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["experiment", "output_dir", *CANONICAL_FIELDS, "graph_smoothness_norm", "graph_smoothness_raw", "shared_explained_variance", "connect_num_components", "connect_largest_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
