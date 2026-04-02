import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_schema_fields_present(tmp_path: Path) -> None:
    metrics = {
        "dataset": "synthetic",
        "graph_family": "chain",
        "artifact_id": "synthetic_chain",
        "artifact_version": "v1",
        "data_source": "synthetic",
        "prep_config_hash": "abc",
        "eval_protocol_id": "default",
        "method": "Proposed",
        "method_version": "v1",
        "seed": 0,
        "rank": 1,
        "lambda1": 0.1,
        "lambda2": 0.2,
        "rho": 5.0,
        "corruption_type": "none",
        "corruption_level": 0.0,
        "graph_used_id": "chain_clean",
        "graph_reference_id": "chain_clean",
        "explained_variance": 1.0,
        "smoothness_used_graph": 0.1,
        "smoothness_reference_graph": 0.1,
        "runtime_sec": 0.1,
        "iterations": 10,
        "nnz_loadings": 4,
        "sparsity_ratio": 0.2,
        "final_objective": -0.1,
        "final_coupling_gap": 0.0,
        "final_orthogonality_defect": 0.0,
        "stop_reason": "tol_obj",
        "convergence_flag": True,
        "support_precision": 1.0,
        "support_recall": 1.0,
        "support_f1": 1.0,
        "n_samples": 100,
        "n_features": 20,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    for key in ["artifact_id", "eval_protocol_id", "method_version", "nnz_loadings"]:
        assert key in loaded


def test_collect_metrics_emits_canonical_columns(tmp_path: Path, monkeypatch) -> None:
    import scripts.collect_metrics as collect_metrics

    run_dir = tmp_path / "outputs" / "run1"
    run_dir.mkdir(parents=True)
    artifacts = {
        "config": {
            "graph_family": "chain",
            "support_type": "connected",
            "seed": 0,
            "lambda2": 0.2,
            "output_dir": str(run_dir),
        }
    }
    metrics = {
        "Proposed": {
            "dataset": "synthetic",
            "graph_family": "chain",
            "artifact_id": "synthetic_chain",
            "artifact_version": "v1",
            "data_source": "synthetic",
            "prep_config_hash": "abc",
            "eval_protocol_id": "default",
            "method": "Proposed",
            "method_version": "v1",
            "seed": 0,
            "rank": 1,
            "lambda1": 0.1,
            "lambda2": 0.2,
            "rho": 5.0,
            "corruption_type": "none",
            "corruption_level": 0.0,
            "graph_used_id": "chain_clean",
            "graph_reference_id": "chain_clean",
            "explained_variance": 1.0,
            "smoothness_used_graph": 0.1,
            "smoothness_reference_graph": 0.1,
            "runtime_sec": 0.1,
            "iterations": 10,
            "nnz_loadings": 4,
            "sparsity_ratio": 0.2,
            "final_objective": -0.1,
            "final_coupling_gap": 0.0,
            "final_orthogonality_defect": 0.0,
            "stop_reason": "tol_obj",
            "convergence_flag": True,
            "support_precision": 1.0,
            "support_recall": 1.0,
            "support_f1": 1.0,
            "n_samples": 100,
            "n_features": 20,
        }
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    monkeypatch.setattr(collect_metrics, "ROOT", tmp_path)
    monkeypatch.setattr(collect_metrics, "OUTPUTS", tmp_path / "outputs")

    collect_metrics.main()

    out_path = tmp_path / "results" / "metrics_summary.csv"
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["dataset"] == "synthetic"
    assert row["artifact_id"] == "synthetic_chain"
    assert row["nnz_loadings"] == "4"


def test_validate_config_allows_artifact_runs_without_synthetic_fields() -> None:
    from src.grpca_gd.runner import _validate_config

    cfg = {
        "seed": 0,
        "r": 2,
        "lambda1": 0.1,
        "lambda2": 0.2,
        "rho": 5.0,
        "max_iters": 10,
        "tol_obj": 1e-6,
        "tol_gap": 1e-6,
        "tol_orth": 1e-6,
        "eta_A": 0.05,
        "baseline": "PCA",
        "output_dir": "outputs/run",
        "artifact_dir": "artifacts/run",
    }

    _validate_config(cfg)
