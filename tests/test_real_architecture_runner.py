from __future__ import annotations

import json
from pathlib import Path


def test_real_experiment_runner_writes_summary_bundle(tmp_path: Path) -> None:
    from nc_spca.api.factory import build_backend, build_experiment
    from nc_spca.config.schema import (
        BackendConfig,
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        ObjectiveConfig,
        OptimizerConfig,
        TrackingConfig,
    )

    backend = build_backend(BackendConfig(name="numpy"))
    experiment = build_experiment(
        experiment_cfg=ExperimentConfig(name="pitprop_real", repeats=1, seed=3),
        data_cfg=DataConfig(
            name="pitprop",
            graph_type="knn",
            knn_k=3,
            random_state=3,
        ),
        model_cfg=ModelConfig(name="nc_spca_single", n_components=1, random_state=3),
        objective_cfg=ObjectiveConfig(name="nc_spca_single", lambda1=0.05, lambda2=0.05),
        optimizer_cfg=OptimizerConfig(name="pg", max_iter=20, learning_rate="auto"),
        tracking_cfg=TrackingConfig(root_dir=str(tmp_path), project="nc_spca_real_test"),
        backend=backend,
    )

    result = experiment.run()

    assert result.records
    assert result.records[0]["dataset"] == "pitprop"
    summary_path = Path(result.artifact_paths["summary"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["n_runs"] == 1

