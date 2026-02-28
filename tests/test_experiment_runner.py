from __future__ import annotations

import json
from pathlib import Path


def test_synthetic_experiment_runner_writes_summary_bundle(tmp_path: Path) -> None:
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
        experiment_cfg=ExperimentConfig(name="paper_core", repeats=1, seed=11),
        data_cfg=DataConfig(
            name="synthetic_chain",
            n_samples=40,
            n_features=16,
            support_size=4,
            graph_type="chain",
            random_state=11,
        ),
        model_cfg=ModelConfig(name="nc_spca_single", n_components=1, random_state=11),
        objective_cfg=ObjectiveConfig(name="nc_spca_single", lambda1=0.1, lambda2=0.2),
        optimizer_cfg=OptimizerConfig(name="pg", max_iter=40, learning_rate="auto"),
        tracking_cfg=TrackingConfig(root_dir=str(tmp_path), project="nc_spca_test"),
        backend=backend,
    )

    result = experiment.run()

    assert result.records
    assert "f1_mean" in result.summary
    assert Path(result.artifact_paths["summary"]).is_file()
    payload = json.loads(Path(result.artifact_paths["summary"]).read_text(encoding="utf-8"))
    assert payload["f1_mean"] == result.summary["f1_mean"]
