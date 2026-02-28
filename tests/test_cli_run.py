from __future__ import annotations

import json
from pathlib import Path


def test_run_from_config_executes_end_to_end(tmp_path: Path) -> None:
    from nc_spca.cli.run import run_from_config
    from nc_spca.config.schema import (
        AppConfig,
        BackendConfig,
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        ObjectiveConfig,
        OptimizerConfig,
        TrackingConfig,
    )

    config = AppConfig(
        backend=BackendConfig(name="numpy"),
        data=DataConfig(
            name="synthetic_chain",
            n_samples=32,
            n_features=12,
            support_size=4,
            graph_type="chain",
            random_state=5,
        ),
        objective=ObjectiveConfig(name="nc_spca_single", lambda1=0.05, lambda2=0.2),
        optimizer=OptimizerConfig(name="pg", max_iter=25, learning_rate="auto"),
        model=ModelConfig(name="nc_spca_single", n_components=1, random_state=5),
        experiment=ExperimentConfig(name="cli_smoke", repeats=1, seed=5),
        tracking=TrackingConfig(root_dir=str(tmp_path), project="nc_spca_cli_test"),
    )

    result = run_from_config(config)

    assert result.records
    summary_path = Path(result.artifact_paths["summary"])
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["n_runs"] == 1

