from __future__ import annotations

from pathlib import Path


def test_run_from_config_executes_real_dataset(tmp_path: Path) -> None:
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
        data=DataConfig(name="pitprop", graph_type="chain", random_state=2),
        objective=ObjectiveConfig(name="nc_spca_single", lambda1=0.05, lambda2=0.05),
        optimizer=OptimizerConfig(name="pg", max_iter=20, learning_rate="auto"),
        model=ModelConfig(name="nc_spca_single", n_components=1, random_state=2),
        experiment=ExperimentConfig(name="pitprop_cli", repeats=1, seed=2),
        tracking=TrackingConfig(root_dir=str(tmp_path), project="nc_spca_cli_real"),
    )

    result = run_from_config(config)

    assert result.records
    assert Path(result.artifact_paths["seed_manifest"]).is_file()
