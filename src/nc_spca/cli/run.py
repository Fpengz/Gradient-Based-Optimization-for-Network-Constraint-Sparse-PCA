"""Run a configured NC-SPCA experiment."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig

from ..api.factory import build_backend, build_experiment
from ..config import AppConfig, app_config_from_mapping


def run_from_config(config: AppConfig | DictConfig | dict[str, Any]):
    """Execute a single experiment from typed or Hydra config."""

    app_config = app_config_from_mapping(config)
    backend = build_backend(app_config.backend)
    experiment = build_experiment(
        experiment_cfg=app_config.experiment,
        data_cfg=app_config.data,
        model_cfg=app_config.model,
        objective_cfg=app_config.objective,
        optimizer_cfg=app_config.optimizer,
        tracking_cfg=app_config.tracking,
        backend=backend,
    )
    return experiment.run()


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    result = run_from_config(cfg)
    print(result.artifact_paths["run_dir"])


if __name__ == "__main__":
    main()
