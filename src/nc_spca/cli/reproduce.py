"""Paper reproduction entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .run import run_from_config


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
