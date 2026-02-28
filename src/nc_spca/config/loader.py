"""Config loading helpers for Hydra and programmatic entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .schema import (
    AppConfig,
    BackendConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrackingConfig,
)


def _as_mapping(payload: DictConfig | Mapping[str, Any] | AppConfig) -> Mapping[str, Any]:
    if isinstance(payload, AppConfig):
        return payload.to_dict()
    if isinstance(payload, DictConfig):
        return OmegaConf.to_container(payload, resolve=True)  # type: ignore[return-value]
    return payload


def app_config_from_mapping(payload: DictConfig | Mapping[str, Any] | AppConfig) -> AppConfig:
    """Build the typed root config from a mapping-like object."""

    mapping = dict(_as_mapping(payload))
    return AppConfig(
        backend=BackendConfig(**mapping.get("backend", {})),
        data=DataConfig(**mapping.get("data", {})),
        objective=ObjectiveConfig(**mapping.get("objective", {})),
        optimizer=OptimizerConfig(**mapping.get("optimizer", {})),
        model=ModelConfig(**mapping.get("model", {})),
        experiment=ExperimentConfig(**mapping.get("experiment", {})),
        tracking=TrackingConfig(**mapping.get("tracking", {})),
    )
