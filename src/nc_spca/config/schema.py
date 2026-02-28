"""Structured configuration objects for the NC-SPCA package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BackendConfig:
    name: str = "numpy"
    dtype: str = "float64"
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ObjectiveConfig:
    name: str = "nc_spca_single"
    lambda1: float = 0.15
    lambda2: float = 0.25
    support_threshold: float = 1e-6
    sparsity_mode: str = "l1"
    group_lambda: float | None = None
    retraction: str = "polar"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "pg"
    max_iter: int = 400
    learning_rate: float | str = "auto"
    tol: float = 1e-6
    monotone_backtracking: bool = True
    grad_norm_tol: float | None = None
    qn_memory: int = 10
    support_threshold: float = 1e-8
    active_set_window: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelConfig:
    name: str = "nc_spca_single"
    n_components: int = 1
    init: str = "pca"
    random_state: int = 42
    backend: BackendConfig = field(default_factory=BackendConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class DataConfig:
    name: str = "synthetic_chain"
    n_samples: int = 200
    n_features: int = 100
    support_size: int = 20
    n_components: int = 1
    support_overlap_mode: str = "disjoint"
    signal_strength: float = 8.0
    noise_std: float = 1.0
    graph_type: str = "chain"
    graph_laplacian_type: str = "unnormalized"
    knn_k: int = 8
    standardize: bool = True
    data_root: str = "data"
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrackingConfig:
    root_dir: str = "outputs"
    project: str = "nc_spca"
    enable_wandb: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "paper_core"
    repeats: int = 3
    seed: int = 42
    output_subdir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AppConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    data: DataConfig = field(default_factory=DataConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
