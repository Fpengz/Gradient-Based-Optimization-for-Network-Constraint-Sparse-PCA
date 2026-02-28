"""Factories for backends, models, and experiments."""

from __future__ import annotations

from ..backends import NumpyBackend, TorchBackend
from ..config.schema import (
    BackendConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrackingConfig,
)
from ..experiments import BlockSyntheticExperiment, RealExperiment, SyntheticExperiment
from ..models import LegacyEstimatorModel, NCSPCABlockModel, NCSPCAModel
from ..objectives import NCSPCABlockObjective, NCSPCASingleObjective
from ..optimizers import MASPGCAROptimizer, ManifoldPGOptimizer, PGOptimizer, ProxQNOptimizer
from src.models.generalized_power_method import GeneralizedPowerMethod
from src.models.sparse_pca import SparsePCA_L1_ProxGrad, ZouSparsePCA
from src.models.vanilla import PCAEstimator


def build_backend(config: BackendConfig):
    """Instantiate a backend from config."""

    if config.name == "numpy":
        return NumpyBackend(dtype=config.dtype, device=config.device)
    if config.name == "torch":
        return TorchBackend(dtype=config.dtype, device=config.device)
    raise ValueError(f"Unsupported backend: {config.name!r}")


def build_model(
    model_cfg: ModelConfig,
    objective_cfg: ObjectiveConfig,
    optimizer_cfg: OptimizerConfig,
    backend,
):
    """Instantiate a compositional model."""

    if optimizer_cfg.name == "pg":
        optimizer = PGOptimizer(
            max_iter=optimizer_cfg.max_iter,
            learning_rate=optimizer_cfg.learning_rate,
            tol=optimizer_cfg.tol,
            monotone_backtracking=optimizer_cfg.monotone_backtracking,
        )
    elif optimizer_cfg.name == "maspg_car":
        optimizer = MASPGCAROptimizer(
            max_iter=optimizer_cfg.max_iter,
            learning_rate=optimizer_cfg.learning_rate,
            tol=optimizer_cfg.tol,
            monotone_backtracking=optimizer_cfg.monotone_backtracking,
        )
    elif optimizer_cfg.name == "prox_qn":
        optimizer = ProxQNOptimizer(
            max_iter=optimizer_cfg.max_iter,
            learning_rate=optimizer_cfg.learning_rate,
            tol=optimizer_cfg.tol,
            monotone_backtracking=optimizer_cfg.monotone_backtracking,
            qn_memory=optimizer_cfg.qn_memory,
        )
    elif optimizer_cfg.name == "manpg":
        optimizer = ManifoldPGOptimizer(
            max_iter=optimizer_cfg.max_iter,
            learning_rate=optimizer_cfg.learning_rate,
            tol=optimizer_cfg.tol,
            monotone_backtracking=optimizer_cfg.monotone_backtracking,
            grad_norm_tol=optimizer_cfg.grad_norm_tol,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name!r}")

    if objective_cfg.name == "nc_spca_single" and model_cfg.name == "nc_spca_single":
        objective = NCSPCASingleObjective(
            lambda1=objective_cfg.lambda1,
            lambda2=objective_cfg.lambda2,
            support_threshold=objective_cfg.support_threshold,
        )
        return NCSPCAModel(
            objective=objective,
            optimizer=optimizer,
            backend=backend,
            n_components=model_cfg.n_components,
            init=model_cfg.init,
            random_state=model_cfg.random_state,
            name=model_cfg.name,
        )

    if objective_cfg.name == "nc_spca_block" and model_cfg.name == "nc_spca_block":
        objective = NCSPCABlockObjective(
            lambda1=objective_cfg.lambda1,
            lambda2=objective_cfg.lambda2,
            support_threshold=objective_cfg.support_threshold,
            sparsity_mode=objective_cfg.sparsity_mode,
            group_lambda=objective_cfg.group_lambda,
            retraction=objective_cfg.retraction,
        )
        return NCSPCABlockModel(
            objective=objective,
            optimizer=optimizer,
            backend=backend,
            n_components=model_cfg.n_components,
            init=model_cfg.init,
            random_state=model_cfg.random_state,
            name=model_cfg.name,
        )

    if model_cfg.name == "pca":
        return LegacyEstimatorModel(
            builder=lambda: PCAEstimator(n_components=model_cfg.n_components),
            backend=backend,
            name="pca",
        )
    if model_cfg.name == "l1_spca":
        return LegacyEstimatorModel(
            builder=lambda: SparsePCA_L1_ProxGrad(
                n_components=model_cfg.n_components,
                lambda1=objective_cfg.lambda1,
                max_iter=optimizer_cfg.max_iter,
                learning_rate=optimizer_cfg.learning_rate,
                tol=optimizer_cfg.tol,
                init=model_cfg.init,
                monotone_backtracking=optimizer_cfg.monotone_backtracking,
            ),
            backend=backend,
            name="l1_spca",
        )
    if model_cfg.name == "gpower":
        return LegacyEstimatorModel(
            builder=lambda: GeneralizedPowerMethod(
                n_components=model_cfg.n_components,
                gamma=objective_cfg.lambda1,
                max_iter=optimizer_cfg.max_iter,
                tol=optimizer_cfg.tol,
            ),
            backend=backend,
            name="gpower",
        )
    if model_cfg.name == "elastic_net_spca":
        return LegacyEstimatorModel(
            builder=lambda: ZouSparsePCA(
                n_components=model_cfg.n_components,
                alpha=max(objective_cfg.lambda1 * 40.0, 1e-6),
                lambda_l2=1e-3,
                max_iter=min(optimizer_cfg.max_iter, 200),
            ),
            backend=backend,
            name="elastic_net_spca",
        )

    raise ValueError(
        f"Unsupported model/objective combination: {model_cfg.name!r} / {objective_cfg.name!r}"
    )


def build_experiment(
    experiment_cfg: ExperimentConfig,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    objective_cfg: ObjectiveConfig,
    optimizer_cfg: OptimizerConfig,
    tracking_cfg: TrackingConfig,
    backend,
):
    """Instantiate an experiment runner."""

    if data_cfg.name.startswith("synthetic"):
        if model_cfg.name == "nc_spca_block" or data_cfg.n_components > 1:
            return BlockSyntheticExperiment(
                experiment_cfg=experiment_cfg,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                objective_cfg=objective_cfg,
                optimizer_cfg=optimizer_cfg,
                tracking_cfg=tracking_cfg,
                backend=backend,
                model_builder=build_model,
            )
        return SyntheticExperiment(
            experiment_cfg=experiment_cfg,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            optimizer_cfg=optimizer_cfg,
            tracking_cfg=tracking_cfg,
            backend=backend,
            model_builder=build_model,
        )
    if data_cfg.name in {"colon", "pitprop"}:
        return RealExperiment(
            experiment_cfg=experiment_cfg,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            objective_cfg=objective_cfg,
            optimizer_cfg=optimizer_cfg,
            tracking_cfg=tracking_cfg,
            backend=backend,
            model_builder=build_model,
        )
    raise ValueError(f"Unsupported experiment data source: {data_cfg.name!r}")
