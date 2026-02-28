from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warnings


def test_factory_builds_numpy_nc_spca_components() -> None:
    from nc_spca.api.factory import build_backend, build_model
    from nc_spca.config.schema import (
        BackendConfig,
        ModelConfig,
        ObjectiveConfig,
        OptimizerConfig,
    )

    backend = build_backend(BackendConfig(name="numpy"))
    model = build_model(
        model_cfg=ModelConfig(name="nc_spca_single", n_components=1),
        objective_cfg=ObjectiveConfig(name="nc_spca_single", lambda1=0.2, lambda2=0.3),
        optimizer_cfg=OptimizerConfig(name="pg", max_iter=25, learning_rate="auto"),
        backend=backend,
    )

    assert backend.name == "numpy"
    assert model.name == "nc_spca_single"
    assert model.optimizer.name == "pg"
    assert model.objective.name == "nc_spca_single"
    vec = np.array([1.0, -0.5, 0.1])
    np.testing.assert_allclose(
        backend.soft_threshold(vec, 0.25),
        np.array([0.75, -0.25, 0.0]),
    )


def test_filesystem_tracker_writes_manifest_and_metrics(tmp_path: Path) -> None:
    from nc_spca.config.schema import TrackingConfig
    from nc_spca.tracking.filesystem import FilesystemTracker

    tracker = FilesystemTracker(
        TrackingConfig(root_dir=str(tmp_path), project="test_project", enable_wandb=False)
    )
    tracker.start_run(experiment_name="smoke")
    tracker.log_config({"seed": 7, "backend": "numpy"})
    tracker.log_metric("objective", 1.23, step=0, context={"split": "train"})
    tracker.log_event("fit_started", {"model": "nc_spca_single"})
    tracker.finalize({"best_objective": 1.23})

    assert tracker.run_dir is not None
    config_path = tracker.run_dir / "resolved_config.json"
    metrics_path = tracker.run_dir / "metrics.jsonl"
    summary_path = tracker.run_dir / "summary.json"

    assert config_path.is_file()
    assert metrics_path.is_file()
    assert summary_path.is_file()

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["backend"] == "numpy"


def test_filesystem_tracker_run_ids_are_unique(tmp_path: Path) -> None:
    from nc_spca.config.schema import TrackingConfig
    from nc_spca.tracking.filesystem import FilesystemTracker

    tracker = FilesystemTracker(TrackingConfig(root_dir=str(tmp_path), project="unique_test"))
    first = tracker.start_run("smoke")
    second = tracker.start_run("smoke")

    assert first != second


def test_nc_spca_explained_variance_is_finite_and_warning_free() -> None:
    from nc_spca.metrics import explained_variance

    rng = np.random.default_rng(11)
    X = rng.normal(size=(64, 25))
    weight = rng.normal(size=25)
    weight /= np.linalg.norm(weight) + 1e-12

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        value = explained_variance(X, weight)

    assert np.isfinite(value)
    assert caught == []
