from __future__ import annotations

import numpy as np
import pytest


def _block_dataset(seed: int = 5, n_features: int = 16) -> dict[str, object]:
    from nc_spca.config.schema import DataConfig
    from nc_spca.data.synthetic.generators import generate_synthetic_dataset

    return generate_synthetic_dataset(
        DataConfig(
            name="synthetic_grid",
            n_samples=48,
            n_features=n_features,
            support_size=4,
            graph_type="grid",
            random_state=seed,
        ),
        seed=seed,
    )


def _orthonormal_loadings(X: np.ndarray, n_components: int) -> np.ndarray:
    X_centered = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
    return np.asarray(vt[:n_components].T, dtype=float)


def test_block_objective_exposes_native_manifold_ops() -> None:
    from nc_spca.objectives import NCSPCABlockObjective

    dataset = _block_dataset()
    X = np.asarray(dataset["X"], dtype=float)
    loadings = _orthonormal_loadings(X, n_components=2)
    batch = {
        "X": X - X.mean(axis=0, keepdims=True),
        "graph": dataset["graph"],
    }
    objective = NCSPCABlockObjective(
        lambda1=0.05,
        lambda2=0.1,
        sparsity_mode="l1",
    )

    smooth_value, grad = objective.smooth_value_and_grad({"loadings": loadings}, batch)
    evaluation = objective.evaluate({"loadings": loadings}, batch)
    riem = objective.riemannian_grad({"loadings": loadings}, batch)

    assert np.isfinite(smooth_value)
    assert evaluation.total >= evaluation.smooth
    assert grad["loadings"].shape == loadings.shape
    assert riem["loadings"].shape == loadings.shape

    tangent_residual = loadings.T @ riem["loadings"] + riem["loadings"].T @ loadings
    np.testing.assert_allclose(tangent_residual, np.zeros((2, 2)), atol=1e-6)


def test_block_objective_row_group_prox_zeroes_small_rows() -> None:
    from nc_spca.backends import NumpyBackend
    from nc_spca.objectives import NCSPCABlockObjective

    objective = NCSPCABlockObjective(
        lambda1=0.2,
        lambda2=0.1,
        sparsity_mode="l21",
        group_lambda=0.2,
    )
    backend = NumpyBackend()
    loadings = np.array(
        [
            [0.30, 0.40],
            [0.05, 0.00],
            [0.00, 0.00],
        ],
        dtype=float,
    )

    prox = objective.prox({"loadings": loadings}, step_size=1.0, backend=backend)

    assert prox["loadings"].shape == loadings.shape
    np.testing.assert_allclose(prox["loadings"][1], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(prox["loadings"][2], np.zeros(2), atol=1e-12)
    assert np.linalg.norm(prox["loadings"][0]) < np.linalg.norm(loadings[0])


def test_block_model_uses_native_manpg_path() -> None:
    from nc_spca.api.factory import build_backend, build_model
    from nc_spca.config.schema import BackendConfig, ModelConfig, ObjectiveConfig, OptimizerConfig

    backend = build_backend(BackendConfig(name="numpy"))
    model = build_model(
        model_cfg=ModelConfig(name="nc_spca_block", n_components=2, random_state=7),
        objective_cfg=ObjectiveConfig(
            name="nc_spca_block",
            lambda1=0.05,
            lambda2=0.1,
            sparsity_mode="l21",
            group_lambda=0.05,
        ),
        optimizer_cfg=OptimizerConfig(name="manpg", max_iter=30, learning_rate="auto"),
        backend=backend,
    )

    fit = model.fit(_block_dataset(seed=7))

    assert fit.components.shape == (2, 16)
    assert fit.params["loadings"].shape == (16, 2)
    assert fit.metadata["backend"] == "numpy"
    assert fit.metadata["implementation"] == "native_manpg"
    assert "riemannian_grad_norm" in fit.history
    assert "orthogonality_error" in fit.history
    gram = fit.components @ fit.components.T
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-4)


def test_block_model_runs_on_torch_backend_without_geoopt_dependency() -> None:
    pytest.importorskip("torch")

    from nc_spca.api.factory import build_backend, build_model
    from nc_spca.config.schema import BackendConfig, ModelConfig, ObjectiveConfig, OptimizerConfig

    backend = build_backend(BackendConfig(name="torch"))
    model = build_model(
        model_cfg=ModelConfig(name="nc_spca_block", n_components=2, random_state=11),
        objective_cfg=ObjectiveConfig(
            name="nc_spca_block",
            lambda1=0.05,
            lambda2=0.1,
            sparsity_mode="l1",
        ),
        optimizer_cfg=OptimizerConfig(name="manpg", max_iter=20, learning_rate="auto"),
        backend=backend,
    )

    fit = model.fit(_block_dataset(seed=11))

    assert fit.metadata["backend"] == "torch"
    assert fit.metadata["implementation"] == "native_manpg"
    assert fit.components.shape == (2, 16)
    gram = fit.components @ fit.components.T
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-4)
