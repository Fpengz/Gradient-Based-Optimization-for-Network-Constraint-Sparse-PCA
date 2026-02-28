"""Block Stiefel-model implementation for NC-SPCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import FitResult


@dataclass(slots=True)
class NCSPCABlockModel:
    """Multi-component NC-SPCA model using a native manifold PG optimizer."""

    objective: Any
    optimizer: Any
    backend: Any
    n_components: int = 2
    init: str = "pca"
    random_state: int = 42
    name: str = "nc_spca_block"

    @staticmethod
    def _orthonormalize(loadings: np.ndarray) -> np.ndarray:
        try:
            U, _, Vt = np.linalg.svd(loadings, full_matrices=False)
            return U @ Vt
        except np.linalg.LinAlgError:
            Q, _ = np.linalg.qr(loadings)
            return Q[:, : loadings.shape[1]]

    def _initialize_loadings(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        k = min(self.n_components, n_features)
        if self.init == "pca":
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            loadings = np.asarray(vt[:k].T, dtype=float)
        elif self.init == "random":
            rng = np.random.default_rng(self.random_state)
            loadings = rng.normal(size=(n_features, k))
        else:
            raise ValueError(f"Unsupported init strategy: {self.init!r}")
        return self._orthonormalize(loadings)

    def fit(self, dataset: dict[str, Any], tracker: Any | None = None) -> FitResult:
        X = np.asarray(dataset["X"], dtype=float)
        X_centered = X - X.mean(axis=0, keepdims=True)
        params = {"loadings": self._initialize_loadings(X_centered)}
        batch = {"X": X_centered, "graph": dataset["graph"]}
        artifacts = self.optimizer.optimize(params, self.objective, batch, self.backend)
        loadings = np.asarray(artifacts.params["loadings"], dtype=float)
        components = loadings.T
        result = FitResult(
            params={"loadings": loadings},
            components=components,
            objective=float(artifacts.objective),
            n_iter=int(artifacts.state.iteration),
            converged=bool(artifacts.state.converged),
            history=artifacts.state.history,
            metadata={
                "backend": self.backend.name,
                "implementation": "native_manpg",
                "sparsity_mode": getattr(self.objective, "sparsity_mode", "l1"),
            },
        )
        if tracker is not None:
            tracker.log_metric("objective", result.objective, step=result.n_iter)
            tracker.log_metric(
                "orthogonality_error",
                float(self.objective.orthogonality_error({"loadings": loadings})),
                step=result.n_iter,
            )
        return result
