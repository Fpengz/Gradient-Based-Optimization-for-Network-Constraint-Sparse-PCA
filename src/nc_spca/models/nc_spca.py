"""Composable NC-SPCA model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import FitResult


@dataclass(slots=True)
class NCSPCAModel:
    """Single-component NC-SPCA model composed from objective and optimizer."""

    objective: Any
    optimizer: Any
    backend: Any
    n_components: int = 1
    init: str = "pca"
    random_state: int = 42
    name: str = "nc_spca_single"

    def _initialize_weight(self, X: np.ndarray) -> np.ndarray:
        if self.init == "pca":
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            weight = np.asarray(vt[0], dtype=float)
        elif self.init == "random":
            rng = np.random.default_rng(self.random_state)
            weight = rng.normal(size=X.shape[1])
        else:
            raise ValueError(f"Unsupported init strategy: {self.init!r}")
        norm = np.linalg.norm(weight)
        if norm > 1e-12:
            weight = weight / norm
        return weight

    def fit(self, dataset: dict[str, Any], tracker: Any | None = None) -> FitResult:
        X = np.asarray(dataset["X"], dtype=float)
        X_centered = X - X.mean(axis=0, keepdims=True)
        params = {"weight": self._initialize_weight(X_centered)}
        batch = {"X": X_centered, "graph": dataset["graph"]}
        artifacts = self.optimizer.optimize(params, self.objective, batch, self.backend)
        components = np.asarray(artifacts.params["weight"], dtype=float).reshape(1, -1)
        result = FitResult(
            params=artifacts.params,
            components=components,
            objective=float(artifacts.objective),
            n_iter=artifacts.state.iteration,
            converged=bool(artifacts.state.converged),
            history=artifacts.state.history,
            metadata={"backend": self.backend.name},
        )
        if tracker is not None:
            tracker.log_metric("objective", result.objective, step=result.n_iter)
        return result
