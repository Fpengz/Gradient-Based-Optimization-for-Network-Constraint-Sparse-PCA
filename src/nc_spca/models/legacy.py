"""Adapters for legacy baseline estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import FitResult


@dataclass(slots=True)
class LegacyEstimatorModel:
    """Adapter that exposes a legacy estimator through the new model API."""

    builder: Callable[[], Any]
    backend: Any
    name: str

    def fit(self, dataset: dict[str, Any], tracker: Any | None = None) -> FitResult:
        estimator = self.builder()
        X = np.asarray(dataset["X"], dtype=float)
        graph = dataset.get("graph")
        try:
            estimator.fit(X, graph=graph)
        except TypeError:
            estimator.fit(X)
        components = np.asarray(estimator.components_, dtype=float)
        n_iter_raw = getattr(estimator, "n_iter_", 0)
        if isinstance(n_iter_raw, list):
            n_iter = int(max(n_iter_raw)) if n_iter_raw else 0
        else:
            n_iter = int(n_iter_raw)
        history_raw = getattr(estimator, "history_", {})
        result = FitResult(
            params={"components": components},
            components=components,
            objective=float(getattr(estimator, "objective_", np.nan))
            if getattr(estimator, "objective_", None) is not None
            else float("nan"),
            n_iter=n_iter,
            converged=bool(getattr(estimator, "converged_", False)),
            history={key: value for key, value in history_raw.items() if isinstance(value, list)},
            metadata={"backend": self.backend.name},
        )
        if tracker is not None and np.isfinite(result.objective):
            tracker.log_metric("objective", result.objective, step=result.n_iter)
        return result
