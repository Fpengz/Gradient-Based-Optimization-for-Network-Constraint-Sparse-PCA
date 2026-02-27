"""Shared estimator utilities for SPCA-style models."""

from __future__ import annotations

from typing import Any


class EstimatorStateMixin:
    """Mixin that standardizes post-fit estimator state.

    The project roadmap expects all estimators to expose common attributes
    (`history_`, `converged_`, `n_iter_`, `objective_`).
    """

    def _init_fit_state(self) -> None:
        self.history_: dict[str, list[Any]] = {}
        self.converged_ = False
        self.n_iter_ = 0
        self.objective_ = None

    def _push_history(self, key: str, value: Any) -> None:
        self.history_.setdefault(key, []).append(value)
