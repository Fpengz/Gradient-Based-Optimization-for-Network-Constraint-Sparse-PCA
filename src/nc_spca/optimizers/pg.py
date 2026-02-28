"""Proximal-gradient optimizer for single-component NC-SPCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .base import FitArtifacts, FitState


@dataclass(slots=True)
class PGOptimizer:
    """Baseline proximal-gradient optimizer."""

    max_iter: int = 400
    learning_rate: float | str = "auto"
    tol: float = 1e-6
    monotone_backtracking: bool = True
    name: str = "pg"

    def _estimate_lipschitz(
        self,
        X: np.ndarray,
        laplacian: sp.spmatrix | np.ndarray,
        lambda2: float,
    ) -> float:
        _, singular_values, _ = np.linalg.svd(X, full_matrices=False)
        sigma_norm = (float(singular_values[0]) ** 2) / max(X.shape[0], 1)
        if sp.issparse(laplacian):
            laplacian_csr = sp.csr_matrix(laplacian)
            degrees = np.asarray(laplacian_csr.diagonal(), dtype=float).reshape(-1)
            lap_norm = float(2.0 * degrees.max()) if degrees.size else 0.0
        else:
            lap_norm = float(np.linalg.norm(np.asarray(laplacian, dtype=float), 2))
        return 2.0 * sigma_norm + 2.0 * lambda2 * lap_norm

    def optimize(
        self,
        params: dict[str, np.ndarray],
        objective: Any,
        batch: dict[str, Any],
        backend: Any,
    ) -> FitArtifacts:
        weight = np.asarray(params["weight"], dtype=float).reshape(-1)
        X = np.asarray(batch["X"], dtype=float)
        laplacian = batch["graph"].laplacian
        eta = (
            1.0 / max(self._estimate_lipschitz(X, laplacian, objective.lambda2), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )
        state = FitState(history={"objective": [], "step_size": [], "rel_change": []})
        current = {"weight": weight}

        for iteration in range(self.max_iter):
            current_weight = np.asarray(current["weight"], dtype=float).reshape(-1)
            smooth_value, grad = objective.smooth_value_and_grad(current, batch)
            grad_weight = np.asarray(grad["weight"], dtype=float).reshape(-1)
            trial_step = eta
            while True:
                trial = {"weight": current_weight - trial_step * grad_weight}
                prox = objective.prox(trial, trial_step, backend)
                new_weight = np.asarray(prox["weight"], dtype=float).reshape(-1)
                if not self.monotone_backtracking:
                    break
                old_obj = objective.evaluate({"weight": current_weight}, batch).total
                new_obj = objective.evaluate({"weight": new_weight}, batch).total
                if new_obj <= old_obj + 1e-12 or trial_step < 1e-16:
                    break
                trial_step *= 0.5

            rel_change = float(
                np.linalg.norm(new_weight - current_weight)
                / max(np.linalg.norm(current_weight), 1e-12)
            )
            current = {"weight": new_weight}
            total = objective.evaluate(current, batch).total
            state.history["objective"].append(float(total))
            state.history["step_size"].append(float(trial_step))
            state.history["rel_change"].append(rel_change)
            state.iteration = iteration + 1
            eta = trial_step
            if rel_change < self.tol:
                state.converged = True
                break

        final_objective = objective.evaluate(current, batch).total
        return FitArtifacts(params=current, state=state, objective=float(final_objective))
