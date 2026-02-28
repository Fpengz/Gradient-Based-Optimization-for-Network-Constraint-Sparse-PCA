"""Manifold proximal-gradient optimizer for block NC-SPCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .base import FitArtifacts, FitState
from .pg import PGOptimizer


@dataclass(slots=True)
class ManifoldPGOptimizer(PGOptimizer):
    """Monotone block manifold proximal-gradient optimizer."""

    grad_norm_tol: float | None = None
    name: str = "manpg"

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
        loadings = np.asarray(params["loadings"], dtype=float)
        X = np.asarray(batch["X"], dtype=float)
        laplacian = batch["graph"].laplacian
        eta = (
            1.0 / max(self._estimate_lipschitz(X, laplacian, objective.lambda2), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )
        current = objective.retract({"loadings": loadings})
        state = FitState(
            history={
                "objective": [],
                "step_size": [],
                "rel_change": [],
                "riemannian_grad_norm": [],
                "orthogonality_error": [],
                "support_size": [],
            }
        )

        for iteration in range(self.max_iter):
            current_loadings = np.asarray(current["loadings"], dtype=float)
            riemannian = objective.riemannian_grad(current, batch)
            riemannian_grad = np.asarray(riemannian["loadings"], dtype=float)
            grad_norm = float(np.linalg.norm(riemannian_grad, ord="fro"))
            trial_step = eta
            old_obj = objective.evaluate(current, batch).total

            while True:
                candidate = {"loadings": current_loadings - trial_step * riemannian_grad}
                prox = objective.prox(candidate, trial_step, backend)
                trial = objective.retract(prox)
                new_loadings = np.asarray(trial["loadings"], dtype=float)
                new_obj = objective.evaluate({"loadings": new_loadings}, batch).total
                if not self.monotone_backtracking:
                    break
                if new_obj <= old_obj + 1e-12 or trial_step < 1e-16:
                    break
                trial_step *= 0.5

            rel_change = float(
                np.linalg.norm(new_loadings - current_loadings, ord="fro")
                / max(np.linalg.norm(current_loadings, ord="fro"), 1e-12)
            )
            ortho_error = objective.orthogonality_error({"loadings": new_loadings})
            support_size = int(
                np.count_nonzero(
                    np.linalg.norm(new_loadings, axis=1)
                    > getattr(objective, "support_threshold", 1e-6)
                )
            )

            current = {"loadings": new_loadings}
            state.history["objective"].append(float(new_obj))
            state.history["step_size"].append(float(trial_step))
            state.history["rel_change"].append(rel_change)
            state.history["riemannian_grad_norm"].append(grad_norm)
            state.history["orthogonality_error"].append(float(ortho_error))
            state.history["support_size"].append(float(support_size))
            state.iteration = iteration + 1
            eta = trial_step

            grad_tol = self.grad_norm_tol if self.grad_norm_tol is not None else self.tol
            if rel_change < self.tol or grad_norm < grad_tol:
                state.converged = True
                break

        final_objective = objective.evaluate(current, batch).total
        return FitArtifacts(params=current, state=state, objective=float(final_objective))
