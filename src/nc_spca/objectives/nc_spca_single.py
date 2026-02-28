"""Single-component NC-SPCA objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .base import ObjectiveEval


@dataclass(slots=True)
class NCSPCASingleObjective:
    """Single-component NC-SPCA objective.

    The mathematical source of truth is

        min_w -w^T Sigmahat w + lambda1 ||w||_1 + lambda2 w^T L w
        subject to ||w||_2 <= 1.
    """

    lambda1: float = 0.15
    lambda2: float = 0.25
    support_threshold: float = 1e-6
    name: str = "nc_spca_single"

    def _laplacian_value(
        self, vector: np.ndarray, laplacian: sp.spmatrix | np.ndarray
    ) -> float:
        if sp.issparse(laplacian):
            return float(vector @ (laplacian @ vector))
        dense = np.asarray(laplacian, dtype=float)
        return float(vector @ (dense @ vector))

    def smooth_value_and_grad(
        self,
        params: dict[str, np.ndarray],
        batch: dict[str, Any],
    ) -> tuple[float, dict[str, np.ndarray]]:
        weight = np.asarray(params["weight"], dtype=float).reshape(-1)
        X = np.asarray(batch["X"], dtype=float)
        laplacian = batch["graph"].laplacian
        n = X.shape[0]
        sigma_grad = -(2.0 / n) * (X.T @ (X @ weight))
        if sp.issparse(laplacian):
            graph_grad = 2.0 * self.lambda2 * np.asarray(laplacian @ weight).reshape(-1)
        else:
            graph_grad = 2.0 * self.lambda2 * (np.asarray(laplacian, dtype=float) @ weight)
        grad = sigma_grad + graph_grad
        sigma_value = -float(weight @ (X.T @ (X @ weight))) / float(n)
        graph_value = self.lambda2 * self._laplacian_value(weight, laplacian)
        return sigma_value + graph_value, {"weight": grad}

    def prox(self, params: dict[str, np.ndarray], step_size: float, backend: Any) -> dict[str, np.ndarray]:
        weight = backend.soft_threshold(params["weight"], step_size * self.lambda1)
        weight = backend.project_l2_ball(weight)
        return {"weight": backend.to_numpy(weight)}

    def evaluate(self, params: dict[str, np.ndarray], batch: dict[str, Any]) -> ObjectiveEval:
        weight = np.asarray(params["weight"], dtype=float).reshape(-1)
        smooth_value, _ = self.smooth_value_and_grad({"weight": weight}, batch)
        nonsmooth = self.lambda1 * float(np.linalg.norm(weight, 1))
        total = smooth_value + nonsmooth
        return ObjectiveEval(
            total=total,
            smooth=smooth_value,
            nonsmooth=nonsmooth,
            terms={
                "l1_penalty": nonsmooth,
                "weight_norm": float(np.linalg.norm(weight)),
            },
        )
