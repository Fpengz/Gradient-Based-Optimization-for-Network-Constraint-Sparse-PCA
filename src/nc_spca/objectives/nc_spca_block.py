"""Block NC-SPCA objective on the Stiefel manifold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .base import ObjectiveEval


@dataclass(slots=True)
class NCSPCABlockObjective:
    """Block NC-SPCA objective configuration and geometry helpers."""

    lambda1: float = 0.15
    lambda2: float = 0.25
    support_threshold: float = 1e-6
    sparsity_mode: str = "l1"
    group_lambda: float | None = None
    retraction: str = "polar"
    name: str = "nc_spca_block"

    @staticmethod
    def _sym(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _row_group_soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe = np.maximum(norms, 1e-12)
        scale = np.maximum(1.0 - threshold / safe, 0.0)
        scale[norms <= threshold] = 0.0
        return matrix * scale

    @staticmethod
    def _retract_stiefel(loadings: np.ndarray, method: str = "polar") -> np.ndarray:
        if method not in {"polar", "qr"}:
            raise ValueError(f"Unsupported retraction: {method!r}")
        if method == "polar":
            try:
                U, _, Vt = np.linalg.svd(loadings, full_matrices=False)
                return U @ Vt
            except np.linalg.LinAlgError:
                method = "qr"
        Q, _ = np.linalg.qr(loadings)
        return Q[:, : loadings.shape[1]]

    def _penalty_weight(self) -> float:
        if self.sparsity_mode == "l21":
            return float(self.group_lambda if self.group_lambda is not None else self.lambda1)
        return float(self.lambda1)

    def _laplacian_multiply(
        self,
        loadings: np.ndarray,
        laplacian: sp.spmatrix | np.ndarray,
    ) -> np.ndarray:
        if sp.issparse(laplacian):
            return np.asarray(laplacian @ loadings, dtype=float)
        return np.asarray(laplacian, dtype=float) @ loadings

    def _laplacian_value(
        self,
        loadings: np.ndarray,
        laplacian: sp.spmatrix | np.ndarray,
    ) -> float:
        lap = self._laplacian_multiply(loadings, laplacian)
        return float(np.sum(loadings * lap))

    def _nonsmooth_value(self, loadings: np.ndarray) -> float:
        if self.sparsity_mode == "l21":
            return self._penalty_weight() * float(np.sum(np.linalg.norm(loadings, axis=1)))
        return self.lambda1 * float(np.sum(np.abs(loadings)))

    def smooth_value_and_grad(
        self,
        params: dict[str, np.ndarray],
        batch: dict[str, Any],
    ) -> tuple[float, dict[str, np.ndarray]]:
        loadings = np.asarray(params["loadings"], dtype=float)
        X = np.asarray(batch["X"], dtype=float)
        laplacian = batch["graph"].laplacian
        n = max(X.shape[0], 1)
        XV = X @ loadings
        sigma_value = -float(np.sum(XV * XV) / n)
        sigma_grad = -(2.0 / n) * (X.T @ XV)
        laplacian_mult = self._laplacian_multiply(loadings, laplacian)
        laplacian_value = self.lambda2 * float(np.sum(loadings * laplacian_mult))
        laplacian_grad = 2.0 * self.lambda2 * laplacian_mult
        grad = np.asarray(sigma_grad + laplacian_grad, dtype=float)
        return sigma_value + laplacian_value, {"loadings": grad}

    def riemannian_grad(
        self,
        params: dict[str, np.ndarray],
        batch: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        loadings = np.asarray(params["loadings"], dtype=float)
        _, grad = self.smooth_value_and_grad(params, batch)
        euclidean = np.asarray(grad["loadings"], dtype=float)
        riemannian = euclidean - loadings @ self._sym(loadings.T @ euclidean)
        return {"loadings": riemannian}

    def prox(
        self,
        params: dict[str, np.ndarray],
        step_size: float,
        backend: Any,
    ) -> dict[str, np.ndarray]:
        loadings = np.asarray(params["loadings"], dtype=float)
        if self.sparsity_mode == "l21":
            prox = self._row_group_soft_threshold(loadings, step_size * self._penalty_weight())
            return {"loadings": prox}
        prox = backend.soft_threshold(loadings, step_size * self.lambda1)
        return {"loadings": backend.to_numpy(prox)}

    def retract(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        loadings = np.asarray(params["loadings"], dtype=float)
        return {"loadings": self._retract_stiefel(loadings, method=self.retraction)}

    def orthogonality_error(self, params: dict[str, np.ndarray]) -> float:
        loadings = np.asarray(params["loadings"], dtype=float)
        n_components = loadings.shape[1]
        gram = loadings.T @ loadings
        return float(np.linalg.norm(gram - np.eye(n_components), ord="fro"))

    def evaluate(self, params: dict[str, np.ndarray], batch: dict[str, Any]) -> ObjectiveEval:
        loadings = np.asarray(params["loadings"], dtype=float)
        smooth_value, _ = self.smooth_value_and_grad(params, batch)
        nonsmooth = self._nonsmooth_value(loadings)
        total = smooth_value + nonsmooth
        return ObjectiveEval(
            total=total,
            smooth=smooth_value,
            nonsmooth=nonsmooth,
            terms={
                "sparsity_penalty": nonsmooth,
                "laplacian_energy": self.lambda2
                * self._laplacian_value(loadings, batch["graph"].laplacian),
                "orthogonality_error": self.orthogonality_error(params),
            },
        )
