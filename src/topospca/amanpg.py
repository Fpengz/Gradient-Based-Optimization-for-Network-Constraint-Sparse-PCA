from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .objective import objective_terms
from .solver import soft_threshold
from .stiefel import qr_retraction, rgrad
from .metrics import orthogonality_error, sparsity_fraction


@dataclass
class AmanpgConfig:
    lambda1: float
    eta_A: float
    max_iters: int
    tol_obj: float
    tol_orth: float


@dataclass
class AmanpgResult:
    A: np.ndarray
    history: Dict[str, np.ndarray]


def solve_amanpg(A0: np.ndarray, sigma_hat: np.ndarray, cfg: AmanpgConfig) -> AmanpgResult:
    A = A0.copy()
    p = A.shape[0]
    zero_laplacian = np.zeros((p, p), dtype=A.dtype)

    history: Dict[str, List[float]] = {
        "total_objective": [],
        "negative_variance_term": [],
        "sparsity_penalty": [],
        "orthogonality_error": [],
        "sparsity_fraction": [],
    }

    prev_obj = None
    for _ in range(cfg.max_iters):
        grad = -2.0 * (sigma_hat @ A)
        A_step = A - cfg.eta_A * rgrad(A, grad)
        A_sparse = soft_threshold(A_step, cfg.eta_A * cfg.lambda1)
        A = qr_retraction(A_sparse)

        terms = objective_terms(A, A, sigma_hat, zero_laplacian, cfg.lambda1, 0.0, 0.0)
        obj = terms["total_objective"]

        history["total_objective"].append(obj)
        history["negative_variance_term"].append(terms["negative_variance_term"])
        history["sparsity_penalty"].append(terms["sparsity_penalty"])
        history["orthogonality_error"].append(orthogonality_error(A))
        history["sparsity_fraction"].append(sparsity_fraction(A))

        if not np.isfinite(obj):
            raise FloatingPointError("Objective became non-finite")

        if prev_obj is not None:
            rel = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
            if rel <= cfg.tol_obj and history["orthogonality_error"][-1] <= cfg.tol_orth:
                break
        prev_obj = obj

    history_np = {k: np.array(v, dtype=float) for k, v in history.items()}
    return AmanpgResult(A=A, history=history_np)
