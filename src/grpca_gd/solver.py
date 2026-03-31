from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .objective import objective_terms, objective_total
from .stiefel import qr_retraction, rgrad
from .metrics import coupling_gap, laplacian_energy, orthogonality_error, sparsity_fraction


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


@dataclass
class SolverConfig:
    lambda1: float
    lambda2: float
    rho: float
    eta_A: float
    max_iters: int
    tol_obj: float
    tol_gap: float
    tol_orth: float


@dataclass
class SolverResult:
    A: np.ndarray
    B: np.ndarray
    history: Dict[str, np.ndarray]


def solve(
    A0: np.ndarray,
    B0: np.ndarray,
    sigma_hat: np.ndarray,
    L: np.ndarray,
    cfg: SolverConfig,
) -> SolverResult:
    A = A0.copy()
    B = B0.copy()

    L_norm = float(np.linalg.eigvalsh(L).max())
    eta_B = 1.0 / (2.0 * cfg.lambda2 * L_norm + cfg.rho)

    history: Dict[str, List[float]] = {
        "total_objective": [],
        "negative_variance_term": [],
        "sparsity_penalty": [],
        "graph_penalty": [],
        "coupling_penalty": [],
        "coupling_gap": [],
        "orthogonality_error": [],
        "sparsity_fraction": [],
        "laplacian_energy": [],
    }

    prev_obj = None
    for _ in range(cfg.max_iters):
        G_A = -2.0 * (sigma_hat @ A) + cfg.rho * (A - B)
        A = qr_retraction(A - cfg.eta_A * rgrad(A, G_A))

        grad_B = 2.0 * cfg.lambda2 * (L @ B) + cfg.rho * (B - A)
        B_new = soft_threshold(B - eta_B * grad_B, eta_B * cfg.lambda1)

        if not np.isfinite(B_new).all():
            back_eta = eta_B
            for _ in range(10):
                back_eta *= 0.5
                B_new = soft_threshold(B - back_eta * grad_B, back_eta * cfg.lambda1)
                if np.isfinite(B_new).all():
                    eta_B = back_eta
                    break
        B = B_new

        terms = objective_terms(A, B, sigma_hat, L, cfg.lambda1, cfg.lambda2, cfg.rho)
        obj = terms["total_objective"]

        history["total_objective"].append(obj)
        history["negative_variance_term"].append(terms["negative_variance_term"])
        history["sparsity_penalty"].append(terms["sparsity_penalty"])
        history["graph_penalty"].append(terms["graph_penalty"])
        history["coupling_penalty"].append(terms["coupling_penalty"])
        history["coupling_gap"].append(coupling_gap(A, B))
        history["orthogonality_error"].append(orthogonality_error(A))
        history["sparsity_fraction"].append(sparsity_fraction(B))
        history["laplacian_energy"].append(laplacian_energy(B, L))

        if prev_obj is not None:
            rel = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
            if (
                rel <= cfg.tol_obj
                and history["coupling_gap"][-1] <= cfg.tol_gap
                and history["orthogonality_error"][-1] <= cfg.tol_orth
            ):
                break
        prev_obj = obj

        if not np.isfinite(obj):
            raise FloatingPointError("Objective became non-finite")

    history_np = {k: np.array(v, dtype=float) for k, v in history.items()}
    return SolverResult(A=A, B=B, history=history_np)
