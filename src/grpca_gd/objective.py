from __future__ import annotations

from typing import Dict

import numpy as np


def negative_variance_term(A: np.ndarray, sigma_hat: np.ndarray) -> float:
    return -float(np.trace(A.T @ sigma_hat @ A))


def sparsity_penalty(B: np.ndarray, lambda1: float) -> float:
    return float(lambda1 * np.sum(np.abs(B)))


def graph_penalty(B: np.ndarray, L: np.ndarray, lambda2: float) -> float:
    return float(lambda2 * np.trace(B.T @ L @ B))


def coupling_penalty(A: np.ndarray, B: np.ndarray, rho: float) -> float:
    diff = A - B
    return float(0.5 * rho * np.sum(diff * diff))


def objective_terms(
    A: np.ndarray,
    B: np.ndarray,
    sigma_hat: np.ndarray,
    L: np.ndarray,
    lambda1: float,
    lambda2: float,
    rho: float,
) -> Dict[str, float]:
    neg_var = negative_variance_term(A, sigma_hat)
    sparse = sparsity_penalty(B, lambda1)
    graph = graph_penalty(B, L, lambda2)
    coupling = coupling_penalty(A, B, rho)
    total = neg_var + sparse + graph + coupling
    return {
        "negative_variance_term": neg_var,
        "sparsity_penalty": sparse,
        "graph_penalty": graph,
        "coupling_penalty": coupling,
        "total_objective": total,
    }


def objective_total(
    A: np.ndarray,
    B: np.ndarray,
    sigma_hat: np.ndarray,
    L: np.ndarray,
    lambda1: float,
    lambda2: float,
    rho: float,
) -> float:
    return objective_terms(A, B, sigma_hat, L, lambda1, lambda2, rho)["total_objective"]
