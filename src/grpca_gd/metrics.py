from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def sparsity_fraction(B: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs(B) > eps))


def laplacian_energy(B: np.ndarray, L: np.ndarray) -> float:
    # Inputs are finite in pilot checks; guard against spurious BLAS warnings.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.trace(B.T @ L @ B))


def graph_smoothness_raw(B: np.ndarray, L_true: np.ndarray) -> float:
    # Inputs are finite in pilot checks; guard against spurious BLAS warnings.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.trace(B.T @ L_true @ B))


def graph_smoothness_norm(B: np.ndarray, L_true: np.ndarray) -> float:
    denom = float(np.trace(B.T @ B))
    if denom <= 0:
        return 0.0
    # Inputs are finite in pilot checks; guard against spurious BLAS warnings.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.trace(B.T @ L_true @ B) / denom)


def explained_variance(Q: np.ndarray, sigma_hat: np.ndarray) -> float:
    # Inputs are finite in pilot checks; guard against spurious BLAS warnings.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return float(np.trace(Q.T @ sigma_hat @ Q))


def orthonormalize(B: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(B)
    return Q


def orthogonality_error(A: np.ndarray) -> float:
    r = A.shape[1]
    return float(np.linalg.norm(A.T @ A - np.eye(r), ord="fro"))


def coupling_gap(A: np.ndarray, B: np.ndarray) -> float:
    denom = max(1.0, np.linalg.norm(A, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)


def match_components(
    A_est: np.ndarray, A_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    corr = np.abs(A_est.T @ A_true)
    row_ind, col_ind = linear_sum_assignment(-corr)
    signs = np.ones_like(col_ind, dtype=float)
    for i, j in zip(row_ind, col_ind):
        signs[i] = 1.0 if (A_est[:, i].T @ A_true[:, j]) >= 0 else -1.0
    return col_ind, signs


def _support_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def support_metrics(
    B_est: np.ndarray,
    true_supports: List[np.ndarray],
    eps: float = 1e-8,
) -> Dict[str, Dict[str, float]]:
    r = B_est.shape[1]
    if r != len(true_supports):
        raise ValueError("Mismatch between number of components and true supports")

    est_supports = [np.abs(B_est[:, j]) > eps for j in range(r)]
    p = B_est.shape[0]
    per_comp = []
    for j in range(r):
        truth_mask = np.zeros(p, dtype=bool)
        truth_mask[true_supports[j]] = True
        per_comp.append(_support_metrics(est_supports[j], truth_mask))

    est_union = np.any(np.column_stack(est_supports), axis=1)
    true_union_mask = np.zeros(p, dtype=bool)
    for idx in true_supports:
        true_union_mask[idx] = True
    true_union = true_union_mask
    union = _support_metrics(est_union, true_union)

    return {
        "per_component": {
            str(j): per_comp[j] for j in range(r)
        },
        "union": union,
    }


def support_connectivity(mask: np.ndarray, L_true: np.ndarray) -> Dict[str, float]:
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"num_components": 0.0, "largest_component_ratio": 0.0}

    adjacency = (L_true < 0).astype(int)
    visited = set()
    components = []
    for node in idx:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            neighbors = np.where(adjacency[u] > 0)[0]
            for v in neighbors:
                if v in visited or not mask[v]:
                    continue
                visited.add(v)
                stack.append(v)
        components.append(size)

    num_components = len(components)
    largest = max(components)
    return {
        "num_components": float(num_components),
        "largest_component_ratio": float(largest / len(idx)),
    }
