"""Evaluation metrics for SPCA experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def explained_variance(
    X: np.ndarray, component: np.ndarray, centered: bool = False
) -> float:
    X = np.asarray(X, dtype=float)
    w = np.asarray(component, dtype=float).reshape(-1)
    if not np.all(np.isfinite(w)):
        return float("nan")
    if not np.all(np.isfinite(X)):
        return float("nan")
    if not centered:
        X = X - X.mean(axis=0)
    n = X.shape[0]
    if n <= 0:
        return float("nan")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        xw = X @ w
        value = np.dot(xw, xw) / n
    return float(value) if np.isfinite(value) else float("nan")


def support_metrics(
    estimated_support: Iterable[int] | np.ndarray,
    true_support: Iterable[int] | np.ndarray,
) -> dict[str, float]:
    est = set(int(i) for i in estimated_support)
    tru = set(int(i) for i in true_support)
    tp = len(est & tru)
    fp = len(est - tru)
    fn = len(tru - est)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def laplacian_energy(vector: np.ndarray, laplacian: sp.spmatrix | np.ndarray) -> float:
    x = np.asarray(vector, dtype=float).reshape(-1)
    L = laplacian if sp.issparse(laplacian) else np.asarray(laplacian, dtype=float)
    if sp.issparse(L):
        return float(x @ (L @ x))
    return float(x @ L @ x)


def connected_support_lcc_ratio(
    vector_or_support: np.ndarray | Iterable[int],
    adjacency: sp.spmatrix | np.ndarray,
    threshold: float = 1e-8,
) -> float:
    if isinstance(vector_or_support, np.ndarray):
        x = np.asarray(vector_or_support).reshape(-1)
        support = np.flatnonzero(np.abs(x) > threshold)
    else:
        support = np.array(sorted(set(int(i) for i in vector_or_support)), dtype=int)
    if support.size == 0:
        return 0.0
    if support.size == 1:
        return 1.0

    A = (
        adjacency
        if sp.issparse(adjacency)
        else sp.csr_matrix(np.asarray(adjacency, dtype=float))
    )
    sub = A[support][:, support]
    n_comp, labels = connected_components(sub, directed=False, return_labels=True)
    if n_comp == 0:
        return 0.0
    counts = np.bincount(labels)
    return float(counts.max() / support.size)
