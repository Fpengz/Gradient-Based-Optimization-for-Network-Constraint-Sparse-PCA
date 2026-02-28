"""Support recovery metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def support_metrics(
    estimated_support: Iterable[int] | np.ndarray,
    true_support: Iterable[int] | np.ndarray,
) -> dict[str, float]:
    estimated = set(int(idx) for idx in estimated_support)
    truth = set(int(idx) for idx in true_support)
    tp = len(estimated & truth)
    fp = len(estimated - truth)
    fn = len(truth - estimated)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def topk_support_metrics(
    weights: np.ndarray,
    true_support: Iterable[int] | np.ndarray,
    k: int,
) -> dict[str, float]:
    """Compute support metrics from the top-k absolute coefficients."""

    flattened = np.asarray(weights, dtype=float).reshape(-1)
    if k <= 0 or flattened.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    k_eff = min(int(k), flattened.size)
    indices = np.argpartition(np.abs(flattened), -k_eff)[-k_eff:]
    return support_metrics(indices, true_support)


def component_support_metrics(
    estimated_loadings: np.ndarray,
    true_loadings: np.ndarray,
    threshold: float = 1e-8,
) -> dict[str, float | list[tuple[int, int]]]:
    """Match components by support overlap and report macro recovery metrics."""

    estimated = np.asarray(estimated_loadings, dtype=float)
    truth = np.asarray(true_loadings, dtype=float)
    if estimated.ndim != 2 or truth.ndim != 2:
        raise ValueError("estimated_loadings and true_loadings must be matrices.")
    if estimated.shape[1] != truth.shape[1]:
        raise ValueError("estimated_loadings and true_loadings must have the same number of columns.")

    n_components = truth.shape[1]
    precision = np.zeros((n_components, n_components), dtype=float)
    recall = np.zeros((n_components, n_components), dtype=float)
    f1 = np.zeros((n_components, n_components), dtype=float)
    for i in range(n_components):
        est_support = np.flatnonzero(np.abs(estimated[:, i]) > threshold)
        for j in range(n_components):
            true_support = np.flatnonzero(np.abs(truth[:, j]) > threshold)
            metrics = support_metrics(est_support, true_support)
            precision[i, j] = metrics["precision"]
            recall[i, j] = metrics["recall"]
            f1[i, j] = metrics["f1"]

    row_ind, col_ind = linear_sum_assignment(1.0 - f1)
    matched_pairs = [(int(i), int(j)) for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=False)]
    return {
        "mean_precision": float(np.mean(precision[row_ind, col_ind])) if matched_pairs else 0.0,
        "mean_recall": float(np.mean(recall[row_ind, col_ind])) if matched_pairs else 0.0,
        "mean_f1": float(np.mean(f1[row_ind, col_ind])) if matched_pairs else 0.0,
        "matched_pairs": matched_pairs,
    }


def shared_support_metrics(
    estimated_loadings: np.ndarray,
    true_loadings: np.ndarray,
    threshold: float = 1e-8,
) -> dict[str, float]:
    """Compute row-support recovery for multi-component loadings."""

    estimated = np.asarray(estimated_loadings, dtype=float)
    truth = np.asarray(true_loadings, dtype=float)
    est_rows = np.flatnonzero(np.linalg.norm(estimated, axis=1) > threshold)
    true_rows = np.flatnonzero(np.linalg.norm(truth, axis=1) > threshold)
    return support_metrics(est_rows, true_rows)
