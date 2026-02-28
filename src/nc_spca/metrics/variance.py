"""Variance-oriented evaluation metrics."""

from __future__ import annotations

import numpy as np


def explained_variance(
    X: np.ndarray,
    component: np.ndarray,
    centered: bool = False,
) -> float:
    """Compute empirical explained variance for a single component."""

    X_array = np.asarray(X, dtype=float)
    weight = np.asarray(component, dtype=float).reshape(-1)
    if not np.all(np.isfinite(X_array)) or not np.all(np.isfinite(weight)):
        return float("nan")
    centered_X = X_array if centered else X_array - X_array.mean(axis=0, keepdims=True)
    n_samples = centered_X.shape[0]
    if n_samples <= 0:
        return float("nan")
    # Prefer einsum here because some BLAS backends emit spurious overflow
    # warnings on small matrix-vector products even when inputs are well scaled.
    projection = np.einsum("ij,j->i", centered_X, weight, optimize=True)
    value = np.dot(projection, projection) / n_samples
    return float(value) if np.isfinite(value) else float("nan")
