"""Graph-structure evaluation metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def laplacian_energy(vector: np.ndarray, laplacian: sp.spmatrix | np.ndarray) -> float:
    """Compute the graph Dirichlet energy x^T L x."""

    values = np.asarray(vector, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        return float("nan")
    operator = laplacian if sp.issparse(laplacian) else np.asarray(laplacian, dtype=float)
    if sp.issparse(operator):
        value = float(values @ (operator @ values))
    else:
        value = float(values @ operator @ values)
    return value if np.isfinite(value) else float("nan")


def connected_support_lcc_ratio(
    vector_or_support: np.ndarray | Iterable[int],
    adjacency: sp.spmatrix | np.ndarray,
    threshold: float = 1e-8,
) -> float:
    """Largest connected component ratio inside the selected support."""

    if isinstance(vector_or_support, np.ndarray):
        support = np.flatnonzero(np.abs(np.asarray(vector_or_support, dtype=float)) > threshold)
    else:
        support = np.array(sorted(set(int(idx) for idx in vector_or_support)), dtype=int)
    if support.size == 0:
        return 0.0
    if support.size == 1:
        return 1.0

    adjacency_csr = adjacency if sp.issparse(adjacency) else sp.csr_matrix(adjacency)
    subgraph = sp.csr_matrix(adjacency_csr)[support][:, support]
    n_components, labels = connected_components(subgraph, directed=False, return_labels=True)
    if n_components == 0:
        return 0.0
    return float(np.bincount(labels).max() / support.size)
