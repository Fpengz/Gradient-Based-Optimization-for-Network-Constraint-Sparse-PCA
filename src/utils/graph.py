"""Graph construction helpers and Laplacian utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphData:
    adjacency: sp.csr_matrix
    laplacian: sp.csr_matrix
    laplacian_type: str = "unnormalized"
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_csr(matrix: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def adjacency_to_laplacian(
    adjacency: sp.spmatrix | np.ndarray,
    laplacian_type: str = "unnormalized",
) -> sp.csr_matrix:
    """Construct a graph Laplacian from a symmetric adjacency matrix."""
    A = _to_csr(adjacency)
    if laplacian_type not in {"unnormalized", "sym_norm"}:
        raise ValueError(f"Unknown laplacian_type={laplacian_type!r}")

    degrees = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(degrees, offsets=0, format="csr")

    if laplacian_type == "unnormalized":
        return (D - A).tocsr()

    inv_sqrt = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    D_inv_sqrt = sp.diags(inv_sqrt, offsets=0, format="csr")
    eye = sp.eye(A.shape[0], format="csr")
    return (eye - D_inv_sqrt @ A @ D_inv_sqrt).tocsr()


def _package_graph(
    adjacency: sp.spmatrix | np.ndarray,
    laplacian_type: str,
    metadata: dict[str, Any],
) -> GraphData:
    A = _to_csr(adjacency)
    # Ensure symmetry for downstream metrics/solvers.
    A = ((A + A.T) * 0.5).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    L = adjacency_to_laplacian(A, laplacian_type=laplacian_type)
    return GraphData(
        adjacency=A, laplacian=L, laplacian_type=laplacian_type, metadata=metadata
    )


def chain_graph(n_nodes: int, laplacian_type: str = "unnormalized") -> GraphData:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    rows = np.arange(n_nodes - 1)
    cols = rows + 1
    data = np.ones(n_nodes - 1, dtype=float)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    A = A + A.T
    return _package_graph(A, laplacian_type, {"family": "chain", "n_nodes": n_nodes})


def grid_graph(
    n_rows: int, n_cols: int, laplacian_type: str = "unnormalized"
) -> GraphData:
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    n = n_rows * n_cols
    rows: list[int] = []
    cols: list[int] = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if r + 1 < n_rows:
                rows.append(idx)
                cols.append((r + 1) * n_cols + c)
            if c + 1 < n_cols:
                rows.append(idx)
                cols.append(r * n_cols + (c + 1))
    data = np.ones(len(rows), dtype=float)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T
    return _package_graph(
        A, laplacian_type, {"family": "grid", "n_rows": n_rows, "n_cols": n_cols}
    )


def er_graph(
    n_nodes: int,
    p_edge: float,
    random_state: int | None = None,
    laplacian_type: str = "unnormalized",
) -> GraphData:
    if not (0 <= p_edge <= 1):
        raise ValueError("p_edge must be in [0, 1]")
    rng = np.random.default_rng(random_state)
    upper = rng.random((n_nodes, n_nodes))
    mask = np.triu(upper < p_edge, k=1)
    A = sp.csr_matrix(mask.astype(float))
    A = A + A.T
    return _package_graph(
        A, laplacian_type, {"family": "er", "n_nodes": n_nodes, "p_edge": p_edge}
    )


def sbm_graph(
    block_sizes: list[int] | tuple[int, ...],
    p_in: float,
    p_out: float,
    random_state: int | None = None,
    laplacian_type: str = "unnormalized",
) -> GraphData:
    if p_in < 0 or p_out < 0 or p_in > 1 or p_out > 1:
        raise ValueError("p_in and p_out must be in [0, 1]")
    sizes = list(block_sizes)
    if any(s <= 0 for s in sizes):
        raise ValueError("All block sizes must be positive")
    n = sum(sizes)
    labels = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    rng = np.random.default_rng(random_state)
    upper = rng.random((n, n))
    probs = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    mask = np.triu(upper < probs, k=1)
    A = sp.csr_matrix(mask.astype(float))
    A = A + A.T
    return _package_graph(
        A,
        laplacian_type,
        {"family": "sbm", "block_sizes": sizes, "p_in": p_in, "p_out": p_out},
    )


def knn_graph(
    X: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "euclidean",
    mode: str = "distance",
    laplacian_type: str = "unnormalized",
) -> GraphData:
    """Construct a symmetric kNN graph from row-wise samples/features."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, X.shape[0]), metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(X.shape[0]):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):  # skip self
            rows.append(i)
            cols.append(int(j))
            if mode == "distance":
                vals.append(float(np.exp(-(dist**2))))
            elif mode == "connectivity":
                vals.append(1.0)
            else:
                raise ValueError(f"Unknown mode={mode!r}")

    A = sp.csr_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    return _package_graph(
        A,
        laplacian_type,
        {"family": "knn", "n_neighbors": n_neighbors, "metric": metric, "mode": mode},
    )
