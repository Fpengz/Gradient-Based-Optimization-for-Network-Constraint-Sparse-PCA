"""Graph construction helpers and Laplacian utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


@dataclass(slots=True)
class GraphData:
    adjacency: sp.csr_matrix
    laplacian: sp.csr_matrix
    laplacian_type: str = "unnormalized"
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_csr(matrix: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return sp.csr_matrix(matrix)
    return sp.csr_matrix(np.asarray(matrix, dtype=float))


def adjacency_to_laplacian(
    adjacency: sp.spmatrix | np.ndarray,
    laplacian_type: str = "unnormalized",
) -> sp.csr_matrix:
    """Construct a Laplacian from a symmetric adjacency matrix."""

    adjacency_csr = _to_csr(adjacency)
    if laplacian_type not in {"unnormalized", "sym_norm"}:
        raise ValueError(f"Unknown laplacian_type={laplacian_type!r}")

    degrees = np.asarray(adjacency_csr.sum(axis=1)).ravel()
    degree_matrix = sp.diags(degrees, offsets=0, format="csr")
    if laplacian_type == "unnormalized":
        return (degree_matrix - adjacency_csr).tocsr()

    inv_sqrt = np.zeros_like(degrees, dtype=float)
    mask = degrees > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    degree_inv_sqrt = sp.diags(inv_sqrt, offsets=0, format="csr")
    eye = sp.eye(adjacency_csr.shape[0], format="csr")
    return (eye - degree_inv_sqrt @ adjacency_csr @ degree_inv_sqrt).tocsr()


def _package_graph(
    adjacency: sp.spmatrix | np.ndarray,
    laplacian_type: str,
    metadata: dict[str, Any],
) -> GraphData:
    adjacency_csr = _to_csr(adjacency)
    adjacency_csr = ((adjacency_csr + adjacency_csr.T) * 0.5).tocsr()
    adjacency_csr.setdiag(0.0)
    adjacency_csr.eliminate_zeros()
    laplacian = adjacency_to_laplacian(adjacency_csr, laplacian_type=laplacian_type)
    return GraphData(
        adjacency=adjacency_csr,
        laplacian=laplacian,
        laplacian_type=laplacian_type,
        metadata=metadata,
    )


def chain_graph(n_nodes: int, laplacian_type: str = "unnormalized") -> GraphData:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    rows = np.arange(n_nodes - 1)
    cols = rows + 1
    data = np.ones(n_nodes - 1, dtype=float)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adjacency = adjacency + adjacency.T
    return _package_graph(adjacency, laplacian_type, {"family": "chain", "n_nodes": n_nodes})


def grid_graph(n_rows: int, n_cols: int, laplacian_type: str = "unnormalized") -> GraphData:
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be positive")
    n_nodes = n_rows * n_cols
    rows: list[int] = []
    cols: list[int] = []
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if row + 1 < n_rows:
                rows.append(idx)
                cols.append((row + 1) * n_cols + col)
            if col + 1 < n_cols:
                rows.append(idx)
                cols.append(row * n_cols + (col + 1))
    data = np.ones(len(rows), dtype=float)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adjacency = adjacency + adjacency.T
    return _package_graph(
        adjacency,
        laplacian_type,
        {"family": "grid", "n_rows": n_rows, "n_cols": n_cols},
    )


def er_graph(
    n_nodes: int,
    p_edge: float,
    random_state: int | None = None,
    laplacian_type: str = "unnormalized",
) -> GraphData:
    if not 0.0 <= p_edge <= 1.0:
        raise ValueError("p_edge must be in [0, 1]")
    rng = np.random.default_rng(random_state)
    upper = rng.random((n_nodes, n_nodes))
    mask = np.triu(upper < p_edge, k=1)
    adjacency = sp.csr_matrix(mask.astype(float))
    adjacency = adjacency + adjacency.T
    return _package_graph(
        adjacency,
        laplacian_type,
        {"family": "er", "n_nodes": n_nodes, "p_edge": p_edge},
    )


def random_geometric_graph(
    n_nodes: int,
    radius: float,
    random_state: int | None = None,
    laplacian_type: str = "unnormalized",
) -> GraphData:
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    rng = np.random.default_rng(random_state)
    coordinates = rng.uniform(size=(n_nodes, 2))
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    mask = (dist <= radius) & (~np.eye(n_nodes, dtype=bool))
    return _package_graph(
        sp.csr_matrix(mask.astype(float)),
        laplacian_type,
        {"family": "rgg", "n_nodes": n_nodes, "radius": float(radius)},
    )


def sbm_graph(
    block_sizes: list[int] | tuple[int, ...],
    p_in: float,
    p_out: float,
    random_state: int | None = None,
    laplacian_type: str = "unnormalized",
) -> GraphData:
    if not 0.0 <= p_in <= 1.0 or not 0.0 <= p_out <= 1.0:
        raise ValueError("p_in and p_out must be in [0, 1]")
    sizes = list(block_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("All block sizes must be positive")
    n_nodes = sum(sizes)
    labels = np.concatenate([np.full(size, idx) for idx, size in enumerate(sizes)])
    rng = np.random.default_rng(random_state)
    upper = rng.random((n_nodes, n_nodes))
    probs = np.where(labels[:, None] == labels[None, :], p_in, p_out)
    mask = np.triu(upper < probs, k=1)
    adjacency = sp.csr_matrix(mask.astype(float))
    adjacency = adjacency + adjacency.T
    return _package_graph(
        adjacency,
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
    """Construct a symmetric kNN graph from row-wise samples or features."""

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be 2D")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    neighbors = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, X_array.shape[0]),
        metric=metric,
    )
    neighbors.fit(X_array)
    distances, indices = neighbors.kneighbors(X_array, return_distance=True)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row_idx in range(X_array.shape[0]):
        for distance, col_idx in zip(distances[row_idx, 1:], indices[row_idx, 1:]):
            rows.append(row_idx)
            cols.append(int(col_idx))
            if mode == "distance":
                vals.append(float(np.exp(-(distance**2))))
            elif mode == "connectivity":
                vals.append(1.0)
            else:
                raise ValueError(f"Unknown mode={mode!r}")

    adjacency = sp.csr_matrix((vals, (rows, cols)), shape=(X_array.shape[0], X_array.shape[0]))
    return _package_graph(
        adjacency,
        laplacian_type,
        {
            "family": "knn",
            "n_neighbors": n_neighbors,
            "metric": metric,
            "mode": mode,
        },
    )
