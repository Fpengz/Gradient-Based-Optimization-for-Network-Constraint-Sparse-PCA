from __future__ import annotations

import numpy as np


def chain_graph_laplacian(p: int) -> tuple[np.ndarray, np.ndarray]:
    if p < 2:
        raise ValueError("p must be >= 2 for chain graph")
    W = np.zeros((p, p), dtype=float)
    idx = np.arange(p - 1)
    W[idx, idx + 1] = 1.0
    W[idx + 1, idx] = 1.0
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def chain_graph_artifact(p: int) -> dict:
    L, W = chain_graph_laplacian(p)
    return {
        "adjacency": W,
        "laplacian": L,
        "family": "chain",
        "metadata": {},
    }


def infer_grid_shape(p: int) -> tuple[int, int]:
    side = int(np.sqrt(p))
    if side * side != p:
        raise ValueError("p must be a perfect square to infer grid shape")
    return side, side


def grid_graph_artifact(rows: int, cols: int) -> dict:
    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows and cols must be positive")
    p = rows * cols
    W = np.zeros((p, p), dtype=float)

    def node(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                i, j = node(r, c), node(r + 1, c)
                W[i, j] = 1.0
                W[j, i] = 1.0
            if c + 1 < cols:
                i, j = node(r, c), node(r, c + 1)
                W[i, j] = 1.0
                W[j, i] = 1.0

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return {
        "adjacency": W,
        "laplacian": L,
        "family": "grid",
        "metadata": {"grid_rows": rows, "grid_cols": cols},
    }


def sbm_graph_laplacian(
    p: int,
    blocks: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
    block_sizes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if block_sizes is None:
        base = p // blocks
        sizes = [base] * blocks
        sizes[-1] += p - base * blocks
    else:
        if sum(block_sizes) != p:
            raise ValueError("sum(block_sizes) must equal p")
        sizes = block_sizes

    labels = np.zeros(p, dtype=int)
    start = 0
    for k, sz in enumerate(sizes):
        labels[start : start + sz] = k
        start += sz

    W = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(i + 1, p):
            prob = p_in if labels[i] == labels[j] else p_out
            if rng.random() < prob:
                W[i, j] = 1.0
                W[j, i] = 1.0

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W, labels
