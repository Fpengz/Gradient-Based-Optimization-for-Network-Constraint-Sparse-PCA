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


def sbm_graph_laplacian(
    p: int,
    blocks: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
    block_sizes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    return L, W


def grid_graph_laplacian(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    p = rows * cols
    W = np.zeros((p, p), dtype=float)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if r + 1 < rows:
                j = idx(r + 1, c)
                W[i, j] = 1.0
                W[j, i] = 1.0
            if c + 1 < cols:
                j = idx(r, c + 1)
                W[i, j] = 1.0
                W[j, i] = 1.0

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W
