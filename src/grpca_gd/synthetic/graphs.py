from __future__ import annotations

import numpy as np


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = np.sum(W, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv = np.diag(inv_sqrt)
    return np.eye(W.shape[0]) - D_inv @ W @ D_inv


def grid_graph_laplacian(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    p = rows * cols
    W = np.zeros((p, p), dtype=float)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = rr * cols + cc
                    W[idx, j] = 1.0
    W = np.maximum(W, W.T)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def er_graph_laplacian(
    p: int,
    p_edge: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.triu((rng.random((p, p)) < p_edge).astype(float), 1)
    W = W + W.T
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def knn_graph_laplacian(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(dists[i])[1 : k + 1]
        W[i, nn] = 1.0
    W = np.maximum(W, W.T)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def small_world_laplacian(
    p: int,
    k: int,
    beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(1, k + 1):
            W[i, (i + j) % p] = 1.0
            W[i, (i - j) % p] = 1.0
    max_attempts = p
    for i in range(p):
        for j in range(1, k + 1):
            if rng.random() < beta:
                old = (i + j) % p
                # Best-effort rewiring: try a bounded number of random targets,
                # then keep the original edge if none are valid.
                for _ in range(max_attempts):
                    new = int(rng.integers(0, p))
                    if new != i and W[i, new] == 0.0:
                        W[i, old] = 0.0
                        W[old, i] = 0.0
                        W[i, new] = 1.0
                        W[new, i] = 1.0
                        break
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


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
