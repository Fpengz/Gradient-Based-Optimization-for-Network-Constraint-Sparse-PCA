from __future__ import annotations

import numpy as np


def _upper_edge_pairs(W: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.triu_indices_from(W, k=1)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if W[i, j] > 0]


def delete_edges(W: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    W2 = W.copy()
    edge_idx = _upper_edge_pairs(W2)
    m = int(len(edge_idx) * frac)
    if m <= 0:
        return W2
    remove = rng.choice(len(edge_idx), size=m, replace=False)
    for idx in np.atleast_1d(remove):
        i, j = edge_idx[int(idx)]
        W2[i, j] = 0.0
        W2[j, i] = 0.0
    return W2


def rewire_edges(W: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    W2 = W.copy()
    edge_idx = _upper_edge_pairs(W2)
    m = int(len(edge_idx) * frac)
    if m <= 0 or not edge_idx:
        return W2

    remove = rng.choice(len(edge_idx), size=m, replace=False)
    available = [
        (int(i), int(j))
        for i, j in zip(*np.triu_indices_from(W2, k=1))
        if W2[int(i), int(j)] <= 0
    ]

    for idx in np.atleast_1d(remove):
        i, j = edge_idx[int(idx)]
        weight = W2[i, j]
        W2[i, j] = 0.0
        W2[j, i] = 0.0
        if not available:
            W2[i, j] = weight
            W2[j, i] = weight
            continue
        pick = int(rng.integers(0, len(available)))
        a, b = available.pop(pick)
        W2[a, b] = weight
        W2[b, a] = weight
    return W2


def perturb_weights(W: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=scale, size=W.shape)
    Wn = np.maximum(0.0, W + noise)
    np.fill_diagonal(Wn, 0.0)
    return np.maximum(Wn, Wn.T)
