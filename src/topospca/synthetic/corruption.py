from __future__ import annotations

from typing import Dict

import numpy as np


def _edge_list(A: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.triu_indices_from(A, k=1)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if A[i, j] > 0]


def _non_edge_list(A: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.triu_indices_from(A, k=1)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if A[i, j] == 0]


def _rebuild_artifact(artifact: Dict[str, object], A: np.ndarray) -> Dict[str, object]:
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return {
        "adjacency": A,
        "laplacian": L,
        "family": artifact["family"],
        "metadata": artifact.get("metadata", {}),
    }


def delete_edges(artifact: Dict[str, object], alpha: float, rng: np.random.Generator) -> Dict[str, object]:
    A = np.array(artifact["adjacency"], copy=True)
    edges = _edge_list(A)
    if not edges or alpha <= 0:
        return _rebuild_artifact(artifact, A)
    k = min(len(edges), int(np.floor(alpha * len(edges))))
    if k == 0:
        return _rebuild_artifact(artifact, A)
    idx = rng.choice(len(edges), size=k, replace=False)
    for i in idx:
        u, v = edges[i]
        A[u, v] = 0.0
        A[v, u] = 0.0
    return _rebuild_artifact(artifact, A)


def add_edges(artifact: Dict[str, object], alpha: float, rng: np.random.Generator) -> Dict[str, object]:
    A = np.array(artifact["adjacency"], copy=True)
    non_edges = _non_edge_list(A)
    if not non_edges or alpha <= 0:
        return _rebuild_artifact(artifact, A)
    current_edges = _edge_list(A)
    k = min(len(non_edges), int(np.floor(alpha * len(current_edges))))
    if k == 0:
        return _rebuild_artifact(artifact, A)
    idx = rng.choice(len(non_edges), size=k, replace=False)
    for i in idx:
        u, v = non_edges[i]
        A[u, v] = 1.0
        A[v, u] = 1.0
    return _rebuild_artifact(artifact, A)


def rewire_edges(artifact: Dict[str, object], alpha: float, rng: np.random.Generator) -> Dict[str, object]:
    A = np.array(artifact["adjacency"], copy=True)
    edges = _edge_list(A)
    non_edges = _non_edge_list(A)
    if not edges or not non_edges or alpha <= 0:
        return _rebuild_artifact(artifact, A)
    k = min(len(edges), len(non_edges), int(np.floor(alpha * len(edges))))
    if k == 0:
        return _rebuild_artifact(artifact, A)
    remove_idx = rng.choice(len(edges), size=k, replace=False)
    add_idx = rng.choice(len(non_edges), size=k, replace=False)
    for i in remove_idx:
        u, v = edges[i]
        A[u, v] = 0.0
        A[v, u] = 0.0
    for i in add_idx:
        u, v = non_edges[i]
        A[u, v] = 1.0
        A[v, u] = 1.0
    return _rebuild_artifact(artifact, A)


def corrupt_graph(
    artifact: Dict[str, object],
    corruption_type: str,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    if corruption_type == "delete":
        return delete_edges(artifact, alpha, rng)
    if corruption_type == "add":
        return add_edges(artifact, alpha, rng)
    if corruption_type == "rewire":
        return rewire_edges(artifact, alpha, rng)
    raise ValueError("corruption_type must be 'delete', 'add', or 'rewire'")
