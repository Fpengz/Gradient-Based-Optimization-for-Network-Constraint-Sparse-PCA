"""Synthetic graph-structured data generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config.schema import DataConfig
from .graphs import chain_graph, er_graph, grid_graph, random_geometric_graph, sbm_graph


def _build_graph(cfg: DataConfig, rng: np.random.Generator):
    if cfg.graph_type == "chain":
        return chain_graph(cfg.n_features, laplacian_type=cfg.graph_laplacian_type)
    if cfg.graph_type == "grid":
        side = int(np.sqrt(cfg.n_features))
        if side * side != cfg.n_features:
            raise ValueError("grid graph requires n_features to be a perfect square")
        return grid_graph(side, side, laplacian_type=cfg.graph_laplacian_type)
    if cfg.graph_type == "rgg":
        return random_geometric_graph(
            cfg.n_features,
            radius=0.18,
            random_state=int(rng.integers(0, 2**31 - 1)),
            laplacian_type=cfg.graph_laplacian_type,
        )
    if cfg.graph_type == "sbm":
        n_blocks = min(4, max(2, cfg.n_features // 8))
        base = cfg.n_features // n_blocks
        rem = cfg.n_features % n_blocks
        sizes = [base + (1 if i < rem else 0) for i in range(n_blocks)]
        return sbm_graph(
            sizes,
            p_in=0.25,
            p_out=0.02,
            random_state=int(rng.integers(0, 2**31 - 1)),
            laplacian_type=cfg.graph_laplacian_type,
        )
    if cfg.graph_type == "er":
        return er_graph(
            cfg.n_features,
            p_edge=0.08,
            random_state=int(rng.integers(0, 2**31 - 1)),
            laplacian_type=cfg.graph_laplacian_type,
        )
    raise ValueError(f"Unsupported graph type: {cfg.graph_type!r}")


def _sample_connected_support(adjacency, support_size: int, rng: np.random.Generator) -> np.ndarray:
    n = adjacency.shape[0]
    k = min(max(1, support_size), n)
    start = int(rng.integers(0, n))
    support = [start]
    seen = {start}
    frontier = [start]

    while len(support) < k and frontier:
        node = frontier.pop(0)
        neighbors = adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]
        neighbors = neighbors[rng.permutation(len(neighbors))]
        for neighbor in neighbors:
            idx = int(neighbor)
            if idx in seen:
                continue
            support.append(idx)
            seen.add(idx)
            frontier.append(idx)
            if len(support) >= k:
                break

    if len(support) < k:
        remaining = [i for i in range(n) if i not in seen]
        extra = rng.choice(remaining, size=k - len(support), replace=False)
        support.extend(int(i) for i in extra)

    return np.array(sorted(support), dtype=int)


def _sample_multi_component_supports(
    adjacency,
    support_size: int,
    n_components: int,
    overlap_mode: str,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if n_components <= 1:
        return [_sample_connected_support(adjacency, support_size, rng)]
    if overlap_mode == "shared":
        base_support = _sample_connected_support(adjacency, support_size, rng)
        return [base_support.copy() for _ in range(n_components)]
    if overlap_mode != "disjoint":
        raise ValueError(
            f"Unsupported support overlap mode: {overlap_mode!r}. Expected 'disjoint' or 'shared'."
        )

    supports: list[np.ndarray] = []
    used: set[int] = set()
    for _ in range(n_components):
        for _attempt in range(128):
            candidate = _sample_connected_support(adjacency, support_size, rng)
            if used.isdisjoint(set(int(idx) for idx in candidate)):
                supports.append(candidate)
                used.update(int(idx) for idx in candidate)
                break
        else:
            remaining = [idx for idx in range(adjacency.shape[0]) if idx not in used]
            if len(remaining) < support_size:
                candidate = _sample_connected_support(adjacency, support_size, rng)
            else:
                candidate = np.array(sorted(rng.choice(remaining, size=support_size, replace=False)))
            supports.append(np.asarray(candidate, dtype=int))
            used.update(int(idx) for idx in candidate)
    return supports


def generate_synthetic_dataset(cfg: DataConfig, seed: int | None = None) -> dict[str, Any]:
    """Generate a synthetic graph-structured dataset."""
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    graph = _build_graph(cfg, rng)
    true_supports = _sample_multi_component_supports(
        graph.adjacency,
        cfg.support_size,
        cfg.n_components,
        cfg.support_overlap_mode,
        rng,
    )
    V_true = np.zeros((cfg.n_features, cfg.n_components), dtype=float)
    for component_idx, support in enumerate(true_supports):
        values = rng.normal(size=support.size)
        V_true[support, component_idx] = values
        norm = np.linalg.norm(V_true[:, component_idx]) + 1e-12
        V_true[:, component_idx] /= norm

    latent = rng.normal(size=(cfg.n_samples, cfg.n_components))
    noise = cfg.noise_std * rng.normal(size=(cfg.n_samples, cfg.n_features))
    scale = np.sqrt(cfg.signal_strength / max(cfg.n_components, 1))
    signal = np.zeros((cfg.n_samples, cfg.n_features), dtype=float)
    for component_idx in range(cfg.n_components):
        signal += scale * np.outer(latent[:, component_idx], V_true[:, component_idx])
    X = signal + noise

    w_true = V_true[:, 0]
    true_support = true_supports[0]
    return {
        "X": X,
        "graph": graph,
        "w_true": w_true,
        "V_true": V_true,
        "true_support": true_support,
        "true_supports": [np.asarray(support, dtype=int) for support in true_supports],
        "seed": int(cfg.random_state if seed is None else seed),
    }
