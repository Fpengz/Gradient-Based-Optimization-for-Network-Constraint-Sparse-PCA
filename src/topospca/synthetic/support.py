from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def generate_supports(
    p: int,
    r: int,
    support_size: int,
    support_type: str,
    rng: np.random.Generator,
    adjacency: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> List[np.ndarray]:
    if support_size <= 0 or support_size > p:
        raise ValueError("support_size must be in [1, p]")
    supports: List[np.ndarray] = []
    max_start = max(0, p - support_size)
    step = max(1, (p - support_size) // max(1, r))

    for j in range(r):
        base_start = (j * step) % (max_start + 1)
        jitter = int(rng.integers(0, max(1, step)))
        start = min(base_start + jitter, max_start)

        if support_type == "connected":
            if adjacency is not None:
                idx = _sample_connected(adjacency, support_size, rng)
            else:
                idx = np.arange(start, start + support_size)
        elif support_type == "multi_cluster":
            if adjacency is None:
                raise ValueError("multi_cluster requires adjacency")
            s1 = support_size // 2
            s2 = support_size - s1
            idx1 = _sample_connected(adjacency, s1, rng)
            idx2 = _sample_connected_excluding(adjacency, s2, rng, set(idx1))
            idx = np.concatenate([idx1, idx2])
        elif support_type == "fragmented":
            if adjacency is None:
                idx = rng.choice(p, size=support_size, replace=False)
            else:
                idx = _sample_fragmented(adjacency, support_size, rng)
        elif support_type == "cross_community":
            if metadata is None or "sbm_labels" not in metadata:
                raise ValueError("cross_community requires sbm_labels metadata")
            labels = np.array(metadata["sbm_labels"], dtype=int)
            idx = _sample_cross_community(labels, support_size, rng)
        elif support_type == "connected_disjoint":
            total = r * support_size
            min_gap = 1
            if p < total + (r + 1) * min_gap:
                raise ValueError("p too small for disjoint connected supports")
            gap = (p - total) // (r + 1)
            gap = max(min_gap, gap)
            used = total + (r + 1) * gap
            slack = max(0, p - used)
            offset = int(rng.integers(0, slack + 1)) if slack > 0 else 0
            start = offset + gap + j * (support_size + gap)
            idx = np.arange(start, start + support_size)
        elif support_type == "disconnected":
            s1 = support_size // 2
            s2 = support_size - s1
            if p < s1 + s2 + 1:
                raise ValueError("p too small for disconnected support")
            max_start1 = p - (s1 + s2 + 1)
            start1 = min(start, max_start1)
            gap = 1 + int(rng.integers(0, max(1, p - (start1 + s1 + s2))))
            start2 = start1 + s1 + gap
            idx = np.concatenate(
                [np.arange(start1, start1 + s1), np.arange(start2, start2 + s2)]
            )
        else:
            raise ValueError(
                "support_type must be 'connected', 'multi_cluster', 'fragmented', "
                "'cross_community', 'connected_disjoint', or 'disconnected'"
            )

        supports.append(idx)

    return supports


def _neighbors(adjacency: np.ndarray, node: int) -> np.ndarray:
    return np.where(adjacency[node] > 0)[0]


def _sample_connected(adjacency: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    p = adjacency.shape[0]
    for _ in range(20):
        start = int(rng.integers(0, p))
        visited = set([start])
        queue = [start]
        while queue and len(visited) < size:
            current = queue.pop(0)
            nbrs = _neighbors(adjacency, current)
            rng.shuffle(nbrs)
            for n in nbrs:
                if n not in visited:
                    visited.add(int(n))
                    queue.append(int(n))
                if len(visited) >= size:
                    break
        if len(visited) >= size:
            return np.array(list(visited)[:size], dtype=int)
    return rng.choice(p, size=size, replace=False)


def _sample_connected_excluding(
    adjacency: np.ndarray,
    size: int,
    rng: np.random.Generator,
    excluded: set,
) -> np.ndarray:
    p = adjacency.shape[0]
    candidates = np.array([i for i in range(p) if i not in excluded], dtype=int)
    if len(candidates) < size:
        raise ValueError("Not enough nodes to sample disjoint clusters")
    sub_adj = adjacency[np.ix_(candidates, candidates)]
    idx = _sample_connected(sub_adj, size, rng)
    return candidates[idx]


def _sample_fragmented(adjacency: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    p = adjacency.shape[0]
    selected: List[int] = []
    candidates = list(range(p))
    rng.shuffle(candidates)
    for node in candidates:
        if len(selected) >= size:
            break
        if all(adjacency[node, s] == 0 for s in selected):
            selected.append(node)
    if len(selected) < size:
        remaining = [i for i in range(p) if i not in selected]
        extra = rng.choice(remaining, size=size - len(selected), replace=False).tolist()
        selected.extend(extra)
    return np.array(selected, dtype=int)


def _sample_cross_community(
    labels: np.ndarray, support_size: int, rng: np.random.Generator
) -> np.ndarray:
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("cross_community requires at least 2 communities")
    c1, c2 = rng.choice(unique, size=2, replace=False)
    idx1 = np.where(labels == c1)[0]
    idx2 = np.where(labels == c2)[0]
    s1 = support_size // 2
    s2 = support_size - s1
    if len(idx1) < s1 or len(idx2) < s2:
        raise ValueError("Not enough nodes in selected communities")
    pick1 = rng.choice(idx1, size=s1, replace=False)
    pick2 = rng.choice(idx2, size=s2, replace=False)
    return np.concatenate([pick1, pick2])
