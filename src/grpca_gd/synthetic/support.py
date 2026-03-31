from __future__ import annotations

from typing import List

import numpy as np


def generate_supports(
    p: int,
    r: int,
    support_size: int,
    support_type: str,
    rng: np.random.Generator,
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
            idx = np.arange(start, start + support_size)
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
            idx = np.concatenate([
                np.arange(start1, start1 + s1),
                np.arange(start2, start2 + s2),
            ])
        else:
            raise ValueError(
                "support_type must be 'connected', 'connected_disjoint', or 'disconnected'"
            )

        supports.append(idx)

    return supports
