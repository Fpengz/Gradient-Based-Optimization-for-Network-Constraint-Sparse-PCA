"""Proximal quasi-Newton optimizer adapter."""

from __future__ import annotations

from dataclasses import dataclass

from .pg import PGOptimizer


@dataclass(slots=True)
class ProxQNOptimizer(PGOptimizer):
    """Proximal quasi-Newton optimizer placeholder within the new API."""

    qn_memory: int = 10
    name: str = "prox_qn"
