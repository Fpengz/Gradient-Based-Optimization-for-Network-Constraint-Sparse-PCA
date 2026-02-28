"""MASPG-CAR optimizer adapter."""

from __future__ import annotations

from dataclasses import dataclass

from .pg import PGOptimizer


@dataclass(slots=True)
class MASPGCAROptimizer(PGOptimizer):
    """Practical accelerated variant.

    The current implementation reuses the PG optimizer surface while exposing a
    distinct optimizer identity and configuration slot. This keeps the public
    architecture stable while the full acceleration path migrates out of the
    legacy estimator stack.
    """

    name: str = "maspg_car"
