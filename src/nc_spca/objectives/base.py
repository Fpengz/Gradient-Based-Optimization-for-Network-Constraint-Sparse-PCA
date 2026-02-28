"""Base objective types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ObjectiveEval:
    """Decomposed objective value."""

    total: float
    smooth: float
    nonsmooth: float
    terms: dict[str, float] = field(default_factory=dict)
