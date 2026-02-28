"""Base optimizer types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FitState:
    """Mutable optimizer state."""

    iteration: int = 0
    converged: bool = False
    history: dict[str, list[float]] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitArtifacts:
    """Completed optimizer output."""

    params: dict[str, np.ndarray]
    state: FitState
    objective: float
