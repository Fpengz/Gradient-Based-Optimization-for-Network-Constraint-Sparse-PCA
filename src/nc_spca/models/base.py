"""Base model types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FitResult:
    """Standardized model fit result."""

    params: dict[str, np.ndarray]
    components: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    history: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
