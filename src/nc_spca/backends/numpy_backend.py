"""NumPy backend implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class NumpyBackend:
    """Concrete NumPy backend."""

    dtype: str = "float64"
    device: str = "cpu"
    name: str = "numpy"

    def asarray(self, value: Any, dtype: str | None = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype or self.dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)

    def soft_threshold(self, value: Any, threshold: float) -> np.ndarray:
        array = self.asarray(value)
        return np.sign(array) * np.maximum(np.abs(array) - threshold, 0.0)

    def project_l2_ball(self, value: Any) -> np.ndarray:
        array = self.asarray(value)
        norm = float(np.linalg.norm(array))
        if not np.isfinite(norm):
            return np.zeros_like(array)
        return array / norm if norm > 1.0 else array
