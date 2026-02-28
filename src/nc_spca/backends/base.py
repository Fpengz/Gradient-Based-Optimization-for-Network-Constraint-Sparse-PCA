"""Backend protocol for objective and optimizer implementations."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class BackendProtocol(Protocol):
    """Abstract tensor backend contract."""

    name: str
    dtype: str
    device: str

    def asarray(self, value: Any, dtype: str | None = None) -> Any:
        """Convert a value to the backend array type."""

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a backend tensor to a NumPy array."""

    def soft_threshold(self, value: Any, threshold: float) -> Any:
        """Apply elementwise soft-thresholding."""

    def project_l2_ball(self, value: Any) -> Any:
        """Project a vector onto the unit l2 ball."""
