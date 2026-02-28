"""Torch backend implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TorchBackend:
    """Optional Torch backend.

    The implementation is intentionally thin at this stage. It provides the
    same surface as the NumPy backend and raises a clear error if torch is not
    installed in the environment.
    """

    dtype: str = "float64"
    device: str = "cpu"
    name: str = "torch"

    def _torch(self):
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - guarded in runtime paths.
            raise RuntimeError(
                "Torch backend requested but torch is not installed."
            ) from exc
        return torch

    def asarray(self, value: Any, dtype: str | None = None) -> Any:
        torch = self._torch()
        torch_dtype = getattr(torch, dtype or self.dtype)
        return torch.as_tensor(value, dtype=torch_dtype, device=self.device)

    def to_numpy(self, value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()

    def soft_threshold(self, value: Any, threshold: float) -> Any:
        torch = self._torch()
        tensor = self.asarray(value)
        return torch.sign(tensor) * torch.clamp(torch.abs(tensor) - threshold, min=0.0)

    def project_l2_ball(self, value: Any) -> Any:
        torch = self._torch()
        tensor = self.asarray(value)
        norm = torch.linalg.norm(tensor)
        if not torch.isfinite(norm):
            return torch.zeros_like(tensor)
        return tensor / norm if float(norm) > 1.0 else tensor
