"""Filesystem checkpoint utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np


class NumpyCheckpointIO:
    """Checkpoint serializer for NumPy-backed runs."""

    def save(
        self,
        path: Path,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        payload = {key: np.asarray(value) for key, value in model_state.items()}
        payload.update(
            {
                f"optimizer__{key}": np.asarray(value)
                for key, value in optimizer_state.items()
            }
        )
        np.savez(path / "state.npz", **cast(dict[str, Any], payload))
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def load(self, path: Path) -> dict[str, Any]:
        state = np.load(path / "state.npz", allow_pickle=False)
        model_state: dict[str, np.ndarray] = {}
        optimizer_state: dict[str, np.ndarray] = {}
        for key in state.files:
            value = np.asarray(state[key])
            if key.startswith("optimizer__"):
                optimizer_state[key.removeprefix("optimizer__")] = value
            else:
                model_state[key] = value
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metadata": metadata,
        }
