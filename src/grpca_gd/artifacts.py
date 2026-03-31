from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_npz(path: Path, arrays: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_metrics_csv(path: Path, metrics: Dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = _flatten(metrics)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in flat.items():
            writer.writerow([key, value])


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{key}" if prefix == "" else f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, full_key))
        else:
            out[full_key] = value
    return out
