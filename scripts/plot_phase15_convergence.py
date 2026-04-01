from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "trackB" / "phase1_5"
FIGURES = ROOT / "figures"


def _pick_run_dir() -> Path:
    candidates = sorted(RESULTS.rglob("artifacts.npz"))
    if not candidates:
        raise FileNotFoundError(f"No artifacts.npz found under {RESULTS}")
    preferred = [
        path
        for path in candidates
        if "decoy_high" in path.as_posix() and "lambda2_0p05" in path.as_posix()
    ]
    return preferred[0].parent if preferred else candidates[0].parent


def _load_series(artifact_path: Path, key: str) -> np.ndarray:
    with np.load(artifact_path, allow_pickle=True) as data:
        if key in data:
            return np.asarray(data[key], dtype=float)
        if f"history_{key}" in data:
            return np.asarray(data[f"history_{key}"], dtype=float)
        raise KeyError(f"{key} not found in {artifact_path}")


def _plot_series(ax: plt.Axes, y: np.ndarray, title: str, ylabel: str) -> None:
    x = np.arange(len(y))
    ax.plot(x, y, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def main() -> None:
    run_dir = _pick_run_dir()
    artifact_path = run_dir / "artifacts.npz"

    total_objective = _load_series(artifact_path, "total_objective")
    coupling_gap = _load_series(artifact_path, "coupling_gap")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    _plot_series(axes[0], total_objective, "Objective", "Objective")
    _plot_series(axes[1], coupling_gap, "Coupling gap", "Coupling gap")
    fig.suptitle(f"Phase 1.5 convergence: {run_dir.relative_to(ROOT)}", y=1.02)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase15_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
