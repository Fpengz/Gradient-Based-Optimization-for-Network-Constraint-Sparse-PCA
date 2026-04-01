from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "latex" / "figures" / "phase15_coupling_gap.png"


def _load_history(run_dir: Path) -> np.ndarray:
    data = np.load(run_dir / "artifacts.npz")
    return data["history_coupling_gap"]


def main() -> None:
    runs = [
        ROOT / "results" / "trackB" / "phase1_5" / "seed0" / "lambda2_0p00" / "decoy_high",
        ROOT / "results" / "trackB" / "phase1_5" / "seed0" / "lambda2_0p05" / "decoy_high",
        ROOT / "results" / "trackB" / "phase1_5" / "seed0" / "lambda2_0p10" / "decoy_high",
        ROOT / "results" / "trackB" / "phase1_5" / "seed0" / "lambda2_0p20" / "decoy_high",
    ]
    labels = ["0.00", "0.05", "0.10", "0.20"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for run, label in zip(runs, labels):
        series = _load_history(run)
        ax.plot(series, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Coupling gap")
    ax.grid(True, alpha=0.3)
    ax.legend(title=r"$\lambda_2$")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
