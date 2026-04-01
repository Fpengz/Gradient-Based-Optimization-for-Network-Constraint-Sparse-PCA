from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
FIG = ROOT / "latex" / "figures" / "phase15_pareto.png"


def main() -> None:
    df = pd.read_csv(SUMMARY)
    df = df[df["decoy_intensity"] == "high"]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for method, marker in [("Proposed", "o"), ("SparseNoGraph", "s"), ("A-ManPG", "^")]:
        subset = df[df["method"] == method]
        ax.scatter(
            subset["graph_smoothness_norm"],
            subset["support_f1"],
            alpha=0.6,
            label=method,
            marker=marker,
        )
    ax.set_xlabel("Smoothness (norm)")
    ax.set_ylabel("Support F1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
