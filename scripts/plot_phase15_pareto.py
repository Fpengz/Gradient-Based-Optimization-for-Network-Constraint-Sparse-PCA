from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
FIGURES = ROOT / "figures"


def main() -> None:
    df = pd.read_csv(SUMMARY)
    phase = df[(df["experiment"] == "trackB_phase15") & (df["decoy_intensity"] == "high")]

    metrics = phase.groupby(["method", "lambda2"], as_index=False).agg(
        support_f1=("support_f1", "mean"),
        graph_smoothness_norm=("graph_smoothness_norm", "mean"),
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    methods = ["PCA", "A-ManPG", "SparseNoGraph", "Proposed"]
    colors = {
        "PCA": "#7f7f7f",
        "A-ManPG": "#1f77b4",
        "SparseNoGraph": "#ff7f0e",
        "Proposed": "#2ca02c",
    }

    for method in methods:
        subset = metrics[metrics["method"] == method].sort_values("graph_smoothness_norm")
        if subset.empty:
            continue
        ax.scatter(
            subset["graph_smoothness_norm"],
            subset["support_f1"],
            s=48,
            alpha=0.85,
            label=method,
            color=colors.get(method),
        )
        if method == "Proposed":
            ax.plot(
                subset["graph_smoothness_norm"],
                subset["support_f1"],
                color=colors.get(method),
                alpha=0.5,
                linewidth=1.2,
            )

    ax.set_xlabel("Graph smoothness norm")
    ax.set_ylabel("Support F1")
    ax.set_title("Phase 1.5 high-decoy Pareto scatter")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase15_pareto_scatter.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
