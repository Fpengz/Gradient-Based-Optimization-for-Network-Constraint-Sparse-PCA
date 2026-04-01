from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "metrics_summary.csv"
FIGURES = ROOT / "latex" / "figures"


def _mean_sem(series):
    values = series.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return math.nan, math.nan
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def _plot_metric(ax, grid, df, metric, label):
    means = []
    sems = []
    for val in grid:
        subset = df[df["lambda2"] == val]
        mean, sem = _mean_sem(subset[metric])
        means.append(mean)
        sems.append(sem)
    ax.errorbar(grid, means, yerr=sems, marker="o", label=label)


def main() -> None:
    df = pd.read_csv(SUMMARY)
    real = df[df["experiment"] == "real_data"].copy()
    if real.empty:
        raise SystemExit("No real_data rows found in metrics_summary.csv")

    proposed = real[real["method"] == "Proposed"]
    spca = real[real["method"] == "SparseNoGraph"]
    grid = sorted(proposed["lambda2"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    _plot_metric(axes[0], grid, proposed, "graph_smoothness_norm", "Proposed")
    _plot_metric(axes[0], grid, spca, "graph_smoothness_norm", "SparseNoGraph")
    axes[0].set_title("Graph smoothness norm")
    axes[0].set_xlabel(r"$\lambda_2$")
    axes[0].grid(True, alpha=0.3)

    _plot_metric(axes[1], grid, proposed, "shared_explained_variance", "Proposed")
    _plot_metric(axes[1], grid, spca, "shared_explained_variance", "SparseNoGraph")
    axes[1].set_title("Explained variance")
    axes[1].set_xlabel(r"$\lambda_2$")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "realdata_lambda2_sweep.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
