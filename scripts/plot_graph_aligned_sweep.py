from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "metrics_summary.csv"
FIGURES = ROOT / "figures"


def _mean_std(series) -> Tuple[float, float]:
    values = series.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return math.nan, math.nan
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def _amanpg_reference(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    amanpg = df[df["method"] == "A-ManPG"].copy()
    amanpg = amanpg.drop_duplicates(subset=["seed"])
    mean, std = _mean_std(amanpg[metric])
    return {"mean": mean, "std": std}


def _plot_metric(
    ax,
    grid,
    method_df,
    metric: str,
    label: str,
) -> None:
    means = []
    stds = []
    for val in grid:
        subset = method_df[method_df["lambda2"] == val]
        mean, std = _mean_std(subset[metric])
        means.append(mean)
        stds.append(std)
    ax.errorbar(grid, means, yerr=stds, marker="o", label=label)


def main() -> None:
    df = pd.read_csv(SUMMARY)
    ga = df[df["experiment"] == "graph_aligned"]
    proposed = ga[ga["method"] == "Proposed"]
    spca = ga[ga["method"] == "SparseNoGraph"]
    grid = sorted(proposed["lambda2"].dropna().unique())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [
        ("support_f1", "Support F1"),
        ("graph_smoothness_norm", "Smoothness (norm)"),
        ("connect_largest_ratio", "Largest CC ratio"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        _plot_metric(ax, grid, proposed, metric, "Proposed")
        if not spca.empty:
            _plot_metric(ax, grid, spca, metric, "SparseNoGraph")
        ref = _amanpg_reference(ga, metric)
        if not math.isnan(ref["mean"]):
            ax.axhline(ref["mean"], linestyle="--", color="gray", label="A-ManPG")
            if ref["std"] > 0:
                ax.fill_between(
                    [min(grid), max(grid)],
                    ref["mean"] - ref["std"],
                    ref["mean"] + ref["std"],
                    color="gray",
                    alpha=0.15,
                )
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "graph_aligned_lambda2_sweep_panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
