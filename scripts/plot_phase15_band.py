from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
FIG = ROOT / "latex" / "figures" / "phase15_high_decoy_band.png"


def _mean_sem(series) -> Tuple[float, float]:
    values = series.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return math.nan, math.nan
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def _amanpg_reference(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    amanpg = df[df["method"] == "A-ManPG"].drop_duplicates(subset=["seed"])
    mean, sem = _mean_sem(amanpg[metric])
    return {"mean": mean, "sem": sem}


def main() -> None:
    df = pd.read_csv(SUMMARY)
    df = df[df["decoy_intensity"] == "high"]
    proposed = df[df["method"] == "Proposed"]
    spca = df[df["method"] == "SparseNoGraph"]
    grid = sorted(proposed["lambda2"].dropna().unique())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [
        ("support_f1", "Support F1"),
        ("graph_smoothness_norm", "Smoothness (norm)"),
        ("connect_largest_ratio", "Largest CC ratio"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for label, data in [("Proposed", proposed), ("SparseNoGraph", spca)]:
            means = []
            sems = []
            for val in grid:
                subset = data[data["lambda2"] == val]
                mean, sem = _mean_sem(subset[metric])
                means.append(mean)
                sems.append(sem)
            ax.errorbar(grid, means, yerr=sems, marker="o", label=label)
        ref = _amanpg_reference(df, metric)
        if not math.isnan(ref["mean"]):
            ax.axhline(ref["mean"], linestyle="--", color="gray", label="A-ManPG")
            if ref["sem"] > 0:
                ax.fill_between(
                    [min(grid), max(grid)],
                    ref["mean"] - ref["sem"],
                    ref["mean"] + ref["sem"],
                    color="gray",
                    alpha=0.15,
                )
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.grid(True, alpha=0.3)
        ax.axvspan(0.05, 0.10, color="orange", alpha=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
