from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "realdata"


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df["dataset"].isin(["mnist", "tcga", "sp500"]) & (df["method"] == "Proposed")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["explained_variance", "smoothness_used_graph", "sparsity_ratio"],
        ["Explained variance", "Smoothness", "Sparsity"],
    ):
        for dataset in ["mnist", "tcga", "sp500"]:
            sub = df[df["dataset"] == dataset]
            ax.plot(sub["lambda2"], sub[metric], label=dataset)
        ax.set_ylabel(title)
        ax.legend()
    axes[-1].set_xlabel("lambda2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "realdata_unified_panels.png", dpi=200)

    summary = (
        df.groupby("dataset")["smoothness_used_graph"]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "realdata_summary.csv", index=False)


if __name__ == "__main__":
    main()
