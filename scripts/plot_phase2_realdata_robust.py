from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results" / "phase2" / "realdata_robust" / "metrics_summary.csv"
DEFAULT_OUT = ROOT / "figures" / "phase2" / "realdata" / "realdata_subsample_lambda2_sweep.png"


def write_realdata_robust_plot(df: pd.DataFrame, out_path: Path) -> None:
    base = df[(df["dataset_name"] == "tcga_brca_string")]
    methods = ["Proposed", "SparseNoGraph"]
    base = base[base["method"].isin(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    metrics = [
        ("graph_smoothness_norm", "Smoothness (norm)"),
        ("shared_explained_variance", "Shared explained var"),
    ]
    colors = {"Proposed": "#2b6cb0", "SparseNoGraph": "#c53030"}

    for ax, (metric, title) in zip(axes, metrics):
        for method in methods:
            subset = base[base["method"] == method]
            grouped = subset.groupby("lambda2")[metric].mean().reset_index()
            ax.plot(
                grouped["lambda2"],
                grouped[metric],
                marker="o",
                label=method,
                color=colors[method],
            )
        ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DEFAULT_TABLE)
    write_realdata_robust_plot(df, DEFAULT_OUT)


if __name__ == "__main__":
    main()
