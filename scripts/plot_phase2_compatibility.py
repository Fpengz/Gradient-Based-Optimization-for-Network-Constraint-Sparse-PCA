from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results" / "phase2" / "compatibility" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "phase2" / "compatibility"


def _plot_family(df: pd.DataFrame, graph_family: str, support_types: list[str]) -> None:
    base = df[(df["graph_family"] == graph_family) & (df["method"] == "Proposed")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [
        ("support_f1", "Support F1"),
        ("graph_smoothness_norm", "Smoothness (norm)"),
        ("shared_explained_variance", "Shared explained var"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for support_type in support_types:
            subset = base[base["support_type"] == support_type]
            grouped = subset.groupby("lambda2")[metric].mean().reset_index()
            ax.plot(
                grouped["lambda2"],
                grouped[metric],
                marker="o",
                label=support_type,
            )
        ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(support_types))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"compat_{graph_family}_lambda2_sweep.png", dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DEFAULT_TABLE)
    _plot_family(df, "chain", ["connected", "multi_cluster", "fragmented"])
    _plot_family(df, "grid", ["connected", "multi_cluster", "fragmented"])
    _plot_family(df, "sbm", ["connected", "multi_cluster", "fragmented", "cross_community"])


if __name__ == "__main__":
    main()
