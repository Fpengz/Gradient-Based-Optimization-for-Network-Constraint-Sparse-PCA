from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results" / "phase2" / "multicomponent" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "phase2" / "multicomponent"


def _plot_support_type(df: pd.DataFrame, support_type: str) -> None:
    base = df[(df["method"] == "Proposed") & (df["support_type"] == support_type)]
    ks = sorted(base["k"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    metrics = [
        ("per_component_f1_mean", "Mean per-component F1"),
        ("orthogonality_error", "Orthogonality Error"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for k in ks:
            subset = base[base["k"] == k]
            grouped = subset.groupby("lambda2")[metric].mean().reset_index()
            ax.plot(
                grouped["lambda2"],
                grouped[metric],
                marker="o",
                label=f"k={int(k)}",
            )
        ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ks))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"multicomp_{support_type}_lambda2_sweep.png", dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DEFAULT_TABLE)
    for support_type in ["connected", "fragmented", "cross_community"]:
        _plot_support_type(df, support_type)


if __name__ == "__main__":
    main()
