from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results" / "phase2" / "blockA_alpha" / "metrics_summary.csv"
DEFAULT_OUT = (
    ROOT
    / "figures"
    / "phase2"
    / "compare"
    / "phase2_chain_alpha_lambda2_phase_diagram.png"
)


def _pivot(df: pd.DataFrame, corruption_type: str) -> pd.DataFrame:
    subset = df[
        (df["graph_family"] == "chain")
        & (df["method"] == "Proposed")
        & (df["corruption_type"] == corruption_type)
    ]
    pivot = subset.pivot_table(
        index="corruption_level", columns="lambda2", values="support_f1", aggfunc="mean"
    )
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def write_phase_diagram(df: pd.DataFrame, out_path: Path) -> None:
    delete = _pivot(df, "delete")
    rewire = _pivot(df, "rewire")
    vmin = np.nanmin([delete.values.min(), rewire.values.min()])
    vmax = np.nanmax([delete.values.max(), rewire.values.max()])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pivot, title in zip(
        axes, [delete, rewire], ["Delete", "Rewire"]
    ):
        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])
        ax.set_ylabel(r"$\alpha$" if title == "Delete" else "")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Support F1")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DEFAULT_TABLE)
    write_phase_diagram(df, DEFAULT_OUT)


if __name__ == "__main__":
    main()
