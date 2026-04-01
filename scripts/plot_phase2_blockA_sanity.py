from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results" / "phase2" / "blockA" / "metrics_summary.csv"
DEFAULT_OUT = ROOT / "figures" / "phase2" / "blockA" / "blockA_sanity_lambda2_sweep.png"


def write_blockA_sanity_plot(df: pd.DataFrame, out_path: Path) -> None:
    base = df[(df["graph_family"] == "chain") & (df["method"] == "Proposed")]
    clean = base[
        (base["corruption_level"] == 0.0)
        & (base["prior_graph_state"] == "clean")
    ]
    delete = base[
        (base["corruption_type"] == "delete")
        & (base["corruption_level"] == 0.2)
        & (base["prior_graph_state"] == "corrupted")
    ]
    rewire = base[
        (base["corruption_type"] == "rewire")
        & (base["corruption_level"] == 0.2)
        & (base["prior_graph_state"] == "corrupted")
    ]
    if clean.empty or delete.empty or rewire.empty:
        raise ValueError("Missing clean/delete/rewire data for blockA sanity plot")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [
        ("support_f1", "Support F1"),
        ("graph_smoothness_norm", "Smoothness (norm)"),
        ("shared_explained_variance", "Shared explained var"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        clean_group = clean.groupby("lambda2")[metric].mean().reset_index()
        delete_group = delete.groupby("lambda2")[metric].mean().reset_index()
        rewire_group = rewire.groupby("lambda2")[metric].mean().reset_index()
        ax.plot(clean_group["lambda2"], clean_group[metric], marker="o", label="clean")
        ax.plot(delete_group["lambda2"], delete_group[metric], marker="o", label="delete")
        ax.plot(rewire_group["lambda2"], rewire_group[metric], marker="o", label="rewire")
        ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(r"$\lambda_2$")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DEFAULT_TABLE)
    write_blockA_sanity_plot(df, DEFAULT_OUT)


if __name__ == "__main__":
    main()
