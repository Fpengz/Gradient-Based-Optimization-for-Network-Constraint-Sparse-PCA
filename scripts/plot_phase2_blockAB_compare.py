from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLOCKA_TABLE = ROOT / "results" / "phase2" / "blockA" / "metrics_summary.csv"
BLOCKB_TABLE = ROOT / "results" / "phase2" / "blockB" / "metrics_summary.csv"
DEFAULT_OUT = (
    ROOT
    / "figures"
    / "phase2"
    / "compare"
    / "blockAB_chain_grid_lambda2_sweep.png"
)


COLORS = {
    "clean": "#1f1f1f",
    "delete": "#2b6cb0",
    "rewire": "#c53030",
}


def _extract_series(df: pd.DataFrame, graph_family: str) -> dict[str, pd.DataFrame]:
    base = df[(df["graph_family"] == graph_family) & (df["method"] == "Proposed")]
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
        raise ValueError(f"Missing clean/delete/rewire data for {graph_family}")

    return {
        "clean": clean,
        "delete": delete,
        "rewire": rewire,
    }


def _metric_limits(
    chain: dict[str, pd.DataFrame],
    grid: dict[str, pd.DataFrame],
    metric: str,
) -> tuple[float, float]:
    values = []
    for bundle in (chain, grid):
        for df in bundle.values():
            values.append(df.groupby("lambda2")[metric].mean())
    combined = pd.concat(values, axis=0)
    lo = float(combined.min())
    hi = float(combined.max())
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def write_blockAB_compare_plot(
    blockA_df: pd.DataFrame, blockB_df: pd.DataFrame, out_path: Path
) -> None:
    chain = _extract_series(blockA_df, "chain")
    grid = _extract_series(blockB_df, "grid")

    metrics = [
        ("support_f1", "Support F1"),
        ("graph_smoothness_norm", "Smoothness (norm)"),
        ("shared_explained_variance", "Shared explained var"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for row, (metric, title) in enumerate(metrics):
        y_min, y_max = _metric_limits(chain, grid, metric)
        for col, (graph_name, bundle) in enumerate(
            [("Chain", chain), ("Grid", grid)]
        ):
            ax = axes[row, col]
            for label in ("clean", "delete", "rewire"):
                grouped = bundle[label].groupby("lambda2")[metric].mean().reset_index()
                ax.plot(
                    grouped["lambda2"],
                    grouped[metric],
                    marker="o",
                    label=label,
                    color=COLORS[label],
                )
            ax.axvline(0.0, linestyle="--", color="gray", alpha=0.6)
            ax.set_ylim(y_min, y_max)
            if row == 0:
                ax.set_title(graph_name)
            if col == 0:
                ax.set_ylabel(title)
            if row == len(metrics) - 1:
                ax.set_xlabel(r"$\lambda_2$")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    blockA_df = pd.read_csv(BLOCKA_TABLE)
    blockB_df = pd.read_csv(BLOCKB_TABLE)
    write_blockAB_compare_plot(blockA_df, blockB_df, DEFAULT_OUT)


if __name__ == "__main__":
    main()
