from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
FIGURES = ROOT / "figures"
METRICS_CACHE: Dict[Path, Dict[str, object]] = {}


def _mean_sem(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return math.nan, math.nan
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


def _metric_series(df: pd.DataFrame, method: str, lambda2: float, metric: str) -> pd.Series:
    subset = df[(df["method"] == method) & (df["lambda2"] == lambda2)]
    return subset[metric]


def _load_metrics(output_dir: str) -> Dict[str, object]:
    path = Path(output_dir) / "metrics.json"
    if path not in METRICS_CACHE:
        with path.open("r", encoding="utf-8") as f:
            METRICS_CACHE[path] = json.load(f)
    return METRICS_CACHE[path]


def _connectivity_series(df: pd.DataFrame, method: str, lambda2: float) -> pd.Series:
    subset = df[(df["method"] == method) & (df["lambda2"] == lambda2)]
    if subset.empty:
        return subset.get("connect_largest_ratio", pd.Series(dtype=float))

    values = []
    for _, row in subset.iterrows():
        metrics = _load_metrics(str(row["output_dir"]))
        payload = metrics.get(method, {})
        conn = payload.get("support_connectivity_union", {})
        values.append(conn.get("largest_component_ratio"))
    return pd.Series(values, dtype=float)


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    methods: Iterable[str],
) -> None:
    grid = sorted(float(v) for v in df["lambda2"].dropna().unique())
    for method in methods:
        means = []
        sems = []
        for lam in grid:
            if metric == "connect_largest_ratio":
                series = _connectivity_series(df, method, lam)
            else:
                series = _metric_series(df, method, lam, metric)
            mean, sem = _mean_sem(series)
            means.append(mean)
            sems.append(sem)
        ax.errorbar(grid, means, yerr=sems, marker="o", capsize=3, label=method)

    ax.axvspan(0.05, 0.10, color="gray", alpha=0.15, zorder=0)
    ax.set_title(title)
    ax.set_xlabel(r"$\lambda_2$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def main() -> None:
    df = pd.read_csv(SUMMARY)
    phase = df[df["experiment"] == "trackB_phase15"].copy()
    high = phase[phase["decoy_intensity"] == "high"].copy()
    methods = ["SparseNoGraph", "A-ManPG", "Proposed"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharex=True)
    panels = [
        ("support_f1", "Support F1", "Support F1"),
        ("graph_smoothness_norm", "Graph smoothness norm", "Smoothness (norm)"),
        ("connect_largest_ratio", "Largest CC ratio", "Largest CC ratio"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        _plot_panel(ax, high, metric, title, ylabel, methods)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "phase15_band_panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
