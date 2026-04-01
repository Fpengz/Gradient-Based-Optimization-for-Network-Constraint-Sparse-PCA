"""
Generates Figure: high_decoy_band.png (Phase 1.5)

Inputs:
- Phase 1.5 CSV (default: ../phase15/... or local fallback)

Outputs:
- latex/figures/high_decoy_band.png

Panels:
1. Support F1
2. Graph smoothness norm
3. Largest CC ratio
4. \u0394 Support F1 (Proposed \u2212 SparseNoGraph)

This script is the single source of truth for the main graph-aligned result.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    Path("/Users/zhoufuwang/Projects/GRPCA-GD-phase15")
    / "results"
    / "trackB"
    / "phase1_5"
    / "aggregated_phase15.csv"
)
FALLBACK_DATA = ROOT / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
OUTPUT_FIG = ROOT / "latex" / "figures" / "high_decoy_band.png"


def _sem(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_data_path() -> Path:
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA
    if FALLBACK_DATA.exists():
        return FALLBACK_DATA
    raise FileNotFoundError("No aggregated_phase15.csv found in expected locations.")


def _collect_metric(
    rows: List[dict],
    metric: str,
    method: str,
) -> Dict[float, List[float]]:
    data: Dict[float, List[float]] = defaultdict(list)
    for r in rows:
        if r["decoy_intensity"] != "high":
            continue
        if r["method"] != method:
            continue
        if r["status"] != "success":
            continue
        l2 = float(r["lambda2"])
        data[l2].append(float(r[metric]))
    return data


def _collect_delta_f1(rows: List[dict]) -> Dict[float, List[float]]:
    # paired per seed: Proposed - SparseNoGraph
    by_key: Dict[Tuple[int, float, str], float] = {}
    for r in rows:
        if r["decoy_intensity"] != "high":
            continue
        if r["status"] != "success":
            continue
        if r["method"] not in {"Proposed", "SparseNoGraph"}:
            continue
        seed = int(r["seed"])
        l2 = float(r["lambda2"])
        key = (seed, l2, r["method"])
        by_key[key] = float(r["support_f1"])

    deltas: Dict[float, List[float]] = defaultdict(list)
    for seed in {k[0] for k in by_key}:
        for l2 in {k[1] for k in by_key}:
            prop = by_key.get((seed, l2, "Proposed"))
            sparse = by_key.get((seed, l2, "SparseNoGraph"))
            if prop is None or sparse is None:
                continue
            deltas[l2].append(prop - sparse)
    return deltas


def _plot_series(ax, grid: List[float], data: Dict[float, List[float]], label: str) -> None:
    means = [_mean(data.get(l2, [])) for l2 in grid]
    sems = [_sem(data.get(l2, [])) for l2 in grid]
    ax.errorbar(grid, means, yerr=sems, marker="o", label=label)


def main() -> None:
    csv_path = _get_data_path()
    rows = _load_rows(csv_path)

    proposed_f1 = _collect_metric(rows, "support_f1", "Proposed")
    sparse_f1 = _collect_metric(rows, "support_f1", "SparseNoGraph")
    proposed_smooth = _collect_metric(rows, "graph_smoothness_norm", "Proposed")
    sparse_smooth = _collect_metric(rows, "graph_smoothness_norm", "SparseNoGraph")
    proposed_conn = _collect_metric(rows, "connect_largest_ratio", "Proposed")
    sparse_conn = _collect_metric(rows, "connect_largest_ratio", "SparseNoGraph")
    delta_f1 = _collect_delta_f1(rows)

    grid = sorted({*proposed_f1.keys(), *sparse_f1.keys()})

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    axes = axes.ravel()

    # Support F1
    _plot_series(axes[0], grid, proposed_f1, "Proposed")
    _plot_series(axes[0], grid, sparse_f1, "SparseNoGraph")
    axes[0].set_title("Support F1")
    axes[0].set_xlabel(r"$\lambda_2$")
    axes[0].grid(True, alpha=0.3)

    # Smoothness norm
    _plot_series(axes[1], grid, proposed_smooth, "Proposed")
    _plot_series(axes[1], grid, sparse_smooth, "SparseNoGraph")
    axes[1].set_title("Graph smoothness norm")
    axes[1].set_xlabel(r"$\lambda_2$")
    axes[1].grid(True, alpha=0.3)

    # Connectivity
    _plot_series(axes[2], grid, proposed_conn, "Proposed")
    _plot_series(axes[2], grid, sparse_conn, "SparseNoGraph")
    axes[2].set_title("Largest CC ratio")
    axes[2].set_xlabel(r"$\lambda_2$")
    axes[2].grid(True, alpha=0.3)

    # Delta F1
    means = [_mean(delta_f1.get(l2, [])) for l2 in grid]
    sems = [_sem(delta_f1.get(l2, [])) for l2 in grid]
    axes[3].errorbar(grid, means, yerr=sems, marker="o", label="Proposed - SparseNoGraph")
    axes[3].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[3].set_title(r"$\Delta$ Support F1")
    axes[3].set_xlabel(r"$\lambda_2$")
    axes[3].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
