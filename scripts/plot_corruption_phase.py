from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures"


def _compute_delta(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df[df["corruption_level"] == 0.0]
        .groupby(["graph_family", "lambda2"])["support_f1"]
        .mean()
        .reset_index()
        .rename(columns={"support_f1": "base_f1"})
    )
    merged = df.merge(base, on=["graph_family", "lambda2"], how="left")
    merged["delta_f1"] = merged["support_f1"] - merged["base_f1"]
    return merged


def _phase_panels(delta: pd.DataFrame, families: list[str]) -> None:
    agg = (
        delta.groupby(["graph_family", "lambda2", "corruption_level"])["delta_f1"]
        .mean()
        .reset_index()
    )
    vmin = float(agg["delta_f1"].min())
    vmax = float(agg["delta_f1"].max())
    lim = max(abs(vmin), abs(vmax))

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, fam in zip(axes, families):
        sub = agg[agg["graph_family"] == fam]
        if sub.empty:
            ax.set_axis_off()
            continue
        pivot = sub.pivot(index="corruption_level", columns="lambda2", values="delta_f1")
        im = ax.imshow(
            pivot.values,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
        )
        ax.set_title(fam)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("lambda2")
        ax.set_ylabel("corruption level α")

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, label="ΔF1 vs clean")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "corruption_phase_panels.png", dpi=200)


def _summary_plot(delta: pd.DataFrame, families: list[str]) -> None:
    target = delta[(delta["corruption_level"] == 0.4) & (delta["lambda2"] == 0.5)]
    stats = (
        target.groupby("graph_family")["delta_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
    stats = stats.set_index("graph_family").reindex(families).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        stats["graph_family"],
        stats["mean"],
        yerr=stats["se"],
        capsize=4,
        color="slategray",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("ΔF1 vs clean (α=0.4, λ2=0.5)")
    ax.set_xlabel("graph family")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "corruption_delta_summary.png", dpi=200)


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[
        (df["method"] == "Proposed")
        & (df["output_dir"].str.contains("outputs/corruption"))
    ].copy()
    families = ["chain", "grid", "sbm", "knn"]
    delta = _compute_delta(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _phase_panels(delta, families)
    _summary_plot(delta, families)


if __name__ == "__main__":
    main()
