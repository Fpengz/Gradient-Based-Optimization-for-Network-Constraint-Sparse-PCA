"""Run lambda sweeps for NetSPCA and export paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.experiments.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    build_baselines,
    run_repeated_benchmark,
)


def _parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def _to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df[
        df["method"].isin(
            ["NetSPCA-PG", "NetSPCA-MASPG-CAR", "NetSPCA-ProxQN"]
        )
    ].copy()


def _plot_sweep(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return
    agg = (
        df.groupby(["method", "lambda1", "lambda2"], as_index=False)
        .agg(
            explained_variance=("explained_variance", "mean"),
            f1=("f1", "mean"),
            lcc_ratio=("lcc_ratio", "mean"),
            support_size=("support_size", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(["method", "lambda1", "lambda2"])
    )

    plt.figure(figsize=(7.2, 4.8))
    for method, sub in agg.groupby("method"):
        plt.plot(
            sub["support_size"],
            sub["explained_variance"],
            marker="o",
            label=method,
        )
    plt.xlabel("Estimated support size")
    plt.ylabel("Explained variance")
    plt.title("Variance vs Sparsity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "variance_vs_sparsity.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for method, sub in agg.groupby("method"):
        by_l2 = sub.groupby("lambda2", as_index=False)["lcc_ratio"].mean()
        plt.plot(by_l2["lambda2"], by_l2["lcc_ratio"], marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("lambda2 (graph regularization)")
    plt.ylabel("LCC ratio")
    plt.title("Connectivity vs Graph Regularization")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "connectivity_vs_lambda2.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--lambda1-grid", type=str, default="0.01,0.05,0.1,0.2,0.5")
    parser.add_argument("--lambda2-grid", type=str, default="0.0,0.01,0.05,0.1,0.5,1.0")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config matching SyntheticBenchmarkConfig.",
    )
    parser.add_argument(
        "--graph-misspec-rate",
        type=float,
        default=None,
        help="Optional synthetic-graph perturbation rate in [0, 1].",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch", "torch-geoopt"],
        help="Backend for graph-constrained methods.",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="cpu",
        help="Torch device when using torch backends.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Torch dtype when using torch backends.",
    )
    return parser.parse_args()


def _load_config(path: str | None, seed: int) -> SyntheticBenchmarkConfig:
    if path is None:
        cfg = SyntheticBenchmarkConfig()
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = SyntheticBenchmarkConfig(**payload)
    cfg.random_state = seed
    return cfg


def main() -> None:
    args = parse_args()
    lambda1_grid = _parse_float_list(args.lambda1_grid)
    lambda2_grid = _parse_float_list(args.lambda2_grid)
    cfg = _load_config(args.config, args.seed)
    cfg.n_components = args.n_components
    if args.graph_misspec_rate is not None:
        cfg.graph_misspec_rate = float(args.graph_misspec_rate)

    all_records: list[dict[str, Any]] = []
    for lambda1 in lambda1_grid:
        for lambda2 in lambda2_grid:
            methods = build_baselines(
                lambda1=lambda1,
                lambda2=lambda2,
                max_iter=args.max_iter,
                random_state=args.seed,
                n_components=args.n_components,
                backend=args.backend,
                torch_device=args.torch_device,
                torch_dtype=args.torch_dtype,
            )
            methods = {
                "NetSPCA-PG": methods["NetSPCA-PG"],
                "NetSPCA-MASPG-CAR": methods["NetSPCA-MASPG-CAR"],
                "NetSPCA-ProxQN": methods["NetSPCA-ProxQN"],
            }
            records = run_repeated_benchmark(
                cfg=cfg,
                methods=methods,
                n_repeats=args.n_repeats,
                base_seed=args.seed,
            )
            for row in records:
                row["lambda1"] = lambda1
                row["lambda2"] = lambda2
            all_records.extend(records)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.output_dir) / f"synth-sweep-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2), encoding="utf-8"
    )

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(outdir / "records.csv", index=False)
    df_net = _to_df(all_records)
    df_net.to_csv(outdir / "netspca_records.csv", index=False)

    if not df_net.empty:
        grouped = (
            df_net.groupby(["method", "lambda1", "lambda2"], as_index=False)
            .agg(
                explained_variance=("explained_variance", "mean"),
                f1=("f1", "mean"),
                lcc_ratio=("lcc_ratio", "mean"),
                runtime_sec=("runtime_sec", "mean"),
            )
            .sort_values(["method", "f1"], ascending=[True, False])
        )
        grouped.to_csv(outdir / "netspca_summary.csv", index=False)
        grouped.to_latex(
            outdir / "netspca_summary.tex",
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    _plot_sweep(df_net, outdir)
    print(f"Saved sweep artifacts to: {outdir}")


if __name__ == "__main__":
    main()
