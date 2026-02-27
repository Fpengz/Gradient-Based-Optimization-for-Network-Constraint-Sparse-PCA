"""Run reproducible synthetic SPCA method comparisons."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiments.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    build_baselines,
    run_repeated_benchmark,
    summarize_records,
)
from src.experiments.real_benchmark import (
    RealBenchmarkConfig,
    run_real_benchmark,
    summarize_real_records,
)


def _load_config(path: str | None) -> SyntheticBenchmarkConfig:
    if path is None:
        return SyntheticBenchmarkConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SyntheticBenchmarkConfig(**payload)


def _to_latex_table(df: pd.DataFrame) -> str:
    cols = [
        "method",
        "explained_variance_mean",
        "f1_mean",
        "lcc_ratio_mean",
        "runtime_sec_mean",
    ]
    view = df[cols].copy()
    view.columns = ["Method", "Expl.Var", "F1", "LCC Ratio", "Runtime(s)"]
    return view.to_latex(index=False, float_format=lambda x: f"{x:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config matching SyntheticBenchmarkConfig fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root output directory.",
    )
    parser.add_argument("--n-repeats", type=int, default=3, help="Independent repeats.")
    parser.add_argument(
        "--n-components",
        type=int,
        default=1,
        help="Number of components for each method.",
    )
    parser.add_argument(
        "--lambda1", type=float, default=0.15, help="L1 sparsity level."
    )
    parser.add_argument(
        "--lambda2", type=float, default=0.25, help="Graph smoothness level."
    )
    parser.add_argument("--max-iter", type=int, default=400, help="Max iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--graph-misspec-rate",
        type=float,
        default=None,
        help="Optional synthetic-graph perturbation rate in [0, 1].",
    )
    parser.add_argument(
        "--include-stiefel-manifold",
        action="store_true",
        help="Include NetSPCA manifold multi-component baseline.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "colon", "pitprop"],
        help="Benchmark dataset mode.",
    )
    parser.add_argument(
        "--graph-type-real",
        type=str,
        default="chain",
        choices=["chain", "knn"],
        help="Feature-graph construction used for real datasets.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=8,
        help="k for kNN feature graph when --graph-type-real knn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "synthetic":
        cfg = _load_config(args.config)
        cfg.random_state = args.seed
        cfg.n_components = args.n_components
        if args.graph_misspec_rate is not None:
            cfg.graph_misspec_rate = float(args.graph_misspec_rate)
        methods = build_baselines(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            max_iter=args.max_iter,
            random_state=args.seed,
            n_components=args.n_components,
            include_stiefel_manifold=args.include_stiefel_manifold,
        )
        records = run_repeated_benchmark(
            cfg=cfg,
            methods=methods,
            n_repeats=args.n_repeats,
            base_seed=args.seed,
        )
        summary = summarize_records(records)
        cfg_dict = cfg.to_dict()
    else:
        cfg_real = RealBenchmarkConfig(
            dataset=args.dataset,
            n_components=args.n_components,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            max_iter=args.max_iter,
            random_state=args.seed,
            graph_type=args.graph_type_real,
            knn_k=args.knn_k,
        )
        records = run_real_benchmark(cfg_real)
        summary = summarize_real_records(records)
        cfg_dict = cfg_real.to_dict()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = (
        "synth-comparison"
        if args.dataset == "synthetic"
        else f"{args.dataset}-comparison"
    )
    outdir = Path(args.output_dir) / f"{prefix}-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )
    (outdir / "records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    df_records = pd.DataFrame(records)
    df_summary = pd.DataFrame(summary)
    df_records.to_csv(outdir / "records.csv", index=False)
    df_summary.to_csv(outdir / "summary.csv", index=False)
    if args.dataset == "synthetic" and not df_summary.empty:
        (outdir / "summary_table.tex").write_text(
            _to_latex_table(df_summary), encoding="utf-8"
        )

    print(f"Saved experiment artifacts to: {outdir}")
    if summary:
        best = summary[0]
        if args.dataset == "synthetic":
            print(
                "Top method by F1: "
                f"{best['method']} (F1={best['f1_mean']:.4f}, "
                f"LCC={best['lcc_ratio_mean']:.4f})"
            )
        else:
            print(
                "Top method by explained variance: "
                f"{best['method']} (EV={best['explained_variance']:.4f})"
            )


if __name__ == "__main__":
    main()
