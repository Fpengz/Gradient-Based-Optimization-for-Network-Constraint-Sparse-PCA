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
from src.experiments.stats import paired_significance
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


def _compute_significance(
    records: list[dict[str, object]],
    reference_method: str = "NetSPCA-PG",
    metrics: tuple[str, ...] = ("f1", "f1_topk", "explained_variance", "runtime_sec"),
) -> pd.DataFrame:
    """Paired significance by seed/repeat versus a reference method."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "method" not in df.columns:
        return pd.DataFrame()
    if reference_method not in set(df["method"].astype(str)):
        return pd.DataFrame()
    key_cols = [c for c in ("seed", "repeat", "graph_type") if c in df.columns]
    ref = df[df["method"] == reference_method]
    rows: list[dict[str, object]] = []
    for method in sorted(set(df["method"].astype(str))):
        if method == reference_method:
            continue
        cur = df[df["method"] == method]
        merged = ref.merge(cur, on=key_cols, suffixes=("_ref", "_cur"))
        if merged.empty:
            continue
        for metric in metrics:
            ref_col = f"{metric}_ref"
            cur_col = f"{metric}_cur"
            if ref_col not in merged.columns or cur_col not in merged.columns:
                continue
            res = paired_significance(
                values_a=merged[cur_col].astype(float).values,
                values_b=merged[ref_col].astype(float).values,
                metric=metric,
                method_a=method,
                method_b=reference_method,
            )
            rows.append(
                {
                    "metric": res.metric,
                    "method_a": res.method_a,
                    "method_b": res.method_b,
                    "n_pairs": res.n_pairs,
                    "mean_diff": res.mean_diff,
                    "p_value": res.p_value,
                    "test": res.test,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                }
            )
    return pd.DataFrame(rows)


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
        help="Torch device when --backend is torch or torch-geoopt.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Torch dtype when --backend is torch or torch-geoopt.",
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
            backend=args.backend,
            torch_device=args.torch_device,
            torch_dtype=args.torch_dtype,
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
            backend=args.backend,
            torch_device=args.torch_device,
            torch_dtype=args.torch_dtype,
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
    if args.dataset == "synthetic":
        df_sig = _compute_significance(records)
        if not df_sig.empty:
            df_sig.to_csv(outdir / "significance.csv", index=False)
            (outdir / "significance.json").write_text(
                df_sig.to_json(orient="records", indent=2), encoding="utf-8"
            )
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
