"""Run large-scale stress benchmarks (p >= 2000) and export summaries."""

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


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty integer list.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument(
        "--n-features-grid",
        type=str,
        default="2000,3000",
        help="Comma-separated feature dimensions for stress tests.",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default="chain",
        choices=["chain", "grid", "er", "rgg", "sbm"],
    )
    parser.add_argument("--lambda1", type=float, default=0.15)
    parser.add_argument("--lambda2", type=float, default=0.25)
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch", "torch-geoopt"],
    )
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_grid = _parse_int_list(args.n_features_grid)
    all_records: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []

    for p in p_grid:
        support_size = max(20, p // 20)
        cfg = SyntheticBenchmarkConfig(
            n_samples=args.n_samples,
            n_features=p,
            support_size=support_size,
            graph_type=args.graph_type,
            random_state=args.seed,
        )
        methods = build_baselines(
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            max_iter=args.max_iter,
            random_state=args.seed,
            backend=args.backend,
            torch_device=args.torch_device,
            torch_dtype=args.torch_dtype,
        )
        methods = {
            name: methods[name]
            for name in [
                "PCA",
                "L1-SPCA-ProxGrad",
                "Graph-PCA",
                "NetSPCA-PG",
                "NetSPCA-MASPG-CAR",
                "NetSPCA-ProxQN",
            ]
        }
        records = run_repeated_benchmark(
            cfg=cfg, methods=methods, n_repeats=args.n_repeats, base_seed=args.seed
        )
        for row in records:
            row["stress_n_features"] = p
            row["stress_graph_type"] = args.graph_type
        all_records.extend(records)

        summary = summarize_records(records)
        for row in summary:
            row["stress_n_features"] = p
            row["stress_graph_type"] = args.graph_type
        all_summary.extend(summary)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.output_dir) / f"large-scale-stress-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "seed": args.seed,
        "n_repeats": args.n_repeats,
        "max_iter": args.max_iter,
        "n_samples": args.n_samples,
        "n_features_grid": p_grid,
        "graph_type": args.graph_type,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "backend": args.backend,
        "torch_device": args.torch_device,
        "torch_dtype": args.torch_dtype,
    }
    (outdir / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    (outdir / "records.json").write_text(
        json.dumps(all_records, indent=2), encoding="utf-8"
    )

    df_records = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(all_summary)
    df_records.to_csv(outdir / "records.csv", index=False)
    df_summary.to_csv(outdir / "summary.csv", index=False)

    if not df_summary.empty:
        view = df_summary[
            [
                "stress_n_features",
                "method",
                "f1_mean",
                "explained_variance_mean",
                "runtime_sec_mean",
                "pg_residual_last_mean",
                "objective_monotone_rate",
            ]
        ].copy()
        (outdir / "summary_table.md").write_text(
            _to_markdown_table(view), encoding="utf-8"
        )

    print(f"Saved large-scale stress artifacts to: {outdir}")


if __name__ == "__main__":
    main()
