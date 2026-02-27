"""Compare NumPy vs Torch backends on aligned synthetic benchmarks."""

from __future__ import annotations

import argparse
import importlib
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


def _available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--graph-type", type=str, default="chain")
    parser.add_argument("--lambda1", type=float, default=0.15)
    parser.add_argument("--lambda2", type=float, default=0.25)
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument(
        "--torch-dtype", type=str, default="float64", choices=["float32", "float64"]
    )
    return parser.parse_args()


def _run_backend(
    backend: str,
    cfg: SyntheticBenchmarkConfig,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    methods = build_baselines(
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_iter=args.max_iter,
        random_state=args.seed,
        backend=backend,
        torch_device=args.torch_device,
        torch_dtype=args.torch_dtype,
    )
    methods = {
        name: methods[name]
        for name in ["Graph-PCA", "NetSPCA-PG", "NetSPCA-MASPG-CAR"]
    }
    records = run_repeated_benchmark(
        cfg=cfg, methods=methods, n_repeats=args.n_repeats, base_seed=args.seed
    )
    for row in records:
        row["backend"] = backend
    return records


def _significance(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["f1", "f1_topk", "explained_variance", "runtime_sec"]
    key_cols = ["seed", "repeat", "graph_type", "method"]
    backends = sorted(df["backend"].unique())
    if "numpy" not in backends:
        return pd.DataFrame()
    base = df[df["backend"] == "numpy"]
    for backend in backends:
        if backend == "numpy":
            continue
        cur = df[df["backend"] == backend]
        merged = base.merge(cur, on=key_cols, suffixes=("_np", "_b"))
        if merged.empty:
            continue
        for method in sorted(merged["method"].unique()):
            sub = merged[merged["method"] == method]
            for metric in metrics:
                res = paired_significance(
                    values_a=sub[f"{metric}_b"].astype(float).values,
                    values_b=sub[f"{metric}_np"].astype(float).values,
                    metric=f"{method}:{metric}",
                    method_a=backend,
                    method_b="numpy",
                )
                rows.append(
                    {
                        "metric": res.metric,
                        "backend_a": backend,
                        "backend_b": "numpy",
                        "n_pairs": res.n_pairs,
                        "mean_diff": res.mean_diff,
                        "p_value": res.p_value,
                        "test": res.test,
                        "ci_low": res.ci_low,
                        "ci_high": res.ci_high,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = SyntheticBenchmarkConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        support_size=max(20, args.n_features // 10),
        graph_type=args.graph_type,
        random_state=args.seed,
    )
    backends = ["numpy"]
    if _available("torch"):
        backends.append("torch")
    if _available("torch") and _available("geoopt"):
        backends.append("torch-geoopt")

    all_records: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for backend in backends:
        rec = _run_backend(backend, cfg, args)
        summ = summarize_records(rec)
        for row in summ:
            row["backend"] = backend
        all_records.extend(rec)
        all_summary.extend(summ)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.output_dir) / f"backend-comparison-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_repeats": args.n_repeats,
                "max_iter": args.max_iter,
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "graph_type": args.graph_type,
                "lambda1": args.lambda1,
                "lambda2": args.lambda2,
                "torch_device": args.torch_device,
                "torch_dtype": args.torch_dtype,
                "backends_run": backends,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    df_records = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(all_summary)
    df_records.to_csv(outdir / "records.csv", index=False)
    df_summary.to_csv(outdir / "summary.csv", index=False)
    sig = _significance(df_records)
    if not sig.empty:
        sig.to_csv(outdir / "significance.csv", index=False)
        (outdir / "significance.json").write_text(
            sig.to_json(orient="records", indent=2), encoding="utf-8"
        )

    print(f"Saved backend comparison artifacts to: {outdir}")
    print("Backends executed:", ", ".join(backends))


if __name__ == "__main__":
    main()
