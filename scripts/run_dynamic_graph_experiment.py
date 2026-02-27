"""Dynamic-graph robustness experiment with time-varying feature graphs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiments.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    build_baselines,
    generate_graph_structured_data,
    run_benchmark_once,
)
from src.utils.graph import er_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--support-size", type=int, default=20)
    parser.add_argument("--lambda1", type=float, default=0.15)
    parser.add_argument("--lambda2", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch", "torch-geoopt"])
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument("--torch-dtype", type=str, default="float64", choices=["float32", "float64"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = SyntheticBenchmarkConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        support_size=args.support_size,
        graph_type="chain",
        random_state=args.seed,
    )
    sample = generate_graph_structured_data(base_cfg, random_state=args.seed)
    X = sample["X"]
    w_true = sample["w_true"]

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
        k: methods[k]
        for k in ["Graph-PCA", "NetSPCA-PG", "NetSPCA-MASPG-CAR", "NetSPCA-ProxQN"]
    }

    rows: list[dict[str, object]] = []
    for step in range(args.n_steps):
        # Time-varying ER connectivity as a simple dynamic-graph proxy.
        p_edge = 0.02 + 0.06 * step / max(args.n_steps - 1, 1)
        graph_t = er_graph(args.n_features, p_edge=p_edge, random_state=args.seed + step)
        rec = run_benchmark_once(
            X,
            graph=graph_t,
            w_true=w_true,
            methods=methods,
            support_threshold=base_cfg.support_threshold,
            random_state=args.seed + step,
        )
        for row in rec:
            row["time_step"] = step
            row["dynamic_p_edge"] = p_edge
        rows.extend(rec)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["method", "time_step", "dynamic_p_edge"], as_index=False)
        .agg(
            f1=("f1", "mean"),
            f1_topk=("f1_topk", "mean"),
            explained_variance=("explained_variance", "mean"),
            laplacian_energy=("laplacian_energy", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(["method", "time_step"])
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.output_dir) / f"dynamic-graph-{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_steps": args.n_steps,
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "support_size": args.support_size,
                "lambda1": args.lambda1,
                "lambda2": args.lambda2,
                "max_iter": args.max_iter,
                "backend": args.backend,
                "torch_device": args.torch_device,
                "torch_dtype": args.torch_dtype,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    df.to_csv(outdir / "records.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    print(f"Saved dynamic graph artifacts to: {outdir}")


if __name__ == "__main__":
    main()
