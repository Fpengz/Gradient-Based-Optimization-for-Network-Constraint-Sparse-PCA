"""Experiment utilities for reproducible SPCA benchmarking."""

from .synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    build_baselines,
    generate_graph_structured_data,
    run_benchmark_once,
    run_repeated_benchmark,
    summarize_records,
)
from .real_benchmark import (
    RealBenchmarkConfig,
    build_feature_graph,
    load_real_dataset,
    run_real_benchmark,
    summarize_real_records,
)
from .stats import SignificanceResult, bootstrap_mean_diff_ci, paired_significance

__all__ = [
    "SyntheticBenchmarkConfig",
    "build_baselines",
    "generate_graph_structured_data",
    "run_benchmark_once",
    "run_repeated_benchmark",
    "summarize_records",
    "RealBenchmarkConfig",
    "load_real_dataset",
    "build_feature_graph",
    "run_real_benchmark",
    "summarize_real_records",
    "SignificanceResult",
    "bootstrap_mean_diff_ci",
    "paired_significance",
]
