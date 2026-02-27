import numpy as np

from src.experiments.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    build_baselines,
    generate_graph_structured_data,
    run_benchmark_once,
    run_repeated_benchmark,
    summarize_records,
)


def test_generate_graph_structured_data_shapes():
    cfg = SyntheticBenchmarkConfig(
        n_samples=32,
        n_features=25,
        support_size=6,
        graph_type="grid",
        random_state=7,
    )
    sample = generate_graph_structured_data(cfg)
    assert sample["X"].shape == (32, 25)
    assert sample["w_true"].shape == (25,)
    assert np.count_nonzero(sample["w_true"]) == 6
    assert sample["graph"].laplacian.shape == (25, 25)


def test_run_benchmark_once_returns_expected_metrics():
    cfg = SyntheticBenchmarkConfig(
        n_samples=40, n_features=20, support_size=5, random_state=1
    )
    sample = generate_graph_structured_data(cfg)
    methods = {
        "PCA": build_baselines(lambda1=0.1, lambda2=0.2, max_iter=40, random_state=0)[
            "PCA"
        ],
        "NetSPCA-PG": build_baselines(
            lambda1=0.1, lambda2=0.2, max_iter=40, random_state=0
        )["NetSPCA-PG"],
    }
    records = run_benchmark_once(
        sample["X"], graph=sample["graph"], w_true=sample["w_true"], methods=methods
    )
    assert len(records) == 2
    for row in records:
        assert "method" in row
        assert "f1" in row
        assert "lcc_ratio" in row
        assert row["support_size"] >= 0


def test_repeated_benchmark_summary_non_empty():
    cfg = SyntheticBenchmarkConfig(
        n_samples=30, n_features=18, support_size=4, random_state=3
    )
    methods = {
        "PCA": build_baselines(max_iter=30)["PCA"],
        "L1-SPCA-ProxGrad": build_baselines(max_iter=30)["L1-SPCA-ProxGrad"],
    }
    records = run_repeated_benchmark(
        cfg=cfg, methods=methods, n_repeats=2, base_seed=11
    )
    summary = summarize_records(records)
    assert len(records) == 4
    assert len(summary) == 2
    assert all("f1_mean" in row for row in summary)
