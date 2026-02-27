import numpy as np

from src.experiments.synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    _perturb_graph_for_misspecification,
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


def test_generate_graph_structured_data_rgg():
    cfg = SyntheticBenchmarkConfig(
        n_samples=24,
        n_features=16,
        support_size=4,
        graph_type="rgg",
        graph_rgg_radius=0.4,
        random_state=9,
    )
    sample = generate_graph_structured_data(cfg)
    assert sample["X"].shape == (24, 16)
    assert sample["graph"].adjacency.shape == (16, 16)


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


def test_graph_misspecification_perturbs_adjacency():
    cfg = SyntheticBenchmarkConfig(n_features=25, graph_type="grid", random_state=5)
    sample = generate_graph_structured_data(cfg)
    graph = sample["graph"]
    graph_perturbed = _perturb_graph_for_misspecification(
        graph,
        perturb_rate=0.2,
        random_state=7,
    )
    diff = (graph.adjacency != graph_perturbed.adjacency).nnz
    assert diff > 0


def test_run_benchmark_once_reports_topk_and_misspecification_metrics():
    cfg = SyntheticBenchmarkConfig(
        n_samples=40,
        n_features=20,
        support_size=5,
        graph_type="chain",
        graph_misspec_rate=0.15,
        random_state=3,
    )
    sample = generate_graph_structured_data(cfg)
    methods = {
        "NetSPCA-PG": build_baselines(
            lambda1=0.1, lambda2=0.2, max_iter=40, random_state=0
        )["NetSPCA-PG"],
    }
    records = run_benchmark_once(
        sample["X"],
        graph=sample["graph"],
        w_true=sample["w_true"],
        methods=methods,
        support_threshold=cfg.support_threshold,
        graph_misspec_rate=cfg.graph_misspec_rate,
        random_state=cfg.random_state,
    )
    row = records[0]
    assert "f1_topk" in row
    assert "graph_misspec_rate" in row
    assert "objective_curve" in row
    assert "pg_residual_curve" in row


def test_build_baselines_can_include_stiefel_solver():
    methods = build_baselines(
        lambda1=0.1,
        lambda2=0.2,
        max_iter=20,
        random_state=0,
        n_components=2,
        include_stiefel_manifold=True,
    )
    assert "NetSPCA-Stiefel" in methods


def test_build_baselines_includes_proxqn_numpy():
    methods = build_baselines(
        lambda1=0.1,
        lambda2=0.2,
        max_iter=20,
        random_state=0,
        n_components=1,
        backend="numpy",
    )
    assert "NetSPCA-ProxQN" in methods
    assert type(methods["NetSPCA-ProxQN"]).__name__ == "NetworkSparsePCA_ProxQN"


def test_build_baselines_torch_backend_wiring():
    methods = build_baselines(
        lambda1=0.1,
        lambda2=0.2,
        max_iter=20,
        random_state=0,
        n_components=1,
        backend="torch",
    )
    assert type(methods["NetSPCA-PG"]).__name__ == "TorchNetworkSparsePCA"


def test_build_baselines_torch_geoopt_stiefel_wiring():
    methods = build_baselines(
        lambda1=0.1,
        lambda2=0.2,
        max_iter=20,
        random_state=0,
        n_components=2,
        include_stiefel_manifold=True,
        backend="torch-geoopt",
    )
    assert type(methods["NetSPCA-Stiefel"]).__name__ == "TorchNetworkSparsePCA_GeooptStiefel"


def test_summarize_records_includes_stationarity_fields():
    cfg = SyntheticBenchmarkConfig(
        n_samples=36, n_features=16, support_size=4, random_state=12
    )
    methods = {"NetSPCA-PG": build_baselines(max_iter=25)["NetSPCA-PG"]}
    records = run_repeated_benchmark(cfg=cfg, methods=methods, n_repeats=2, base_seed=12)
    summary = summarize_records(records)
    row = summary[0]
    assert "pg_residual_last_mean" in row
    assert "pg_residual_ratio_mean" in row
    assert "objective_monotone_rate" in row
