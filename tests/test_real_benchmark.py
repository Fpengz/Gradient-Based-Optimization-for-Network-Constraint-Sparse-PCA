from src.experiments.real_benchmark import (
    RealBenchmarkConfig,
    build_feature_graph,
    load_real_dataset,
    run_real_benchmark,
)


def test_load_pitprop_dataset_shape():
    X = load_real_dataset("pitprop")
    assert X.ndim == 2
    assert X.shape[0] == 13
    assert X.shape[1] == 13


def test_build_feature_graph_knn():
    X = load_real_dataset("pitprop")
    graph = build_feature_graph(X, graph_type="knn", knn_k=3)
    assert graph.adjacency.shape == (13, 13)
    assert graph.laplacian.shape == (13, 13)


def test_run_real_benchmark_pitprop():
    cfg = RealBenchmarkConfig(
        dataset="pitprop",
        max_iter=20,
        lambda1=0.05,
        lambda2=0.05,
        graph_type="chain",
        random_state=0,
    )
    records = run_real_benchmark(cfg)
    assert len(records) >= 4
    methods = {r["method"] for r in records}
    assert "NetSPCA-PG" in methods
    assert "PCA" in methods
