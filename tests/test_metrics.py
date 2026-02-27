import numpy as np

from src.utils.graph import chain_graph
from src.utils.metrics import (
    connected_support_lcc_ratio,
    explained_variance,
    laplacian_energy,
    support_metrics,
)


def test_support_metrics_basic():
    metrics = support_metrics([0, 1, 2], [1, 2, 3])
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["recall"], 2 / 3)
    assert np.isclose(metrics["f1"], 2 / 3)


def test_laplacian_energy_nonnegative_on_chain():
    graph = chain_graph(5)
    x = np.array([1.0, 0.5, 0.5, 0.0, 0.0])
    assert laplacian_energy(x, graph.laplacian) >= -1e-10


def test_connected_support_lcc_ratio():
    graph = chain_graph(6)
    ratio_connected = connected_support_lcc_ratio([1, 2, 3], graph.adjacency)
    ratio_split = connected_support_lcc_ratio([0, 2, 5], graph.adjacency)
    assert np.isclose(ratio_connected, 1.0)
    assert ratio_split < 1.0


def test_explained_variance_scalar():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    w = np.array([1.0, 0.0])
    val = explained_variance(X, w, centered=True)
    assert val >= 0.0
