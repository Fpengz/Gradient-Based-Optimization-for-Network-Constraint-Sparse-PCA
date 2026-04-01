import numpy as np

from grpca_gd.synthetic.graphs import (
    er_graph_laplacian,
    grid_graph_laplacian,
    knn_graph_laplacian,
    normalized_laplacian,
    small_world_laplacian,
)


def test_grid_graph_shape() -> None:
    L, W = grid_graph_laplacian(rows=3, cols=4)
    assert L.shape == (12, 12)
    assert W.shape == (12, 12)


def test_er_graph_shape() -> None:
    L, W = er_graph_laplacian(p=10, p_edge=0.2, rng=np.random.default_rng(0))
    assert L.shape == (10, 10)
    assert W.shape == (10, 10)


def test_knn_graph_shape() -> None:
    points = np.random.default_rng(1).normal(size=(8, 2))
    L, W = knn_graph_laplacian(points, k=3)
    assert L.shape == (8, 8)
    assert W.shape == (8, 8)


def test_small_world_shape() -> None:
    L, W = small_world_laplacian(p=12, k=2, beta=0.2, rng=np.random.default_rng(2))
    assert L.shape == (12, 12)
    assert W.shape == (12, 12)


def test_normalized_laplacian_diagonal() -> None:
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    L = normalized_laplacian(W)
    assert np.allclose(np.diag(L), np.ones(2))
