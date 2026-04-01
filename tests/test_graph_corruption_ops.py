import numpy as np

from grpca_gd.synthetic.corruption import delete_edges, perturb_weights, rewire_edges


def test_delete_edges_reduces_edges() -> None:
    W = np.ones((5, 5)) - np.eye(5)
    W2 = delete_edges(W, frac=0.2, rng=np.random.default_rng(0))
    assert W2.sum() < W.sum()


def test_rewire_preserves_edge_count() -> None:
    W = np.zeros((6, 6))
    W[0, 1] = W[1, 0] = 1
    W[2, 3] = W[3, 2] = 1
    W2 = rewire_edges(W, frac=0.5, rng=np.random.default_rng(1))
    assert np.isclose(W2.sum(), W.sum())


def test_perturb_weights_changes_values() -> None:
    W = np.ones((4, 4)) - np.eye(4)
    W2 = perturb_weights(W, scale=0.1, rng=np.random.default_rng(2))
    assert not np.allclose(W, W2)
