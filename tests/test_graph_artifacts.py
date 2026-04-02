import numpy as np

from topospca.synthetic.graphs import chain_graph_artifact, sbm_graph_laplacian


def test_graph_artifact_shapes():
    artifact = chain_graph_artifact(10)
    assert artifact["adjacency"].shape == (10, 10)
    assert artifact["laplacian"].shape == (10, 10)
    assert artifact["family"] == "chain"
    assert "metadata" in artifact


def test_graph_artifact_symmetry():
    artifact = chain_graph_artifact(10)
    A = artifact["adjacency"]
    assert np.allclose(A, A.T)


def test_sbm_graph_laplacian_labels():
    rng = np.random.default_rng(0)
    L, W, labels = sbm_graph_laplacian(12, 3, 0.4, 0.1, rng)
    assert L.shape == (12, 12)
    assert W.shape == (12, 12)
    assert labels.shape == (12,)
