import numpy as np

from src.utils.graph import (
    adjacency_to_laplacian,
    chain_graph,
    grid_graph,
    random_geometric_graph,
    sbm_graph,
)


def test_chain_graph_laplacian_is_symmetric_psd():
    graph = chain_graph(8)
    L = graph.laplacian.toarray()
    assert np.allclose(L, L.T)
    eigvals = np.linalg.eigvalsh(L)
    assert eigvals.min() >= -1e-10


def test_grid_graph_node_count():
    graph = grid_graph(3, 4)
    assert graph.adjacency.shape == (12, 12)
    assert graph.laplacian.shape == (12, 12)


def test_symmetric_normalized_laplacian_shape():
    graph = sbm_graph(
        [4, 4], p_in=0.8, p_out=0.1, random_state=0, laplacian_type="sym_norm"
    )
    L = adjacency_to_laplacian(graph.adjacency, laplacian_type="sym_norm").toarray()
    assert L.shape == (8, 8)
    assert np.allclose(L, L.T, atol=1e-10)


def test_random_geometric_graph_properties():
    graph = random_geometric_graph(
        n_nodes=20, radius=0.35, random_state=2, laplacian_type="unnormalized"
    )
    A = graph.adjacency.toarray()
    L = graph.laplacian.toarray()
    assert A.shape == (20, 20)
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)
    assert np.allclose(L, L.T)
