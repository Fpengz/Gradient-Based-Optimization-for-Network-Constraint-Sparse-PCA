import numpy as np

from src.models import (
    GeneralizedPowerMethod,
    NetworkSparsePCA,
    NetworkSparsePCA_ProxQN,
    NetworkSparsePCA_StiefelManifold,
    NetworkSparsePCA_MASPG_CAR,
    PCAEstimator,
    SparsePCA_L1_ProxGrad,
    TorchNetworkSparsePCA,
    TorchNetworkSparsePCA_GeooptStiefel,
    ZouSparsePCA,
)
from src.utils.graph import chain_graph


def _toy_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 10))
    X[:, 0] += 2.0 * X[:, 1]
    return X


def _assert_common_attrs(model):
    assert hasattr(model, "components_")
    assert hasattr(model, "history_")
    assert hasattr(model, "converged_")
    assert hasattr(model, "n_iter_")
    assert hasattr(model, "objective_")


def test_pca_estimator_api():
    model = PCAEstimator(n_components=2).fit(_toy_data())
    _assert_common_attrs(model)
    assert model.components_.shape == (2, 10)


def test_l1_spca_proxgrad_api():
    model = SparsePCA_L1_ProxGrad(n_components=1, max_iter=50, tol=1e-4).fit(
        _toy_data()
    )
    _assert_common_attrs(model)
    assert model.components_.shape == (1, 10)


def test_zou_sparse_pca_api():
    model = ZouSparsePCA(n_components=2, max_iter=3).fit(_toy_data())
    _assert_common_attrs(model)
    assert model.components_.shape[0] == 2


def test_gpower_api():
    model = GeneralizedPowerMethod(n_components=1, max_iter=20).fit(_toy_data())
    _assert_common_attrs(model)
    assert model.components_.shape == (1, 10)


def test_network_sparse_pca_pg_and_maspg_api():
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    pg = NetworkSparsePCA(n_components=1, max_iter=30, tol=1e-4, random_state=0).fit(
        X, graph=graph
    )
    maspg = NetworkSparsePCA_MASPG_CAR(
        n_components=1, max_iter=30, tol=1e-4, random_state=0
    ).fit(X, graph=graph)
    for model in (pg, maspg):
        _assert_common_attrs(model)
        assert model.components_.shape == (1, 10)
        hist = model.history_
        assert "pg_residual_history_by_component" in hist
        assert len(hist["pg_residual_history_by_component"]) == 1


def test_network_sparse_pca_proxqn_api():
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    proxqn = NetworkSparsePCA_ProxQN(
        n_components=1, max_iter=30, tol=1e-4, random_state=0
    ).fit(X, graph=graph)
    _assert_common_attrs(proxqn)
    assert proxqn.components_.shape == (1, 10)
    hist = proxqn.history_
    assert "pg_residual_history_by_component" in hist
    assert len(hist["pg_residual_history_by_component"]) == 1


def test_network_sparse_pca_handles_disconnected_graph():
    X = _toy_data()
    # Zero Laplacian (all nodes disconnected) should not crash auto-step estimation.
    import scipy.sparse as sp

    L0 = sp.csr_matrix((X.shape[1], X.shape[1]))
    model = NetworkSparsePCA(
        n_components=1, max_iter=20, tol=1e-4, random_state=0, learning_rate="auto"
    ).fit(X, L=L0)
    _assert_common_attrs(model)
    assert model.components_.shape == (1, 10)


def test_network_sparse_pca_stiefel_multi_component_api():
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    model = NetworkSparsePCA_StiefelManifold(
        n_components=3,
        max_iter=40,
        tol=1e-4,
        random_state=0,
    ).fit(X, graph=graph)
    _assert_common_attrs(model)
    assert model.components_.shape == (3, 10)
    gram = model.components_ @ model.components_.T
    assert np.allclose(gram, np.eye(3), atol=5e-3)


def test_network_sparse_pca_fit_path_returns_warm_started_models():
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    base = NetworkSparsePCA(
        n_components=1,
        max_iter=30,
        tol=1e-4,
        random_state=0,
    )
    path = base.fit_path(X, graph=graph, lambda1_grid=[0.1, 0.2], lambda2_grid=[0.0, 0.3])
    assert len(path) == 4
    assert all("lambda1" in row and "lambda2" in row for row in path)
    assert all("model" in row for row in path)
    assert all(path[i]["warm_started"] for i in range(1, len(path)))


def test_network_sparse_pca_fit_path_serpentine_ordering():
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    base = NetworkSparsePCA(
        n_components=1,
        max_iter=20,
        tol=1e-4,
        random_state=0,
    )
    path = base.fit_path(
        X,
        graph=graph,
        lambda1_grid=[0.3, 0.1],
        lambda2_grid=[0.4, 0.0, 0.2],
    )
    pairs = [(row["lambda1"], row["lambda2"]) for row in path]
    assert pairs == [
        (0.1, 0.0),
        (0.1, 0.2),
        (0.1, 0.4),
        (0.3, 0.4),
        (0.3, 0.2),
        (0.3, 0.0),
    ]


def test_torch_network_sparse_pca_api_if_torch_available():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    model = TorchNetworkSparsePCA(
        n_components=1,
        max_iter=25,
        tol=1e-4,
        random_state=0,
        backend="pg",
    ).fit(X, graph=graph)
    _assert_common_attrs(model)
    assert model.components_.shape == (1, 10)
    assert "pg_residual_history_by_component" in model.history_


def test_torch_geoopt_stiefel_api_if_geoopt_available():
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    pytest.importorskip("geoopt")
    X = _toy_data()
    graph = chain_graph(X.shape[1])
    model = TorchNetworkSparsePCA_GeooptStiefel(
        n_components=2,
        max_iter=25,
        tol=1e-4,
        random_state=0,
    ).fit(X, graph=graph)
    _assert_common_attrs(model)
    assert model.components_.shape == (2, 10)
    gram = model.components_ @ model.components_.T
    assert np.allclose(gram, np.eye(2), atol=1e-2)
