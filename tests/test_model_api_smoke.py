import numpy as np

from src.models import (
    GeneralizedPowerMethod,
    NetworkSparsePCA,
    NetworkSparsePCA_MASPG_CAR,
    PCAEstimator,
    SparsePCA_L1_ProxGrad,
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
