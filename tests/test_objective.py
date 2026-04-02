import numpy as np

from topospca.objective import objective_terms


def test_objective_terms_sum():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 2))
    B = rng.normal(size=(5, 2))
    sigma_hat = rng.normal(size=(5, 5))
    sigma_hat = sigma_hat.T @ sigma_hat
    L = np.eye(5)
    terms = objective_terms(A, B, sigma_hat, L, lambda1=0.1, lambda2=0.2, rho=0.3)
    total = (
        terms["negative_variance_term"]
        + terms["sparsity_penalty"]
        + terms["graph_penalty"]
        + terms["coupling_penalty"]
    )
    assert np.isclose(total, terms["total_objective"], atol=1e-8)
