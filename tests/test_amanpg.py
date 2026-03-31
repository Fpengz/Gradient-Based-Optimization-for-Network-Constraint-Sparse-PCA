import numpy as np

from grpca_gd.amanpg import AmanpgConfig, solve_amanpg
from grpca_gd.objective import objective_terms


def test_amanpg_returns_orthonormal_columns():
    rng = np.random.default_rng(0)
    p, r = 10, 3
    Sigma_hat = np.eye(p)
    A0 = rng.standard_normal((p, r))
    cfg = AmanpgConfig(lambda1=0.1, eta_A=0.05, max_iters=5, tol_obj=1e-12, tol_orth=1e-8)
    result = solve_amanpg(A0, Sigma_hat, cfg)

    gram = result.A.T @ result.A
    assert np.allclose(gram, np.eye(r), atol=1e-6)


def test_amanpg_objective_terms_sum():
    rng = np.random.default_rng(1)
    p, r = 8, 2
    Sigma_hat = np.eye(p)
    A = rng.standard_normal((p, r))
    terms = objective_terms(A, A, Sigma_hat, np.zeros((p, p)), lambda1=0.1, lambda2=0.0, rho=0.0)
    total = terms["negative_variance_term"] + terms["sparsity_penalty"]
    assert np.isclose(total, terms["total_objective"])
