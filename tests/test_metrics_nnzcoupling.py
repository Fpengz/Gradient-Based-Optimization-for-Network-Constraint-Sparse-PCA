import numpy as np

from grpca_gd.metrics import coupling_gap, nnz_loadings


def test_nnz_loadings_counts() -> None:
    B = np.array([[0.0, 1.0], [2.0, 0.0]])
    assert nnz_loadings(B) == 2


def test_coupling_gap_zero() -> None:
    A = np.eye(2)
    B = np.eye(2)
    assert coupling_gap(A, B) == 0.0
