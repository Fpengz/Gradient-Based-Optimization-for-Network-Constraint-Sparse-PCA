import numpy as np

from grpca_gd.metrics import match_components


def test_matching_sign_and_perm():
    rng = np.random.default_rng(1)
    A_true = rng.normal(size=(6, 3))
    A_true, _ = np.linalg.qr(A_true)

    perm = [1, 2, 0]
    signs = np.array([1.0, -1.0, 1.0])
    A_est = A_true[:, perm] * signs

    matched_perm, matched_signs = match_components(A_est, A_true)
    assert np.array_equal(matched_perm, np.array(perm))
    assert np.array_equal(np.sign(matched_signs), np.sign(signs))
