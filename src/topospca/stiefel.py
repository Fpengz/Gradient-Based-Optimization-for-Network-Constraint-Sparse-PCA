from __future__ import annotations

import numpy as np


def sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def rgrad(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    return G - A @ sym(A.T @ G)


def qr_retraction(A: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(A)
    diag = np.sign(np.diag(R))
    diag[diag == 0] = 1.0
    return Q * diag
