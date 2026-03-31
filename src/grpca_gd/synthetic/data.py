from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .support import generate_supports


@dataclass
class SyntheticDataset:
    X: np.ndarray
    Sigma_true: np.ndarray
    Sigma_hat: np.ndarray
    L: np.ndarray
    true_loadings: np.ndarray
    true_supports: List[np.ndarray]
    metadata: Dict[str, object]


def build_loadings(
    p: int,
    r: int,
    support_size: int,
    support_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    supports = generate_supports(p, r, support_size, support_type, rng)
    U = np.zeros((p, r), dtype=float)
    for j, idx in enumerate(supports):
        U[idx, j] = rng.normal(size=len(idx))
        norm = np.linalg.norm(U[:, j])
        if norm > 0:
            U[:, j] /= norm
    return U, supports


def build_covariance(
    U: np.ndarray,
    signal_eigs: Optional[List[float]],
    snr: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r = U.shape[1]
    if signal_eigs is None:
        signal_eigs = [float(v) for v in range(r, 0, -1)]
    if len(signal_eigs) != r:
        raise ValueError("signal_eigs length must match r")
    Lambda = np.diag(signal_eigs)
    Sigma_signal = U @ Lambda @ U.T
    signal_power = float(np.trace(Sigma_signal)) / U.shape[0]
    if snr <= 0:
        raise ValueError("snr must be positive")
    sigma2 = signal_power / snr
    Sigma_true = Sigma_signal + sigma2 * np.eye(U.shape[0])
    return Sigma_true, Lambda, sigma2


def sample_data(
    Sigma_true: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    try:
        Lc = np.linalg.cholesky(Sigma_true)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.eye(Sigma_true.shape[0])
        Lc = np.linalg.cholesky(Sigma_true + jitter)
    Z = rng.normal(size=(n, Sigma_true.shape[0]))
    return Z @ Lc.T


def generate_dataset(
    n: int,
    p: int,
    r: int,
    support_size: int,
    support_type: str,
    L: np.ndarray,
    snr: float,
    signal_eigs: Optional[List[float]],
    seed: int,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    U, supports = build_loadings(p, r, support_size, support_type, rng)
    Sigma_true, Lambda, sigma2 = build_covariance(U, signal_eigs, snr)
    X = sample_data(Sigma_true, n, rng)
    Sigma_hat = (X.T @ X) / n
    metadata = {
        "seed": seed,
        "support_size": support_size,
        "support_type": support_type,
        "snr": snr,
        "signal_eigs": list(np.diag(Lambda)),
        "sigma2": sigma2,
    }
    return SyntheticDataset(
        X=X,
        Sigma_true=Sigma_true,
        Sigma_hat=Sigma_hat,
        L=L,
        true_loadings=U,
        true_supports=supports,
        metadata=metadata,
    )
