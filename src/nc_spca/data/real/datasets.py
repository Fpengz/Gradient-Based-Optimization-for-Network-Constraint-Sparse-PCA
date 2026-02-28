"""Real dataset loading and feature-graph helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.pitprop import get_pitprop_correlation_matrix

from ...config.schema import DataConfig
from ..synthetic.graphs import GraphData, chain_graph, knn_graph


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _standardize(X: np.ndarray) -> np.ndarray:
    X_array = np.asarray(X, dtype=float)
    centered = X_array - X_array.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return centered / scale


def load_real_dataset(cfg: DataConfig) -> np.ndarray:
    """Load a real dataset configured by name."""

    dataset = cfg.name.lower()
    if dataset == "colon":
        path = _repo_root() / cfg.data_root / "colon_x.csv"
        X = pd.read_csv(path, index_col=0).values
    elif dataset == "pitprop":
        corr = get_pitprop_correlation_matrix()
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 0.0)
        X = np.einsum("ik,k,jk->ij", eigvecs, np.sqrt(eigvals), eigvecs, optimize=True)
    else:
        raise ValueError(f"Unsupported real dataset: {cfg.name!r}")
    if cfg.standardize:
        return _standardize(X)
    return np.asarray(X, dtype=float)


def build_feature_graph(X: np.ndarray, cfg: DataConfig) -> GraphData:
    """Build a feature graph for a real dataset."""

    n_features = X.shape[1]
    if cfg.graph_type == "chain":
        return chain_graph(n_features, laplacian_type=cfg.graph_laplacian_type)
    if cfg.graph_type == "knn":
        return knn_graph(
            X.T,
            n_neighbors=cfg.knn_k,
            mode="distance",
            laplacian_type=cfg.graph_laplacian_type,
        )
    raise ValueError(f"Unsupported real-data graph type: {cfg.graph_type!r}")
