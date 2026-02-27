"""Real-data benchmark helpers for NC-SPCA comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from data.pitprop import get_pitprop_correlation_matrix
from src.experiments.synthetic_benchmark import (
    _estimated_support,
    _sanitize_component,
    build_baselines,
)
from src.models.network_sparse_pca import NetworkSparsePCA
from src.utils.graph import GraphData, chain_graph, knn_graph
from src.utils.metrics import explained_variance, laplacian_energy


@dataclass
class RealBenchmarkConfig:
    dataset: str = "colon"
    n_components: int = 1
    lambda1: float = 0.15
    lambda2: float = 0.25
    max_iter: int = 400
    random_state: int = 42
    graph_type: str = "chain"
    knn_k: int = 8
    support_threshold: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return X / scale


def load_real_dataset(name: str) -> np.ndarray:
    dataset = name.lower()
    if dataset == "colon":
        X = pd.read_csv("data/colon_x.csv", index_col=0).values
        return _standardize(X)
    if dataset == "pitprop":
        corr = get_pitprop_correlation_matrix()
        vals, vecs = np.linalg.eigh(corr)
        vals = np.maximum(vals, 0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scales = np.sqrt(vals)
            X_pseudo = np.einsum("ik,k,jk->ij", vecs, scales, vecs, optimize=True)
        X_pseudo = np.nan_to_num(X_pseudo, nan=0.0, posinf=0.0, neginf=0.0)
        return _standardize(X_pseudo)
    raise ValueError(f"Unsupported dataset={name!r}. Use 'colon' or 'pitprop'.")


def build_feature_graph(
    X: np.ndarray,
    graph_type: str = "chain",
    knn_k: int = 8,
) -> GraphData:
    p = X.shape[1]
    if graph_type == "chain":
        return chain_graph(p)
    if graph_type == "knn":
        return knn_graph(X.T, n_neighbors=knn_k, mode="distance")
    raise ValueError(f"Unsupported graph_type={graph_type!r}. Use 'chain' or 'knn'.")


def run_real_benchmark(cfg: RealBenchmarkConfig) -> list[dict[str, Any]]:
    X = load_real_dataset(cfg.dataset)
    graph = build_feature_graph(X, graph_type=cfg.graph_type, knn_k=cfg.knn_k)
    methods = build_baselines(
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        n_components=cfg.n_components,
    )

    records: list[dict[str, Any]] = []
    L = graph.laplacian
    for method_name, estimator in methods.items():
        tic = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            if isinstance(estimator, NetworkSparsePCA):
                estimator.fit(X, graph=graph)
            else:
                estimator.fit(X)
        runtime_sec = perf_counter() - tic

        w_hat = _sanitize_component(np.asarray(estimator.components_[0], dtype=float))
        support_size = int(
            _estimated_support(w_hat, support_threshold=cfg.support_threshold).size
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ev = explained_variance(X, w_hat)
            le = laplacian_energy(w_hat, L)

        records.append(
            {
                "dataset": cfg.dataset,
                "method": method_name,
                "runtime_sec": float(runtime_sec),
                "explained_variance": float(ev) if np.isfinite(ev) else float("nan"),
                "support_size": support_size,
                "laplacian_energy": float(le) if np.isfinite(le) else float("nan"),
                "converged": bool(getattr(estimator, "converged_", False)),
                "n_iter": getattr(estimator, "n_iter_", None),
                "objective": getattr(estimator, "objective_", None),
                "graph_type": cfg.graph_type,
            }
        )
    return records


def summarize_real_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    out: list[dict[str, Any]] = []
    for row in records:
        out.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "explained_variance": row["explained_variance"],
                "support_size": row["support_size"],
                "laplacian_energy": row["laplacian_energy"],
                "runtime_sec": row["runtime_sec"],
                "converged": row["converged"],
            }
        )
    out.sort(key=lambda r: r["explained_variance"], reverse=True)
    return out
