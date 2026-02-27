"""Synthetic benchmark utilities for network-constrained SPCA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning

from src.models import (
    GeneralizedPowerMethod,
    NetworkSparsePCA,
    NetworkSparsePCA_MASPG_CAR,
    PCAEstimator,
    SparsePCA_L1_ProxGrad,
    ZouSparsePCA,
)
from src.utils.graph import GraphData, chain_graph, er_graph, grid_graph, sbm_graph
from src.utils.metrics import (
    connected_support_lcc_ratio,
    explained_variance,
    laplacian_energy,
    support_metrics,
)


@dataclass
class SyntheticBenchmarkConfig:
    """Configuration for synthetic graph-structured SPCA benchmarks."""

    n_samples: int = 200
    n_features: int = 100
    support_size: int = 20
    signal_strength: float = 8.0
    noise_std: float = 1.0
    graph_type: str = "chain"
    graph_laplacian_type: str = "unnormalized"
    graph_er_p: float = 0.08
    graph_sbm_p_in: float = 0.25
    graph_sbm_p_out: float = 0.02
    support_threshold: float = 1e-6
    n_components: int = 1
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_graph(cfg: SyntheticBenchmarkConfig, rng: np.random.Generator) -> GraphData:
    if cfg.graph_type == "chain":
        return chain_graph(cfg.n_features, laplacian_type=cfg.graph_laplacian_type)
    if cfg.graph_type == "grid":
        side = int(np.sqrt(cfg.n_features))
        if side * side != cfg.n_features:
            raise ValueError(
                "For graph_type='grid', n_features must be a perfect square."
            )
        return grid_graph(side, side, laplacian_type=cfg.graph_laplacian_type)
    if cfg.graph_type == "er":
        seed = int(rng.integers(0, 2**31 - 1))
        return er_graph(
            cfg.n_features,
            p_edge=cfg.graph_er_p,
            random_state=seed,
            laplacian_type=cfg.graph_laplacian_type,
        )
    if cfg.graph_type == "sbm":
        # Near-uniform blocks for a compact default community structure.
        n_blocks = min(4, max(2, cfg.n_features // 25))
        base = cfg.n_features // n_blocks
        rem = cfg.n_features % n_blocks
        sizes = [base + (1 if i < rem else 0) for i in range(n_blocks)]
        seed = int(rng.integers(0, 2**31 - 1))
        return sbm_graph(
            sizes,
            p_in=cfg.graph_sbm_p_in,
            p_out=cfg.graph_sbm_p_out,
            random_state=seed,
            laplacian_type=cfg.graph_laplacian_type,
        )
    raise ValueError(f"Unsupported graph_type={cfg.graph_type!r}")


def _sample_connected_support(
    adjacency: sp.csr_matrix,
    support_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = adjacency.shape[0]
    k = min(max(1, support_size), n)
    start = int(rng.integers(0, n))

    support = [start]
    support_set = {start}
    frontier = [start]

    while len(support) < k and frontier:
        node = frontier.pop(0)
        neighbors = adjacency.indices[
            adjacency.indptr[node] : adjacency.indptr[node + 1]
        ]
        neighbors = neighbors[rng.permutation(len(neighbors))]
        for nei in neighbors:
            idx = int(nei)
            if idx in support_set:
                continue
            support.append(idx)
            support_set.add(idx)
            frontier.append(idx)
            if len(support) >= k:
                break

    # If graph disconnected / sparse, fill remaining coordinates uniformly.
    if len(support) < k:
        remaining = [i for i in range(n) if i not in support_set]
        if remaining:
            extra = rng.choice(remaining, size=k - len(support), replace=False)
            support.extend(int(i) for i in extra)

    return np.array(sorted(set(support)), dtype=int)


def generate_graph_structured_data(
    cfg: SyntheticBenchmarkConfig,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Generate graph-structured synthetic data with sparse connected truth."""
    seed = cfg.random_state if random_state is None else random_state
    rng = np.random.default_rng(seed)
    graph = _build_graph(cfg, rng)
    support = _sample_connected_support(graph.adjacency, cfg.support_size, rng)

    w_true = np.zeros(cfg.n_features, dtype=float)
    w_true[support] = rng.normal(size=support.size)
    w_true /= np.linalg.norm(w_true) + 1e-12

    z = rng.normal(size=(cfg.n_samples, 1))
    eps = cfg.noise_std * rng.normal(size=(cfg.n_samples, cfg.n_features))
    X = np.sqrt(cfg.signal_strength) * (z @ w_true.reshape(1, -1)) + eps

    return {
        "X": X,
        "w_true": w_true,
        "true_support": support,
        "graph": graph,
        "seed": seed,
        "config": cfg.to_dict(),
    }


def _safe_scalar(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return int(max(value)) if value else None
    return None


def _first_component(model: Any) -> np.ndarray:
    comp = np.asarray(model.components_, dtype=float)
    if comp.ndim != 2 or comp.shape[0] == 0:
        raise ValueError("Model does not expose a valid components_ matrix.")
    return comp[0]


def _sanitize_component(component: np.ndarray) -> np.ndarray:
    w = np.asarray(component, dtype=float).reshape(-1)
    if not np.all(np.isfinite(w)):
        return np.zeros_like(w)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return np.zeros_like(w)
    return w / norm


def build_baselines(
    lambda1: float = 0.15,
    lambda2: float = 0.25,
    max_iter: int = 400,
    random_state: int = 0,
) -> dict[str, Any]:
    """Build a targeted comparison set spanning sparsity/graph/optimizer axes."""
    return {
        "PCA": PCAEstimator(n_components=1),
        "L1-SPCA-ProxGrad": SparsePCA_L1_ProxGrad(
            n_components=1,
            lambda1=lambda1,
            max_iter=max_iter,
            monotone_backtracking=True,
        ),
        "Graph-PCA": NetworkSparsePCA(
            n_components=1,
            lambda1=0.0,
            lambda2=lambda2,
            max_iter=max_iter,
            algorithm="pg",
            random_state=random_state,
        ),
        "NetSPCA-PG": NetworkSparsePCA(
            n_components=1,
            lambda1=lambda1,
            lambda2=lambda2,
            max_iter=max_iter,
            algorithm="pg",
            random_state=random_state,
        ),
        "NetSPCA-MASPG-CAR": NetworkSparsePCA_MASPG_CAR(
            n_components=1,
            lambda1=lambda1,
            lambda2=lambda2,
            max_iter=max_iter,
            random_state=random_state,
        ),
        "GPower": GeneralizedPowerMethod(
            n_components=1,
            gamma=lambda1,
            max_iter=max_iter,
        ),
        "ElasticNet-SPCA": ZouSparsePCA(
            n_components=1,
            alpha=max(lambda1 * 40.0, 1e-6),
            lambda_l2=1e-3,
            max_iter=min(max_iter, 200),
        ),
    }


def run_benchmark_once(
    X: np.ndarray,
    graph: GraphData,
    w_true: np.ndarray,
    methods: dict[str, Any],
    support_threshold: float = 1e-6,
) -> list[dict[str, Any]]:
    """Run one benchmark pass and return per-method metric records."""
    records: list[dict[str, Any]] = []
    true_support = np.flatnonzero(np.abs(w_true) > support_threshold)
    L = graph.laplacian
    A = graph.adjacency

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            w_hat = _sanitize_component(_first_component(estimator))
        est_support = np.flatnonzero(np.abs(w_hat) > support_threshold)
        s_metrics = support_metrics(est_support, true_support)

        record = {
            "method": method_name,
            "runtime_sec": float(runtime_sec),
            "explained_variance": explained_variance(X, w_hat),
            "support_size": int(est_support.size),
            "precision": float(s_metrics["precision"]),
            "recall": float(s_metrics["recall"]),
            "f1": float(s_metrics["f1"]),
            "lcc_ratio": connected_support_lcc_ratio(est_support, A),
            "laplacian_energy": laplacian_energy(w_hat, L),
            "converged": bool(getattr(estimator, "converged_", False)),
            "n_iter": _safe_scalar(getattr(estimator, "n_iter_", None)),
            "objective": _safe_scalar(getattr(estimator, "objective_", None)),
        }
        records.append(record)

    return records


def run_repeated_benchmark(
    cfg: SyntheticBenchmarkConfig,
    methods: dict[str, Any],
    n_repeats: int = 1,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run repeated independent synthetic trials and collect records."""
    records: list[dict[str, Any]] = []
    start_seed = cfg.random_state if base_seed is None else base_seed

    for rep in range(n_repeats):
        rep_seed = start_seed + rep
        sample = generate_graph_structured_data(cfg, random_state=rep_seed)
        X = sample["X"]
        graph = sample["graph"]
        w_true = sample["w_true"]
        run_records = run_benchmark_once(
            X,
            graph=graph,
            w_true=w_true,
            methods=methods,
            support_threshold=cfg.support_threshold,
        )
        for row in run_records:
            row["repeat"] = rep
            row["seed"] = rep_seed
            row["graph_type"] = cfg.graph_type
            row["n_samples"] = cfg.n_samples
            row["n_features"] = cfg.n_features
            row["support_size_true"] = cfg.support_size
            row["noise_std"] = cfg.noise_std
            row["signal_strength"] = cfg.signal_strength
        records.extend(run_records)

    return records


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated benchmark records into mean/std summaries."""
    if not records:
        return []
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_method.setdefault(str(row["method"]), []).append(row)

    metrics = [
        "explained_variance",
        "support_size",
        "precision",
        "recall",
        "f1",
        "lcc_ratio",
        "laplacian_energy",
        "runtime_sec",
    ]
    summary: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        out: dict[str, Any] = {"method": method, "n_runs": len(rows)}
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in rows], dtype=float)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std(ddof=0))
        converged_rate = np.mean(
            [1.0 if bool(r.get("converged")) else 0.0 for r in rows]
        )
        out["converged_rate"] = float(converged_rate)
        summary.append(out)

    summary.sort(key=lambda r: r["f1_mean"], reverse=True)
    return summary
