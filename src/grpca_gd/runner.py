from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from .artifacts import save_json, save_metrics_csv, save_npz
from .amanpg import AmanpgConfig, solve_amanpg
from .datasets import load_artifact
from .metrics import (
    explained_variance,
    graph_smoothness_norm,
    graph_smoothness_raw,
    laplacian_energy,
    match_components,
    nnz_loadings,
    orthonormalize,
    orthogonality_error,
    sparsity_fraction,
    support_connectivity,
    support_metrics,
)
from .objective import objective_terms
from .solver import GraphSparseConfig, solve_graph_sparse, SolverConfig, solve, soft_threshold
from .synthetic.data import generate_dataset
from .synthetic.graphs import (
    chain_graph_laplacian,
    grid_graph_laplacian,
    knn_graph_laplacian,
    sbm_graph_laplacian,
    small_world_laplacian,
)
from .synthetic.corruption import delete_edges, rewire_edges, perturb_weights


def _hash_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    return _hash_bytes(arr.tobytes())


def _hash_config(cfg: Dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _env_info() -> Dict[str, Any]:
    import platform
    import importlib.metadata as md

    deps = {dist.metadata["Name"]: dist.version for dist in md.distributions()}
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": deps,
        "hostname": platform.node(),
    }


def _validate_config(cfg: Dict[str, Any]) -> None:
    required = [
        "seed",
        "r",
        "lambda1",
        "lambda2",
        "rho",
        "max_iters",
        "tol_obj",
        "tol_gap",
        "tol_orth",
        "eta_A",
        "baseline",
        "output_dir",
    ]
    if not cfg.get("artifact_dir"):
        required.extend(
            [
                "n",
                "p",
                "support_size",
                "snr",
                "graph_family",
                "support_type",
            ]
        )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config fields: {missing}")


def _alignment(
    A_est: np.ndarray, B_est: np.ndarray, true_loadings: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm, signs = match_components(A_est, true_loadings)
    r = B_est.shape[1]
    B_aligned = np.zeros_like(B_est)
    for i in range(r):
        B_aligned[:, perm[i]] = signs[i] * B_est[:, i]
    return B_aligned, perm, signs


def _pca_top_r(sigma_hat: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(sigma_hat)
    order = np.argsort(vals)[::-1]
    vals = vals[order][:r]
    vecs = vecs[:, order][:, :r]
    return vals, vecs


def _run_summary(
    history: Dict[str, np.ndarray],
    *,
    tol_obj: float,
    tol_orth: float,
    tol_gap: Optional[float] = None,
    final_gap: Optional[float] = None,
) -> Dict[str, Any]:
    iterations = int(len(history["total_objective"]))
    final_objective = float(history["total_objective"][-1]) if iterations else float("nan")
    final_orthogonality_defect = (
        float(history["orthogonality_error"][-1]) if iterations else float("nan")
    )
    if final_gap is None and "coupling_gap" in history and iterations:
        final_gap = float(history["coupling_gap"][-1])
    final_gap = 0.0 if final_gap is None else float(final_gap)

    convergence_flag = False
    if iterations >= 2:
        prev_obj = float(history["total_objective"][-2])
        rel = abs(prev_obj - final_objective) / max(1.0, abs(prev_obj))
        gap_ok = tol_gap is None or final_gap <= tol_gap
        convergence_flag = (
            rel <= tol_obj
            and gap_ok
            and final_orthogonality_defect <= tol_orth
        )

    return {
        "iterations": iterations,
        "final_objective": final_objective,
        "final_coupling_gap": final_gap,
        "final_orthogonality_defect": final_orthogonality_defect,
        "stop_reason": "tol_obj" if convergence_flag else "max_iters",
        "convergence_flag": convergence_flag,
    }


def _canonical_method_metrics(
    *,
    dataset: str,
    graph_family: str,
    artifact_id: str,
    artifact_version: str,
    data_source: str,
    prep_config_hash: str,
    eval_protocol_id: str,
    method: str,
    method_version: str,
    seed: int,
    rank: int,
    lambda1: float,
    lambda2: float,
    rho: float,
    corruption_type: str,
    corruption_level: float,
    graph_used_id: str,
    graph_reference_id: str,
    explained_variance_value: float,
    smoothness_used_graph: float,
    smoothness_reference_graph: float,
    runtime_sec: float,
    iterations: int,
    nnz: int,
    sparsity_ratio_value: float,
    final_objective: float,
    final_coupling_gap: float,
    final_orthogonality_defect: float,
    stop_reason: str,
    convergence_flag: bool,
    support: Optional[Dict[str, float]],
    n_samples: int,
    n_features: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "graph_family": graph_family,
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "data_source": data_source,
        "prep_config_hash": prep_config_hash,
        "eval_protocol_id": eval_protocol_id,
        "method": method,
        "method_version": method_version,
        "seed": seed,
        "rank": rank,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "rho": rho,
        "corruption_type": corruption_type,
        "corruption_level": corruption_level,
        "graph_used_id": graph_used_id,
        "graph_reference_id": graph_reference_id,
        "explained_variance": explained_variance_value,
        "smoothness_used_graph": smoothness_used_graph,
        "smoothness_reference_graph": smoothness_reference_graph,
        "runtime_sec": runtime_sec,
        "iterations": iterations,
        "nnz_loadings": nnz,
        "sparsity_ratio": sparsity_ratio_value,
        "final_objective": final_objective,
        "final_coupling_gap": final_coupling_gap,
        "final_orthogonality_defect": final_orthogonality_defect,
        "stop_reason": stop_reason,
        "convergence_flag": convergence_flag,
        "support_precision": support.get("precision") if support else None,
        "support_recall": support.get("recall") if support else None,
        "support_f1": support.get("f1") if support else None,
        "n_samples": n_samples,
        "n_features": n_features,
    }
    if extra:
        payload.update(extra)
    return payload


def _lambda1_candidates(base: float) -> List[float]:
    grid = [base * scale for scale in (0.25, 0.5, 1.0, 2.0, 4.0)]
    return sorted({max(1e-8, float(x)) for x in grid})


def _match_lambda1(
    target_nnz: int,
    candidates: List[float],
    run_fn,
) -> Tuple[float, Any, int]:
    best = None
    best_score = None
    for lam in candidates:
        result = run_fn(lam)
        nnz = nnz_loadings(result.B)
        score = abs(nnz - target_nnz)
        if best_score is None or score < best_score:
            best_score = score
            best = (lam, result, nnz)
    if best is None:
        raise RuntimeError("Failed to match lambda1 candidates")
    return best


def run(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    status = "success"
    failure_reason = None
    start_time = time.time()

    history_arrays = {}
    arrays = {}
    metrics_out: Dict[str, Any] = {}

    dataset = None
    diagnostics = None
    dataset_meta: Dict[str, Any] = {}
    try:
        artifact_dir = cfg.get("artifact_dir")
        if artifact_dir:
            artifact = load_artifact(Path(artifact_dir))
            X = artifact.X
            L = artifact.L
            dataset_meta = dict(artifact.metadata)
            dataset_name = artifact.dataset
            graph_family = artifact.graph_family
            artifact_id = artifact.artifact_id
            artifact_version = artifact.artifact_version
            data_source = artifact.data_source
            prep_config_hash = artifact.prep_config_hash
            eval_protocol_id = artifact.eval_protocol_id
            true_loadings = None
            true_supports = None
        else:
            graph_family = cfg["graph_family"]
            if graph_family == "chain":
                L, W = chain_graph_laplacian(cfg["p"])
            elif graph_family == "grid":
                rows = int(cfg.get("grid_rows", 10))
                cols = int(cfg.get("grid_cols", 10))
                if rows * cols != cfg["p"]:
                    raise ValueError("grid_rows * grid_cols must equal p")
                L, W = grid_graph_laplacian(rows, cols)
            elif graph_family == "knn":
                knn_k = int(cfg.get("knn_k", 5))
                knn_dim = int(cfg.get("knn_dim", 2))
                rng = np.random.default_rng(cfg["seed"] + 123)
                points = rng.normal(size=(cfg["p"], knn_dim))
                L, W = knn_graph_laplacian(points, knn_k)
            elif graph_family == "small_world":
                sw_k = int(cfg.get("small_world_k", 2))
                sw_beta = float(cfg.get("small_world_beta", 0.1))
                rng = np.random.default_rng(cfg["seed"] + 123)
                L, W = small_world_laplacian(cfg["p"], sw_k, sw_beta, rng)
            elif graph_family == "sbm":
                rng = np.random.default_rng(cfg["seed"] + 123)
                blocks = int(cfg.get("sbm_blocks", 3))
                p_in = float(cfg.get("sbm_p_in", 0.2))
                p_out = float(cfg.get("sbm_p_out", 0.02))
                block_sizes = cfg.get("sbm_block_sizes")
                if block_sizes is not None:
                    block_sizes = [int(x) for x in block_sizes]
                L, W = sbm_graph_laplacian(
                    cfg["p"], blocks, p_in, p_out, rng, block_sizes=block_sizes
                )
            else:
                raise ValueError(
                    "graph_family must be 'chain', 'grid', 'knn', 'small_world', or 'sbm'"
                )
            L_ref = L
            corruption_type = str(cfg.get("corruption_type", "none"))
            corruption_level = float(cfg.get("corruption_level", 0.0))
            if corruption_type != "none" and corruption_level > 0.0:
                rng = np.random.default_rng(cfg["seed"] + 999)
                if corruption_type == "delete":
                    W_used = delete_edges(W, corruption_level, rng)
                elif corruption_type == "rewire":
                    W_used = rewire_edges(W, corruption_level, rng)
                elif corruption_type == "perturb":
                    W_used = perturb_weights(W, corruption_level, rng)
                else:
                    raise ValueError("corruption_type must be none/delete/rewire/perturb")
                D_used = np.diag(W_used.sum(axis=1))
                L = D_used - W_used
                dataset_meta = {
                    "corruption_type": corruption_type,
                    "corruption_level": corruption_level,
                    "graph_used_id": f"{graph_family}_{corruption_type}_{corruption_level:.2f}",
                    "graph_reference_id": f"{graph_family}_clean",
                }
            else:
                L_ref = L
            dataset = generate_dataset(
                n=cfg["n"],
                p=cfg["p"],
                r=cfg["r"],
                support_size=cfg["support_size"],
                support_type=cfg["support_type"],
                L=L,
                snr=cfg["snr"],
                signal_eigs=cfg.get("signal_eigs"),
                seed=cfg["seed"],
                decoy_count=int(cfg.get("decoy_count", 0)),
                decoy_variance_factor=float(cfg.get("decoy_variance_factor", 0.0)),
            )
            X = dataset.X
            dataset_meta = {**dataset_meta, **dataset.metadata}
            dataset_name = "synthetic"
            artifact_id = f"synthetic_{graph_family}"
            artifact_version = "v1"
            data_source = "synthetic"
            prep_config_hash = _hash_config({k: v for k, v in cfg.items() if k != "output_dir"})
            eval_protocol_id = "default"
            true_loadings = dataset.true_loadings
            true_supports = dataset.true_supports

        Sigma_hat = dataset.Sigma_hat if dataset is not None else (X.T @ X) / int(X.shape[0])
        n_samples = int(X.shape[0])
        n_features = int(X.shape[1])
        eigvals, V = _pca_top_r(Sigma_hat, cfg["r"])
        A_pca = V
        B_pca = V

        if true_loadings is not None and true_supports is not None:
            B_aligned_pca, perm_pca, signs_pca = _alignment(A_pca, B_pca, true_loadings)
            support_pca = support_metrics(B_aligned_pca, true_supports)
        else:
            B_aligned_pca = B_pca
            perm_pca = np.arange(B_pca.shape[1])
            signs_pca = np.ones(B_pca.shape[1])
            support_pca = None
        pca_support_union = support_pca["union"] if support_pca else None
        pca_union_mask = np.any(np.abs(B_aligned_pca) > 1e-8, axis=1)
        pca_terms = objective_terms(
            A_pca,
            B_pca,
            Sigma_hat,
            L,
            cfg["lambda1"],
            cfg["lambda2"],
            cfg["rho"],
        )
        pca_smoothness_raw = graph_smoothness_raw(B_pca, L)
        pca_smoothness_norm = graph_smoothness_norm(B_pca, L)
        pca_eval = _canonical_method_metrics(
            dataset=dataset_name,
            graph_family=graph_family,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            data_source=data_source,
            prep_config_hash=prep_config_hash,
            eval_protocol_id=eval_protocol_id,
            method="PCA",
            method_version="v1",
            seed=cfg["seed"],
            rank=cfg["r"],
            lambda1=cfg["lambda1"],
            lambda2=cfg["lambda2"],
            rho=cfg["rho"],
            corruption_type=dataset_meta.get("corruption_type", "none"),
            corruption_level=float(dataset_meta.get("corruption_level", 0.0)),
            graph_used_id=dataset_meta.get("graph_used_id", f"{graph_family}_clean"),
            graph_reference_id=dataset_meta.get("graph_reference_id", f"{graph_family}_clean"),
            explained_variance_value=float(np.sum(eigvals)),
            smoothness_used_graph=pca_smoothness_norm,
            smoothness_reference_graph=pca_smoothness_norm,
            runtime_sec=0.0,
            iterations=1,
            nnz=nnz_loadings(B_pca),
            sparsity_ratio_value=sparsity_fraction(B_pca),
            final_objective=float(pca_terms["total_objective"]),
            final_coupling_gap=0.0,
            final_orthogonality_defect=orthogonality_error(A_pca),
            stop_reason="closed_form",
            convergence_flag=True,
            support=pca_support_union,
            n_samples=n_samples,
            n_features=n_features,
            extra={
                "method_name": "PCA",
                "objective_terms": pca_terms,
                "sparsity_fraction": sparsity_fraction(B_pca),
                "orthogonality_error": orthogonality_error(A_pca),
                "laplacian_energy": laplacian_energy(B_pca, L),
                "support_metrics": support_pca,
                "support_metrics_note": "Dense baseline diagnostics; interpret with caution.",
                "graph_smoothness_raw_trueL": pca_smoothness_raw,
                "graph_smoothness_norm_trueL": pca_smoothness_norm,
                "shared_explained_variance": explained_variance(A_pca, Sigma_hat),
            },
        )
        pca_eval["support_connectivity_union"] = support_connectivity(pca_union_mask, L)
        metrics_out["PCA"] = pca_eval

        A0 = A_pca
        B0 = soft_threshold(A0, cfg["lambda1"] / max(cfg["rho"], 1e-8))
        amanpg_cfg = AmanpgConfig(
            lambda1=cfg["lambda1"],
            eta_A=cfg["eta_A"],
            max_iters=cfg["max_iters"],
            tol_obj=cfg["tol_obj"],
            tol_orth=cfg["tol_orth"],
        )
        amanpg_result = solve_amanpg(A0, Sigma_hat, amanpg_cfg)
        amanpg_B = amanpg_result.A
        amanpg_summary = _run_summary(
            amanpg_result.history,
            tol_obj=amanpg_cfg.tol_obj,
            tol_orth=amanpg_cfg.tol_orth,
            final_gap=0.0,
        )
        if true_loadings is not None and true_supports is not None:
            amanpg_aligned, amanpg_perm, amanpg_signs = _alignment(
                amanpg_result.A, amanpg_B, true_loadings
            )
            amanpg_support = support_metrics(amanpg_aligned, true_supports)
        else:
            amanpg_aligned = amanpg_B
            amanpg_perm = np.arange(amanpg_B.shape[1])
            amanpg_signs = np.ones(amanpg_B.shape[1])
            amanpg_support = None
        amanpg_support_union = amanpg_support["union"] if amanpg_support else None
        amanpg_union_mask = np.any(np.abs(amanpg_aligned) > 1e-8, axis=1)
        amanpg_smoothness_raw = graph_smoothness_raw(amanpg_B, L)
        amanpg_smoothness_norm = graph_smoothness_norm(amanpg_B, L)
        amanpg_eval = _canonical_method_metrics(
            dataset=dataset_name,
            graph_family=graph_family,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            data_source=data_source,
            prep_config_hash=prep_config_hash,
            eval_protocol_id=eval_protocol_id,
            method="A-ManPG",
            method_version="v1",
            seed=cfg["seed"],
            rank=cfg["r"],
            lambda1=cfg["lambda1"],
            lambda2=0.0,
            rho=0.0,
            corruption_type=dataset_meta.get("corruption_type", "none"),
            corruption_level=float(dataset_meta.get("corruption_level", 0.0)),
            graph_used_id=dataset_meta.get("graph_used_id", f"{graph_family}_clean"),
            graph_reference_id=dataset_meta.get("graph_reference_id", f"{graph_family}_clean"),
            explained_variance_value=explained_variance(orthonormalize(amanpg_B), Sigma_hat),
            smoothness_used_graph=amanpg_smoothness_norm,
            smoothness_reference_graph=amanpg_smoothness_norm,
            runtime_sec=0.0,
            iterations=int(len(amanpg_result.history["total_objective"])),
            nnz=nnz_loadings(amanpg_B),
            sparsity_ratio_value=sparsity_fraction(amanpg_B),
            final_objective=float(amanpg_summary["final_objective"]),
            final_coupling_gap=float(amanpg_summary["final_coupling_gap"]),
            final_orthogonality_defect=float(amanpg_summary["final_orthogonality_defect"]),
            stop_reason=amanpg_summary["stop_reason"],
            convergence_flag=amanpg_summary["convergence_flag"],
            support=amanpg_support_union,
            n_samples=n_samples,
            n_features=n_features,
            extra={
                "method_name": "A-ManPG",
                "objective_terms": objective_terms(
                    amanpg_result.A,
                    amanpg_B,
                    Sigma_hat,
                    L,
                    cfg["lambda1"],
                    0.0,
                    0.0,
                ),
                "sparsity_fraction": sparsity_fraction(amanpg_B),
                "orthogonality_error": orthogonality_error(amanpg_result.A),
                "laplacian_energy": laplacian_energy(amanpg_B, L),
                "support_metrics": amanpg_support,
                "graph_smoothness_raw_trueL": amanpg_smoothness_raw,
                "graph_smoothness_norm_trueL": amanpg_smoothness_norm,
                "shared_explained_variance": explained_variance(
                    orthonormalize(amanpg_B), Sigma_hat
                ),
            },
        )
        amanpg_eval["support_connectivity_union"] = support_connectivity(
            amanpg_union_mask, L
        )
        metrics_out["A-ManPG"] = amanpg_eval

        solver_cfg = SolverConfig(
            lambda1=cfg["lambda1"],
            lambda2=cfg["lambda2"],
            rho=cfg["rho"],
            eta_A=cfg["eta_A"],
            max_iters=cfg["max_iters"],
            tol_obj=cfg["tol_obj"],
            tol_gap=cfg["tol_gap"],
            tol_orth=cfg["tol_orth"],
        )
        result = solve(A0, B0, Sigma_hat, L, solver_cfg)
        solver_summary = _run_summary(
            result.history,
            tol_obj=solver_cfg.tol_obj,
            tol_orth=solver_cfg.tol_orth,
            tol_gap=solver_cfg.tol_gap,
        )
        if true_loadings is not None and true_supports is not None:
            B_aligned, perm, signs = _alignment(result.A, result.B, true_loadings)
            support = support_metrics(B_aligned, true_supports)
        else:
            B_aligned = result.B
            perm = np.arange(result.B.shape[1])
            signs = np.ones(result.B.shape[1])
            support = None
        support_union = support["union"] if support else None
        proposed_union_mask = np.any(np.abs(B_aligned) > 1e-8, axis=1)
        smooth_L_ref = L_ref if "L_ref" in locals() else L
        proposed_smoothness_raw = graph_smoothness_raw(result.B, L)
        proposed_smoothness_norm = graph_smoothness_norm(result.B, L)
        proposed_smoothness_ref = graph_smoothness_norm(result.B, smooth_L_ref)
        proposed_eval = _canonical_method_metrics(
            dataset=dataset_name,
            graph_family=graph_family,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            data_source=data_source,
            prep_config_hash=prep_config_hash,
            eval_protocol_id=eval_protocol_id,
            method="Proposed",
            method_version="v1",
            seed=cfg["seed"],
            rank=cfg["r"],
            lambda1=cfg["lambda1"],
            lambda2=cfg["lambda2"],
            rho=cfg["rho"],
            corruption_type=dataset_meta.get("corruption_type", "none"),
            corruption_level=float(dataset_meta.get("corruption_level", 0.0)),
            graph_used_id=dataset_meta.get("graph_used_id", f"{graph_family}_clean"),
            graph_reference_id=dataset_meta.get("graph_reference_id", f"{graph_family}_clean"),
            explained_variance_value=explained_variance(orthonormalize(result.B), Sigma_hat),
            smoothness_used_graph=proposed_smoothness_norm,
            smoothness_reference_graph=proposed_smoothness_ref,
            runtime_sec=0.0,
            iterations=solver_summary["iterations"],
            nnz=nnz_loadings(result.B),
            sparsity_ratio_value=sparsity_fraction(result.B),
            final_objective=float(solver_summary["final_objective"]),
            final_coupling_gap=float(solver_summary["final_coupling_gap"]),
            final_orthogonality_defect=float(solver_summary["final_orthogonality_defect"]),
            stop_reason=solver_summary["stop_reason"],
            convergence_flag=solver_summary["convergence_flag"],
            support=support_union,
            n_samples=n_samples,
            n_features=n_features,
            extra={
                "method_name": "Proposed",
                "objective_terms": objective_terms(
                    result.A,
                    result.B,
                    Sigma_hat,
                    L,
                    cfg["lambda1"],
                    cfg["lambda2"],
                    cfg["rho"],
                ),
                "sparsity_fraction": sparsity_fraction(result.B),
                "orthogonality_error": orthogonality_error(result.A),
                "laplacian_energy": laplacian_energy(result.B, L),
                "support_metrics": support,
                "graph_smoothness_raw_trueL": proposed_smoothness_raw,
                "graph_smoothness_norm_trueL": proposed_smoothness_norm,
                "shared_explained_variance": explained_variance(
                    orthonormalize(result.B), Sigma_hat
                ),
            },
        )
        proposed_eval["support_connectivity_union"] = support_connectivity(
            proposed_union_mask, L
        )
        metrics_out["Proposed"] = proposed_eval
        target_nnz = nnz_loadings(result.B)
        lambda1_candidates = _lambda1_candidates(cfg["lambda1"])

        # Graph-only PCA baseline (lambda1=0)
        graph_only_cfg = SolverConfig(
            lambda1=0.0,
            lambda2=cfg["lambda2"],
            rho=cfg["rho"],
            eta_A=cfg["eta_A"],
            max_iters=cfg["max_iters"],
            tol_obj=cfg["tol_obj"],
            tol_gap=cfg["tol_gap"],
            tol_orth=cfg["tol_orth"],
        )
        graph_only_result = solve(A0, A0, Sigma_hat, L, graph_only_cfg)
        graph_only_summary = _run_summary(
            graph_only_result.history,
            tol_obj=graph_only_cfg.tol_obj,
            tol_orth=graph_only_cfg.tol_orth,
            tol_gap=graph_only_cfg.tol_gap,
        )
        if true_loadings is not None and true_supports is not None:
            graph_only_aligned, graph_only_perm, graph_only_signs = _alignment(
                graph_only_result.A, graph_only_result.B, true_loadings
            )
            graph_only_support = support_metrics(graph_only_aligned, true_supports)
        else:
            graph_only_aligned = graph_only_result.B
            graph_only_perm = np.arange(graph_only_result.B.shape[1])
            graph_only_signs = np.ones(graph_only_result.B.shape[1])
            graph_only_support = None
        graph_only_support_union = graph_only_support["union"] if graph_only_support else None
        graph_only_union_mask = np.any(np.abs(graph_only_aligned) > 1e-8, axis=1)
        graph_only_smoothness_raw = graph_smoothness_raw(graph_only_result.B, L)
        graph_only_smoothness_norm = graph_smoothness_norm(graph_only_result.B, L)
        graph_only_smoothness_ref = graph_smoothness_norm(graph_only_result.B, smooth_L_ref)
        graph_only_eval = _canonical_method_metrics(
            dataset=dataset_name,
            graph_family=graph_family,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            data_source=data_source,
            prep_config_hash=prep_config_hash,
            eval_protocol_id=eval_protocol_id,
            method="GraphOnlyPCA",
            method_version="v1",
            seed=cfg["seed"],
            rank=cfg["r"],
            lambda1=0.0,
            lambda2=cfg["lambda2"],
            rho=cfg["rho"],
            corruption_type=dataset_meta.get("corruption_type", "none"),
            corruption_level=float(dataset_meta.get("corruption_level", 0.0)),
            graph_used_id=dataset_meta.get("graph_used_id", f"{graph_family}_clean"),
            graph_reference_id=dataset_meta.get("graph_reference_id", f"{graph_family}_clean"),
            explained_variance_value=explained_variance(
                orthonormalize(graph_only_result.B), Sigma_hat
            ),
            smoothness_used_graph=graph_only_smoothness_norm,
            smoothness_reference_graph=graph_only_smoothness_ref,
            runtime_sec=0.0,
            iterations=graph_only_summary["iterations"],
            nnz=nnz_loadings(graph_only_result.B),
            sparsity_ratio_value=sparsity_fraction(graph_only_result.B),
            final_objective=float(graph_only_summary["final_objective"]),
            final_coupling_gap=float(graph_only_summary["final_coupling_gap"]),
            final_orthogonality_defect=float(graph_only_summary["final_orthogonality_defect"]),
            stop_reason=graph_only_summary["stop_reason"],
            convergence_flag=graph_only_summary["convergence_flag"],
            support=graph_only_support_union,
            n_samples=n_samples,
            n_features=n_features,
            extra={
                "method_name": "GraphOnlyPCA",
                "objective_terms": objective_terms(
                    graph_only_result.A,
                    graph_only_result.B,
                    Sigma_hat,
                    L,
                    0.0,
                    cfg["lambda2"],
                    cfg["rho"],
                ),
                "sparsity_fraction": sparsity_fraction(graph_only_result.B),
                "orthogonality_error": orthogonality_error(graph_only_result.A),
                "laplacian_energy": laplacian_energy(graph_only_result.B, L),
                "support_metrics": graph_only_support,
                "support_metrics_note": "Dense graph-only baseline; interpret support metrics with caution.",
                "graph_smoothness_raw_trueL": graph_only_smoothness_raw,
                "graph_smoothness_norm_trueL": graph_only_smoothness_norm,
                "shared_explained_variance": explained_variance(
                    orthonormalize(graph_only_result.B), Sigma_hat
                ),
            },
        )
        graph_only_eval["support_connectivity_union"] = support_connectivity(
            graph_only_union_mask, L
        )
        metrics_out["GraphOnlyPCA"] = graph_only_eval

        # Graph + sparsity baseline without orthogonality (non-manifold)
        def _run_graph_sparse(lam: float):
            cfg_local = GraphSparseConfig(
                lambda1=lam,
                lambda2=cfg["lambda2"],
                max_iters=cfg["max_iters"],
                tol_obj=cfg["tol_obj"],
            )
            return solve_graph_sparse(B0, Sigma_hat, L, cfg_local)

        graph_sparse_lambda1, graph_sparse_result, graph_sparse_nnz = _match_lambda1(
            target_nnz, lambda1_candidates, _run_graph_sparse
        )
        graph_sparse_cfg = GraphSparseConfig(
            lambda1=graph_sparse_lambda1,
            lambda2=cfg["lambda2"],
            max_iters=cfg["max_iters"],
            tol_obj=cfg["tol_obj"],
        )
        graph_sparse_summary = _run_summary(
            graph_sparse_result.history,
            tol_obj=graph_sparse_cfg.tol_obj,
            tol_orth=cfg["tol_orth"],
            tol_gap=None,
            final_gap=0.0,
        )
        if true_loadings is not None and true_supports is not None:
            graph_sparse_aligned, graph_sparse_perm, graph_sparse_signs = _alignment(
                graph_sparse_result.B, graph_sparse_result.B, true_loadings
            )
            graph_sparse_support = support_metrics(graph_sparse_aligned, true_supports)
        else:
            graph_sparse_aligned = graph_sparse_result.B
            graph_sparse_perm = np.arange(graph_sparse_result.B.shape[1])
            graph_sparse_signs = np.ones(graph_sparse_result.B.shape[1])
            graph_sparse_support = None
        graph_sparse_support_union = (
            graph_sparse_support["union"] if graph_sparse_support else None
        )
        graph_sparse_union_mask = np.any(np.abs(graph_sparse_aligned) > 1e-8, axis=1)
        graph_sparse_smoothness_raw = graph_smoothness_raw(graph_sparse_result.B, L)
        graph_sparse_smoothness_norm = graph_smoothness_norm(graph_sparse_result.B, L)
        graph_sparse_smoothness_ref = graph_smoothness_norm(graph_sparse_result.B, smooth_L_ref)
        graph_sparse_eval = _canonical_method_metrics(
            dataset=dataset_name,
            graph_family=graph_family,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            data_source=data_source,
            prep_config_hash=prep_config_hash,
            eval_protocol_id=eval_protocol_id,
            method="GraphSparseNoOrth",
            method_version="v1",
            seed=cfg["seed"],
            rank=cfg["r"],
            lambda1=graph_sparse_lambda1,
            lambda2=cfg["lambda2"],
            rho=0.0,
            corruption_type=dataset_meta.get("corruption_type", "none"),
            corruption_level=float(dataset_meta.get("corruption_level", 0.0)),
            graph_used_id=dataset_meta.get("graph_used_id", f"{graph_family}_clean"),
            graph_reference_id=dataset_meta.get("graph_reference_id", f"{graph_family}_clean"),
            explained_variance_value=explained_variance(
                orthonormalize(graph_sparse_result.B), Sigma_hat
            ),
            smoothness_used_graph=graph_sparse_smoothness_norm,
            smoothness_reference_graph=graph_sparse_smoothness_ref,
            runtime_sec=0.0,
            iterations=graph_sparse_summary["iterations"],
            nnz=nnz_loadings(graph_sparse_result.B),
            sparsity_ratio_value=sparsity_fraction(graph_sparse_result.B),
            final_objective=float(graph_sparse_summary["final_objective"]),
            final_coupling_gap=float(graph_sparse_summary["final_coupling_gap"]),
            final_orthogonality_defect=float(graph_sparse_summary["final_orthogonality_defect"]),
            stop_reason=graph_sparse_summary["stop_reason"],
            convergence_flag=graph_sparse_summary["convergence_flag"],
            support=graph_sparse_support_union,
            n_samples=n_samples,
            n_features=n_features,
            extra={
                "method_name": "GraphSparseNoOrth",
                "objective_terms": objective_terms(
                    graph_sparse_result.B,
                    graph_sparse_result.B,
                    Sigma_hat,
                    L,
                    cfg["lambda1"],
                    cfg["lambda2"],
                    0.0,
                ),
                "sparsity_fraction": sparsity_fraction(graph_sparse_result.B),
                "orthogonality_error": orthogonality_error(
                    orthonormalize(graph_sparse_result.B)
                ),
                "laplacian_energy": laplacian_energy(graph_sparse_result.B, L),
                "support_metrics": graph_sparse_support,
                "graph_smoothness_raw_trueL": graph_sparse_smoothness_raw,
                "graph_smoothness_norm_trueL": graph_sparse_smoothness_norm,
                "shared_explained_variance": explained_variance(
                    orthonormalize(graph_sparse_result.B), Sigma_hat
                ),
                "optimization_note": "Non-manifold graph+sparsity baseline without orthogonality constraints.",
                "lambda1_match_target_nnz": float(target_nnz),
                "lambda1_matched_nnz": float(graph_sparse_nnz),
            },
        )
        graph_sparse_eval["support_connectivity_union"] = support_connectivity(
            graph_sparse_union_mask, L
        )
        metrics_out["GraphSparseNoOrth"] = graph_sparse_eval

        # Sparse PCA baseline (no graph regularization)
        L_zero = np.zeros_like(L)
        def _run_spca(lam: float) -> SolverResult:
            spca_cfg_local = SolverConfig(
                lambda1=lam,
                lambda2=0.0,
                rho=cfg["rho"],
                eta_A=cfg["eta_A"],
                max_iters=cfg["max_iters"],
                tol_obj=cfg["tol_obj"],
                tol_gap=cfg["tol_gap"],
                tol_orth=cfg["tol_orth"],
            )
            B0_local = soft_threshold(A0, lam / max(cfg["rho"], 1e-8))
            return solve(A0, B0_local, Sigma_hat, L_zero, spca_cfg_local)

        spca_lambda1, spca_result, spca_nnz = _match_lambda1(
            target_nnz, lambda1_candidates, _run_spca
        )
        spca_cfg = SolverConfig(
            lambda1=spca_lambda1,
            lambda2=0.0,
            rho=cfg["rho"],
            eta_A=cfg["eta_A"],
            max_iters=cfg["max_iters"],
            tol_obj=cfg["tol_obj"],
            tol_gap=cfg["tol_gap"],
            tol_orth=cfg["tol_orth"],
        )
        spca_summary = _run_summary(
            spca_result.history,
            tol_obj=spca_cfg.tol_obj,
            tol_orth=spca_cfg.tol_orth,
            tol_gap=spca_cfg.tol_gap,
        )
        if true_loadings is not None and true_supports is not None:
            spca_aligned, spca_perm, spca_signs = _alignment(
                spca_result.A, spca_result.B, true_loadings
            )
            spca_support = support_metrics(spca_aligned, true_supports)
        else:
            spca_aligned = spca_result.B
            spca_perm = np.arange(spca_result.B.shape[1])
            spca_signs = np.ones(spca_result.B.shape[1])
            spca_support = None
        spca_support_union = spca_support["union"] if spca_support else None
        spca_union_mask = np.any(np.abs(spca_aligned) > 1e-8, axis=1)
        spca_smoothness_raw = graph_smoothness_raw(spca_result.B, L)
        spca_smoothness_norm = graph_smoothness_norm(spca_result.B, L)
        spca_smoothness_ref = graph_smoothness_norm(spca_result.B, smooth_L_ref)
        spca_eval = _canonical_method_metrics(
            dataset=dataset_name,
            graph_family=graph_family,
            artifact_id=artifact_id,
            artifact_version=artifact_version,
            data_source=data_source,
            prep_config_hash=prep_config_hash,
            eval_protocol_id=eval_protocol_id,
            method="SparseNoGraph",
            method_version="v1",
            seed=cfg["seed"],
            rank=cfg["r"],
            lambda1=spca_lambda1,
            lambda2=0.0,
            rho=cfg["rho"],
            corruption_type=dataset_meta.get("corruption_type", "none"),
            corruption_level=float(dataset_meta.get("corruption_level", 0.0)),
            graph_used_id=dataset_meta.get("graph_used_id", f"{graph_family}_clean"),
            graph_reference_id=dataset_meta.get("graph_reference_id", f"{graph_family}_clean"),
            explained_variance_value=explained_variance(
                orthonormalize(spca_result.B), Sigma_hat
            ),
            smoothness_used_graph=spca_smoothness_norm,
            smoothness_reference_graph=spca_smoothness_ref,
            runtime_sec=0.0,
            iterations=spca_summary["iterations"],
            nnz=nnz_loadings(spca_result.B),
            sparsity_ratio_value=sparsity_fraction(spca_result.B),
            final_objective=float(spca_summary["final_objective"]),
            final_coupling_gap=float(spca_summary["final_coupling_gap"]),
            final_orthogonality_defect=float(spca_summary["final_orthogonality_defect"]),
            stop_reason=spca_summary["stop_reason"],
            convergence_flag=spca_summary["convergence_flag"],
            support=spca_support_union,
            n_samples=n_samples,
            n_features=n_features,
            extra={
                "method_name": "SparseNoGraph",
                "objective_terms": objective_terms(
                    spca_result.A,
                    spca_result.B,
                    Sigma_hat,
                    L_zero,
                    cfg["lambda1"],
                    0.0,
                    cfg["rho"],
                ),
                "sparsity_fraction": sparsity_fraction(spca_result.B),
                "orthogonality_error": orthogonality_error(spca_result.A),
                "laplacian_energy": laplacian_energy(spca_result.B, L_zero),
                "support_metrics": spca_support,
                "graph_smoothness_raw_trueL": spca_smoothness_raw,
                "graph_smoothness_norm_trueL": spca_smoothness_norm,
                "shared_explained_variance": explained_variance(
                    orthonormalize(spca_result.B), Sigma_hat
                ),
                "lambda1_match_target_nnz": float(target_nnz),
                "lambda1_matched_nnz": float(spca_nnz),
            },
        )
        spca_eval["support_connectivity_union"] = support_connectivity(
            spca_union_mask, L
        )
        metrics_out["SparseNoGraph"] = spca_eval

        history_arrays = result.history
        arrays = {
            "A": result.A,
            "B": result.B,
            "A_init": A0,
            "B_init": B0,
            "amanpg_A": amanpg_result.A,
            "amanpg_B": amanpg_B,
            "amanpg_matching_perm": amanpg_perm,
            "amanpg_matching_signs": amanpg_signs,
            "pca_A": A_pca,
            "pca_B": B_pca,
            "Sigma_hat": Sigma_hat,
            "L": L,
            "matching_perm": perm,
            "matching_signs": signs,
            "pca_matching_perm": perm_pca,
            "pca_matching_signs": signs_pca,
            **{f"amanpg_history_{k}": v for k, v in amanpg_result.history.items()},
            **{f"history_{k}": v for k, v in history_arrays.items()},
        }
        if dataset is not None:
            arrays.update(
                {
                    "Sigma_true": dataset.Sigma_true,
                    "true_loadings": dataset.true_loadings,
                    "true_support": np.array(dataset.true_supports, dtype=object),
                }
            )
        else:
            arrays["X"] = X

        _write_convergence_plots(output_dir / "plots", history_arrays)

    except Exception as exc:
        status = "failed"
        failure_reason = f"{exc}"
        diagnostics = traceback.format_exc()

    end_time = time.time()
    runtime = end_time - start_time

    for payload in metrics_out.values():
        payload["runtime_sec"] = runtime

    manifest = {
        "status": status,
        "failure_reason": failure_reason,
        "runtime_sec": runtime,
        "config_hash": _hash_config(cfg),
        "dataset_hash": _hash_array(dataset.Sigma_true) if dataset is not None else None,
        "git_hash": _git_hash(),
        "tuning_note": "Defaults rho=5.0 and eta_A=0.05 selected for improved coupling behavior on r=1 and r=3 smoke runs without harming recovery.",
        **_env_info(),
    }

    artifacts_meta = {
        "method_name": "Proposed",
        "baseline_methods": [
            "PCA",
            "A-ManPG",
            "GraphOnlyPCA",
            "GraphSparseNoOrth",
            "SparseNoGraph",
        ],
        "status": status,
        "failure_reason": failure_reason,
        "manifest": manifest,
        "config": cfg,
        "dataset_metadata": dataset_meta,
        "baseline_placeholder_note": "PCA artifacts use B=V for schema compatibility.",
        "diagnostics": diagnostics,
    }

    if arrays:
        save_npz(output_dir / "artifacts.npz", arrays)

    save_json(output_dir / "artifacts.json", artifacts_meta)
    save_json(output_dir / "manifest.json", manifest)
    if status == "success":
        save_json(output_dir / "metrics.json", metrics_out)
        save_metrics_csv(output_dir / "metrics.csv", metrics_out)

    if status != "success":
        raise SystemExit(1)

def _git_hash() -> Optional[str]:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return None


def _write_convergence_plots(plot_dir: Path, history: Dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    series = [
        ("total_objective", "Objective"),
        ("coupling_gap", "Coupling Gap"),
        ("orthogonality_error", "Orthogonality Error"),
        ("sparsity_fraction", "Sparsity Fraction"),
        ("laplacian_energy", "Laplacian Energy"),
    ]
    for key, title in series:
        if key not in history:
            continue
        plt.figure()
        plt.plot(history[key])
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{key}.png")
        plt.close()
