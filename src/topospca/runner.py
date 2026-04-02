from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .artifacts import save_json, save_metrics_csv, save_npz
from .amanpg import AmanpgConfig, solve_amanpg
from .metrics import (
    explained_variance,
    graph_smoothness_norm,
    graph_smoothness_raw,
    laplacian_energy,
    match_components,
    orthonormalize,
    orthogonality_error,
    sparsity_fraction,
    support_connectivity,
    support_metrics,
)
from .objective import objective_terms
from .solver import SolverConfig, solve, soft_threshold
from .synthetic.data import generate_dataset
from .synthetic.corruption import corrupt_graph
from .synthetic.graphs import (
    chain_graph_artifact,
    grid_graph_artifact,
    infer_grid_shape,
    sbm_graph_laplacian,
)
from .real_data import load_tcga_brca_string


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
    dataset_type = cfg.get("dataset_type", "synthetic")
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
    if dataset_type == "synthetic":
        required += [
            "n",
            "p",
            "support_size",
            "snr",
            "graph_family",
            "support_type",
        ]
    else:
        required += ["dataset_name"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config fields: {missing}")

    graph_family = cfg.get("graph_family")
    if dataset_type == "real":
        if cfg.get("corruption_type") or cfg.get("corruption_level"):
            raise ValueError("Corruption not supported for real-data configs")
        return

    if graph_family == "grid":
        rows = cfg.get("grid_rows")
        cols = cfg.get("grid_cols")
        if (rows is None) != (cols is None):
            raise ValueError("grid_rows and grid_cols must both be provided")
        if rows is None and cols is None:
            p = int(cfg["p"])
            side = int(np.sqrt(p))
            if side * side != p:
                raise ValueError("p must be a perfect square to infer grid shape")

    corruption_type = cfg.get("corruption_type")
    corruption_level = cfg.get("corruption_level")
    if corruption_type is not None:
        if corruption_level is None:
            raise ValueError("corruption_level required when corruption_type is set")
    if corruption_level is not None:
        if not (0.0 <= float(corruption_level) <= 1.0):
            raise ValueError("corruption_level must be in [0, 1]")


def _build_graph_artifact(cfg: Dict[str, Any]) -> Dict[str, Any]:
    graph_family = cfg["graph_family"]
    if graph_family == "chain":
        return chain_graph_artifact(cfg["p"])
    if graph_family == "grid":
        rows = cfg.get("grid_rows")
        cols = cfg.get("grid_cols")
        if rows is None and cols is None:
            rows, cols = infer_grid_shape(cfg["p"])
        return grid_graph_artifact(int(rows), int(cols))
    if graph_family == "sbm":
        rng = np.random.default_rng(cfg["seed"] + 123)
        blocks = int(cfg.get("sbm_blocks", 3))
        p_in = float(cfg.get("sbm_p_in", 0.2))
        p_out = float(cfg.get("sbm_p_out", 0.02))
        block_sizes = cfg.get("sbm_block_sizes")
        if block_sizes is not None:
            block_sizes = [int(x) for x in block_sizes]
        L, W, labels = sbm_graph_laplacian(
            cfg["p"], blocks, p_in, p_out, rng, block_sizes=block_sizes
        )
        return {
            "adjacency": W,
            "laplacian": L,
            "family": "sbm",
            "metadata": {
                "sbm_blocks": blocks,
                "sbm_labels": labels.tolist(),
                "sbm_block_sizes": block_sizes if block_sizes is not None else None,
            },
        }
    raise ValueError("graph_family must be 'chain', 'grid', or 'sbm'")


def _maybe_corrupt_graph(
    artifact: Dict[str, Any],
    cfg: Dict[str, Any],
    corruption_type: Optional[str] = None,
    corruption_level: Optional[float] = None,
) -> Dict[str, Any]:
    if corruption_type is None:
        corruption_type = cfg.get("corruption_type")
    if corruption_level is None:
        corruption_level = cfg.get("corruption_level")
    if corruption_type is None or corruption_level is None:
        return artifact
    rng = np.random.default_rng(cfg["seed"] + 987)
    return corrupt_graph(artifact, str(corruption_type), float(corruption_level), rng)


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
    try:
        dataset_type = cfg.get("dataset_type", "synthetic")
        if dataset_type == "real":
            data_dir = Path(cfg.get("real_data_dir", "data/real"))
            max_genes = int(cfg.get("real_max_genes", 500))
            score_threshold = int(cfg.get("string_score_threshold", 700))
            subsample_frac = cfg.get("real_subsample_frac")
            dataset = load_tcga_brca_string(
                data_dir=data_dir,
                max_genes=max_genes,
                score_threshold=score_threshold,
                seed=cfg.get("seed"),
                subsample_frac=subsample_frac,
            )
            data_graph_artifact = {
                "adjacency": dataset.W,
                "laplacian": dataset.L,
                "family": "real",
                "metadata": dataset.metadata,
            }
            prior_graph_artifact = data_graph_artifact
            L = prior_graph_artifact["laplacian"]
            Sigma_hat = dataset.Sigma_hat
            eigvals, V = _pca_top_r(Sigma_hat, cfg["r"])
            A_pca = V
            B_pca = V
            B_aligned_pca = B_pca
            perm_pca = None
            signs_pca = None
            support_pca = {}
        else:
            data_graph_artifact = _build_graph_artifact(cfg)
            prior_graph_artifact = _maybe_corrupt_graph(data_graph_artifact, cfg)
            L = prior_graph_artifact["laplacian"]
            dataset = generate_dataset(
                n=cfg["n"],
                p=cfg["p"],
                r=cfg["r"],
                support_size=cfg["support_size"],
                support_type=cfg["support_type"],
                L=data_graph_artifact["laplacian"],
                snr=cfg["snr"],
                signal_eigs=cfg.get("signal_eigs"),
                seed=cfg["seed"],
                adjacency=data_graph_artifact.get("adjacency"),
                graph_metadata=data_graph_artifact.get("metadata"),
                decoy_count=int(cfg.get("decoy_count", 0)),
                decoy_variance_factor=float(cfg.get("decoy_variance_factor", 0.0)),
            )

            Sigma_hat = dataset.Sigma_hat
            eigvals, V = _pca_top_r(Sigma_hat, cfg["r"])
            A_pca = V
            B_pca = V

            B_aligned_pca, perm_pca, signs_pca = _alignment(
                A_pca, B_pca, dataset.true_loadings
            )
            support_pca = support_metrics(B_aligned_pca, dataset.true_supports)

        pca_eval = {
            "method_name": "PCA",
            "objective_terms": objective_terms(
                A_pca,
                B_pca,
                Sigma_hat,
                prior_graph_artifact["laplacian"],
                cfg["lambda1"],
                cfg["lambda2"],
                cfg["rho"],
            ),
            "explained_variance": float(np.sum(eigvals)),
            "sparsity_fraction": sparsity_fraction(B_pca),
            "orthogonality_error": orthogonality_error(A_pca),
            "laplacian_energy": laplacian_energy(
                B_pca, prior_graph_artifact["laplacian"]
            ),
            "support_metrics": support_pca,
            "support_metrics_note": "Dense baseline diagnostics; interpret with caution.",
            "graph_smoothness_raw_trueL": graph_smoothness_raw(
                B_pca, data_graph_artifact["laplacian"]
            ),
            "graph_smoothness_norm_trueL": graph_smoothness_norm(
                B_pca, data_graph_artifact["laplacian"]
            ),
            "shared_explained_variance": explained_variance(A_pca, Sigma_hat),
        }
        pca_union_mask = np.any(np.abs(B_aligned_pca) > 1e-8, axis=1)
        pca_eval["support_connectivity_union"] = support_connectivity(
            pca_union_mask, data_graph_artifact["laplacian"]
        )
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
        if dataset_type == "real":
            rng = np.random.default_rng(cfg["seed"])
            init_noise = float(cfg.get("real_init_noise", 0.0))
            if init_noise > 0.0:
                A0 = A0 + init_noise * rng.standard_normal(A0.shape)
                A0 = orthonormalize(A0)
        amanpg_result = solve_amanpg(A0, Sigma_hat, amanpg_cfg)
        amanpg_B = amanpg_result.A
        if dataset_type == "real":
            amanpg_aligned = amanpg_B
            amanpg_perm = None
            amanpg_signs = None
            amanpg_support = {}
        else:
            amanpg_aligned, amanpg_perm, amanpg_signs = _alignment(
                amanpg_result.A, amanpg_B, dataset.true_loadings
            )
            amanpg_support = support_metrics(amanpg_aligned, dataset.true_supports)
        amanpg_eval = {
            "method_name": "A-ManPG",
            "objective_terms": objective_terms(
                amanpg_result.A,
                amanpg_B,
                Sigma_hat,
                prior_graph_artifact["laplacian"],
                cfg["lambda1"],
                0.0,
                0.0,
            ),
            "sparsity_fraction": sparsity_fraction(amanpg_B),
            "orthogonality_error": orthogonality_error(amanpg_result.A),
            "laplacian_energy": laplacian_energy(
                amanpg_B, prior_graph_artifact["laplacian"]
            ),
            "support_metrics": amanpg_support,
            "graph_smoothness_raw_trueL": graph_smoothness_raw(
                amanpg_B, data_graph_artifact["laplacian"]
            ),
            "graph_smoothness_norm_trueL": graph_smoothness_norm(
                amanpg_B, data_graph_artifact["laplacian"]
            ),
            "shared_explained_variance": explained_variance(
                orthonormalize(amanpg_B), Sigma_hat
            ),
        }
        amanpg_union_mask = np.any(np.abs(amanpg_aligned) > 1e-8, axis=1)
        amanpg_eval["support_connectivity_union"] = support_connectivity(
            amanpg_union_mask, data_graph_artifact["laplacian"]
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

        if dataset_type == "real":
            B_aligned = result.B
            perm = None
            signs = None
            support = {}
        else:
            B_aligned, perm, signs = _alignment(
                result.A, result.B, dataset.true_loadings
            )
            support = support_metrics(B_aligned, dataset.true_supports)

        # Sparse PCA baseline (no graph regularization)
        L_zero = np.zeros_like(prior_graph_artifact["laplacian"])
        spca_cfg = SolverConfig(
            lambda1=cfg["lambda1"],
            lambda2=0.0,
            rho=cfg["rho"],
            eta_A=cfg["eta_A"],
            max_iters=cfg["max_iters"],
            tol_obj=cfg["tol_obj"],
            tol_gap=cfg["tol_gap"],
            tol_orth=cfg["tol_orth"],
        )
        spca_result = solve(A0, B0, Sigma_hat, L_zero, spca_cfg)
        if dataset_type == "real":
            spca_aligned = spca_result.B
            spca_perm = None
            spca_signs = None
            spca_support = {}
        else:
            spca_aligned, spca_perm, spca_signs = _alignment(
                spca_result.A, spca_result.B, dataset.true_loadings
            )
            spca_support = support_metrics(spca_aligned, dataset.true_supports)

        proposed_eval = {
            "method_name": "Proposed",
            "objective_terms": objective_terms(
                result.A,
                result.B,
                Sigma_hat,
                prior_graph_artifact["laplacian"],
                cfg["lambda1"],
                cfg["lambda2"],
                cfg["rho"],
            ),
            "sparsity_fraction": sparsity_fraction(result.B),
            "orthogonality_error": orthogonality_error(result.A),
            "laplacian_energy": laplacian_energy(
                result.B, prior_graph_artifact["laplacian"]
            ),
            "support_metrics": support,
            "graph_smoothness_raw_trueL": graph_smoothness_raw(
                result.B, data_graph_artifact["laplacian"]
            ),
            "graph_smoothness_norm_trueL": graph_smoothness_norm(
                result.B, data_graph_artifact["laplacian"]
            ),
            "shared_explained_variance": explained_variance(
                orthonormalize(result.B), Sigma_hat
            ),
        }
        proposed_union_mask = np.any(np.abs(B_aligned) > 1e-8, axis=1)
        proposed_eval["support_connectivity_union"] = support_connectivity(
            proposed_union_mask, data_graph_artifact["laplacian"]
        )
        metrics_out["Proposed"] = proposed_eval

        spca_eval = {
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
            "graph_smoothness_raw_trueL": graph_smoothness_raw(
                spca_result.B, data_graph_artifact["laplacian"]
            ),
            "graph_smoothness_norm_trueL": graph_smoothness_norm(
                spca_result.B, data_graph_artifact["laplacian"]
            ),
            "shared_explained_variance": explained_variance(
                orthonormalize(spca_result.B), Sigma_hat
            ),
        }
        spca_union_mask = np.any(np.abs(spca_aligned) > 1e-8, axis=1)
        spca_eval["support_connectivity_union"] = support_connectivity(
            spca_union_mask, data_graph_artifact["laplacian"]
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
            "pca_A": A_pca,
            "pca_B": B_pca,
            "Sigma_hat": Sigma_hat,
            "L": data_graph_artifact["laplacian"],
            **{f"amanpg_history_{k}": v for k, v in amanpg_result.history.items()},
            **{f"history_{k}": v for k, v in history_arrays.items()},
        }
        if perm is not None:
            arrays["matching_perm"] = perm
        if signs is not None:
            arrays["matching_signs"] = signs
        if perm_pca is not None:
            arrays["pca_matching_perm"] = perm_pca
        if signs_pca is not None:
            arrays["pca_matching_signs"] = signs_pca
        if dataset_type != "real":
            arrays.update(
                {
                    "Sigma_true": dataset.Sigma_true,
                    "true_loadings": dataset.true_loadings,
                    "true_support": np.array(dataset.true_supports, dtype=object),
                    "amanpg_matching_perm": amanpg_perm,
                    "amanpg_matching_signs": amanpg_signs,
                }
            )

        _write_convergence_plots(output_dir / "plots", history_arrays)

    except Exception as exc:
        status = "failed"
        failure_reason = f"{exc}"
        diagnostics = traceback.format_exc()

    end_time = time.time()
    runtime = end_time - start_time

    manifest = {
        "status": status,
        "failure_reason": failure_reason,
        "runtime_sec": runtime,
        "config_hash": _hash_config(cfg),
        "dataset_hash": _hash_array(dataset.Sigma_true)
        if dataset is not None and hasattr(dataset, "Sigma_true")
        else _hash_array(dataset.Sigma_hat)
        if dataset is not None and hasattr(dataset, "Sigma_hat")
        else None,
        "git_hash": _git_hash(),
        "tuning_note": "Defaults rho=5.0 and eta_A=0.05 selected for improved coupling behavior on r=1 and r=3 smoke runs without harming recovery.",
        **_env_info(),
    }

    artifacts_meta = {
        "method_name": "Proposed",
        "baseline_methods": ["PCA", "A-ManPG", "SparseNoGraph"],
        "status": status,
        "failure_reason": failure_reason,
        "manifest": manifest,
        "config": cfg,
        "dataset_metadata": dataset.metadata if dataset is not None else None,
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
