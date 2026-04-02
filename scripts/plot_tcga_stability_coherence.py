from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grpca_gd.datasets.artifacts import load_artifact
from grpca_gd.metrics import orthonormalize


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures" / "realdata"
RESULTS = ROOT / "results"


def _load_tcga_B(seed: int, lambda2_tag: str) -> np.ndarray:
    run_dir = OUTPUTS / "realdata" / "tcga_brca" / f"seed{seed}" / f"lambda2_{lambda2_tag}"
    arrays = np.load(run_dir / "artifacts.npz")
    return arrays["B"]


def _pairwise_indices(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def _support_union(B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.any(np.abs(B) > eps, axis=1)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union > 0 else 0.0


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 0:
        return 0.0
    return float(np.dot(u, v) / denom)


def _subspace_similarity(B1: np.ndarray, B2: np.ndarray) -> float:
    Q1 = orthonormalize(B1)
    Q2 = orthonormalize(B2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(s**2))


def _adjacency_from_laplacian(L: np.ndarray) -> np.ndarray:
    W = -L.copy()
    np.fill_diagonal(W, 0.0)
    return (W > 0).astype(int)


def _coherence_metrics(mask: np.ndarray, adj: np.ndarray) -> Dict[str, float]:
    idx = np.where(mask)[0]
    if idx.size <= 1:
        return {"density": 0.0, "clustering": 0.0}
    sub = adj[np.ix_(idx, idx)]
    k = sub.shape[0]
    edges = float(np.sum(sub) / 2.0)
    density = edges / max(1.0, k * (k - 1) / 2.0)

    deg = sub.sum(axis=1)
    avg_degree = float(np.mean(deg)) if deg.size else 0.0
    return {
        "density": float(density),
        "avg_degree": avg_degree,
    }


def _degree_matched_sample(
    degrees: np.ndarray,
    target_mask: np.ndarray,
    rng: np.random.Generator,
    n_bins: int = 5,
) -> np.ndarray:
    n = degrees.size
    target_idx = np.where(target_mask)[0]
    if target_idx.size == 0:
        return np.zeros(n, dtype=bool)
    bins = np.quantile(degrees, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-6
    sampled = []
    remaining = set(range(n))
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = np.where((degrees >= lo) & (degrees < hi))[0]
        in_bin = [i for i in in_bin if i in remaining]
        target_count = int(np.sum((degrees[target_idx] >= lo) & (degrees[target_idx] < hi)))
        if target_count <= 0:
            continue
        if len(in_bin) < target_count:
            sample = rng.choice(n, size=target_count, replace=False).tolist()
        else:
            sample = rng.choice(in_bin, size=target_count, replace=False).tolist()
        sampled.extend(sample)
        for s in sample:
            remaining.discard(s)
    if len(sampled) < target_idx.size:
        extra = rng.choice(list(remaining), size=target_idx.size - len(sampled), replace=False)
        sampled.extend(extra.tolist())
    mask = np.zeros(n, dtype=bool)
    mask[np.array(sampled, dtype=int)] = True
    return mask


def main() -> None:
    lambda2_tag = "0p10"
    seeds = [0, 1, 2]
    B_list = [_load_tcga_B(seed, lambda2_tag) for seed in seeds]
    supports = [_support_union(B) for B in B_list]

    pairs = _pairwise_indices(len(B_list))
    jaccards = []
    cosines = []
    subspaces = []
    for i, j in pairs:
        jaccards.append(_jaccard(supports[i], supports[j]))
        cosines.append(_cosine(B_list[i].ravel(), B_list[j].ravel()))
        subspaces.append(_subspace_similarity(B_list[i], B_list[j]))

    stability = {
        "stability_jaccard": float(np.mean(jaccards)),
        "stability_cosine": float(np.mean(cosines)),
        "stability_subspace": float(np.mean(subspaces)),
    }

    artifact = load_artifact(ROOT / "artifacts" / "real" / "tcga_brca_string")
    adj = _adjacency_from_laplacian(artifact.L)
    degrees = adj.sum(axis=1)

    rng = np.random.default_rng(42)
    n_null = 20
    observed = []
    uniform_null = []
    degree_null = []
    for mask in supports:
        observed.append(_coherence_metrics(mask, adj))
        uniform_metrics = []
        degree_metrics = []
        for _ in range(n_null):
            sample_uniform = np.zeros_like(mask)
            idx = rng.choice(len(mask), size=int(mask.sum()), replace=False)
            sample_uniform[idx] = True
            uniform_metrics.append(_coherence_metrics(sample_uniform, adj))
            sample_degree = _degree_matched_sample(degrees, mask, rng)
            degree_metrics.append(_coherence_metrics(sample_degree, adj))
        uniform_null.append(
            {k: float(np.mean([m[k] for m in uniform_metrics])) for k in uniform_metrics[0]}
        )
        degree_null.append(
            {k: float(np.mean([m[k] for m in degree_metrics])) for k in degree_metrics[0]}
        )

    def _avg_dict(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    observed_avg = _avg_dict(observed)
    uniform_avg = _avg_dict(uniform_null)
    degree_avg = _avg_dict(degree_null)

    summary = {
        "lambda2": 0.10,
        **stability,
        **{f"observed_{k}": v for k, v in observed_avg.items()},
        **{f"uniform_{k}": v for k, v in uniform_avg.items()},
        **{f"degree_{k}": v for k, v in degree_avg.items()},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "tcga_stability_coherence.csv"
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].bar(["Jaccard", "Cosine", "Subspace"], [
        summary["stability_jaccard"],
        summary["stability_cosine"],
        summary["stability_subspace"],
    ], color="#4C72B0")
    axes[0].set_title("TCGA Stability (Proposed, $\lambda_2=0.1$)")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Similarity")

    metrics = ["density", "avg_degree"]
    x = np.arange(len(metrics))
    width = 0.25
    axes[1].bar(x - width, [observed_avg[m] for m in metrics], width, label="Observed")
    axes[1].bar(x, [uniform_avg[m] for m in metrics], width, label="Uniform null")
    axes[1].bar(x + width, [degree_avg[m] for m in metrics], width, label="Degree-matched null")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Density", "Avg. degree"])
    axes[1].set_title("TCGA Coherence vs Nulls")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "tcga_stability_coherence.png", dpi=200)


if __name__ == "__main__":
    main()
