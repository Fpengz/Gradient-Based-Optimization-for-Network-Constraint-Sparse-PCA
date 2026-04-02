from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class RealDataset:
    X: np.ndarray
    Sigma_hat: np.ndarray
    L: np.ndarray
    W: np.ndarray
    genes: List[str]
    samples: List[str]
    metadata: Dict[str, object]


def _load_expression(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, sep="\t", compression="gzip", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    samples = list(df.columns)
    return df, samples


def _load_gene_mapping(alias_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    protein_to_gene: Dict[str, str] = {}
    gene_to_protein: Dict[str, str] = {}
    with gzip.open(alias_path, "rt") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            protein, alias, source = parts
            if source != "Ensembl_HGNC":
                continue
            if protein not in protein_to_gene:
                protein_to_gene[protein] = alias
            if alias not in gene_to_protein:
                gene_to_protein[alias] = protein
    return protein_to_gene, gene_to_protein


def _select_genes(
    expr: pd.DataFrame, gene_to_protein: Dict[str, str], max_genes: int
) -> List[str]:
    candidates = [g for g in expr.index if g in gene_to_protein]
    if not candidates:
        raise ValueError("No overlap between expression genes and STRING mapping.")
    expr = expr.loc[candidates]
    variances = expr.var(axis=1)
    top = variances.sort_values(ascending=False).head(max_genes)
    return list(top.index)


def _build_adjacency(
    links_path: Path,
    protein_to_gene: Dict[str, str],
    gene_index: Dict[str, int],
    score_threshold: int,
) -> np.ndarray:
    p = len(gene_index)
    W = np.zeros((p, p), dtype=float)
    selected_genes = set(gene_index)
    selected_proteins = {
        prot for prot, gene in protein_to_gene.items() if gene in selected_genes
    }
    with gzip.open(links_path, "rt") as f:
        header = next(f, None)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            prot1, prot2, score_str = parts[:3]
            if prot1 not in selected_proteins or prot2 not in selected_proteins:
                continue
            score = int(score_str)
            if score < score_threshold:
                continue
            gene1 = protein_to_gene.get(prot1)
            gene2 = protein_to_gene.get(prot2)
            if gene1 is None or gene2 is None or gene1 == gene2:
                continue
            i = gene_index.get(gene1)
            j = gene_index.get(gene2)
            if i is None or j is None:
                continue
            weight = score / 1000.0
            if weight > W[i, j]:
                W[i, j] = weight
                W[j, i] = weight
    return W


def load_tcga_brca_string(
    data_dir: Path,
    max_genes: int = 500,
    score_threshold: int = 700,
    seed: int | None = None,
    subsample_frac: float | None = None,
) -> RealDataset:
    expr_path = data_dir / "HiSeqV2.gz"
    aliases_path = data_dir / "9606.protein.aliases.v11.5.txt.gz"
    links_path = data_dir / "9606.protein.links.v11.5.txt.gz"
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"tcga_brca_string_p{max_genes}_thr{score_threshold}.npz"

    use_cache = subsample_frac is None or subsample_frac >= 1.0
    if cache_path.exists() and use_cache:
        cached = np.load(cache_path, allow_pickle=True)
        X = cached["X"]
        L = cached["L"]
        W = cached["W"]
        genes = cached["genes"].tolist()
        samples = cached["samples"].tolist()
        Sigma_hat = cached["Sigma_hat"]
        return RealDataset(
            X=X,
            Sigma_hat=Sigma_hat,
            L=L,
            W=W,
            genes=genes,
            samples=samples,
            metadata={
                "dataset": "TCGA-BRCA",
                "graph": "STRING",
                "max_genes": max_genes,
                "score_threshold": score_threshold,
                "cached": True,
            },
        )

    expr, samples = _load_expression(expr_path)
    protein_to_gene, gene_to_protein = _load_gene_mapping(aliases_path)
    selected_genes = _select_genes(expr, gene_to_protein, max_genes)
    expr = expr.loc[selected_genes]
    genes = list(expr.index)
    gene_index = {g: i for i, g in enumerate(genes)}
    W = _build_adjacency(links_path, protein_to_gene, gene_index, score_threshold)
    D = np.diag(W.sum(axis=1))
    L = D - W

    X = expr.to_numpy(dtype=float).T
    if subsample_frac is not None and subsample_frac < 1.0:
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        k = max(1, int(round(n * subsample_frac)))
        idx = rng.choice(n, size=k, replace=False)
        X = X[idx]
        samples = [samples[i] for i in idx]
    X = X - X.mean(axis=0, keepdims=True)
    Sigma_hat = (X.T @ X) / max(1, X.shape[0])

    if use_cache:
        np.savez(
            cache_path,
            X=X,
            Sigma_hat=Sigma_hat,
            L=L,
            W=W,
            genes=np.array(genes, dtype=object),
            samples=np.array(samples, dtype=object),
        )

    return RealDataset(
        X=X,
        Sigma_hat=Sigma_hat,
        L=L,
        W=W,
        genes=genes,
        samples=samples,
        metadata={
            "dataset": "TCGA-BRCA",
            "graph": "STRING",
            "max_genes": max_genes,
            "score_threshold": score_threshold,
            "cached": use_cache,
            "subsample_frac": subsample_frac,
        },
    )
