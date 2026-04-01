from __future__ import annotations

import csv
import gzip
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


XENA_TCGA_HUB = "https://tcga.xenahubs.net"
XENA_BRCA_HISEQV2 = f"{XENA_TCGA_HUB}/download/TCGA.BRCA.sampleMap/HiSeqV2.gz"

STRING_BASE = "https://stringdb-downloads.org/download"
STRING_VERSION = "11.5"
STRING_LINKS = (
    f"{STRING_BASE}/protein.links.v{STRING_VERSION}/9606.protein.links.v{STRING_VERSION}.txt.gz"
)
STRING_ALIASES = (
    f"{STRING_BASE}/protein.aliases.v{STRING_VERSION}/9606.protein.aliases.v{STRING_VERSION}.txt.gz"
)


@dataclass
class RealDataset:
    X: np.ndarray
    L: np.ndarray
    gene_symbols: List[str]
    sample_ids: List[str]
    metadata: Dict[str, object]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest)


def _normalize_gene_symbols(index: Iterable[str]) -> List[str]:
    symbols = []
    for gene in index:
        if isinstance(gene, str) and "|" in gene:
            symbols.append(gene.split("|")[0])
        else:
            symbols.append(str(gene))
    return symbols


def load_tcga_brca_expression(
    cache_dir: Path,
    max_genes: int = 2000,
    seed: int = 0,
    url: str = XENA_BRCA_HISEQV2,
) -> Tuple[np.ndarray, List[str], List[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(url).name
    _download(url, dest)

    df = pd.read_csv(dest, sep="\t", compression="gzip", index_col=0)
    df.index = _normalize_gene_symbols(df.index)

    # Drop duplicated gene symbols by keeping the row with highest variance
    df["_var"] = df.var(axis=1, numeric_only=True)
    df = df.sort_values("_var", ascending=False)
    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(columns=["_var"])

    if max_genes is not None and max_genes > 0 and df.shape[0] > max_genes:
        df = df.head(max_genes)

    # transpose to samples x genes
    X = df.to_numpy(dtype=float).T
    sample_ids = df.columns.tolist()
    gene_symbols = df.index.tolist()

    rng = np.random.default_rng(seed)
    order = np.arange(X.shape[0])
    rng.shuffle(order)
    X = X[order]
    sample_ids = [sample_ids[i] for i in order]

    # center columns
    X = X - X.mean(axis=0, keepdims=True)

    return X, gene_symbols, sample_ids


def _iter_alias_rows(path: Path) -> Iterable[Tuple[str, str, str]]:
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            yield row[0], row[1], row[2]


def _iter_link_rows(path: Path) -> Iterable[Tuple[str, str, int]]:
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if len(row) < 3 or row[0].startswith("protein1"):
                continue
            yield row[0], row[1], int(row[2])


def load_string_ppi(
    cache_dir: Path,
    gene_symbols: List[str],
    score_threshold: int = 700,
    version: str = STRING_VERSION,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    links_url = STRING_LINKS.replace(STRING_VERSION, version)
    aliases_url = STRING_ALIASES.replace(STRING_VERSION, version)

    links_path = cache_dir / Path(links_url).name
    aliases_path = cache_dir / Path(aliases_url).name
    _download(links_url, links_path)
    _download(aliases_url, aliases_path)

    gene_set = set(gene_symbols)
    protein_to_gene: Dict[str, str] = {}
    for protein, alias, source in _iter_alias_rows(aliases_path):
        if protein in protein_to_gene:
            continue
        if alias not in gene_set:
            continue
        if any(tag in source for tag in ["Gene_Name", "Ensembl_HGNC", "HGNC"]):
            protein_to_gene[protein] = alias

    index = {g: i for i, g in enumerate(gene_symbols)}
    p = len(gene_symbols)
    W = np.zeros((p, p), dtype=float)

    for p1, p2, score in _iter_link_rows(links_path):
        if score < score_threshold:
            continue
        g1 = protein_to_gene.get(p1)
        g2 = protein_to_gene.get(p2)
        if g1 is None or g2 is None:
            continue
        if g1 == g2:
            continue
        i = index[g1]
        j = index[g2]
        weight = score / 1000.0
        W[i, j] = max(W[i, j], weight)
        W[j, i] = max(W[j, i], weight)

    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def load_tcga_brca_with_string(
    cache_dir: Path,
    max_genes: int = 2000,
    seed: int = 0,
    score_threshold: int = 700,
    string_version: str = STRING_VERSION,
) -> RealDataset:
    X, genes, samples = load_tcga_brca_expression(
        cache_dir, max_genes=max_genes, seed=seed
    )
    L = load_string_ppi(
        cache_dir,
        genes,
        score_threshold=score_threshold,
        version=string_version,
    )
    metadata = {
        "dataset": "TCGA-BRCA",
        "source": "UCSC Xena TCGA Hub: HiSeqV2",
        "string_version": string_version,
        "string_score_threshold": score_threshold,
        "max_genes": max_genes,
    }
    return RealDataset(X=X, L=L, gene_symbols=genes, sample_ids=samples, metadata=metadata)
