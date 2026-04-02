from __future__ import annotations

from pathlib import Path
import hashlib

from grpca_gd.datasets.artifacts import DatasetArtifact, save_artifact
from grpca_gd.real_data import load_tcga_brca_string


def _hash_config(payload: dict) -> str:
    data = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def prepare_tcga_brca_artifact(
    out_dir: Path,
    data_dir: Path,
    max_genes: int,
    score_threshold: int,
) -> None:
    dataset = load_tcga_brca_string(
        data_dir=data_dir,
        max_genes=max_genes,
        score_threshold=score_threshold,
    )

    prep_cfg = {
        "max_genes": max_genes,
        "score_threshold": score_threshold,
    }
    artifact = DatasetArtifact(
        artifact_id=f"tcga_brca_string_p{max_genes}_thr{score_threshold}_v1",
        artifact_version="v1",
        dataset="tcga_brca",
        graph_family="string",
        data_source="ucsc_xena+string",
        prep_config_hash=_hash_config(prep_cfg),
        eval_protocol_id="default",
        X=dataset.X,
        L=dataset.L,
        metadata={
            "max_genes": max_genes,
            "score_threshold": score_threshold,
            "genes": dataset.genes,
        },
    )
    save_artifact(artifact, out_dir)
