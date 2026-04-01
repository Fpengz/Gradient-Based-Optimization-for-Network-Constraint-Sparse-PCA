from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json
import numpy as np


@dataclass
class DatasetArtifact:
    artifact_id: str
    artifact_version: str
    dataset: str
    graph_family: str
    data_source: str
    prep_config_hash: str
    eval_protocol_id: str
    X: np.ndarray
    L: np.ndarray
    metadata: Dict[str, Any]


def save_artifact(artifact: DatasetArtifact, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "data.npz", X=artifact.X, L=artifact.L)
    meta = {
        "artifact_id": artifact.artifact_id,
        "artifact_version": artifact.artifact_version,
        "dataset": artifact.dataset,
        "graph_family": artifact.graph_family,
        "data_source": artifact.data_source,
        "prep_config_hash": artifact.prep_config_hash,
        "eval_protocol_id": artifact.eval_protocol_id,
        "metadata": artifact.metadata,
    }
    (out_dir / "artifact.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_artifact(out_dir: Path) -> DatasetArtifact:
    payload = json.loads((out_dir / "artifact.json").read_text(encoding="utf-8"))
    arrays = np.load(out_dir / "data.npz")
    return DatasetArtifact(
        artifact_id=payload["artifact_id"],
        artifact_version=payload["artifact_version"],
        dataset=payload["dataset"],
        graph_family=payload["graph_family"],
        data_source=payload["data_source"],
        prep_config_hash=payload["prep_config_hash"],
        eval_protocol_id=payload["eval_protocol_id"],
        X=arrays["X"],
        L=arrays["L"],
        metadata=payload.get("metadata", {}),
    )
