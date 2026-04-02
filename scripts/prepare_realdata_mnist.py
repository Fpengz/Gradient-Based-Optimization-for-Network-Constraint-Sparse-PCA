from __future__ import annotations

from pathlib import Path
import hashlib

from sklearn.datasets import fetch_openml

from grpca_gd.datasets.artifacts import DatasetArtifact, save_artifact
from grpca_gd.synthetic.graphs import grid_graph_laplacian


def _hash_config(payload: dict) -> str:
    data = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def prepare_mnist_artifact(out_dir: Path, max_samples: int, resolution: int) -> None:
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data.to_numpy()[:max_samples] / 255.0
    X = X[:, : resolution * resolution]

    L, _ = grid_graph_laplacian(resolution, resolution)

    prep_cfg = {"max_samples": max_samples, "resolution": resolution}
    artifact = DatasetArtifact(
        artifact_id=f"mnist_grid_{resolution}x{resolution}_v1",
        artifact_version="v1",
        dataset="mnist",
        graph_family="grid",
        data_source="openml",
        prep_config_hash=_hash_config(prep_cfg),
        eval_protocol_id="default",
        X=X,
        L=L,
        metadata={"resolution": resolution},
    )
    save_artifact(artifact, out_dir)
