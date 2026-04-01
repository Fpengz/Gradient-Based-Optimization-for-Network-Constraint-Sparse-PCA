from pathlib import Path

import numpy as np

from grpca_gd.datasets.artifacts import DatasetArtifact, load_artifact, save_artifact


def test_artifact_roundtrip(tmp_path: Path) -> None:
    X = np.eye(3)
    L = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    artifact = DatasetArtifact(
        artifact_id="mnist_grid_28x28_v1",
        artifact_version="v1",
        dataset="mnist",
        graph_family="grid",
        data_source="openml",
        prep_config_hash="abc123",
        eval_protocol_id="default",
        X=X,
        L=L,
        metadata={"resolution": 28},
    )
    out_dir = tmp_path / "artifact"
    save_artifact(artifact, out_dir)

    loaded = load_artifact(out_dir)
    assert loaded.artifact_id == "mnist_grid_28x28_v1"
    assert loaded.graph_family == "grid"
    assert np.allclose(loaded.X, X)
    assert np.allclose(loaded.L, L)
    assert loaded.metadata["resolution"] == 28
