from pathlib import Path

import numpy as np
import pytest

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


def test_load_artifact_closes_npz_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "artifact.json").write_text(
        """
        {
          "artifact_id": "a",
          "artifact_version": "v1",
          "dataset": "mnist",
          "graph_family": "grid",
          "data_source": "openml",
          "prep_config_hash": "abc123",
          "eval_protocol_id": "default",
          "metadata": {}
        }
        """.strip(),
        encoding="utf-8",
    )

    exited = False

    class FakeArrays:
        def __getitem__(self, key: str) -> np.ndarray:
            return np.array([1.0]) if key == "X" else np.array([2.0])

        def __enter__(self) -> "FakeArrays":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            nonlocal exited
            exited = True

    def fake_load(path: Path) -> FakeArrays:
        assert path == artifact_dir / "data.npz"
        return FakeArrays()

    monkeypatch.setattr("grpca_gd.datasets.artifacts.np.load", fake_load)

    loaded = load_artifact(artifact_dir)

    assert exited is True
    assert np.allclose(loaded.X, np.array([1.0]))
    assert np.allclose(loaded.L, np.array([2.0]))
