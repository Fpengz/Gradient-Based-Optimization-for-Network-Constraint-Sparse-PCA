import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpca_gd.datasets.artifacts import load_artifact


def test_prepare_mnist_artifact(tmp_path: Path) -> None:
    import scripts.prepare_realdata_mnist as prepare_realdata_mnist

    fake_mnist = SimpleNamespace(
        data=SimpleNamespace(
            to_numpy=lambda: np.arange(100 * 28 * 28, dtype=np.float64).reshape(100, 28 * 28)
        )
    )

    out_dir = tmp_path / "mnist_artifact"
    with patch.object(prepare_realdata_mnist, "fetch_openml", return_value=fake_mnist):
        prepare_realdata_mnist.prepare_mnist_artifact(out_dir, max_samples=50, resolution=28)
    artifact = load_artifact(out_dir)
    assert artifact.dataset == "mnist"
    assert artifact.X.shape[0] == 50
    assert artifact.X.shape[1] == 28 * 28
