import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpca_gd.datasets.artifacts import load_artifact


def test_prepare_mnist_artifact(tmp_path: Path) -> None:
    from scripts.prepare_realdata_mnist import prepare_mnist_artifact

    out_dir = tmp_path / "mnist_artifact"
    prepare_mnist_artifact(out_dir, max_samples=50, resolution=28)
    artifact = load_artifact(out_dir)
    assert artifact.dataset == "mnist"
    assert artifact.X.shape[0] == 50
    assert artifact.X.shape[1] == 28 * 28
