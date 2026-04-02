from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpca_gd.datasets.artifacts import load_artifact


def test_prepare_sp500_artifact(tmp_path: Path) -> None:
    from scripts.prepare_realdata_sp500 import prepare_sp500_artifact

    out_dir = tmp_path / "sp500_artifact"
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 10.5, 11.0],
            "BBB": [20.0, 19.5, 19.0],
            "CCC": [30.0, 30.2, 30.4],
        }
    )
    prepare_sp500_artifact(out_dir, prices=prices, corr_threshold=0.1)
    artifact = load_artifact(out_dir)
    assert artifact.dataset == "sp500"
    assert artifact.X.shape[1] == 3
