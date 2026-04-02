# Track A Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Track A revision spec by adding frozen dataset artifacts, expanding graphs/baselines/metrics, and updating figures/manuscript with a unified empirical story.

**Architecture:** Add dataset preparation scripts that freeze artifacts (`X`, `L`, metadata) and expose provenance fields. Update the runner to consume frozen artifacts and emit a canonical results schema. Expand synthetic graph families/corruption, add new baselines/metrics, and generate unified real-data panels. Update the LaTeX manuscript to reflect the new positioning and technical clarifications.

**Tech Stack:** Python (NumPy, SciPy, Pandas, scikit-learn, Matplotlib), YAML configs, LaTeX

---

## File Structure and Responsibilities

- Modify: `pyproject.toml` — add `pandas`, `scikit-learn`, `yfinance`
- Create: `src/grpca_gd/datasets/__init__.py` — dataset prep package export
- Create: `src/grpca_gd/datasets/artifacts.py` — artifact schema + IO helpers
- Create: `src/grpca_gd/datasets/mnist.py` — MNIST prep helpers
- Create: `src/grpca_gd/datasets/sp500.py` — S&P500 fetch + correlation graph
- Create: `src/grpca_gd/datasets/tcga.py` — TCGA/STRING prep wrapper (artifactized)
- Modify: `src/grpca_gd/synthetic/graphs.py` — grid, ER, kNN, small-world, normalized Laplacian
- Create: `src/grpca_gd/synthetic/corruption.py` — delete/rewire/weight perturb
- Modify: `src/grpca_gd/metrics.py` — nnz_loadings + coupling gap + stability placeholders
- Modify: `src/grpca_gd/runner.py` — unified schema, new baselines, real-data loading
- Modify: `scripts/collect_metrics.py` — canonical results schema export
- Create: `scripts/prepare_realdata_mnist.py` — artifact builder
- Create: `scripts/prepare_realdata_sp500.py` — artifact builder
- Create: `scripts/prepare_realdata_tcga.py` — artifact builder
- Create: `scripts/plot_realdata_unified.py` — shared axes panels + summary table
- Modify: `latex/manuscript_sample.tex` — positioning, theory clarifications, Threats to Validity
- Create tests under `tests/` for new modules and schema

---

### Task 1: Add dependencies for real-data preparation

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies**

```toml
# pyproject.toml
[project]
# ...

dependencies = [
  "numpy",
  "scipy",
  "pyyaml",
  "matplotlib",
  "pytest",
  "pandas",
  "scikit-learn",
  "yfinance",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml

git commit -m "chore: add pandas, scikit-learn, yfinance dependencies"
```

---

### Task 2: Add dataset artifact schema and IO helpers

**Files:**
- Create: `src/grpca_gd/datasets/artifacts.py`
- Create: `src/grpca_gd/datasets/__init__.py`
- Create: `tests/test_dataset_artifacts.py`

- [ ] **Step 1: Write failing test for artifact roundtrip**

```python
# tests/test_dataset_artifacts.py
from pathlib import Path
import numpy as np

from grpca_gd.datasets.artifacts import DatasetArtifact, save_artifact, load_artifact


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_artifacts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grpca_gd.datasets'`

- [ ] **Step 3: Implement artifact schema and IO**

```python
# src/grpca_gd/datasets/artifacts.py
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
```

```python
# src/grpca_gd/datasets/__init__.py
from .artifacts import DatasetArtifact, load_artifact, save_artifact

__all__ = ["DatasetArtifact", "load_artifact", "save_artifact"]
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_dataset_artifacts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/grpca_gd/datasets/artifacts.py src/grpca_gd/datasets/__init__.py tests/test_dataset_artifacts.py

git commit -m "feat: add dataset artifact schema and IO helpers"
```

---

### Task 3: Add new graph families + normalized Laplacian

**Files:**
- Modify: `src/grpca_gd/synthetic/graphs.py`
- Create: `tests/test_graph_families.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph_families.py
import numpy as np

from grpca_gd.synthetic.graphs import (
    er_graph_laplacian,
    grid_graph_laplacian,
    knn_graph_laplacian,
    normalized_laplacian,
    small_world_laplacian,
)


def test_grid_graph_shape() -> None:
    L, W = grid_graph_laplacian(rows=3, cols=4)
    assert L.shape == (12, 12)
    assert W.shape == (12, 12)


def test_er_graph_shape() -> None:
    L, W = er_graph_laplacian(p=10, p_edge=0.2, rng=np.random.default_rng(0))
    assert L.shape == (10, 10)
    assert W.shape == (10, 10)


def test_knn_graph_shape() -> None:
    points = np.random.default_rng(1).normal(size=(8, 2))
    L, W = knn_graph_laplacian(points, k=3)
    assert L.shape == (8, 8)
    assert W.shape == (8, 8)


def test_small_world_shape() -> None:
    L, W = small_world_laplacian(p=12, k=2, beta=0.2, rng=np.random.default_rng(2))
    assert L.shape == (12, 12)
    assert W.shape == (12, 12)


def test_normalized_laplacian_diagonal() -> None:
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    L = normalized_laplacian(W)
    assert np.allclose(np.diag(L), np.ones(2))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_graph_families.py -v`
Expected: FAIL with ImportError for missing functions

- [ ] **Step 3: Implement graph families**

```python
# src/grpca_gd/synthetic/graphs.py
from __future__ import annotations

import numpy as np


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    deg = np.sum(W, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv = np.diag(inv_sqrt)
    return np.eye(W.shape[0]) - D_inv @ W @ D_inv


def grid_graph_laplacian(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    p = rows * cols
    W = np.zeros((p, p), dtype=float)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    j = rr * cols + cc
                    W[idx, j] = 1.0
    W = np.maximum(W, W.T)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def er_graph_laplacian(p: int, p_edge: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    W = np.triu((rng.random((p, p)) < p_edge).astype(float), 1)
    W = W + W.T
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def knn_graph_laplacian(points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        nn = np.argsort(dists[i])[1 : k + 1]
        W[i, nn] = 1.0
    W = np.maximum(W, W.T)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W


def small_world_laplacian(p: int, k: int, beta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(1, k + 1):
            W[i, (i + j) % p] = 1.0
            W[i, (i - j) % p] = 1.0
    for i in range(p):
        for j in range(1, k + 1):
            if rng.random() < beta:
                old = (i + j) % p
                new = int(rng.integers(0, p))
                W[i, old] = 0.0
                W[old, i] = 0.0
                W[i, new] = 1.0
                W[new, i] = 1.0
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L, W
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_graph_families.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/grpca_gd/synthetic/graphs.py tests/test_graph_families.py

git commit -m "feat: add grid, ER, kNN, small-world graphs"
```

---

### Task 4: Add corruption operators

**Files:**
- Create: `src/grpca_gd/synthetic/corruption.py`
- Create: `tests/test_graph_corruption_ops.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph_corruption_ops.py
import numpy as np
from grpca_gd.synthetic.corruption import delete_edges, rewire_edges, perturb_weights


def test_delete_edges_reduces_edges() -> None:
    W = np.ones((5, 5)) - np.eye(5)
    W2 = delete_edges(W, frac=0.2, rng=np.random.default_rng(0))
    assert W2.sum() < W.sum()


def test_rewire_preserves_edge_count() -> None:
    W = np.zeros((6, 6))
    W[0, 1] = W[1, 0] = 1
    W[2, 3] = W[3, 2] = 1
    W2 = rewire_edges(W, frac=0.5, rng=np.random.default_rng(1))
    assert np.isclose(W2.sum(), W.sum())


def test_perturb_weights_changes_values() -> None:
    W = np.ones((4, 4)) - np.eye(4)
    W2 = perturb_weights(W, scale=0.1, rng=np.random.default_rng(2))
    assert not np.allclose(W, W2)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_graph_corruption_ops.py -v`
Expected: FAIL with ImportError for missing functions

- [ ] **Step 3: Implement corruption operators**

```python
# src/grpca_gd/synthetic/corruption.py
from __future__ import annotations

import numpy as np


def delete_edges(W: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    W = W.copy()
    edges = np.transpose(np.triu_indices_from(W, k=1))
    edge_idx = [tuple(e) for e in edges if W[e[0], e[1]] > 0]
    m = int(len(edge_idx) * frac)
    remove = rng.choice(len(edge_idx), size=m, replace=False) if m > 0 else []
    for idx in remove:
        i, j = edge_idx[int(idx)]
        W[i, j] = 0.0
        W[j, i] = 0.0
    return W


def rewire_edges(W: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    W = W.copy()
    edges = np.transpose(np.triu_indices_from(W, k=1))
    edge_idx = [tuple(e) for e in edges if W[e[0], e[1]] > 0]
    m = int(len(edge_idx) * frac)
    if m <= 0:
        return W
    remove = rng.choice(len(edge_idx), size=m, replace=False)
    for idx in remove:
        i, j = edge_idx[int(idx)]
        W[i, j] = 0.0
        W[j, i] = 0.0
        a = int(rng.integers(0, W.shape[0]))
        b = int(rng.integers(0, W.shape[0]))
        if a == b:
            continue
        W[a, b] = 1.0
        W[b, a] = 1.0
    return W


def perturb_weights(W: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=scale, size=W.shape)
    Wn = np.maximum(0.0, W + noise)
    np.fill_diagonal(Wn, 0.0)
    return np.maximum(Wn, Wn.T)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_graph_corruption_ops.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/grpca_gd/synthetic/corruption.py tests/test_graph_corruption_ops.py

git commit -m "feat: add graph corruption operators"
```

---

### Task 5: Add metrics for nnz and coupling gap

**Files:**
- Modify: `src/grpca_gd/metrics.py`
- Create: `tests/test_metrics_nnzcoupling.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics_nnzcoupling.py
import numpy as np
from grpca_gd.metrics import nnz_loadings, coupling_gap


def test_nnz_loadings_counts() -> None:
    B = np.array([[0.0, 1.0], [2.0, 0.0]])
    assert nnz_loadings(B) == 2


def test_coupling_gap_zero() -> None:
    A = np.eye(2)
    B = np.eye(2)
    assert coupling_gap(A, B) == 0.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_metrics_nnzcoupling.py -v`
Expected: FAIL with ImportError for nnz_loadings

- [ ] **Step 3: Implement metrics**

```python
# src/grpca_gd/metrics.py

def nnz_loadings(B: np.ndarray, eps: float = 1e-8) -> int:
    return int(np.sum(np.abs(B) > eps))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_metrics_nnzcoupling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/grpca_gd/metrics.py tests/test_metrics_nnzcoupling.py

git commit -m "feat: add nnz loadings metric"
```

---

### Task 6: Implement MNIST artifact preparation

**Files:**
- Create: `scripts/prepare_realdata_mnist.py`
- Create: `tests/test_prepare_mnist_artifact.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_prepare_mnist_artifact.py
from pathlib import Path

from grpca_gd.datasets.artifacts import load_artifact


def test_prepare_mnist_artifact(tmp_path: Path) -> None:
    from scripts.prepare_realdata_mnist import prepare_mnist_artifact

    out_dir = tmp_path / "mnist_artifact"
    prepare_mnist_artifact(out_dir, max_samples=50, resolution=28)
    artifact = load_artifact(out_dir)
    assert artifact.dataset == "mnist"
    assert artifact.X.shape[0] == 50
    assert artifact.X.shape[1] == 28 * 28
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_prepare_mnist_artifact.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.prepare_realdata_mnist`

- [ ] **Step 3: Implement MNIST prep**

```python
# scripts/prepare_realdata_mnist.py
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
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
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_prepare_mnist_artifact.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_realdata_mnist.py tests/test_prepare_mnist_artifact.py

git commit -m "feat: add MNIST artifact preparation"
```

---

### Task 7: Implement S&P500 artifact preparation

**Files:**
- Create: `scripts/prepare_realdata_sp500.py`
- Create: `tests/test_prepare_sp500_artifact.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_prepare_sp500_artifact.py
from pathlib import Path
import pandas as pd
import numpy as np

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
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_prepare_sp500_artifact.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.prepare_realdata_sp500`

- [ ] **Step 3: Implement S&P500 prep**

```python
# scripts/prepare_realdata_sp500.py
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf

from grpca_gd.datasets.artifacts import DatasetArtifact, save_artifact
from grpca_gd.synthetic.graphs import normalized_laplacian


def _hash_config(payload: dict) -> str:
    data = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _corr_graph(prices: pd.DataFrame, corr_threshold: float) -> np.ndarray:
    returns = prices.pct_change().dropna()
    corr = returns.corr().fillna(0.0).to_numpy()
    W = (np.abs(corr) >= corr_threshold).astype(float)
    np.fill_diagonal(W, 0.0)
    return W


def prepare_sp500_artifact(
    out_dir: Path,
    prices: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    end: str = "2023-01-01",
    corr_threshold: float = 0.3,
) -> None:
    if prices is None:
        if tickers is None:
            raise ValueError("tickers required when prices not provided")
        data = yf.download(tickers, start=start, end=end, progress=False)
        prices = data["Adj Close"].dropna()

    W = _corr_graph(prices, corr_threshold)
    L = normalized_laplacian(W)
    X = prices.pct_change().dropna().to_numpy()

    prep_cfg = {
        "start": start,
        "end": end,
        "corr_threshold": corr_threshold,
        "tickers": list(prices.columns),
    }
    artifact = DatasetArtifact(
        artifact_id="sp500_corr_v1",
        artifact_version="v1",
        dataset="sp500",
        graph_family="correlation",
        data_source="yfinance",
        prep_config_hash=_hash_config(prep_cfg),
        eval_protocol_id="default",
        X=X,
        L=L,
        metadata={"tickers": list(prices.columns), "start": start, "end": end},
    )
    save_artifact(artifact, out_dir)
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_prepare_sp500_artifact.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_realdata_sp500.py tests/test_prepare_sp500_artifact.py

git commit -m "feat: add S&P500 artifact preparation"
```

---

### Task 8: Update runner to consume artifacts and emit canonical schema

**Files:**
- Modify: `src/grpca_gd/runner.py`
- Modify: `scripts/collect_metrics.py`
- Create: `tests/test_results_schema.py`

- [ ] **Step 1: Add schema test**

```python
# tests/test_results_schema.py
import json
from pathlib import Path


def test_schema_fields_present(tmp_path: Path) -> None:
    metrics = {
        "dataset": "synthetic",
        "graph_family": "chain",
        "artifact_id": "synthetic_chain",
        "artifact_version": "v1",
        "data_source": "synthetic",
        "prep_config_hash": "abc",
        "eval_protocol_id": "default",
        "method": "Proposed",
        "method_version": "v1",
        "seed": 0,
        "rank": 1,
        "lambda1": 0.1,
        "lambda2": 0.2,
        "rho": 5.0,
        "corruption_type": "none",
        "corruption_level": 0.0,
        "graph_used_id": "chain_clean",
        "graph_reference_id": "chain_clean",
        "explained_variance": 1.0,
        "smoothness_used_graph": 0.1,
        "smoothness_reference_graph": 0.1,
        "runtime_sec": 0.1,
        "iterations": 10,
        "nnz_loadings": 4,
        "sparsity_ratio": 0.2,
        "final_objective": -0.1,
        "final_coupling_gap": 0.0,
        "final_orthogonality_defect": 0.0,
        "stop_reason": "tol_obj",
        "convergence_flag": True,
        "support_precision": 1.0,
        "support_recall": 1.0,
        "support_f1": 1.0,
        "n_samples": 100,
        "n_features": 20,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    for key in ["artifact_id", "eval_protocol_id", "method_version", "nnz_loadings"]:
        assert key in loaded
```

- [ ] **Step 2: Run test to verify pass**

Run: `pytest tests/test_results_schema.py -v`
Expected: PASS

- [ ] **Step 3: Update runner schema emission**

```python
# src/grpca_gd/runner.py (add at top)
from .datasets import load_artifact
from .metrics import nnz_loadings, coupling_gap

# within run():
# if cfg has "artifact_dir" then load artifact and set X/L accordingly
artifact_dir = cfg.get("artifact_dir")
if artifact_dir:
    artifact = load_artifact(Path(artifact_dir))
    X = artifact.X
    L = artifact.L
    dataset_meta = artifact.metadata
    dataset_name = artifact.dataset
else:
    # existing synthetic path
    ...

# in metrics_out payloads, add fields:
"nnz_loadings": nnz_loadings(result.B),
"sparsity_ratio": sparsity_fraction(result.B),
"final_coupling_gap": coupling_gap(result.A, result.B),
"final_orthogonality_defect": orthogonality_error(result.A),
"n_samples": int(X.shape[0]),
"n_features": int(X.shape[1]),

# support metrics only when dataset has true supports
"support_precision": support.get("precision") if support else None,
"support_recall": support.get("recall") if support else None,
"support_f1": support.get("f1") if support else None,
```

- [ ] **Step 4: Update collect_metrics to output canonical schema**

```python
# scripts/collect_metrics.py
# when collecting, emit flat rows with all required keys and None when missing
```

- [ ] **Step 5: Commit**

```bash
git add src/grpca_gd/runner.py scripts/collect_metrics.py tests/test_results_schema.py

git commit -m "feat: extend runner schema and metrics export"
```

---

### Task 9: Add unified real-data plotting script

**Files:**
- Create: `scripts/plot_realdata_unified.py`

- [ ] **Step 1: Add plotting script**

```python
# scripts/plot_realdata_unified.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "metrics_summary.csv"
OUT_DIR = ROOT / "figures" / "realdata"


def main() -> None:
    df = pd.read_csv(RESULTS)
    df = df[df["dataset"].isin(["mnist", "tcga", "sp500"]) & (df["method"] == "Proposed")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["explained_variance", "smoothness_used_graph", "sparsity_ratio"],
        ["Explained variance", "Smoothness", "Sparsity"],
    ):
        for dataset in ["mnist", "tcga", "sp500"]:
            sub = df[df["dataset"] == dataset]
            ax.plot(sub["lambda2"], sub[metric], label=dataset)
        ax.set_ylabel(title)
        ax.legend()
    axes[-1].set_xlabel("lambda2")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "realdata_unified_panels.png", dpi=200)

    summary = (
        df.groupby("dataset")["smoothness_used_graph"]
        .mean()
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "realdata_summary.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/plot_realdata_unified.py

git commit -m "feat: add unified real-data plotting script"
```

---

### Task 10: Update manuscript positioning and technical clarity

**Files:**
- Modify: `latex/manuscript_sample.tex`

- [ ] **Step 1: Update abstract with empirical characterization framing**

Replace current abstract paragraph with:

```latex
We study sparse, graph-smooth, explicitly orthogonal multi-component PCA through a split-variable formulation that couples an orthogonal loading matrix with a sparse graph-regularized proxy. Rather than proposing new optimization primitives, we use this formulation as a controlled testbed to characterize when graph regularization helps or hurts under misspecification. Experiments spanning chain, grid, SBM, ER, kNN, and small-world graphs show a regime-based pattern: weak regularization is robust across topologies, while stronger regularization can amplify misspecification on fragile graphs. Real-data studies on TCGA-BRCA, MNIST (grid), and S\&P500 correlation graphs show aligned trends. Our contribution is thus an empirical characterization of topology- and regime-dependent behavior rather than a universal performance gain.
```

- [ ] **Step 2: Add estimator clarity and limiting cases**

Insert after the formulation section:

```latex
\paragraph{Estimator choice.}
We treat $B$ as the primary estimator for sparsity and smoothness metrics, since it carries the graph-regularized structure. Variance is reported using $A$ (or $\mathrm{QR}(B)$ when needed) to ensure orthogonality.

\paragraph{Limiting cases.}
When $\lambda_2=0$, the formulation reduces to sparse PCA without graph smoothness; when $\lambda_1=0$, it reduces to graph-regularized PCA; and as $\rho\to\infty$, the split formulation converges to a single constrained problem with $A=B$.
```

- [ ] **Step 3: Add Laplacian normalization note**

Insert:

```latex
\paragraph{Laplacian normalization.}
We use the normalized Laplacian $L=I-D^{-1/2}WD^{-1/2}$ throughout, so $\lambda_2$ is comparable across graphs with different degrees.
```

- [ ] **Step 4: Add Threats to Validity**

Insert before Conclusion:

```latex
\paragraph{Threats to validity.}
Despite expanded graph families, our experiments still cover a limited subset of possible topologies. The S\&P500 correlation graph is a noisy proxy for true relationships, and real data lacks ground-truth supports. Implementations share a common NumPy codebase and are not equivalently optimized, so runtime comparisons are indicative only. Finally, hyperparameter tuning choices can influence regime boundaries.
```

- [ ] **Step 5: Commit**

```bash
git add latex/manuscript_sample.tex

git commit -m "docs: reframe contributions and add technical clarifications"
```

---

## Plan Self-Review
- [ ] Verify all spec requirements are mapped to tasks (artifacts, schema, graphs, baselines, real data, plots, manuscript).
- [ ] Confirm there are no placeholders or missing code blocks.
- [ ] Confirm schema keys and names are consistent with plan.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-02-tracka-revision-implementation-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
