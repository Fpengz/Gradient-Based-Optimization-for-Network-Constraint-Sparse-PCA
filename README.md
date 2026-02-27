# Gradient-Based Optimization for Network-Constrained Sparse PCA

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides a unified framework for **Sparse Principal Component Analysis (SPCA)** and **Network-Constrained Sparse PCA (NC-SPCA)**. It includes implementations of classic and state-of-the-art algorithms, evaluated on statistical benchmarks and high-dimensional biological data.

---

## 🚀 Quick Start

This project uses `uv` for modern, fast Python package management.

### Installation

1.  **Install `uv`** (if not already installed):
    ```bash
    pip install uv
    ```
2.  **Sync Dependencies:**
    ```bash
    uv sync
    ```
    For development checks (`ty`, `ruff`, `pytest`, `black`), use:
    ```bash
    uv sync --dev
    ```

3.  **Optional Torch backend dependencies** (only if you want PyTorch/Geoopt models):
    ```bash
    pip install torch geoopt
    ```

### Running Benchmarks

*   **Core synthetic comparison suite (paper baseline set):**
    ```bash
    uv run python scripts/run_experiment.py --n-repeats 3
    ```
    This runs: `PCA`, `L1-SPCA-ProxGrad`, `Graph-PCA`, `NetSPCA-PG`, `NetSPCA-MASPG-CAR`, `GPower`, and `ElasticNet-SPCA`.

*   **Run graph methods with Torch backend:**
    ```bash
    uv run python scripts/run_experiment.py --backend torch --n-repeats 3
    ```

*   **Run manifold baseline with Torch+Geoopt:**
    ```bash
    uv run python scripts/run_experiment.py --backend torch-geoopt --include-stiefel-manifold --n-components 3 --n-repeats 3
    ```

*   **Stress graph misspecification (graph-quality robustness):**
    ```bash
    uv run python scripts/run_experiment.py --dataset synthetic --graph-misspec-rate 0.15 --n-repeats 3
    ```

*   **Multi-component benchmark with manifold baseline:**
    ```bash
    uv run python scripts/run_experiment.py --dataset synthetic --n-components 3 --include-stiefel-manifold --n-repeats 3
    ```

*   **Hyperparameter sweeps (`\lambda_1`, `\lambda_2`) with plots + LaTeX tables:**
    ```bash
    uv run python scripts/run_sweep.py --n-repeats 2
    ```

*   **Real dataset comparison (Colon / Pitprop):**
    ```bash
    uv run python scripts/run_experiment.py --dataset colon
    uv run python scripts/run_experiment.py --dataset pitprop
    ```

*   **One-command figure reproduction bundle:**
    ```bash
    uv run python scripts/reproduce_figures.py
    ```

---

## 🛠 Implemented Algorithms

| Algorithm | Method | Reference |
| :--- | :--- | :--- |
| **Zou SPCA** | Elastic Net / Regression | Zou et al. (2006) |
| **GPM** | Generalized Power Method | Journée et al. (2010) |
| **NC-SPCA** | Laplacian Regularization / Proximal Grad | Wang Zhoufu (2026) |
| **NC-SPCA (Stiefel)** | Manifold proximal gradient + Stiefel retraction | Chen et al. (ManPG lineage) |
| **NC-SPCA (Torch)** | PyTorch PG/MASPG-CAR backend | This repository |
| **NC-SPCA (Torch + Geoopt)** | Geoopt Stiefel manifold backend | This repository |

---

## 📊 Supported Datasets

1.  **Pitprop Dataset**: 13x13 correlation matrix of physical timber properties (Jeffers 1967).
2.  **Colon Cancer**: Gene expression data (2000 genes, 62 samples) from Alon et al. (1999).
3.  **Graph-Structured Synthetic Data**: configurable generators for **Chain, Grid, ER, RGG, and SBM** feature graphs with connected sparse supports.

---

## 📂 Project Structure

*   `src/models/`: Implementation of SPCA variants.
*   `src/experiments/`: Reusable synthetic benchmark and comparison utilities.
*   `data/`: Data loaders.
*   `scripts/`: Reproducible experiment/sweep runners.
*   `doc/`: Theoretical documentation, derivations, and publication drafts.

---

## ✅ CI

CI is configured in `.github/workflows/ci.yml` and runs:

- `uv run ruff check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests`
- `uv run black --check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests`
- `uv run pytest -q`

## 🔒 Pre-commit Hook

Install the local git hook:

```bash
bash scripts/install_git_hooks.sh
```

The hook runs on every commit:

- `uv run ty check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests`
- `uv run ruff check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests`
- `uv run pytest -q`

---

## Core Motivation

Classical PCA produces dense loading vectors, which are difficult to interpret in high-dimensional settings. Sparse PCA enforces sparsity for feature selection, but often ignores known relational structures between features (e.g., gene interaction networks). 

**Network-Constrained SPCA** incorporates prior graph information via a Laplacian penalty:
$$ \min_{\|w\|_2 \le 1} -w^\top \hat\Sigma w + \lambda_1 \|w\|_1 + \lambda_2 w^\top L w $$
This encourages the selection of **connected, smooth supports** on the feature network, leading to more scientifically valid factor discovery.

For regularization-path workflows, `NetworkSparsePCA.fit_path(...)` provides warm-start continuation across `lambda1`/`lambda2` grids.

Torch side-by-side backends are available as:

- `TorchNetworkSparsePCA` (single-component PG/MASPG-CAR style)
- `TorchNetworkSparsePCA_GeooptStiefel` (multi-component Stiefel manifold)

---

## References

*   Zou, H., Hastie, T., & Tibshirani, R. (2006). *Sparse Principal Component Analysis*.
*   Journée, M., Nesterov, Y., Richtárik, P., & Sepulchre, R. (2010). *Generalized Power Method for Sparse Principal Component Analysis*.
*   Qiu, Y., Lei, J., & Roeder, K. (2023). *Gradient-Based Sparse Principal Component Analysis with Extensions to Online Learning*.
