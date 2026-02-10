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

### Running Benchmarks

We provide several reproducible benchmarks from seminal papers:

*   **General Comparison**: Compare all implemented models on synthetic data.
    ```bash
    uv run experiments/benchmark_comparison.py
    ```
*   **Biological Data (p=2000)**: Run the Alon et al. Colon Cancer benchmark.
    ```bash
    export PYTHONPATH=$PYTHONPATH:. && uv run experiments/colon_benchmark.py
    ```
*   **Graph-Structured Data**: Test NC-SPCA on Chain and Grid feature networks.
    ```bash
    export PYTHONPATH=$PYTHONPATH:. && uv run experiments/nc_spca_benchmark.py
    ```

---

## 🛠 Implemented Algorithms

| Algorithm | Method | Reference |
| :--- | :--- | :--- |
| **Zou SPCA** | Elastic Net / Regression | Zou et al. (2006) |
| **GPM** | Generalized Power Method | Journée et al. (2010) |
| **GradFPS** | Fantope Projection / Proximal Grad | Qiu et al. (2023) |
| **NC-SPCA** | Laplacian Regularization / Proximal Grad | *Wang Zhoufu (2026)* |

---

## 📊 Supported Datasets

1.  **Pitprop Dataset**: 13x13 correlation matrix of physical timber properties (Jeffers 1967).
2.  **Colon Cancer**: Gene expression data (2000 genes, 62 samples) from Alon et al. (1999).
3.  **Zou Example 1**: 10-variable synthetic model with 3 hidden factors.
4.  **High-Dim Spiked Model**: $p=500, n=200$ model for subspace recovery tests.
5.  **Graph Synthesis**: Generators for **Chain, Grid, Random Geometric (RGG), and Stochastic Block (SBM)** feature graphs.

---

## 📂 Project Structure

*   `src/models/`: Implementation of SPCA variants.
*   `data/`: Data loaders and synthetic generators.
*   `experiments/`: Benchmark and replication scripts.
*   `doc/`: Theoretical documentation, derivations, and publication drafts.

---

## Core Motivation

Classical PCA produces dense loading vectors, which are difficult to interpret in high-dimensional settings. Sparse PCA enforces sparsity for feature selection, but often ignores known relational structures between features (e.g., gene interaction networks). 

**Network-Constrained SPCA** incorporates prior graph information via a Laplacian penalty:
$$ \min_{\|w\|_2 \le 1} -w^\top \hat\Sigma w + \lambda_1 \|w\|_1 + \lambda_2 w^\top L w $$
This encourages the selection of **connected, smooth supports** on the feature network, leading to more scientifically valid factor discovery.

---

## References

*   Zou, H., Hastie, T., & Tibshirani, R. (2006). *Sparse Principal Component Analysis*.
*   Journée, M., Nesterov, Y., Richtárik, P., & Sepulchre, R. (2010). *Generalized Power Method for Sparse Principal Component Analysis*.
*   Qiu, Y., Lei, J., & Roeder, K. (2023). *Gradient-Based Sparse Principal Component Analysis with Extensions to Online Learning*.