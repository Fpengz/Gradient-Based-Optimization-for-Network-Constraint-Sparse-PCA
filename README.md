# Gradient-Based Optimization for Network-Constrained Sparse PCA

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a research-oriented implementation of **Sparse Principal Component Analysis (SPCA)** and its **network-constrained variants**, with a particular focus on **gradient-based and proximal optimization methods**. The project is designed to be modular, reproducible, and extensible.

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

### Usage

*   **Run the Main Script:**
    ```bash
    uv run main.py
    ```
*   **Run Experiments:**
    ```bash
    uv run scripts/run_experiment.py
    ```
*   **Run Tests:**
    ```bash
    uv run pytest
    ```

---

## Motivation

Classical PCA is a fundamental tool for dimensionality reduction and factor discovery, but it suffers from two major limitations in high-dimensional, structured settings:

1. **Dense loadings**: Principal components typically involve all variables, making interpretation difficult and deployment costly.
2. **Lack of structural awareness**: PCA ignores known relationships between variables (e.g., graphs, networks, or spatial structure).

Sparse PCA addresses the first issue by enforcing sparsity, while **network-constrained Sparse PCA** incorporates prior structural information via graph-based regularization. This project studies how such models can be solved efficiently using modern gradient-based optimization techniques.

---

## Core Research Goals

* Implement and compare **gradient descent, proximal gradient, and projected gradient** methods for Sparse PCA
* Study the effect of **graph Laplacian regularization** on sparsity, stability, and interpretability
* Analyze **convergence behavior** and optimization trade-offs in non-convex settings
* Provide a clean experimental framework suitable for paper-quality results

---

## Possible Future Applications

Although the core focus of this repository is methodological, the techniques developed here have direct relevance to several high-impact applied domains—especially in **quantitative finance**.

### 1. Portfolio Construction via Sparse PCA

In finance, standard PCA is widely used to construct *eigen-portfolios*—portfolios corresponding to the principal components of asset return covariance matrices. However, classical PCA produces **dense eigenvectors**, meaning that a portfolio based on these components requires holding positions in *every* asset in the universe.

**Application:** Sparse PCA enables the construction of **sparse eigen-portfolios**, where only a small subset of assets have non-zero weights.

**Why it matters:**

* Fewer assets lead to **lower transaction costs**
* Reduced turnover and **less market impact** during rebalancing
* Improved interpretability of latent risk factors (e.g., value, momentum, sector exposure)

In practice, sparse eigen-portfolios are actively used by systematic and quantitative funds as a direct way to improve **P&L efficiency**.

---

### 2. Modeling Market Structure with Graph Laplacian Regularization

Financial markets are inherently **networked systems**. Assets are linked through:

* Sector and industry classifications
* Supply-chain relationships
* Shared macroeconomic sensitivities

Standard covariance-based models struggle to encode this explicit structure.

**Application:** The **graph Laplacian penalty**
[
v^\top L v
]
encourages portfolio weights or factor loadings to be *smooth* over a predefined market graph. If two stocks are strongly connected in the graph, their loadings are encouraged to behave similarly.

**Why it matters:**

* Improves stability of learned factors
* Reduces spurious idiosyncratic exposures
* Aligns statistical models with known economic structure

**Related research:** Recent quantitative finance literature explores *Financial Graph Laplacian Regularization* to improve return prediction and factor discovery by jointly modeling correlations and network topology.

---

### 3. Alpha Generation via Graph Signal Processing

The theoretical foundation of this project overlaps strongly with **Graph Signal Processing (GSP)**, which treats data as signals evolving over a graph rather than over time alone.

**Application:** In quantitative finance, GSP has been used to model:

* **Realized volatility** as a signal on a market graph
* **Volatility spillovers** across sectors (e.g., how shocks propagate from Tech to Energy)

Compared to classical models such as GARCH, graph-based approaches can capture **cross-sectional propagation effects** more naturally.

**Momentum strategies:** GSP-based filtering has also been applied to denoise momentum signals, yielding cleaner trend-following strategies with improved risk-adjusted returns.

---

### 4. Advanced Optimization Skills: Manifold and Proximal Methods

Beyond the specific application of PCA, this project emphasizes **optimization on constrained and non-convex domains**, including:

* Unit-norm constraints
* Sparsity-inducing regularizers
* Graph-structured penalties

**Application:** Many financial problems require similar tools, such as:

* Robust covariance estimation
* Correlation matrix calibration
* Optimization over positive semi-definite manifolds

**Career signal:** Experience with **Manifold Optimization** and **Manifold Proximal Gradient (ManPG)** methods demonstrates the ability to design custom solvers for complex constraints—going well beyond off-the-shelf tools like `sklearn` or `cvxpy`.

---

## Status

This is an active research project. The current focus is on building strong baselines, validating optimization behavior, and preparing for large-scale experiments and theoretical analysis.

---

## Disclaimer

This repository is for research and educational purposes only. It does not constitute financial advice or an investment recommendation.
