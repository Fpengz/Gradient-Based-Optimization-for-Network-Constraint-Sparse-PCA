# Gradient-Based Optimization for Network-Constrained Sparse PCA

## Project Overview

This project implements a comprehensive suite of modern and classic Sparse Principal Component Analysis (SPCA) algorithms, with a specialized focus on **Network-Constrained Sparse PCA (NC-SPCA)**.

The goal is to provide a unified framework for sparse dimensionality reduction that respects prior relational knowledge (graph structures) and scales to high-dimensional streaming data.

### Mathematical Formulations

#### 1. NC-SPCA (Network-Constrained)
Solves the graph-regularized variance maximization:
$$ \min_{\|w\|_2 \le 1} \left( - w^\top \hat\Sigma w + \lambda_1 \|w\|_1 + \lambda_2 w^\top L w \right) $$
Where $L$ is the graph Laplacian. Solved via Proximal Gradient Descent.

#### 2. Zou-SPCA (Regression-based)
Reformulates PCA as a regression problem with Elastic Net penalty:
$$ \min_{A, B} \sum_{i=1}^n \|x_i - AB^\top x_i\|^2 + \lambda \|B\|^2_2 + \sum_{j=1}^k \lambda_{1,j} \|\beta_j\|_1 \text{ s.t. } A^\top A = I $$

#### 3. GradFPS (Fantope Projection)
A convex relaxation optimizing over the Fantope $\mathcal{F}^d$:
$$ \min_{X \in \mathcal{F}^d} -\langle \hat\Sigma, X \rangle + \rho \|X\|_{1,1} $$
Where $\mathcal{F}^d = \{X : 0 \preceq X \preceq I, \text{tr}(X) = d\}$.

#### 4. GPM (Generalized Power Method)
Efficient single-unit iteration:
$$ \max_{\|z\|_2 = 1} \|\text{soft}(X^\top z, \gamma)\|_2 $$

## Directory Structure

*   `src/models/`: Implementation of `ZouSparsePCA`, `NetworkSparsePCA`, `GradFPS`, and `GeneralizedPowerMethod`.
*   `data/`:
    *   `synthetic/`: Generators for Zou Example 1, Qiu spiked model, and graph-structured data.
    *   `pitprop.py`: Pitprop dataset reconstruction.
    *   `colon_x.csv`: Alon et al. Colon Cancer dataset.
*   `experiments/`:
    *   `benchmark_comparison.py`: Cross-model verification.
    *   `colon_benchmark.py`: High-dimensional biological data performance.
    *   `nc_spca_benchmark.py`: Recovery of contiguous support on feature networks.

## Commands

*   **Install Dependencies**: `uv sync`
*   **Run All Benchmarks**: `uv run experiments/benchmark_comparison.py`
*   **Run biological Test**: `uv run experiments/colon_benchmark.py`

## Development Roadmap

1.  **[DONE] Algorithm Implementation**: All four major SPCA variants are implemented and tested.
2.  **[DONE] Dataset Integration**: Successfully integrated Pitprop, Colon Cancer, and multiple synthetic models.
3.  **[DONE] Metric Suite**: Implemented support recovery (F1), connectivity (LCC Ratio), and subspace distance metrics.
4.  **[TODO] Online Learning**: Implement `Online-T` and `Online-P` stochastic gradient variants for streaming data.
5.  **[TODO] Manuscript Completion**: Finalize LaTeX draft in `doc/latex/main.tex` with final benchmark results.

## References
*   Zou et al. (2006): Sparse PCA.
*   Journée et al. (2010): Generalized Power Method.
*   Qiu et al. (2023): Gradient-based SPCA with Online Learning.