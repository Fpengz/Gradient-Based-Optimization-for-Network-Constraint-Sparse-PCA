# Gradient-Based Optimization for Network-Constrained Sparse PCA
## Research Starter Guide

## 1. Problem Overview
Sparse Principal Component Analysis (SPCA) seeks principal components with sparse loadings for interpretability and robustness. 
Network-constrained SPCA further enforces structured sparsity using graph information over features.

---

## 2. Baseline: Classical PCA
### Optimization Formulation
Maximize variance:
$$
\max_{w} \; w^\top \Sigma w \quad \text{s.t. } \|w\|_2 = 1
$$

### Why Sparsity?
- Interpretability
- Feature selection
- Noise robustness

---

## 3. Sparse PCA (SPCA)
A common relaxation:
$$
\max_w \; w^\top \Sigma w - \lambda \|w\|_1 \quad \text{s.t. } \|w\|_2 \le 1
$$

Key reference:
- Zou, Hastie, Tibshirani (2006), *Sparse PCA*

---

## 4. Network-Constrained SPCA
Given graph $ G = (V,E) $ over features, introduce Laplacian \(L\):
$$
\max_w \; w^\top \Sigma w - \lambda_1 \|w\|_1 - \lambda_2 w^\top L w \quad \text{s.t. } \|w\|_2 \le 1
$$

Encourages:
- Smoothness over graph
- Connected supports

---

## 5. Optimization Perspective
- Smooth part: variance + Laplacian penalty
- Non-smooth part: sparsity term
- Suitable method: **Proximal Gradient Descent**

---

## 6. Project Milestones

1.  **[DONE] Core Estimators**: Implemented Zou-SPCA (Elastic Net), GradFPS (Fantope), GPM (Power Method), and NC-SPCA.
2.  **[DONE] Synthetic Benchmarks**: Replicated Zou's 10-variable example and Qiu's high-dimensional spiked model.
3.  **[DONE] Real Data Benchmarks**: Replicated Pitprop analysis and Alon et al. Colon Cancer gene expression study.
4.  **[DONE] Graph Generators**: Implemented Chain, Grid, RGG, and SBM generators for feature networks.
5.  **[TODO] Convergence Analysis**: Refine mathematical proofs for stationary point convergence under graph constraints.

---

## 7. Modules

- **Estimators (`src/models/`)**:
    - `ZouSparsePCA`: Regression-based SPCA.
    - `NetworkSparsePCA`: Graph-regularized SPCA using Proximal Gradient.
    - `GradFPS`: Convex Fantope relaxation.
    - `GeneralizedPowerMethod`: Efficient power iteration with thresholding.
- **Data (`data/`)**:
    - `pitprop.py`: Pitprop correlation matrix.
    - `colon_x.csv`: Alon et al. genomic data.
    - `synthetic/`: Generators for spiked covariance and graph-structured data.
- **Benchmarks (`experiments/`)**:
    - `benchmark_comparison.py`: Cross-model comparison.
    - `colon_benchmark.py`: High-dimensional biological data test.
    - `nc_spca_benchmark.py`: Support recovery on feature graphs.
- **Evaluation (`doc/experiments/`)**:
    - Explained variance, F1 score, Largest Connected Component (LCC) ratio, Laplacian smoothness.
