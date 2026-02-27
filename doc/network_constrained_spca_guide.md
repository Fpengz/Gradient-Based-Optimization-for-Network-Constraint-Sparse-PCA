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

1.  **[DONE] Core Estimators**: Implemented Zou-SPCA (Elastic Net), GPM (Power Method), and NC-SPCA.
2.  **[DONE] Synthetic Benchmarks**: Config-driven graph-structured synthetic benchmark with fair baseline comparison.
3.  **[DONE] Real Data Benchmarks**: Unified real-data comparison on Pitprop and Alon et al. Colon Cancer.
4.  **[DONE] Graph Generators**: Implemented Chain, Grid, ER, and SBM generators for feature networks.
5.  **[TODO] Convergence Analysis**: Refine mathematical proofs for stationary point convergence under graph constraints.

---

## 7. Modules

- **Estimators (`src/models/`)**:
    - `ZouSparsePCA`: Regression-based SPCA.
    - `NetworkSparsePCA`: Graph-regularized SPCA using Proximal Gradient.
    - `GeneralizedPowerMethod`: Efficient power iteration with thresholding.
- **Data (`data/`)**:
    - `pitprop.py`: Pitprop correlation matrix.
    - `colon_x.csv`: Alon et al. genomic data.
- **Benchmarks (`scripts/`)**:
    - `run_experiment.py`: synthetic/real comparison runner.
    - `run_sweep.py`: \(\lambda_1,\lambda_2\) sweep + artifact generation.
    - `reproduce_figures.py`: one-command reproduction bundle.
- **Evaluation (`doc/experiments/`)**:
    - Explained variance, F1 score, Largest Connected Component (LCC) ratio, Laplacian smoothness.
