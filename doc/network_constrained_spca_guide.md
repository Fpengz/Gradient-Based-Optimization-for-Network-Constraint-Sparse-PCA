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

Adding sparsity (e.g. $ \|w\|_0 \le k $) makes the problem non-convex and NP-hard.

---

## 3. Sparse PCA (SPCA)
A common relaxation:
$$
\max_w \; w^\top \Sigma w - \lambda \|w\|_1 \quad \text{s.t. } \|w\|_2 \le 1
$$

Key reference:
- Zou, Hastie, Tibshirani (2006), *Sparse PCA*

---

## 4. Constraint Taxonomy
| Constraint Type | Smooth | Gradient-Based | Requires Proximal |
|-----------------|--------|----------------|-------------------|
| L2 norm         | Yes    | Yes            | No                |
| L1 sparsity     | No     | Subgradient    | Yes               |
| Graph Laplacian | Yes    | Yes            | No                |

---

## 5. Network-Constrained SPCA
Given graph $ G = (V,E) $ over features, introduce Laplacian \(L\):
$$
\max_w \; w^\top \Sigma w - \lambda_1 \|w\|_1 - \lambda_2 w^\top L w
$$

Encourages:
- Smoothness over graph
- Connected supports

---

## 6. Optimization Perspective
- Smooth part: variance + Laplacian penalty
- Non-smooth part: sparsity term
- Suitable method: **Proximal Gradient Descent**

---

## 7. Suggested First Milestones
1. Re-derive PCA and SPCA objectives
2. Implement proximal gradient for L1-SPCA
3. Extend to graph-regularized SPCA
4. Analyze convergence and scaling

---

## 8. Next Steps
- Explore nonconvex sparsity (L0, SCAD)
- Study convergence guarantees
- Compare with greedy and SDP approaches


## 9. Modules

- proximal gradient solver (soft-threshold + L2 projection)
- objective + metrics
- synthetic graph generators
- synthetic data generator (spiked covariance with structured sparse support)
- baselines: PCA, L1-SPCA (λ2=0), Graph-PCA (λ1=0)
- evaluation: explained variance, F1, LCC ratio, Laplacian smoothness, runtime