# 05_baselines.md

## Purpose
This document freezes the baseline matrix for paper 1. Experiments should include these baselines and ablations unless explicitly revised.

## Baseline Inclusion Criteria
A baseline is included if it captures at least one of these axes:
- sparsity on loadings,
- graph-aware regularization on loadings,
- explicit orthogonality across multiple components,
- standard PCA as a reference.

## Required Baseline Matrix

### Unstructured
- PCA (standard SVD or eigendecomposition).

### Sparse (Unstructured)
- Sparse PCA (SPCA) baseline using a standard method (e.g., Zou-Hastie-Tibshirani or GPower).

### Orthogonal Sparse
- One joint orthogonal sparse PCA baseline (single method, explicitly enforcing orthogonality across components).

### Graph-Aware (Not Sparse)
- Graph-regularized PCA (GR-PCA style Laplacian smoothness without explicit sparsity).

### Graph + Sparse
- One graph + sparsity PCA baseline from the literature (e.g., graph-Laplacian + l1 or related formulation).

### Manifold Sparse PCA (Optional but Recommended)
- A manifold proximal sparse PCA solver if readily available (used as a sanity check, not as a core comparator).

## Ablation Matrix (Mandatory)
Ablations are applied to our method:
- No graph: set lambda2 = 0.
- No sparsity: set lambda1 = 0.
- Deflation instead of joint extraction: sequential components with orthogonalization.
- Graph corruption: perturb the graph and re-run.

## Reporting Policy
For each baseline and ablation, report:
- explained variance,
- reconstruction error,
- support recovery (precision, recall, F1) on synthetic data,
- graph smoothness (Laplacian energy),
- orthogonality error (when applicable),
- runtime and iterations to convergence.

## Status Tracking
Mark each baseline in the experiment log as:
- implemented,
- in progress,
- or planned.

No baseline should be excluded without an explicit reason documented in the experiment log.
