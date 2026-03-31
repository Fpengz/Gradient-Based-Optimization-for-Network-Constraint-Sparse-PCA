# 03_formulation.md

## Purpose
This document fixes the mathematical objective, variables, and notation for paper 1. Implementation should follow this formulation exactly.

## Notation
- Data matrix: X in R^{n x p}, centered by columns.
- Sample covariance: Sigma_hat = (1/n) X^T X.
- Feature graph: adjacency W, degree D, Laplacian L = D - W.
- Components: r with r << p.

## Primary Objective (Split-Variable Formulation)
We optimize over an orthogonal loading matrix A and a sparse, graph-smooth proxy B:

min_{A^T A = I_r, B in R^{p x r}}
  -tr(A^T Sigma_hat A)
  + lambda1 ||B||_1
  + lambda2 tr(B^T L B)
  + (rho/2) ||A - B||_F^2.

Interpretation:
- A carries orthogonality (Stiefel constraint),
- B carries sparsity and graph smoothness,
- the quadratic coupling aligns A and B.

## Design Choices (Frozen for Paper 1)
- The split-variable objective above is the primary formulation.
- The coupling parameter rho is positive and tuned.
- The Laplacian term uses the combinatorial Laplacian L; no alternate graph operators.
- Sparse penalty is entrywise l1 on B only (no structured group penalties).

## Secondary Reference Variant (Not Primary)
For positioning only, the direct Stiefel form is:

max_{V^T V = I_r}
  tr(V^T Sigma_hat V)
  - lambda1 ||V||_1
  - lambda2 tr(V^T L V).

This variant is referenced for context but is not the implementation target for paper 1.

## Outputs
- Orthogonal loadings: A.
- Sparse, graph-smooth proxy loadings: B.

## Scope Note
This formulation is frozen for paper 1. Any changes require explicit revision to 01_problem_scope.md.
