# 04_algorithm.md

## Purpose
This document freezes the solver for paper 1. Only the algorithm specified here is the primary implementation target.

## Objective and Notation
We solve the split-variable problem from 03_formulation.md:

min_{A^T A = I_r, B}
  -tr(A^T Sigma_hat A)
  + lambda1 ||B||_1
  + lambda2 tr(B^T L B)
  + (rho/2) ||A - B||_F^2.

Let f(A, B) denote the full objective.

## Alternating Solver (Primary)
We alternate between an A-step on the Stiefel manifold and a B-step in Euclidean space.

### A-step (Manifold Update)
Given B^k, update A^k -> A^{k+1} by a Riemannian gradient step on the Stiefel manifold.

Euclidean gradient with respect to A:
G_A = -2 Sigma_hat A + rho (A - B).

Riemannian gradient on Stiefel:
Grad_A = G_A - A * sym(A^T G_A),
where sym(M) = (M + M^T) / 2.

Update:
A_tilde = A - eta_A * Grad_A.
Retraction:
A^{k+1} = qf(A_tilde), where qf denotes the Q factor of a thin QR decomposition.

Step size:
- Default: fixed eta_A tuned by validation.
- Optional: Armijo backtracking line search on f(A, B^k) (not required for v1).

Stopping check for A-step:
- ||A^{k+1} - A||_F / max(1, ||A||_F) <= tol_A.

### B-step (Proximal Gradient)
Given A^{k+1}, update B^k -> B^{k+1} by proximal gradient.

Smooth part:
  g(B) = lambda2 tr(B^T L B) + (rho/2) ||A^{k+1} - B||_F^2.
Nonsmooth part:
  h(B) = lambda1 ||B||_1.

Gradient of the smooth part:
Grad_B = 2 lambda2 L B + rho (B - A^{k+1}).

Lipschitz constant estimate:
L_g = 2 lambda2 ||L||_2 + rho.

Update:
B_tilde = B - eta_B * Grad_B,
B^{k+1} = soft(B_tilde, eta_B * lambda1),
where soft is entrywise soft-thresholding.

Step size:
- Default: eta_B = 1 / L_g.
- Optional: backtracking line search if L_g is not well estimated.

Stopping check for B-step:
- ||B^{k+1} - B||_F / max(1, ||B||_F) <= tol_B.

### Outer Stopping Criteria
Stop alternating when all are satisfied:
- relative objective change: |f^{k+1} - f^k| / max(1, |f^k|) <= tol_obj,
- coupling gap: ||A^{k+1} - B^{k+1}||_F / max(1, ||A^{k+1}||_F) <= tol_gap,
- orthogonality violation: ||(A^{k+1})^T A^{k+1} - I_r||_F <= tol_orth.

## Initialization
- A^0: top r eigenvectors of Sigma_hat (standard PCA), orthonormalized.
- B^0: A^0 or soft(A^0, lambda1 / rho) as a sparsified warm start.

## Hyperparameters
- lambda1: sparsity strength.
- lambda2: graph smoothness strength.
- rho: coupling strength between A and B.
- eta_A: A-step step size (fixed).
- eta_B: B-step step size (1 / L_g or backtracking).
- tol_A, tol_B, tol_obj, tol_gap, tol_orth.
- max_iters and max_inner_iters (if inner iterations are used).

## Implementation Notes
- Precompute Sigma_hat and L.
- Estimate ||L||_2 by power iteration; reuse across iterations.
- Track objective values and orthogonality error each iteration.
- Use a deterministic random seed for reproducibility in synthetic runs.
- Default to one A-step and one B-step per outer iteration (no inner loops in v1).

## Non-Primary Variants (Deferred)
- Primal-dual updates for the B-step.
- Semismooth Newton acceleration.
- Direct Stiefel formulation solvers.

These are not part of the paper-1 implementation target.
