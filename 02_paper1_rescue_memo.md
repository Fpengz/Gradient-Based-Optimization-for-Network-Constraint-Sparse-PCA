# Paper-1 Rescue Memo: Graph-Smooth Sparse Orthogonal PCA

## Executive diagnosis
The current draft overreaches relative to the program guide. Scope is too broad, the solver is underspecified, theory statements overclaim relative to what is proved, and the experimental plan reads as a wishlist rather than an execution plan. This combination invites implementation drift and makes the paper vulnerable to indefensible claims. Paper 1 must be narrowed to a single primary formulation, a single concrete solver path, a synthetic-only evaluation protocol, and a modest theory roadmap.

## Frozen Paper-1 scope
Paper 1 is frozen to a split-variable formulation, an alternating solver with a manifold A-step and proximal-gradient B-step, and a synthetic-only evaluation protocol against strong baselines and mandatory ablations. Real-data evaluation is excluded. Theory is limited to convergence-to-stationarity style guarantees or a clearly stated roadmap with explicit assumptions.

## Primary claim and non-goals
Primary claim: We study sparse, graph-smooth, explicitly orthogonal multi-component PCA and evaluate when it improves support structure and interpretability relative to strong baselines.

Non-goals:
- joint graph learning
- strong statistical consistency theory
- broad optimization-theory novelty claims
- real-data evaluation
- universal superiority claims

## Revised contributions
- A PCA-specific split-variable objective that couples explicit orthogonality with sparse graph-smooth loadings.
- A concrete alternating solver with a Stiefel manifold A-step and proximal-gradient B-step, fully specified.
- A synthetic-first evaluation protocol with a complete baseline matrix and mandatory ablations that isolate graph regularization, sparsity, and joint orthogonality.

## Algorithm freeze
Objective:
- Minimize the split-variable formulation in 03_formulation.md, with A on the Stiefel manifold and B carrying sparsity and graph smoothness.

A-step:
- Euclidean gradient: G_A = -2 Sigma_hat A + rho (A - B).
- Riemannian gradient: Grad_A = G_A - A * sym(A^T G_A).
- Update: A_tilde = A - eta_A * Grad_A.
- Retraction: A^+ = qf(A_tilde) via thin QR.
- Step size: fixed eta_A (Armijo line search optional, not required for v1).
- Stop A-step if relative change <= tol_A.

B-step:
- Smooth part: g(B) = lambda2 tr(B^T L B) + (rho/2) ||A - B||_F^2.
- Nonsmooth part: h(B) = lambda1 ||B||_1.
- Gradient: Grad_B = 2 lambda2 L B + rho (B - A).
- Lipschitz estimate: L_g = 2 lambda2 ||L||_2 + rho.
- Update: B_tilde = B - eta_B * Grad_B.
- Prox: B^+ = soft(B_tilde, eta_B * lambda1).
- Step size: eta_B = 1 / L_g (backtracking optional).
- Stop B-step if relative change <= tol_B.

Outer stopping:
- relative objective change <= tol_obj
- coupling gap <= tol_gap
- orthogonality violation <= tol_orth

Initialization:
- A^0 from top r eigenvectors of Sigma_hat.
- B^0 = A^0 or soft(A^0, lambda1 / rho).

Hyperparameters:
- lambda1, lambda2, rho, eta_A, eta_B, tolerances, max iterations.

## Theory repair
Paper 1 does not claim full theoretical guarantees. The theory section is restricted to a modest roadmap and clear assumptions:
- L is positive semidefinite.
- Step sizes satisfy standard Lipschitz or backtracking conditions.
- Iterates remain in a bounded level set due to quadratic coupling.

Planned analysis targets:
- lower bounded objective
- sufficient decrease or monotone descent
- vanishing successive differences
- stationarity of accumulation points

Anything beyond this is deferred.

## Experiment plan
Synthetic-only, fixed protocol:

Graph families:
- chain graph
- 2D grid graph
- stochastic block model with 3 blocks, intra-p 0.2, inter-p 0.02
- random geometric graph (optional if budget allows)

Support regimes:
- connected support
- nearly connected support (one edge removed)
- disconnected support (two subgraphs)
- overlap regimes: disjoint supports and 50 percent shared support

Parameter ranges:
- n in {200, 500, 1000}
- p in {200, 500, 1000}
- r in {3, 5}
- sparsity s in {10, 20, 40}
- SNR in {0.5, 1, 2}
- graph corruption in {0, 0.1, 0.3}
- lambda1, lambda2 on a 5x5 log grid

Replicates:
- 20 per configuration

Baselines:
- PCA
- standard sparse PCA
- joint orthogonal sparse PCA baseline
- graph-regularized PCA (no explicit sparsity)
- graph + sparsity PCA baseline

Ablations:
- no graph term
- no sparsity term
- deflation instead of joint extraction
- graph corruption stress test

Metrics:
- explained variance
- reconstruction error
- support precision, recall, F1
- subspace distance (synthetic only)
- Laplacian energy
- orthogonality error
- runtime and iteration counts

Compute budget:
- CPU only, cap total runs by halving replicates if runtime exceeds 48 hours.

## Patch-ready rewritten paragraphs
Abstract:
- We study sparse, graph-smooth, explicitly orthogonal multi-component PCA through a split-variable formulation that couples an orthogonal loading matrix with a sparse graph-regularized proxy. We propose a concrete alternating solver consisting of a Stiefel manifold update for the orthogonal factor and a proximal-gradient update for the sparse graph-smooth factor. Our evaluation plan focuses on a synthetic-first protocol with strong baselines and ablations to isolate the roles of graph smoothness, sparsity, and joint orthogonality. This paper targets a narrow, defensible claim about when graph-smooth sparse orthogonal PCA improves structure and interpretability; broader theoretical guarantees and real-data validation are deferred.

Intro contribution paragraph:
- This paper makes three focused contributions. First, we define a PCA-specific split-variable objective that simultaneously enforces orthogonality and graph-smooth sparsity. Second, we specify a single alternating solver with explicit update rules for both the manifold and proximal steps. Third, we lay out a synthetic evaluation protocol with a complete baseline matrix and mandatory ablations that test the contribution of graph regularization, sparsity, and joint orthogonality. These contributions are intentionally narrow to keep Paper 1 executable and defensible.

Related work positioning paragraph:
- Graph-aware PCA and graph-regularized sparse PCA have been studied in multiple forms, and we do not claim novelty in combining graph smoothness with sparsity. The specific gap we target is the joint extraction of multiple explicitly orthogonal components under a split-variable formulation with a concrete, reproducible solver. Our work is positioned as a focused intersection of sparse PCA, graph-regularized PCA, and orthogonality-preserving optimization, rather than as a first instance of any single ingredient.

Algorithm overview paragraph:
- We solve the split-variable objective by alternating between two steps. The A-step updates an orthogonal loading matrix on the Stiefel manifold using a Riemannian gradient step followed by QR-based retraction. The B-step updates the sparse graph-smooth proxy with proximal gradient, treating the Laplacian quadratic and coupling term as the smooth part and the l1 penalty as the nonsmooth part. This single solver path is the primary algorithmic target for Paper 1; more advanced variants are deferred.

Theory framing paragraph:
- Our theoretical objective in Paper 1 is modest: to motivate the alternating method and outline a convergence-to-stationarity roadmap under standard smoothness and boundedness assumptions. We do not claim global optimality, statistical consistency, or broad new manifold optimization theory. Full theoretical guarantees are deferred to later work once the algorithm and experimental results are stabilized.

Experiments overview paragraph:
- We evaluate the method in a synthetic-first protocol designed to isolate when graph smoothness helps. The protocol fixes graph families, support regimes, parameter ranges, and a complete baseline matrix, and includes mandatory ablations such as no-graph, no-sparsity, deflation, and graph corruption. All results report variance metrics, support recovery, smoothness, orthogonality error, and runtime. Real-data evaluation is explicitly excluded from Paper 1.

## Immediate next implementation steps
- Implement the solver skeleton from 04_algorithm.md.
- Implement the synthetic data generator and graph constructors.
- Implement the baseline matrix from 05_baselines.md and track status in the experiment log.
- Implement experiment runner and metrics logging.
- Patch the manuscript with the rewritten paragraphs.

## Quality bar
- No claim exceeds the solver and synthetic protocol.
- Each contribution maps to an implementable method or defined experiment.
- Tense and tone do not imply completed results.
- Real-data claims do not appear.
- The algorithm is single-path and fully specified.
