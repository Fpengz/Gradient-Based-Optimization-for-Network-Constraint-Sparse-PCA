# 01_problem_scope.md

## Purpose
This document freezes the scope of paper 1 for the graph-smooth sparse orthogonal PCA project. It defines the claim, target gaps, non-goals, and implementation boundaries so that method development, experiments, and writing do not drift.

## Frozen Paper-1 Claim
We study sparse, graph-smooth, explicitly orthogonal multi-component PCA and evaluate when it improves support structure and interpretability relative to strong baselines.

## Target Gaps

### Primary Gap
Graph-aware sparse PCA pipelines often do not explicitly enforce simultaneous orthogonality across multiple components.

### Secondary Gap
Current extensions are often described too loosely at the optimization level; paper 1 targets a more defensible optimization framing than naive threshold-then-project approaches.

## What We Are Building
Paper 1 builds:
- a PCA-specific split-variable formulation,
- an alternating solver with manifold A-step and proximal-gradient B-step,
- a synthetic-first evaluation pipeline against strong baselines,
- and a manuscript positioned around sparse + graph-smooth + orthogonal multi-component extraction.

## What We Are Not Building
Paper 1 does not aim to deliver:
- joint graph learning,
- strong statistical consistency theory,
- broad optimization-theory novelty claims,
- many real-data studies,
- or universal superiority claims.

## Scope Boundaries
The paper-1 method scope is frozen to:
- split-variable objective as the primary formulation,
- direct Stiefel formulation only as a secondary reference variant,
- proximal-gradient B-step as the version-1 solver choice,
- synthetic protocol as the primary validation environment,
- theory limited to a convergence-to-stationarity roadmap and modest supporting analysis.

## Success Criteria
Paper 1 is considered successful if it produces:
- a stable implementation of the split-variable method,
- preserved orthogonality across components,
- a complete and credible baseline matrix,
- synthetic results showing when graph smoothness helps,
- and ablations that isolate the roles of graph regularization, sparsity, and joint orthogonality.

## Decision Lock
Unless explicitly revised, implementation, experiments, and manuscript drafting should assume this scope is frozen.
